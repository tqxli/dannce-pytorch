"""Handle training and prediction for DANNCE and COM networks."""
import numpy as np
import os
from datetime import datetime
from typing import Dict, Text
import os, psutil
from tqdm import tqdm

import torch
import dannce.config as config
from dannce.run_utils import *
from dannce.engine.data import serve_data_DANNCE, dataset, generator, processing
from dannce.interface import make_folder
from dannce.engine.models.voxelpose import VoxelPose
from dannce.engine.trainer.voxelpose_trainer import VoxelPoseTrainer
from dannce.engine.logging.logger import setup_logging, get_logger

process = psutil.Process(os.getpid())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def generate_Gaussian_target(all_labels, heatmap_size=[64, 64], ds_fac=4, sigma=2):
    all_targets = []
    for labels in all_labels[0]:
        (x_coord, y_coord) = np.meshgrid(
            np.arange(heatmap_size[1]), np.arange(heatmap_size[0])
        )
        
        targets = []
        for joint in range(labels.shape[-1]):
            if np.isnan(labels[:, joint]).sum() == 0:
                target = np.exp(
                        -(
                            (y_coord - labels[1, joint] // ds_fac) ** 2
                            + (x_coord - labels[0, joint] // ds_fac) ** 2
                        )
                        / (2 * sigma ** 2)
                    )
            else:
                target = np.zeros((heatmap_size[1], heatmap_size[0]))
            targets.append(target)
        # crop out and keep the max to be 1 might still work...
        targets = np.stack(targets, axis=0)
        targets = torch.from_numpy(targets).float()

        all_targets.append(targets)

    targets = np.stack(all_targets, axis=0)

    return torch.from_numpy(targets).unsqueeze(0)

def load_data_into_mem(params, logger, partition, n_cams, generator, train=True):
    n_samples = len(partition["train_sampleIDs"]) if train else len(partition["valid_sampleIDs"]) 
    message = "Loading training data into memory" if train else "Loading validation data into memory"

    # initialize
    X = torch.empty((n_samples, n_cams, params["chan_num"], 256, 256), dtype=torch.float32)
    y2d = torch.empty((n_samples, n_cams, params["n_channels_out"], 64, 64), dtype=torch.float32)
    X_grid = torch.empty((n_samples, params["nvox"] ** 3, 3), dtype=torch.float32)
    y3d = torch.empty((n_samples, 3, params["n_channels_out"]), dtype=torch.float32)
    cameras = []
    # load data from generator
    for i in tqdm(range(n_samples)):
        # print(i, end='\r')
        rr = generator.__getitem__(i)

        X[i] = rr[0][0]
        y2d[i] = generate_Gaussian_target(rr[1][1].numpy())
        X_grid[i] = rr[0][1]
        y3d[i] = rr[1][0]

        # since the heatmaps are even smaller (64x64), resize here
        cam = rr[0][2][0]
        for c in cam:
            c.update_after_resize(image_shape=[256, 256], new_image_shape=[64, 64])

        cameras.append(cam)
    
    return X, y2d, X_grid, y3d, cameras

def collate_fn(items):
    ims = torch.stack([item[0] for item in items], dim=0) #[bs, 6, 3, H, W]
    y2d_gaussian = torch.stack([item[1] for item in items], dim=0) #[bs, 6, H, W]
    camera = [item[2] for item in items]
    grid = torch.stack([item[3] for item in items], dim=0) #[bs, gridsize**3, 3]
    y3d = torch.stack([item[4] for item in items], dim=0) #[bs, 3, n_joints]

    return ims, y2d_gaussian, grid, camera, y3d

def setup_dataloaders(train_dataset, valid_dataset, params):
    valid_batch_size = params["batch_size"] 
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=valid_batch_size, shuffle=True, collate_fn=collate_fn,
        # num_workers=valid_batch_size
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, valid_batch_size, shuffle=False, collate_fn=collate_fn,
        # num_workers=valid_batch_size
    )
    return train_dataloader, valid_dataloader

def train(params: Dict):
    """Train dannce network.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        Exception: Error if training mode is invalid.
    """
    params, base_params, shared_args, shared_args_train, shared_args_valid = config.setup_train(params)

    # Make the training directory if it does not exist.
    make_folder("dannce_train_dir", params)

    # setup logger
    setup_logging(params["dannce_train_dir"])
    logger = get_logger("training.log", verbosity=2)

    assert torch.cuda.is_available(), "No available GPU device."
    params["gpu_id"] = [0]
    device = torch.device("cuda")
    logger.info("***Use {} GPU for training.***".format(params["gpu_id"]))

    # fix random seed if specified
    if params["random_seed"] is not None:
        set_random_seed(params["random_seed"])
        logger.info("***Fix random seed as {}***".format(params["random_seed"]))

    # load in necessary exp & data information
    exps = params["exp"]
    num_experiments = len(exps)
    params["experiment"] = {}
    params["return_full2d"] = True

    (
        samples, 
        datadict, datadict_3d, com3d_dict, 
        cameras, camnames,
        total_chunks,
        temporal_chunks
    ) = processing.load_all_exps(params, logger)
    
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, num_experiments, camnames, cameras
    )

    # Dump the params into file for reproducibility
    processing.save_params_pickle(params)
    logger.info(params)

    # Setup additional variables for later use
    n_cams = len(camnames[0])
    dannce_train_dir = params["dannce_train_dir"]
    outmode = "coordinates" if params["expval"] else "3dprob"
    tifdirs = []  # Training from single images not yet supported in this demo

    vid_exps = np.arange(num_experiments)
    # initialize needed videos
    vids = processing.initialize_all_vids(params, datadict, vid_exps, pathonly=True)

    # make train/valid splits
    partition = processing.make_data_splits(
        samples, params, dannce_train_dir, num_experiments, 
        temporal_chunks=temporal_chunks)

    logger.info("\nTRAIN:VALIDATION SPLIT = {}:{}\n".format(len(partition["train_sampleIDs"]), len(partition["valid_sampleIDs"])))

    base_params = {
        **base_params,
        "camnames": camnames,
        "vidreaders": vids,
        "chunks": total_chunks,
    }
    spec_params = {
        "channel_combo":  params["channel_combo"],
        "expval": params["expval"],
    }
    valid_params = {**base_params, **spec_params}

    genfunc = generator.MultiviewImageGenerator

    train_gen_params = {
        "list_IDs": partition["train_sampleIDs"],
        "labels": datadict,
        "labels_3d": datadict_3d,
        "camera_params": cameras,
        "clusterIDs": partition["train_sampleIDs"],
        "com3d": com3d_dict,
        "tifdirs": tifdirs
    }
    valid_gen_params = {
        "list_IDs": partition["valid_sampleIDs"],
        "labels": datadict,
        "labels_3d": datadict_3d,
        "camera_params": cameras,
        "clusterIDs": partition["valid_sampleIDs"],
        "com3d": com3d_dict,
        "tifdirs": tifdirs
    }

    train_generator = genfunc(**train_gen_params, **valid_params)
    valid_generator = genfunc(**valid_gen_params, **valid_params)
    
    # load everything into memory
    X_train, y2d_train, X_train_grid, y3d_train, cameras_train = load_data_into_mem(params, logger, partition, n_cams, train_generator, train=True)
    X_valid, y2d_valid, X_valid_grid, y3d_valid, cameras_valid = load_data_into_mem(params, logger, partition, n_cams, valid_generator, train=False)

    # initialize datasets and dataloaders
    train_generator = dataset.ImageDataset(
        images=X_train,
        labels=y2d_train,
        num_joints=params["n_channels_out"],
        return_Gaussian=False,
        train=True,
        grids=X_train_grid,
        labels_3d=y3d_train,
        cameras=cameras_train
    )
    valid_generator = dataset.ImageDataset(
        images=X_valid,
        labels=y2d_valid,
        num_joints=params["n_channels_out"],
        return_Gaussian=False,
        train=False,
        grids=X_valid_grid,
        labels_3d=y3d_valid,
        cameras=cameras_valid
    )

    train_dataloader, valid_dataloader = setup_dataloaders(train_generator, valid_generator, params)
    
    # Build network
    logger.info("Initializing Network...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # model = FeatureDANNCE(n_cams=n_cams, output_channels=22, input_shape=params["nvox"], bottleneck_channels=6)
    model = VoxelPose(params["n_channels_out"], params, logger)
    if params["custom_model"]["backbone_pretrained"] is not None:
        ckpt = torch.load(params["custom_model"]["backbone_pretrained"])["state_dict"]
        model.backbone.load_state_dict(ckpt)
        logger.info("Successfully load backbone checkpoint.")
    model = model.to(device)

    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_params, lr=params["lr"])

    logger.info("COMPLETE\n")

    # set up trainer
    trainer = VoxelPoseTrainer(
        params=params,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device,
        logger=logger,
        visualize_batch=False,
        lr_scheduler=None
    )

    trainer.train()

if __name__ == "__main__":
    import argparse
    from dannce import (
        _param_defaults_dannce,
        _param_defaults_shared,
    )
    from dannce.cli import parse_clargs, build_clarg_params

    parser = argparse.ArgumentParser(
        description="Dannce train CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_dannce})
    args = parse_clargs(parser, model_type="dannce", prediction=False)
    params = build_clarg_params(args, dannce_net=True, prediction=False)

    train(params)