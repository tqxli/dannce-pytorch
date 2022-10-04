"""Handle training and prediction for DANNCE and COM networks."""
import numpy as np
import os
import time
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
from dannce.config import print_and_set

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

        # the cropped image is [512, 512], resized to [256, 256]
        # but the heatmaps are even smaller smaller (64x64), resize all here
        cam = rr[0][2][0]
        for c in cam:
            c.update_after_resize(image_shape=[512, 512], new_image_shape=[64, 64])

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


def predict(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    make_folder("dannce_predict_dir", params)

    setup_logging(params["dannce_predict_dir"])
    logger = get_logger("training.log", verbosity=2)
    
    # Because CUDA_VISBILE_DEVICES is already set to a single GPU, the gpu_id here should be "0"
    device = "cuda:0"
    params, valid_params = config.setup_predict(params)
    params["return_full2d"] = True

    samples = []
    datadict = {}
    datadict_3d = {}
    com3d_dict = {}   
    cameras = {}     
    camnames = {}

    num_experiments = len(params["experiment"])
    for e in range(num_experiments):
        (
            params["experiment"][e],
            samples_,
            datadict_,
            datadict_3d_,
            cameras_,
            com3d_dict_,
            _
        ) = processing.do_COM_load(
            params["experiment"][e],
            params["experiment"][e],
            e,
            params,
            training=False,
        )

        # Write 3D COM to file. This might be different from the input com3d file
        # if arena thresholding was applied.
        if e == 0:
            processing.write_com_file(params, samples_, com3d_dict_)


        (samples, datadict, datadict_3d, com3d_dict, _) = serve_data_DANNCE.add_experiment(
            e,
            samples,
            datadict,
            datadict_3d,
            com3d_dict,
            samples_,
            datadict_,
            datadict_3d_,
            com3d_dict_,
        )

        cameras[e] = cameras_
        camnames[e] = params["experiment"][e]["camnames"]

    # Need a '0' experiment ID to work with processing functions.
    # *NOTE* This function modified camnames in place
    # to add the appropriate experiment ID
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, num_experiments, camnames, cameras, dannce_prediction=True
    )

    samples = np.array(samples)

    # Initialize video dictionary. paths to videos only.
    # TODO: Remove this immode option if we decide not
    # to support tifs
    if params["immode"] == "vid":
        vids = {}
        for e in range(num_experiments):
            vids = processing.initialize_vids(params, datadict, 0, vids, pathonly=True)
    
    # Parameters
    valid_params = {
        **valid_params,
        "camnames": camnames,
        "vidreaders": vids,
        "chunks": params["chunks"],
    }

    # Datasets
    valid_inds = np.arange(len(samples))
    partition = {"valid_sampleIDs": samples[valid_inds]}

    # TODO: Remove tifdirs arguments, which are deprecated
    tifdirs = []

    genfunc = generator.MultiviewImageGenerator
    valid_gen_params = {
        "list_IDs": partition["valid_sampleIDs"],
        "labels": datadict,
        "labels_3d": datadict_3d,
        "camera_params": cameras,
        "clusterIDs": partition["valid_sampleIDs"],
        "com3d": com3d_dict,
        "tifdirs": tifdirs
    }
    predict_generator = genfunc(**valid_gen_params, **valid_params)

    print("Initializing Network...")
    model = VoxelPose(params["n_channels_out"], params, logger)
    model = model.to(device)
    model.load_state_dict(torch.load(params["dannce_predict_model"])['state_dict'])
    model.eval()

    if params["maxbatch"] != "max" and params["maxbatch"] > len(predict_generator):
        print(
            "Maxbatch was set to a larger number of matches than exist in the video. Truncating"
        )
        print_and_set(params, "maxbatch", len(predict_generator))

    if params["maxbatch"] == "max":
        print_and_set(params, "maxbatch", len(predict_generator))

    if params["write_npy"] is not None:
        # Instead of running inference, generate all samples
        # from valid_generator and save them to npy files. Useful
        # for working with large datasets (such as Rat 7M) because
        # .npy files can be loaded in quickly with random access
        # during training.
        print("Writing samples to .npy files")
        processing.write_npy(params["write_npy"], predict_generator)
        return
    
    end_time = time.time()
    save_data= {}
    start_ind = params["start_batch"]
    end_ind = params["maxbatch"]

    pbar = tqdm(range(start_ind, end_ind))
    for idx, i in enumerate(pbar):
        if (i - start_ind) % 1000 == 0 and i != start_ind:
            print("Saving checkpoint at {}th batch".format(i))
            p_n = processing.savedata_expval(
                params["dannce_predict_dir"] + "/save_data_AVG.mat",
                params,
                write=True,
                data=save_data,
                tcoord=False,
                num_markers=params["n_channels_out"],
                pmax=False,
            )
        
        rr = predict_generator.__getitem__(idx)
        images = rr[0][0] #[BS, 6, 3, 256, 256]
        grids = rr[0][1] #[BS, nvox**3, 3]
        cam = rr[0][2][0] #
        cameras = []
        for c in cam:
            c.update_after_resize(image_shape=[512, 512], new_image_shape=[64, 64])
            cameras.append(cam)

        images, grids = images.to(device).float(), grids.to(device).float()
        pred = model(images, grids, cameras)[1]
        pred = pred.detach().cpu().numpy()

        for j in range(pred.shape[0]):
            sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]
            save_data[idx * pred.shape[0] + j] = {
                "pred_coord": pred[j],
                "sampleID": sampleID,
            }

        if params["save_tag"] is not None:
            path = os.path.join(
                params["dannce_predict_dir"],
                "save_data_AVG%d.mat" % (params["save_tag"]),
            )
        else:
            path = os.path.join(params["dannce_predict_dir"], "save_data_AVG.mat")
        
        p_n = processing.savedata_expval(
            path,
            params,
            write=True,
            data=save_data,
            tcoord=False,
            num_markers=params["n_channels_out"],
            pmax=False,
        )

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