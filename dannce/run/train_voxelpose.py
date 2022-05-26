"""Handle training and prediction for DANNCE and COM networks."""
import numpy as np
import os
from datetime import datetime
from typing import Dict, Text
import os, psutil
from tqdm import tqdm

import torch
from dannce.engine.data import serve_data_DANNCE, dataset, generator, processing
from dannce.interface import make_folder
from dannce.engine.models.backbone import get_backbone
from dannce.engine.models.voxelpose import FeatureDANNCE
from dannce.engine.trainer.voxelpose_trainer import VoxelPoseTrainer
from dannce.engine.logging.logger import setup_logging, get_logger

process = psutil.Process(os.getpid())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def load_data_into_mem(params, logger, partition, n_cams, generator, train=True):
    n_samples = len(partition["train_sampleIDs"]) if train else len(partition["valid_sampleIDs"]) 
    message = "Loading training data into memory" if train else "Loading validation data into memory"

    # initialize
    X = torch.empty((n_samples, n_cams, params["chan_num"], 512, 512), dtype=torch.float32)
    X_grid = torch.empty((n_samples, params["nvox"] ** 3, 3), dtype=torch.float32)
    y = torch.empty((n_samples, 3, params["n_channels_out"]), dtype=torch.float32)
    cameras = []
    # load data from generator
    for i in tqdm(range(n_samples)):
        # print(i, end='\r')
        rr = generator.__getitem__(i)

        X[i] = rr[0][0]
        X_grid[i] = rr[0][1]
        y[i] = rr[1][0]
        cameras.append(rr[0][2][0])
    
    return X, X_grid, y, cameras

def load_data2d_into_mem(params, logger, partition, n_cams, generator, train=True):
    n_samples = len(partition["train_sampleIDs"]) if train else len(partition["valid_sampleIDs"]) 
    message = "Loading training data into memory" if train else "Loading validation data into memory"

    # initialize
    X = torch.empty((n_samples, n_cams, params["chan_num"], 512, 512), dtype=torch.float32)
    y = torch.empty((n_samples, n_cams, 2, params["n_channels_out"]), dtype=torch.float32)
    
    # load data from generator
    for i in tqdm(range(n_samples)):
        # print(i, end='\r')
        rr = generator.__getitem__(i)

        X[i] = rr[0][0]
        y[i] = rr[1][1]
    
    X = X.reshape(-1, *X.shape[2:])
    y = y.reshape(-1, *y.shape[2:])
    return X, y

def collate_fn(items):
    X = torch.stack([item[0] for item in items], dim=0) #[bs, 6, 3, H, W]
    grid = torch.stack([item[1] for item in items], dim=0) #[bs, gridsize**3, 3]
    camera = [item[2] for item in items]
    
    target = torch.stack([item[3] for item in items], dim=0) #[bs, 3, n_joints]

    return X, grid, camera, target

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
    # turn off currently unavailable features
    params["multi_mode"] = False
    params["depth"] = False
    params["n_views"] = int(params["n_views"])

    # Make the training directory if it does not exist.
    make_folder("dannce_train_dir", params)

    # setup logger
    setup_logging(params["dannce_train_dir"])
    logger = get_logger("training.log", verbosity=2)

    # load in necessary exp & data information
    exps = params["exp"]
    num_experiments = len(exps)
    params["experiment"] = {}

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
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["new_n_channels_out"],
        "out_scale": params["sigma"],
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "vmin": params["vmin"],
        "vmax": params["vmax"],
        "nvox": params["nvox"],
        "interp": params["interp"],
        "depth": params["depth"],
        "mode": outmode,
        "camnames": camnames,
        "immode": params["immode"],
        "shuffle": False,  # will shuffle later
        "rotation": False,  # will rotate later if desired
        "vidreaders": vids,
        "distort": True,
        "crop_im": False,
        "chunks": total_chunks,
        "mono": params["mono"],
        "mirror": params["mirror"],
    }

    genfunc = generator.MultiviewImageGenerator

    # Used to initialize arrays for mono, and also in *frommem (the final generator)
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    spec_params = {
        "channel_combo":  params["channel_combo"],
        "expval": params["expval"],
    }

    valid_params = {**base_params, **spec_params}

    # Setup a generator that will read videos and labels
    train_gen_params = [partition["train_sampleIDs"],
                        datadict,
                        datadict_3d,
                        cameras,
                        partition["train_sampleIDs"],
                        com3d_dict,
                        tifdirs]
    valid_gen_params = [partition["valid_sampleIDs"],
                        datadict,
                        datadict_3d,
                        cameras,
                        partition["valid_sampleIDs"],
                        com3d_dict,
                        tifdirs]

    train_generator = genfunc(*train_gen_params, **valid_params)
    valid_generator = genfunc(*valid_gen_params, **valid_params)
    
    # load everything into memory
    X_train, X_train_grid, y_train, cameras_train = load_data_into_mem(params, logger, partition, n_cams, train_generator, train=True)
    X_valid, X_valid_grid, y_valid, cameras_valid = load_data_into_mem(params, logger, partition, n_cams, valid_generator, train=False)
    
    if params["debug_volume_tifdir"] is not None:
        # When this option is toggled in the config, rather than
        # training, the image volumes are dumped to tif stacks.
        # This can be used for debugging problems with calibration or COM estimation
        processing.save_volumes_into_tif(params, params["debug_volume_tifdir"], X_train, partition["train_sampleIDs"], n_cams, logger)
        return

    # initialize datasets and dataloaders
    train_generator = dataset.MultiViewImageDataset(
        images=X_train,
        grids=X_train_grid,
        labels_3d=y_train,
        cameras=cameras_train 
    )
    valid_generator = dataset.MultiViewImageDataset(
        images=X_valid,
        grids=X_valid_grid,
        labels_3d=y_valid,
        cameras=cameras_valid 
    )

    train_dataloader, valid_dataloader = setup_dataloaders(train_generator, valid_generator, params)
    
    # Build network
    logger.info("Initializing Network...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = FeatureDANNCE(n_cams=n_cams, output_channels=22, input_shape=params["nvox"], bottleneck_channels=6)
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

def train2d(params: Dict):
    """Train dannce network.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        Exception: Error if training mode is invalid.
    """
    # turn off currently unavailable features
    params["multi_mode"] = False
    params["depth"] = False
    params["n_views"] = int(params["n_views"])

    # Make the training directory if it does not exist.
    make_folder("dannce_train_dir", params)

    # setup logger
    setup_logging(params["dannce_train_dir"])
    logger = get_logger("training.log", verbosity=2)

    # load in necessary exp & data information
    exps = params["exp"]
    num_experiments = len(exps)
    params["experiment"] = {}

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
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["new_n_channels_out"],
        "out_scale": params["sigma"],
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "vmin": params["vmin"],
        "vmax": params["vmax"],
        "nvox": params["nvox"],
        "interp": params["interp"],
        "depth": params["depth"],
        "mode": outmode,
        "camnames": camnames,
        "immode": params["immode"],
        "shuffle": False,  # will shuffle later
        "rotation": False,  # will rotate later if desired
        "vidreaders": vids,
        "distort": True,
        "crop_im": False,
        "chunks": total_chunks,
        "mono": params["mono"],
        "mirror": params["mirror"],
    }

    genfunc = generator.MultiviewImageGenerator

    # Used to initialize arrays for mono, and also in *frommem (the final generator)
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    spec_params = {
        "channel_combo":  params["channel_combo"],
        "expval": params["expval"],
    }

    valid_params = {**base_params, **spec_params}

    # Setup a generator that will read videos and labels
    train_gen_params = [partition["train_sampleIDs"],
                        datadict,
                        datadict_3d,
                        cameras,
                        partition["train_sampleIDs"],
                        com3d_dict,
                        tifdirs]
    valid_gen_params = [partition["valid_sampleIDs"],
                        datadict,
                        datadict_3d,
                        cameras,
                        partition["valid_sampleIDs"],
                        com3d_dict,
                        tifdirs]

    train_generator = genfunc(*train_gen_params, **valid_params)
    valid_generator = genfunc(*valid_gen_params, **valid_params)
    
    # load everything into memory
    X_train, y_train = load_data2d_into_mem(params, logger, partition, n_cams, train_generator, train=True)
    X_valid, y_valid = load_data2d_into_mem(params, logger, partition, n_cams, valid_generator, train=False)
    
    if params["debug_volume_tifdir"] is not None:
        # When this option is toggled in the config, rather than
        # training, the image volumes are dumped to tif stacks.
        # This can be used for debugging problems with calibration or COM estimation
        processing.save_train_images(params["debug_volume_tifdir"], X_train, y_train)
        return

    # initialize datasets and dataloaders
    train_generator = dataset.ImageDataset(
        images=X_train,
        labels=y_train,
    )
    valid_generator = dataset.ImageDataset(
        images=X_valid,
        labels=y_valid,
    )

    train_dataloader, valid_dataloader = setup_dataloaders(train_generator, valid_generator, params)
    
    # Build network
    logger.info("Initializing Network...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = FeatureDANNCE(n_cams=n_cams, output_channels=22, input_shape=params["nvox"], bottleneck_channels=6).backbone

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