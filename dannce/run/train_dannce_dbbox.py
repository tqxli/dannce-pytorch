"""Handle training and prediction for DANNCE and COM networks."""
import numpy as np
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, Text
import psutil

import torch

from dannce.engine.data import serve_data_DANNCE, dataset, generator, processing
from dannce.interface import make_folder
import dannce.config as config
import dannce.engine.inference as inference
from dannce.engine.models.nets import initialize_train, initialize_model
from dannce.engine.models.segmentation import get_instance_segmentation_model
from dannce.engine.trainer.dannce_trainer import DannceTrainer, AutoEncoderTrainer
from dannce.config import print_and_set
from dannce.engine.logging.logger import setup_logging, get_logger
from dannce.engine.data.processing import _DEFAULT_SEG_MODEL

process = psutil.Process(os.getpid())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def train(params: Dict):
    """Train dannce network.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        Exception: Error if training mode is invalid.
    """
    (
        params,
        base_params,
        shared_args,
        shared_args_train,
        shared_args_valid
    ) = config.setup_train(params)

    # Make the training directory if it does not exist.
    make_folder("dannce_train_dir", params)

    # setup logger
    setup_logging(params["dannce_train_dir"])
    logger = get_logger("training.log", verbosity=2)

    # set GPU ID
    # Temporarily commented out to test on dsplus gpu
    # if not params["multi_gpu_train"]:
    # os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    # deploy GPU devices
    assert torch.cuda.is_available(), "No available GPU device."
    
    if params["multi_gpu_train"]:
        params["gpu_id"] = list(range(torch.cuda.device_count()))
        device = torch.device("cuda") # use all available GPUs
    else:
        params["gpu_id"] = [0]
        device = torch.device("cuda:0")
    logger.info("***Use {} GPU for training.***".format(params["gpu_id"]))
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

    # Additionally, to keep videos unique across experiments, need to add
    # experiment labels in other places. E.g. experiment 0 CameraE's "camname"
    # Becomes 0_CameraE. *NOTE* This function modified camnames in place
    # to add the appropriate experiment ID
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, num_experiments, camnames, cameras
    )

    # Dump the params into file for reproducibility
    processing.save_params_pickle(params)
    # logger.info(params)

    # Setup additional variables for later use
    n_cams = len(camnames[0])
    tifdirs = []  # Training from single images not yet supported in this demo

    # Two possible data loading schemes:
    # (i) Directly use pre-saved npy volumes
    # (ii) Load images from video files into memory and generate the samples needed for training
    vid_exps = np.arange(num_experiments)
    if params["use_npy"]:
        npydir, missing_npydir, missing_samples = serve_data_DANNCE.examine_npy_training(params, samples)

        if len(missing_samples) != 0:
            logger.info("{} npy files for experiments {} are missing.".format(len(missing_samples), list(missing_npydir.keys())))
        else:
            logger.info("No missing npy files. Ready for training.")
    
    # initialize needed videos
    vids = processing.initialize_all_vids(params, datadict, vid_exps, pathonly=True)

    # make train/valid splits
    partition = processing.make_data_splits(
        samples, params, params["dannce_train_dir"], num_experiments, 
        temporal_chunks=temporal_chunks)
    if params["social_training"]:
        partition, _ = processing.resplit_social(partition)

    logger.info("\nTRAIN:VALIDATION SPLIT = {}:{}\n".format(len(partition["train_sampleIDs"]), len(partition["valid_sampleIDs"])))

    base_params = {
        **base_params,
        "camnames": camnames,
        "vidreaders": vids,
        "chunks": total_chunks,
    }

    if params["social_training"]:
        genfunc = generator.DataGenerator_3Dconv_social
    else:
        genfunc = generator.DataGenerator_3Dconv
    
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
    
    valid_params_sil = deepcopy(valid_params)
    valid_params_sil["vidreaders"] = vids #vids_sil

    # ensure voxel values stay positive, allowing conversion to binary below
    valid_params_sil["norm_im"] = False
    valid_params_sil["expval"] = True

    # prepare segmentation model
    segmentation_model = get_instance_segmentation_model(num_classes=2)
    checkpoints = torch.load(_DEFAULT_SEG_MODEL)["state_dict"]
    segmentation_model.load_state_dict(checkpoints)
    segmentation_model.eval()
    segmentation_model = segmentation_model.to("cuda:0")

    train_generator_sil = genfunc(
        *train_gen_params,
        **valid_params_sil,
        segmentation_model=segmentation_model 
    )

    valid_generator_sil = genfunc(
        *valid_gen_params,
        **valid_params_sil,
        segmentation_model=segmentation_model
    )

    _, X_train_grid, y_train_aux = processing.load_volumes_into_mem(
        params, logger, partition, n_cams, train_generator_sil, 
        train=True, silhouette=True, social=params["social_training"]
    )
    _, X_valid_grid, y_valid_aux = processing.load_volumes_into_mem(
        params, logger, partition, n_cams, valid_generator_sil, 
        train=False, silhouette=True, social=params["social_training"]
    )

    # find tighter bounding box
    mask3d = np.concatenate((y_train_aux, y_valid_aux), axis=0)
    grids = np.concatenate((X_train_grid, X_valid_grid), axis=0)
    new_com3ds, new_dims = processing.compute_bbox_from_3dmask(mask3d, grids)

    if segmentation_model is not None:
        segmentation_model = segmentation_model.cpu()
        del segmentation_model
        torch.cuda.empty_cache()

    com3d_dict, dim_dict = processing.create_new_labels(partition, com3d_dict, new_com3ds, new_dims, params)
    del mask3d, grids, X_train_grid, X_valid_grid, y_train_aux, y_valid_aux

    # now generate volumes
    genfunc = generator.DataGenerator_Dynamic

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

    train_generator = genfunc(*train_gen_params, **valid_params, dim_dict=dim_dict)
    valid_generator = genfunc(*valid_gen_params, **valid_params, dim_dict=dim_dict)
    
    # load everything into memory
    X_train, X_train_grid, y_train = processing.load_volumes_into_mem(
        params, logger, partition, n_cams, train_generator, train=True, social=False
    )
    X_valid, X_valid_grid, y_valid = processing.load_volumes_into_mem(
        params, logger, partition, n_cams, valid_generator, train=False, social=False
    )

    del train_generator, valid_generator

    if params["debug_volume_tifdir"] is not None:
        # When this option is toggled in the config, rather than
        # training, the image volumes are dumped to tif stacks.
        # This can be used for debugging problems with calibration or COM estimation
        processing.save_volumes_into_tif(params, params["debug_volume_tifdir"], X_train, partition["train_sampleIDs"], n_cams, logger)
        return
            
    # if (not params["use_npy"]) and (params["social_training"]):
    #     X_train, X_train_grid, y_train, y_train_aux = processing.align_social_data(X_train, X_train_grid, y_train, y_train_aux)
    #     X_valid, X_valid_grid, y_valid, y_valid_aux = processing.align_social_data(X_valid, X_valid_grid, y_valid, y_valid_aux)
    # processing.save_visual_hull(y_train_aux, partition["train_sampleIDs"])
    # return

    # For AVG+MAX training, need to update the expval flag in the generators
    # and re-generate the 3D training targets
    # TODO: Add code to infer_params
    if params["avg+max"] is not None and params["use_silhouette"]:
        print("******Cannot combine AVG+MAX with silhouette - Using ONLY silhouette*******")

    elif params["avg+max"] is not None:
        y_train_aux, y_valid_aux = processing.initAvgMax(
            y_train, y_valid, X_train_grid, X_valid_grid, params
        )

    # # We apply data augmentation with another data generator class
    genfunc = dataset.PoseDatasetFromMem
    args_train = {
        "list_IDs": np.arange(len(partition["train_sampleIDs"])),
        "data": X_train,
        "labels": y_train,
    }
    args_train = {
                    **args_train,
                    **shared_args_train,
                    **shared_args,
                    "xgrid": X_train_grid,
                    "aux_labels": None,
                    "temporal_chunk_list": partition["train_chunks"] if params["use_temporal"] else None,
                    }

    args_valid = {
        "list_IDs": np.arange(len(partition["valid_sampleIDs"])),
        "data": X_valid,
        "labels": y_valid,
        "aux_labels": None
    }
    args_valid = {
        **args_valid,
        **shared_args_valid,
        **shared_args,
        "xgrid": X_valid_grid,
        "temporal_chunk_list": partition["valid_chunks"] if params["use_temporal"] else None
    }
    
    # if params["social_training"]:
    #     args_train = {**args_train, "pairs": pairs["train_pairs"]}
    #     args_valid = {**args_valid, "pairs": pairs["valid_pairs"]}

    # initialize datasets and dataloaders
    train_generator = genfunc(**args_train)
    valid_generator = genfunc(**args_valid)

    if params["debug_train_volume_tifdir"] is not None:
        processing.save_train_volumes(
            params, 
            params["debug_train_volume_tifdir"], 
            train_generator,
            n_cams,
        )
        return
    params["social_training"] = False
    train_dataloader, valid_dataloader = serve_data_DANNCE.setup_dataloaders(train_generator, valid_generator, params)
    
    # Build network
    logger.info("Initializing Network...")
    model, optimizer, lr_scheduler = initialize_train(params, n_cams, device, logger)
    logger.info("COMPLETE\n")

    # set up trainer
    trainer_class = AutoEncoderTrainer if "ReconstructionLoss" in params["loss"].keys() else DannceTrainer
    trainer = trainer_class(
        params=params,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device,
        logger=logger,
        visualize_batch=False,
        lr_scheduler=lr_scheduler
    )

    trainer.train()