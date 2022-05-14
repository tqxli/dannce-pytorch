"""Handle training and prediction for DANNCE and COM networks."""
from audioop import avg
from email.mime import base
import sys
from matplotlib.pyplot import axis
import numpy as np
import os
from copy import deepcopy
import scipy.io as sio
import imageio
import time
import gc
from datetime import datetime

from dannce.engine.data import serve_data_DANNCE, dataset, generator, processing, ops, io
from dannce.engine.data.processing import savedata_tomat, savedata_expval
from dannce import (
    _param_defaults_dannce,
    _param_defaults_shared,
    _param_defaults_com,
)
import dannce.engine.inference as inference
from typing import Dict, Text
import os, psutil

import torch
from dannce.engine.models.nets import initialize_train, initialize_model
from dannce.engine.models.segmentation import get_instance_segmentation_model
from dannce.engine.trainer.dannce_trainer import DannceTrainer, AutoEncoderTrainer
from dannce.config import print_and_set
from dannce.engine.logging.logger import setup_logging, get_logger

process = psutil.Process(os.getpid())

_DEFAULT_VIDDIR = "videos"
_DEFAULT_VIDDIR_SIL = "videos_sil"
_DEFAULT_COMSTRING = "COM"
_DEFAULT_COMFILENAME = "com3d.mat"
_DEFAULT_SEG_MODEL = 'weights/maskrcnn.pth'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def check_unrecognized_params(params: Dict):
    """Check for invalid keys in the params dict against param defaults.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        ValueError: Error if there are unrecognized keys in the configs.
    """
    # Check if key in any of the defaults
    invalid_keys = []
    for key in params:
        in_com = key in _param_defaults_com
        in_dannce = key in _param_defaults_dannce
        in_shared = key in _param_defaults_shared
        if not (in_com or in_dannce or in_shared):
            invalid_keys.append(key)

    # If there are any keys that are invalid, throw an error and print them out
    if len(invalid_keys) > 0:
        invalid_key_msg = [" %s," % key for key in invalid_keys]
        msg = "Unrecognized keys in the configs: %s" % "".join(invalid_key_msg)
        raise ValueError(msg)


def build_params(base_config: Text, dannce_net: bool):
    """Build parameters dictionary from base config and io.yaml

    Args:
        base_config (Text): Path to base configuration .yaml.
        dannce_net (bool): If True, use dannce net defaults.

    Returns:
        Dict: Parameters dictionary.
    """
    base_params = processing.read_config(base_config)
    base_params = processing.make_paths_safe(base_params)
    params = processing.read_config(base_params["io_config"])
    params = processing.make_paths_safe(params)
    params = processing.inherit_config(params, base_params, list(base_params.keys()))
    check_unrecognized_params(params)
    return params


def make_folder(key: Text, params: Dict):
    """Make the prediction or training directories.

    Args:
        key (Text): Folder descriptor.
        params (Dict): Parameters dictionary.

    Raises:
        ValueError: Error if key is not defined.
    """
    # would be nice to automatically create training folder name

    # Make the prediction directory if it does not exist.
    if params[key] is not None:
        if not os.path.exists(params[key]):            
            os.makedirs(params[key])
    else:
        raise ValueError(key + " must be defined.")

    # if key == "dannce_train_dir":
    #     curr_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    #     new_dir = os.path.join(params[key], curr_time)
    #     os.makedirs(new_dir)
    #     params[key] = new_dir

def dannce_train(params: Dict):
    """Train dannce network.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        Exception: Error if training mode is invalid.
    """
    # turn off currently unavailable features
    params["multi_mode"] = False
    params["depth"] = False

    # Default to 6 views but a smaller number of views can be specified in the
    # DANNCE config. If the legnth of the camera files list is smaller than
    # n_views, relevant lists will be duplicated in order to match n_views, if
    # possible.
    params["n_views"] = int(params["n_views"])

    # turn on flags for losses that require changes in inputs
    if params["use_silhouette_in_volume"]:
        params["use_silhouette"] = True
        params["n_rand_views"] = None
    
    if "SilhouetteLoss" in params["loss"].keys():
        params["use_silhouette"] = True

    if "TemporalLoss" in params["loss"].keys():
        params["use_temporal"] = True
        params["temporal_chunk_size"] = params["loss"]["TemporalLoss"]["temporal_chunk_size"]
    
    if "PairRepulsionLoss" in params["loss"].keys():
        params["social_training"] = True

    # Make the training directory if it does not exist.
    make_folder("dannce_train_dir", params)

    # setup logger
    setup_logging(params["dannce_train_dir"])
    logger = get_logger("training.log", verbosity=2)

    # set GPU ID
    # Temporarily commented out to test on dsplus gpu
    # if not params["multi_gpu_train"]:
    # os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]

    # load in necessary exp & data information
    exps = params["exp"]
    num_experiments = len(exps)
    params["experiment"] = {}

    samples = [] # training sample identifiers
    datadict, datadict_3d, com3d_dict = {}, {}, {} # labels
    cameras, camnames = {}, {} # camera
    total_chunks = {} # video chunks
    temporal_chunks = {} # for temporal training

    for e, expdict in enumerate(exps):

        # load basic exp info
        exp = processing.load_expdict(params, e, expdict, _DEFAULT_VIDDIR, _DEFAULT_VIDDIR_SIL, logger)

        # load corresponding 2D & 3D labels, COMs
        (
            exp,
            samples_,
            datadict_,
            datadict_3d_,
            cameras_,
            com3d_dict_,
            temporal_chunks_
        ) = do_COM_load(exp, expdict, e, params)

        logger.info("Using {} samples total.".format(len(samples_)))

        (
            samples,
            datadict,
            datadict_3d,
            com3d_dict,
            temporal_chunks
        ) = serve_data_DANNCE.add_experiment(
            e,
            samples,
            datadict,
            datadict_3d,
            com3d_dict,
            samples_,
            datadict_,
            datadict_3d_,
            com3d_dict_,
            temporal_chunks,
            temporal_chunks_
        )

        cameras[e] = cameras_
        camnames[e] = exp["camnames"]
        logger.info("Using the following cameras: {}".format(camnames[e]))

        params["experiment"][e] = exp
        for name, chunk in exp["chunks"].items():
            total_chunks[name] = chunk

    # Additionally, to keep videos unique across experiments, need to add
    # experiment labels in other places. E.g. experiment 0 CameraE's "camname"
    # Becomes 0_CameraE. *NOTE* This function modified camnames in place
    # to add the appropriate experiment ID
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, num_experiments, camnames, cameras
    )

    samples = np.array(samples)

    # Dump the params into file for reproducibility
    processing.save_params_pickle(params)
    logger.info(params)

    # Setup additional variables for later use
    n_cams = len(camnames[0])
    dannce_train_dir = params["dannce_train_dir"]
    outmode = "coordinates" if params["expval"] else "3dprob"
    cam3_train = True if params["cam3_train"] else False # only use 3 cameras for training
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
        samples, params, dannce_train_dir, num_experiments, 
        temporal_chunks=temporal_chunks)
    if params["social_training"]:
        partition, pairs = processing.resplit_social(partition)

    logger.info("\nTRAIN:VALIDATION SPLIT = {}:{}\n".format(len(partition["train_sampleIDs"]), len(partition["valid_sampleIDs"])))

    segmentation_model = None

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

    if params["social_training"]:
        genfunc = generator.DataGenerator_3Dconv_social
    else:
        genfunc = generator.DataGenerator_3Dconv

    if params["use_npy"]:
        # mono conversion will happen from RGB npy files, and the generator
        # needs to be aware that the npy files contain RGB content
        params["chan_num"] = params["n_channels_in"]
        spec_params = {
            "channel_combo": None,
            "predict_flag": False,
            "norm_im": False,
            "expval": True,
        }

        valid_params = {**base_params, **spec_params}

        if len(missing_samples) != 0:
            npy_generator = genfunc(
                missing_samples,
                datadict,
                datadict_3d,
                cameras,
                missing_samples,
                com3d_dict,
                tifdirs,
                **valid_params
            )
            processing.save_volumes_into_npy(params, npy_generator, missing_npydir, samples, logger)
    else:
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
        X_train, X_train_grid, y_train = processing.load_volumes_into_mem(params, logger, partition, n_cams, train_generator, train=True, social=params["social_training"])
        X_valid, X_valid_grid, y_valid = processing.load_volumes_into_mem(params, logger, partition, n_cams, valid_generator, train=False, social=params["social_training"])

        if params["debug_volume_tifdir"] is not None:
            # When this option is toggled in the config, rather than
            # training, the image volumes are dumped to tif stacks.
            # This can be used for debugging problems with calibration or COM estimation
            processing.save_volumes_into_tif(params, params["debug_volume_tifdir"], X_train, partition["train_sampleIDs"], n_cams, logger)
            return

    # option for foreground animal segmentation
    y_train_aux, y_valid_aux = None, None
    if params["use_silhouette"]:
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

        if params["use_npy"]:
            npydir, missing_npydir, missing_samples = serve_data_DANNCE.examine_npy_training(params, samples, aux=True)

            if len(missing_samples) != 0:
                logger.info("{} aux npy files for experiments {} are missing.".format(len(missing_samples), list(missing_npydir.keys())))
            else:
                logger.info("No missing aux npy files. Ready for training.")

            if len(missing_samples) != 0:
                npy_generator = genfunc(
                    missing_samples,
                    datadict,
                    datadict_3d,
                    cameras,
                    missing_samples,
                    com3d_dict,
                    tifdirs,
                    segmentation_model=segmentation_model,
                    **valid_params_sil
                )
                processing.save_volumes_into_npy(params, npy_generator, missing_npydir, samples, logger, silhouette=True)
        else:
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

            _, _, y_train_aux = processing.load_volumes_into_mem(
                params, logger, partition, n_cams, train_generator_sil, 
                train=True, silhouette=True, social=params["social_training"]
            )
            _, _, y_valid_aux = processing.load_volumes_into_mem(
                params, logger, partition, n_cams, valid_generator_sil, 
                train=False, silhouette=True,social=params["social_training"]
            )
            if segmentation_model is not None:
                del segmentation_model

            if params["use_silhouette_in_volume"]:
                # concatenate RGB image volumes with silhouette volumes
                X_train = np.concatenate((X_train, y_train_aux, y_train_aux, y_train_aux), axis=-1)
                X_valid = np.concatenate((X_valid, y_valid_aux, y_valid_aux, y_valid_aux), axis=-1)
                logger.info("Input dimension is now {}".format(X_train.shape))

                params["use_silhouette"] = False
                logger.info("Turn off silhouette loss.")
                y_train_aux, y_valid_aux = None, None
            
            if params["social_training"]:
                X_train, X_train_grid, y_train, y_train_aux = processing.align_social_data(X_train, X_train_grid, y_train, y_train_aux)
                X_valid, X_valid_grid, y_valid, y_valid_aux = processing.align_social_data(X_valid, X_valid_grid, y_valid, y_valid_aux)
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

    # We apply data augmentation with another data generator class
    randflag = params["channel_combo"] == "random"

    if cam3_train:
        params["n_rand_views"] = 3
        params["rand_view_replace"] = False
        randflag = True

    if params["n_rand_views"] == 0:
        print(
            "Using default n_rand_views augmentation with {} views and with replacement".format(
                params["n_views"]
            )
        )
        print("To disable n_rand_views augmentation, set it to None in the config.")
        params["n_rand_views"] = params["n_views"]
        params["rand_view_replace"] = True

    shared_args = {
        "chan_num": params["chan_num"],
        "expval": params["expval"],
        "nvox": params["nvox"],
        "heatmap_reg": params["heatmap_reg"],
        "heatmap_reg_coeff": params["heatmap_reg_coeff"],
    }
    shared_args_train = {
        "rotation": params["rotate"],
        "augment_hue": params["augment_hue"],
        "augment_brightness": params["augment_brightness"],
        "augment_continuous_rotation": params["augment_continuous_rotation"],
        "mirror_augmentation": params["mirror_augmentation"],
        "right_keypoints": params["right_keypoints"],
        "left_keypoints": params["left_keypoints"],
        "bright_val": params["augment_bright_val"],
        "hue_val": params["augment_hue_val"],
        "rotation_val": params["augment_rotation_val"],
        "replace": params["rand_view_replace"],
        "random": randflag,
        "n_rand_views": params["n_rand_views"],
    }
    shared_args_valid = {
        "rotation": False,
        "augment_hue": False,
        "augment_brightness": False,
        "augment_continuous_rotation": False,
        "mirror_augmentation": False,
        "shuffle": False,
        "replace": False,
        "n_rand_views": params["n_rand_views"] if cam3_train else None,
        "random": True if cam3_train else False,
    }

    if params["use_npy"]:
        genfunc = dataset.DataGenerator_3Dconv_npy
        args_train = {
            "list_IDs": partition["train_sampleIDs"],
            "labels_3d": datadict_3d,
            "npydir": npydir,
        }
        args_train = {
            **args_train,
            **shared_args_train,
            **shared_args,
            "sigma": params["sigma"],
            "mono": params["mono"],
            "aux_labels": y_train_aux,
            "aux": params["use_silhouette"],
            "temporal_chunk_list": partition["train_chunks"] if params["use_temporal"] else None,
        }

        args_valid = {
            "list_IDs": partition["valid_sampleIDs"],
            "labels_3d": datadict_3d,
            "npydir": npydir,
            "aux_labels": y_valid_aux,
            "aux": params["use_silhouette"],
        }
        args_valid = {
            **args_valid,
            **shared_args_valid,
            **shared_args,
            "sigma": params["sigma"],
            "mono": params["mono"],
            "temporal_chunk_list": partition["valid_chunks"] if params["use_temporal"] else None
        }

    else:
        genfunc = dataset.DataGenerator_3Dconv_frommem
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
                      "aux_labels": y_train_aux,
                      "temporal_chunk_list": partition["train_chunks"] if params["use_temporal"] else None,
                      }

        args_valid = {
            "list_IDs": np.arange(len(partition["valid_sampleIDs"])),
            "data": X_valid,
            "labels": y_valid,
            "aux_labels": y_valid_aux
        }
        args_valid = {
            **args_valid,
            **shared_args_valid,
            **shared_args,
            "xgrid": X_valid_grid,
            "temporal_chunk_list": partition["valid_chunks"] if params["use_temporal"] else None
        }
    
    if params["social_training"]:
        args_train = {**args_train, "pairs": pairs["train_pairs"]}
        args_valid = {**args_valid, "pairs": pairs["valid_pairs"]}

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

    train_dataloader, valid_dataloader = serve_data_DANNCE.setup_dataloaders(train_generator, valid_generator, params)
    
    # Build network
    logger.info("Initializing Network...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

def dannce_predict(params: Dict):
    """Predict with dannce network

    Args:
        params (Dict): Paremeters dictionary.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    make_folder("dannce_predict_dir", params)

    params = setup_dannce_predict(params)

    (
        params["experiment"][0],
        samples_,
        datadict_,
        datadict_3d_,
        cameras_,
        com3d_dict_,
        _
    ) = do_COM_load(
        params["experiment"][0],
        params["experiment"][0],
        0,
        params,
        training=False,
    )

    # Write 3D COM to file. This might be different from the input com3d file
    # if arena thresholding was applied.
    write_com_file(params, samples_, com3d_dict_)

    # The library is configured to be able to train over multiple animals ("experiments")
    # at once. Because supporting code expects to see an experiment ID# prepended to
    # each of these data keys, we need to add a token experiment ID here.
    samples = []
    datadict = {}
    datadict_3d = {}
    com3d_dict = {}
    (samples, datadict, datadict_3d, com3d_dict, _) = serve_data_DANNCE.add_experiment(
        0,
        samples,
        datadict,
        datadict_3d,
        com3d_dict,
        samples_,
        datadict_,
        datadict_3d_,
        com3d_dict_,
    )
    cameras = {}
    cameras[0] = cameras_
    camnames = {}
    camnames[0] = params["experiment"][0]["camnames"]

    # Need a '0' experiment ID to work with processing functions.
    # *NOTE* This function modified camnames in place
    # to add the appropriate experiment ID
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, 1, camnames, cameras, dannce_prediction=True
    )

    samples = np.array(samples)

    # Initialize video dictionary. paths to videos only.
    # TODO: Remove this immode option if we decide not
    # to support tifs
    if params["immode"] == "vid":
        vids = {}
        vids = processing.initialize_vids(params, datadict, 0, vids, pathonly=True)

    # Parameters
    valid_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": params["batch_size"],
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "vmin": params["vmin"],
        "vmax": params["vmax"],
        "nvox": params["nvox"],
        "interp": params["interp"],
        "depth": params["depth"],
        "channel_combo": params["channel_combo"],
        "mode": "coordinates",
        "camnames": camnames,
        "immode": params["immode"],
        "shuffle": False,
        "rotation": False,
        "vidreaders": vids,
        "distort": True,
        "expval": params["expval"],
        "crop_im": False,
        "chunks": params["chunks"],
        "mono": params["mono"],
        "mirror": params["mirror"],
        "predict_flag": True,
    }

    # Datasets
    valid_inds = np.arange(len(samples))
    partition = {"valid_sampleIDs": samples[valid_inds]}

    # TODO: Remove tifdirs arguments, which are deprecated
    tifdirs = []

    # Generators
    # Because CUDA_VISBILE_DEVICES is already set to a single GPU, the gpu_id here should be "0"
    device = "cuda:0"
    genfunc = generator.DataGenerator_3Dconv

    predict_params = [
        partition["valid_sampleIDs"],
        datadict,
        datadict_3d,
        cameras,
        partition["valid_sampleIDs"],
        com3d_dict,
        tifdirs,        
    ]
    predict_generator = genfunc(
        *predict_params,
        **valid_params
    )

    predict_generator_sil = None
    if (params["use_silhouette_in_volume"]) or (params["write_visual_hull"] is not None):
        # require silhouette + RGB volume
        vids_sil = processing.initialize_vids(
            params, datadict, 0, {}, pathonly=True, vidkey="viddir_sil"
        )
        valid_params_sil = deepcopy(valid_params)
        valid_params_sil["vidreaders"] = vids_sil
        valid_params_sil["norm_im"] = False
        valid_params_sil["expval"] = True

        predict_generator_sil = generator.DataGenerator_3Dconv(
            *predict_params,
            **valid_params_sil
        )

    # model = build_model(params, camnames)
    print("Initializing Network...")
    model = initialize_model(params, len(camnames[0]), device)

    # load predict model
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
    
    if params["write_visual_hull"] is not None:
        print("Writing visual hull to .npy files")
        processing.write_sil_npy(params["write_visual_hull"], predict_generator_sil)

    save_data = inference.infer_dannce(
        predict_generator,
        params,
        model,
        partition,
        device,
        params["n_markers"],
        predict_generator_sil,
        save_heatmaps=False
    )

    if params["expval"]:
        if params["save_tag"] is not None:
            path = os.path.join(
                params["dannce_predict_dir"],
                "save_data_AVG%d.mat" % (params["save_tag"]),
            )
        else:
            path = os.path.join(params["dannce_predict_dir"], "save_data_AVG.mat")
        p_n = savedata_expval(
            path,
            params,
            write=True,
            data=save_data,
            tcoord=False,
            num_markers=params["n_markers"],
            pmax=True,
        )
    else:
        if params["save_tag"] is not None:
            path = os.path.join(
                params["dannce_predict_dir"],
                "save_data_MAX%d.mat" % (params["save_tag"]),
            )
        else:
            path = os.path.join(params["dannce_predict_dir"], "save_data_MAX.mat")
        p_n = savedata_tomat(
            path,
            params,
            params["vmin"],
            params["vmax"],
            params["nvox"],
            write=True,
            data=save_data,
            num_markers=params["n_markers"],
            tcoord=False,
        )


def setup_dannce_predict(params):
    # Depth disabled until next release.
    params["depth"] = False
    # Make the prediction directory if it does not exist.
    
    params["net_name"] = params["net"]
    params["n_views"] = int(params["n_views"])

    # While we can use experiment files for DANNCE training,
    # for prediction we use the base data files present in the main config
    # Grab the input file for prediction
    params["label3d_file"] = processing.grab_predict_label3d_file()
    params["base_exp_folder"] = os.path.dirname(params["label3d_file"])
    params["multi_mode"] = False

    print("Using camnames: {}".format(params["camnames"]))
    # Also add parent params under the 'experiment' key for compatibility
    # with DANNCE's video loading function
    if (params["use_silhouette_in_volume"]) or (params["write_visual_hull"] is not None):
        params["viddir_sil"] = os.path.join(params["base_exp_folder"], _DEFAULT_VIDDIR_SIL)
        
    params["experiment"] = {}
    params["experiment"][0] = params

    if params["start_batch"] is None:
        params["start_batch"] = 0
        params["save_tag"] = None
    else:
        params["save_tag"] = params["start_batch"]

    if params["new_n_channels_out"] is not None:
        params["n_markers"] = params["new_n_channels_out"]
    else:
        params["n_markers"] = params["n_channels_out"]

    # For real mono prediction
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    return params


def write_com_file(params, samples_, com3d_dict_):
    cfilename = os.path.join(params["dannce_predict_dir"], "com3d_used.mat")
    print("Saving 3D COM to {}".format(cfilename))
    c3d = np.zeros((len(samples_), 3))
    for i in range(len(samples_)):
        c3d[i] = com3d_dict_[samples_[i]]
    sio.savemat(cfilename, {"sampleID": samples_, "com": c3d})


def do_COM_load(exp: Dict, expdict: Dict, e, params: Dict, training=True):
    """Load and process COMs.

    Args:
        exp (Dict): Parameters dictionary for experiment
        expdict (Dict): Experiment specific overrides (e.g. com_file, vid_dir)
        e (TYPE): Description
        params (Dict): Parameters dictionary.
        training (bool, optional): If true, load COM for training frames.

    Returns:
        TYPE: Description
        exp, samples_, datadict_, datadict_3d_, cameras_, com3d_dict_

    Raises:
        Exception: Exception when invalid com file format.
    """
    (
        samples_,
        datadict_,
        datadict_3d_,
        cameras_,
        temporal_chunks
    ) = serve_data_DANNCE.prepare_data(
        exp, 
        prediction=not training, 
        predict_labeled_only=params["predict_labeled_only"],
        valid=(e in params["valid_exp"]) if params["valid_exp"] is not None else False,
        support=(e in params["support_exp"]) if params["support_exp"] is not None else False,
    )

    # If there is "clean" data (full marker set), can take the
    # 3D COM from the labels
    if exp["com_fromlabels"] and training:
        print("For experiment {}, calculating 3D COM from labels".format(e))
        com3d_dict_ = deepcopy(datadict_3d_)
        for key in com3d_dict_.keys():
            com3d_dict_[key] = np.nanmean(datadict_3d_[key], axis=1, keepdims=True)
    elif "com_file" in expdict and expdict["com_file"] is not None:
        exp["com_file"] = expdict["com_file"]
        if ".mat" in exp["com_file"]:
            c3dfile = sio.loadmat(exp["com_file"])
            com3d_dict_ = check_COM_load(c3dfile, "com", params["medfilt_window"])
        elif ".pickle" in exp["com_file"]:
            datadict_, com3d_dict_ = serve_data_DANNCE.prepare_COM(
                exp["com_file"],
                datadict_,
                comthresh=params["comthresh"],
                weighted=params["weighted"],
                camera_mats=cameras_,
                method=params["com_method"],
            )
            if params["medfilt_window"] is not None:
                raise Exception(
                    "Sorry, median filtering a com pickle is not yet supported. Please use a com3d.mat or *dannce.mat file instead"
                )
        else:
            raise Exception("Not a valid com file format")
    else:
        # Then load COM from the label3d file
        exp["com_file"] = expdict["label3d_file"]
        c3dfile = io.load_com(exp["com_file"])
        com3d_dict_ = check_COM_load(c3dfile, "com3d", params["medfilt_window"])

    print("Experiment {} using com3d: {}".format(e, exp["com_file"]))

    if params["medfilt_window"] is not None:
        print(
            "Median filtering COM trace with window size {}".format(
                params["medfilt_window"]
            )
        )

    # Remove any 3D COMs that are beyond the confines off the 3D arena
    do_cthresh = True if exp["cthresh"] is not None else False

    pre = len(samples_)
    samples_ = serve_data_DANNCE.remove_samples_com(
        samples_,
        com3d_dict_,
        rmc=do_cthresh,
        cthresh=exp["cthresh"],
    )
    msg = "Removed {} samples from the dataset because they either had COM positions over cthresh, or did not have matching sampleIDs in the COM file"
    print(msg.format(pre - len(samples_)))

    return exp, samples_, datadict_, datadict_3d_, cameras_, com3d_dict_, temporal_chunks

def check_COM_load(c3dfile: Dict, kkey: Text, win_size: int):
    """Check that the COM file is of the appropriate format, and filter it.

    Args:
        c3dfile (Dict): Loaded com3d dictionary.
        kkey (Text): Key to use for extracting com.
        wsize (int): Window size.

    Returns:
        Dict: Dictionary containing com data.
    """
    c3d = c3dfile[kkey]

    # do a median filter on the COM traces if indicated
    if win_size is not None:
        if win_size % 2 == 0:
            win_size += 1
            print("medfilt_window was not odd, changing to: {}".format(win_size))

        from scipy.signal import medfilt

        c3d = medfilt(c3d, (win_size, 1))

    c3dsi = np.squeeze(c3dfile["sampleID"])
    com3d_dict = {s: c3d[i] for (i, s) in enumerate(c3dsi)}
    return com3d_dict

def social_dannce_train(params):
    return