"""
High-level wrapper functions for interface
"""
import numpy as np
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, Text

import torch
from dannce import config
from dannce.engine.data import serve_data_DANNCE, dataset, generator, processing
from dannce.engine.models.segmentation import get_instance_segmentation_model
from dannce.engine.data.processing import _DEFAULT_SEG_MODEL

def make_folder(key: Text, params: Dict):
    """Make the prediction or training directories.

    Args:
        key (Text): Folder descriptor.
        params (Dict): Parameters dictionary.

    Raises:
        ValueError: Error if key is not defined.
    """
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

def make_dataset(
        params,  
        base_params,
        shared_args,
        shared_args_train,
        shared_args_valid,
        logger
    ):
    # load in experiments from config file
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

    # make train/valid splits
    partition = processing.make_data_splits(
        samples, params, params["dannce_train_dir"], num_experiments, 
        temporal_chunks=temporal_chunks)
    pairs = None
    if params["social_training"]:
        partition, pairs = processing.resplit_social(partition)

    if params.get("social_joint_training", False):
        # OPTION1: only choose volumes with both animals present
        # partition, com3d_dict, datadict_3d, samples = processing.filter_com3ds(pairs, com3d_dict, datadict_3d)
        # OPTION2: also predict on the other animal if any of its joints is also in the volume
        datadict_3d = processing.prepare_joint_volumes(params, pairs, com3d_dict, datadict_3d)

        # params["social_training"] = False
        params["n_channels_out"] *= 2
        base_params["n_channels_out"] *= 2

    # Dump the params into file for reproducibility
    processing.save_params_pickle(params)

    # Setup additional variables for later use
    tifdirs = []  # Training from single images not yet supported in this demo

    # Two possible data loading schemes:
    # (i) Directly use pre-saved npy volumes
    # (ii) Load images from video files into memory and generate the samples needed for training
    vid_exps = np.arange(num_experiments)
    
    # initialize needed videos
    vids = processing.initialize_all_vids(params, datadict, vid_exps, pathonly=True)

    base_params = {
        **base_params,
        "camnames": camnames,
        "vidreaders": vids,
        "chunks": total_chunks,
    }

    # Prepare datasets and dataloaders
    makedata_func = _make_data_npy if params["use_npy"] else _make_data_mem
    train_generator, valid_generator = makedata_func(
        params, base_params, shared_args, shared_args_train, shared_args_valid,
        datadict, datadict_3d, com3d_dict, 
        cameras, camnames, 
        samples, partition, pairs, tifdirs, vids,
        logger
    )
    train_dataloader, valid_dataloader = serve_data_DANNCE.setup_dataloaders(train_generator, valid_generator, params)
     
    return train_dataloader, valid_dataloader, len(camnames[0])

def _make_data_npy(
        params, base_params, shared_args, shared_args_train, shared_args_valid,
        datadict, datadict_3d, com3d_dict, 
        cameras, camnames, 
        samples, partition, pairs, tifdirs, vids,
        logger
    ):
    """
    Pre-generate the volumetric data and save as .npy.
    Good for large training set that is unable to fit in memory.
    Can be reused for future experiments.
    """

    # Examine through experiments for missing npy data files
    npydir, missing_npydir, missing_samples = serve_data_DANNCE.examine_npy_training(params, samples)

    if len(missing_samples) != 0:
        logger.info("{} npy files for experiments {} are missing.".format(len(missing_samples), list(missing_npydir.keys())))
    else:
        logger.info("No missing npy files. Ready for training.")

    # Generate missing npy files
    # mono conversion will happen from RGB npy files, and the generator
    # needs to be aware that the npy files contain RGB content
    params["chan_num"] = params["n_channels_in"]
    spec_params = {
        "channel_combo": None,
        "predict_flag": False,
        "norm_im": False,
        "expval": True,
    }

    genfunc = generator.DataGenerator_3Dconv
    if params["social_training"]:
        spec_params["occlusion"] = params["downscale_occluded_view"]
        genfunc = generator.DataGenerator_3Dconv_social

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

    # generate segmentation masks if needed
    segmentation_model, valid_params_sil = get_segmentation_model(params, valid_params, vids)

    if segmentation_model is not None:
        npydir, missing_npydir, missing_samples = serve_data_DANNCE.examine_npy_training(params, samples, aux=True)

        if len(missing_samples) != 0:
            logger.info("{} aux npy files for experiments {} are missing.".format(len(missing_samples), list(missing_npydir.keys())))
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
            logger.info("No missing aux npy files. Ready for training.")

    # Use another data generator class for applying augmentation
    genfunc = dataset.PoseDatasetNPY
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
        "aux_labels": None,
        "aux": params["use_silhouette"],
        "temporal_chunk_list": partition["train_chunks"] if params["use_temporal"] else None,
    }

    args_valid = {
        "list_IDs": partition["valid_sampleIDs"],
        "labels_3d": datadict_3d,
        "npydir": npydir,
        "aux_labels": None,
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
    
    if params["social_training"]:
        args_train = {**args_train, "pairs": pairs["train_pairs"]}
        args_valid = {**args_valid, "pairs": pairs["valid_pairs"]}

    # initialize datasets and dataloaders
    train_generator = genfunc(**args_train)
    valid_generator = genfunc(**args_valid)

    return train_generator, valid_generator

def _make_data_mem(        
        params, base_params, shared_args, shared_args_train, shared_args_valid,
        datadict, datadict_3d, com3d_dict, 
        cameras, camnames, 
        samples, partition, pairs, tifdirs, vids,
        logger
    ):
    """
    Training samples are stored in the memory and deployed on the fly.
    """
    n_cams = len(camnames[0])

    genfunc = generator.DataGenerator_3Dconv_social if params["social_training"] else generator.DataGenerator_3Dconv

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
    X_train, X_train_grid, y_train = processing.load_volumes_into_mem(
        params, logger, partition, n_cams, train_generator, train=True, social=params["social_training"]
    )
    X_valid, X_valid_grid, y_valid = processing.load_volumes_into_mem(
        params, logger, partition, n_cams, valid_generator, train=False, social=params["social_training"]
    )

    segmentation_model, valid_params_sil = get_segmentation_model(params, valid_params, vids)

    y_train_aux, y_valid_aux = None, None
    if segmentation_model is not None:
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

    # For AVG+MAX training, need to update the expval flag in the generators
    # and re-generate the 3D training targets
    # TODO: Add code to infer_params
    if params["avg+max"] is not None and params["use_silhouette"]:
        print("******Cannot combine AVG+MAX with silhouette - Using ONLY silhouette*******")

    elif params["avg+max"] is not None:
        y_train_aux, y_valid_aux = processing.initAvgMax(
            y_train, y_valid, X_train_grid, X_valid_grid, params
        )

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

    return train_generator, valid_generator


def get_segmentation_model(params, valid_params, vids):
    valid_params_sil = deepcopy(valid_params)
    valid_params_sil["vidreaders"] = vids #vids_sil

    # ensure voxel values stay positive, allowing conversion to binary below
    valid_params_sil["norm_im"] = False
    valid_params_sil["expval"] = True

    if not params["use_silhouette"]:
        return None, valid_params_sil

    # prepare segmentation model
    segmentation_model = get_instance_segmentation_model(num_classes=2)
    checkpoints = torch.load(_DEFAULT_SEG_MODEL)["state_dict"]
    segmentation_model.load_state_dict(checkpoints)
    segmentation_model.eval()
    segmentation_model = segmentation_model.to("cuda:0")

    return segmentation_model, valid_params_sil

def make_dataset_inference(params, valid_params):
    params, valid_params = config.setup_predict(params)
    # The library is configured to be able to train over multiple animals ("experiments")
    # at once. Because supporting code expects to see an experiment ID# prepended to
    # each of these data keys, we need to add a token experiment ID here.
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

    # Generators
    genfunc = generator.DataGenerator_3Dconv_social if params["social_training"] else generator.DataGenerator_3Dconv

    predict_params = [
        partition["valid_sampleIDs"],
        datadict,
        datadict_3d,
        cameras,
        partition["valid_sampleIDs"],
        com3d_dict,
        tifdirs,        
    ]
    spec_params = {"occlusion": params.get("downscale_occluded_view", False)} if params["social_training"] else {}
    predict_generator = genfunc(
        *predict_params,
        **valid_params,
        **spec_params
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
    
    return predict_generator, predict_generator_sil, camnames, partition