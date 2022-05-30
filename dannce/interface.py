"""Handle training and prediction for DANNCE and COM networks."""
import numpy as np
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, Text
import psutil

import torch

from dannce.engine.data import serve_data_DANNCE, dataset, generator, processing
from dannce.engine.data.processing import savedata_tomat, savedata_expval
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

def dannce_train(params: Dict):
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
        **base_params,
        "camnames": camnames,
        "vidreaders": vids,
        "chunks": total_chunks,
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

        if params["social_training"]:
            spec_params["occlusion"] = params["downscale_occluded_view"],

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
            
    if (not params["use_npy"]) and (params["social_training"]):
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

    # # We apply data augmentation with another data generator class
    if params["use_npy"]:
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

    params, valid_params = config.setup_predict(params)

    (
        params["experiment"][0],
        samples_,
        datadict_,
        datadict_3d_,
        cameras_,
        com3d_dict_,
        _
    ) = processing.do_COM_load(
        params["experiment"][0],
        params["experiment"][0],
        0,
        params,
        training=False,
    )

    # Write 3D COM to file. This might be different from the input com3d file
    # if arena thresholding was applied.
    processing.write_com_file(params, samples_, com3d_dict_)

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

def social_dannce_train(params):
    return