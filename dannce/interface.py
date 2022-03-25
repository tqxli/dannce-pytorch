"""Handle training and prediction for DANNCE and COM networks."""
from audioop import avg
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

import dannce.engine.data.serve_data_DANNCE as serve_data_DANNCE
import dannce.engine.data.generator as generator
import dannce.engine.data.generator_aux as generator_aux
import dannce.engine.data.processing as processing
from dannce.engine.data.processing import savedata_tomat, savedata_expval
from dannce.engine.data import ops, io
from dannce import (
    _param_defaults_dannce,
    _param_defaults_shared,
    _param_defaults_com,
)
import dannce.engine.inference as inference
from typing import List, Dict, Text
import os, psutil, csv

import torch
from dannce.engine.models.nets import DANNCE, initialize_model
from tqdm.auto import tqdm
from dannce.engine.trainer.dannce_trainer import DannceTrainer

process = psutil.Process(os.getpid())

_DEFAULT_VIDDIR = "videos"
_DEFAULT_VIDDIR_SIL = "videos_sil"
_DEFAULT_COMSTRING = "COM"
_DEFAULT_COMFILENAME = "com3d.mat"
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
    # Depth disabled until next release.
    params["depth"] = False
    params["multi_mode"] = False
    # Default to 6 views but a smaller number of views can be specified in the
    # DANNCE config. If the legnth of the camera files list is smaller than
    # n_views, relevant lists will be duplicated in order to match n_views, if
    # possible.
    params["n_views"] = int(params["n_views"])
    if params["use_silhouette_in_volume"]:
        params["use_silhouette"] = True
        params["n_rand_views"] = None

    # Make the training directory if it does not exist.
    make_folder("dannce_train_dir", params)

    # nets.get_losses(params)
    # params["loss"] = getattr(custom_losses, params["loss"])

    # set GPU ID
    # Temporarily commented out to test on dsplus gpu
    # if not params["multi_gpu_train"]:
    # os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]

    samples = []
    datadict = {}
    datadict_3d = {}
    com3d_dict = {}
    cameras = {}
    camnames = {}
    exps = params["exp"]
    num_experiments = len(exps)
    params["experiment"] = {}
    total_chunks = {}
    temporal_chunks = {}

    for e, expdict in enumerate(exps):

        exp = processing.load_expdict(params, e, expdict, _DEFAULT_VIDDIR, _DEFAULT_VIDDIR_SIL)

        (
            exp,
            samples_,
            datadict_,
            datadict_3d_,
            cameras_,
            com3d_dict_,
            temporal_chunks_
        ) = do_COM_load(exp, expdict, e, params)

        print("Using {} samples total.".format(len(samples_)))

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
        print("Using the following cameras: {}".format(camnames[e]))
        params["experiment"][e] = exp
        for name, chunk in exp["chunks"].items():
            total_chunks[name] = chunk

    dannce_train_dir = params["dannce_train_dir"]

    # Dump the params into file for reproducibility
    # processing.save_params(dannce_train_dir, params)

    # Additionally, to keep videos unique across experiments, need to add
    # experiment labels in other places. E.g. experiment 0 CameraE's "camname"
    # Becomes 0_CameraE. *NOTE* This function modified camnames in place
    # to add the appropriate experiment ID
    cameras, datadict, params = serve_data_DANNCE.prepend_experiment(
        params, datadict, num_experiments, camnames, cameras
    )

    samples = np.array(samples)

    if params["use_npy"]:
        # Add all npy volume directories to list, to be used by generator
        npydir = {}
        for e in range(num_experiments):
            npydir[e] = params["experiment"][e]["npy_vol_dir"]

        samples = processing.remove_samples_npy(npydir, samples, params)
    else:
        # Initialize video objects
        vids = {}
        vids_sil = {}
        for e in range(num_experiments):
            if params["immode"] == "vid":
                if params["use_silhouette"]:
                    vids_sil = processing.initialize_vids(
                        params, datadict, e, vids_sil, pathonly=True, vidkey="viddir_sil"
                    )
                vids = processing.initialize_vids(
                    params, datadict, e, vids, pathonly=True
                )

    # Parameters
    if params["expval"]:
        outmode = "coordinates"
    else:
        outmode = "3dprob"

    gridsize = tuple([params["nvox"]] * 3)

    # When this true, the data generator will shuffle the cameras and then select the first 3,
    # to feed to a native 3 camera model
    cam3_train = params["cam3_train"]
    if params["cam3_train"]:
        cam3_train = True
    else:
        cam3_train = False

    partition = processing.make_data_splits(
        samples, params, dannce_train_dir, num_experiments, 
        temporal_chunks=temporal_chunks)

    if params["use_npy"]:
        # mono conversion will happen from RGB npy files, and the generator
        # needs to b aware that the npy files contain RGB content
        params["chan_num"] = params["n_channels_in"]
    else:
        # Used to initialize arrays for mono, and also in *frommem (the final generator)
        params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

        valid_params = {
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
            "channel_combo": params["channel_combo"],
            "mode": outmode,
            "camnames": camnames,
            "immode": params["immode"],
            "shuffle": False,  # We will shuffle later
            "rotation": False,  # We will rotate later if desired
            "vidreaders": vids,
            "distort": True,
            "expval": params["expval"],
            "crop_im": False,
            "chunks": total_chunks,
            "mono": params["mono"],
            "mirror": params["mirror"],
        }

        # Setup a generator that will read videos and labels
        tifdirs = []  # Training from single images not yet supported in this demo

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

        train_generator = generator.DataGenerator_3Dconv(
            *train_gen_params,
            **valid_params
        )
        valid_generator = generator.DataGenerator_3Dconv(
            *valid_gen_params,
            **valid_params
        )

        if params["use_silhouette"]:
            valid_params_sil = deepcopy(valid_params)
            valid_params_sil["vidreaders"] = vids_sil

            # Set this to false so that all of our voxels stay positive, allowing us
            # to convert to binary below
            valid_params_sil["norm_im"] = False

            # expval gets set to True here sop that even in MAX mode the
            # silhouette generator behaves in a predictable way
            valid_params_sil["expval"] = True

            train_generator_sil = generator.DataGenerator_3Dconv(
                *train_gen_params,
                **valid_params_sil
            )

            valid_generator_sil = generator.DataGenerator_3Dconv(
                *valid_gen_params,
                **valid_params_sil
            )

        # # We should be able to load everything into memory...
        n_cams = len(camnames[0])
        X_train, X_train_grid, y_train = processing.load_volumes_into_mem(params, partition, n_cams, train_generator, train=True)
        X_valid, X_valid_grid, y_valid = processing.load_volumes_into_mem(params, partition, n_cams, valid_generator, train=False)
        
        if params["debug_volume_tifdir"] is not None:
            # When this option is toggled in the config, rather than
            # training, the image volumes are dumped to tif stacks.
            # This can be used for debugging problems with calibration or
            # COM estimation
            tifdir = params["debug_volume_tifdir"]
            if not os.path.exists(tifdir):
                os.makedirs(tifdir)
            print("Dump training volumes to {}".format(tifdir))
            for i in range(X_train.shape[0]):
                for j in range(n_cams):
                    im = X_train[
                        i,
                        :,
                        :,
                        :,
                        j * params["chan_num"] : (j + 1) * params["chan_num"],
                    ]
                    im = processing.norm_im(im) * 255
                    im = im.astype("uint8")
                    of = os.path.join(
                        tifdir,
                        partition["train_sampleIDs"][i] + "_cam" + str(j) + ".tif",
                    )
                    imageio.mimwrite(of, np.transpose(im, [2, 0, 1, 3]))
            return

    # For AVG+MAX training, need to update the expval flag in the generators
    # and re-generate the 3D training targets
    # TODO: Add code to infer_params
    if params["avg+max"] is not None and params["use_silhouette"]:
        print("******Cannot combine AVG+MAX with silhouette - Using ONLY silhouette*******")

    y_train_aux = None
    y_valid_aux = None
    if params["use_silhouette"]:
        _, _, y_train_aux = processing.load_volumes_into_mem(params, partition, n_cams, train_generator_sil, train=True, silhouette=True)
        _, _, y_valid_aux = processing.load_volumes_into_mem(params, partition, n_cams, valid_generator_sil, train=False, silhouette=True)

    elif params["avg+max"] is not None:
        y_train_aux, y_valid_aux = processing.initAvgMax(
            y_train, y_valid, X_train_grid, X_valid_grid, params
        )

    if params["use_silhouette_in_volume"]:
        # concatenate RGB image volumes with silhouette volumes
        X_train = np.concatenate((X_train, y_train_aux, y_train_aux, y_train_aux), axis=-1)
        X_valid = np.concatenate((X_valid, y_valid_aux, y_valid_aux, y_valid_aux), axis=-1)
        print("Input dimension is now {}".format(X_train.shape))

        params["use_silhouette"] = False
        print("Turn off silhouette loss.")

        y_train_aux = None
        y_valid_aux = None

    # Now we can generate from memory with shuffling, rotation, etc.
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
        # "batch_size": params["batch_size"],
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
        # "batch_size": params["batch_size"],
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
        genfunc = generator.DataGenerator_3Dconv_npy
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
            "temporal_chunk_list": partition["train_chunks"] if params["use_temporal"] else None,
            "separation_loss": params["use_separation"]
        }

        args_valid = {
            "list_IDs": partition["valid_sampleIDs"],
            "labels_3d": datadict_3d,
            "npydir": npydir,
            "aux_labels": y_valid_aux,
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
        genfunc = generator.DataGenerator_3Dconv_frommem
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
    
    train_generator = genfunc(**args_train)
    valid_generator = genfunc(**args_valid)

    train_dataloader, valid_dataloader = setup_dataloaders(train_generator, valid_generator, params)
    
    # Build net
    print("Initializing Network...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # TODO: initialize optimizer in a getattr way
    # TODO: lr scheduler
    if params["train_mode"] == "new":
        model = initialize_model(params, n_cams, device)
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"])

    elif params["train_mode"] == "finetune" or params["train_mode"] == "continued":
        checkpoints = torch.load(params["dannce_finetune_weights"])
        model = initialize_model(checkpoints["params"], n_cams, device)
        model.load_state_dict(checkpoints["model"])

        model_params = [p for p in model.parameters() if p.requires_grad]
        
        if params["train_mode"] == "continued":
            optimizer = torch.optim.Adam()
            optimizer.load_state_dict(checkpoints["optimizer"])
        else:
            optimizer = torch.optim.Adam(model_params, lr=params["lr"])

    print("COMPLETE\n")

    # set up trainer
    trainer = DannceTrainer(
        params=params,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device
    )

    trainer.train()

    # TODO: set up a logger

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
    if params["use_silhouette_in_volume"]:
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
    model = DANNCE(
        input_channels=(params["chan_num"] + params["depth"])*len(camnames[0]),
        output_channels=params["n_channels_out"],
        norm_method=params["norm_method"],
        return_coords=params["expval"],
        input_shape=params["nvox"]
    )
    model = model.to(device) 

    # load predict model
    model.load_state_dict(torch.load(params["dannce_predict_model"])['state_dict'])
    model.eval()

    if params["maxbatch"] != "max" and params["maxbatch"] > len(predict_generator):
        print(
            "Maxbatch was set to a larger number of matches than exist in the video. Truncating"
        )
        processing.print_and_set(params, "maxbatch", len(predict_generator))

    if params["maxbatch"] == "max":
        processing.print_and_set(params, "maxbatch", len(predict_generator))

    if params["write_npy"] is not None:
        # Instead of running inference, generate all samples
        # from valid_generator and save them to npy files. Useful
        # for working with large datasets (such as Rat 7M) because
        # .npy files can be loaded in quickly with random access
        # during training.
        print("Writing samples to .npy files")
        processing.write_npy(params["write_npy"], predict_generator)
        return

    save_data = inference.infer_dannce(
        predict_generator,
        params,
        model,
        partition,
        device,
        params["n_markers"],
        predict_generator_sil
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

    # Load the appropriate loss function and network 
    # try:
    #     params["loss"] = getattr(losses, params["loss"])
    # except AttributeError:
    #     params["loss"] = getattr(keras_losses, params["loss"])
    
    params["net_name"] = params["net"]
    # params["net"] = getattr(nets, params["net_name"])
    # Default to 6 views but a smaller number of views can be specified in the DANNCE config.
    # If the legnth of the camera files list is smaller than n_views, relevant lists will be
    # duplicated in order to match n_views, if possible.
    params["n_views"] = int(params["n_views"])

    # While we can use experiment files for DANNCE training,
    # for prediction we use the base data files present in the main config
    # Grab the input file for prediction
    params["label3d_file"] = processing.grab_predict_label3d_file()
    params["base_exp_folder"] = os.path.dirname(params["label3d_file"])

    # default to slow numpy backend if there is no predict_mode in config file. I.e. legacy support
    # params["predict_mode"] = (
    #     params["predict_mode"] if params["predict_mode"] is not None else "numpy"
    # )
    params["multi_mode"] = False
    # print("Using {} predict mode".format(params["predict_mode"]))

    print("Using camnames: {}".format(params["camnames"]))
    # Also add parent params under the 'experiment' key for compatibility
    # with DANNCE's video loading function
    if params["use_silhouette_in_volume"]:
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
    ) = serve_data_DANNCE.prepare_data(exp, prediction=not training)

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

def collate_fn(items):
    volumes = torch.cat([item[0] for item in items], dim=0)#.permute(0, 4, 1, 2, 3)
    targets = torch.cat([item[2] for item in items], dim=0)

    try: 
        grids = torch.cat([item[1] for item in items], dim=0)
    except:
        grids = None
    
    try: 
        auxs = torch.cat([item[3] for item in items], dim=0)
    except:
        auxs = None 

    return volumes, grids, targets, auxs 

def prepare_batch(batch, device):
    volumes = batch[0].float().to(device)
    grids = batch[1].float().to(device) if batch[1] is not None else None
    targets = batch[2].float().to(device)
    auxs = batch[3].to(device) if batch[3] is not None else None
    
    return volumes, grids, targets, auxs

def setup_dataloaders(train_dataset, valid_dataset, params):
    # current implementation returns chunked data
    if params["use_temporal"]:
        valid_batch_size = params["batch_size"] // params["temporal_chunk_size"]
    else:
        valid_batch_size = params["batch_size"] 

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=valid_batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=params["batch_size"]
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, valid_batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=params["batch_size"]
    )
    return train_dataloader, valid_dataloader

def social_dannce_train(params):
    return