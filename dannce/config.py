import numpy as np
import imageio
import os, shutil
import yaml
from typing import Dict, Text

from dannce.engine.data import io
from dannce import (
    _param_defaults_dannce,
    _param_defaults_shared,
    _param_defaults_com,
)
import warnings

def grab_predict_label3d_file(defaultdir=""):
    """
    Finds the paths to the training experiment yaml files.
    """
    def_ep = os.path.join(".", defaultdir)
    label3d_files = os.listdir(def_ep)
    label3d_files = [
        os.path.join(def_ep, f) for f in label3d_files if "dannce.mat" in f
    ]
    label3d_files.sort()

    if len(label3d_files) == 0:
        raise Exception("Did not find any *dannce.mat file in {}".format(def_ep))
    print("Using the following *dannce.mat files: {}".format(label3d_files[0]))
    return label3d_files[0]

def infer_params(params, dannce_net, prediction):
    """
    Some parameters that were previously specified in configs can just be inferred
        from others, thus relieving config bloat
    """
    # Grab the camnames from *dannce.mat if not in config
    if params["camnames"] is None:
        f = grab_predict_label3d_file()
        params["camnames"] = io.load_camnames(f)
        if params["camnames"] is None:
            raise Exception("No camnames in config or in *dannce.mat")

    # Infer vid_dir_flag and extension and n_channels_in and chunks
    # from the videos and video folder organization.
    # Look into the video directory / camnames[0]. Is there a video file?
    # If so, vid_dir_flag = True
    viddir = os.path.join(params["viddir"], params["camnames"][0])
    video_files = os.listdir(viddir)

    if any([".mp4" in file for file in video_files]) or any(
        [".avi" in file for file in video_files]
    ):

        print_and_set(params, "vid_dir_flag", True)
    else:
        print_and_set(params, "vid_dir_flag", False)
        viddir = os.path.join(viddir, video_files[0])
        video_files = os.listdir(viddir)

    extension = ".mp4" if any([".mp4" in file for file in video_files]) else ".avi"
    print_and_set(params, "extension", extension)
    video_files = [file for file in video_files if extension in file]

    # Use the camnames to find the chunks for each video
    chunks = {}
    for name in params["camnames"]:
        if params["vid_dir_flag"]:
            camdir = os.path.join(params["viddir"], name)
        else:
            camdir = os.path.join(params["viddir"], name)
            intermediate_folder = os.listdir(camdir)
            camdir = os.path.join(camdir, intermediate_folder[0])
        video_files = os.listdir(camdir)
        video_files = [f for f in video_files if extension in f]
        video_files = sorted(video_files, key=lambda x: int(x.split(".")[0]))
        chunks[name] = np.sort([int(x.split(".")[0]) for x in video_files])

    print_and_set(params, "chunks", chunks)

    firstvid = str(chunks[params["camnames"][0]][0]) + params["extension"]
    camf = os.path.join(viddir, firstvid)

    # Infer n_channels_in from the video info
    v = imageio.get_reader(camf)
    im = v.get_data(0)
    v.close()
    print_and_set(params, "n_channels_in", im.shape[-1])

    # set the raw im height and width
    print_and_set(params, "raw_im_h", im.shape[0])
    print_and_set(params, "raw_im_w", im.shape[1])

    if dannce_net and params["avg+max"] is not None:
        # To use avg+max, need to start with an AVG network
        # In case the net type is not properly specified, set it here
        print_and_set(params, "expval", True)
        print_and_set(params, "net_type", "AVG")

    if dannce_net and params["net"] is None:
        # Here we assume that if the network and expval are specified by the user
        # then there is no reason to infer anything. net + expval compatibility
        # are subsequently verified during check_config()
        #
        # If both the net and expval are unspecified, then we use the simpler
        # 'net_type' + 'train_mode' to select the correct network and set expval.
        # During prediction, the train_mode might be missing, and in any case only the
        # expval needs to be set.
        if params["net_type"] is None:
            raise Exception("Without a net name, net_type must be specified")

        if not prediction and params["train_mode"] is None:
            raise Exception("Need to specific train_mode for DANNCE training")

    print_and_set(params, "expval", True)
    if dannce_net:
        # infer crop_height and crop_width if None. Just use max dims of video, as
        # DANNCE does not need to crop.
        if params["crop_height"] is None or params["crop_width"] is None:
            im_h = []
            im_w = []
            for i in range(len(params["camnames"])):
                viddir = os.path.join(params["viddir"], params["camnames"][i])
                if not params["vid_dir_flag"]:
                    # add intermediate directory to path
                    viddir = os.path.join(
                        params["viddir"], params["camnames"][i], os.listdir(viddir)[0]
                    )
                video_files = sorted(os.listdir(viddir))
                camf = os.path.join(viddir, video_files[0])
                v = imageio.get_reader(camf)
                im = v.get_data(0)
                v.close()
                im_h.append(im.shape[0])
                im_w.append(im.shape[1])

            if params["crop_height"] is None:
                print_and_set(params, "crop_height", [0, np.max(im_h)])
            if params["crop_width"] is None:
                print_and_set(params, "crop_width", [0, np.max(im_w)])

        if params["max_num_samples"] is not None:
            if params["max_num_samples"] == "max":
                print_and_set(params, "maxbatch", "max")
            elif isinstance(params["max_num_samples"], (int, np.integer)):
                print_and_set(
                    params,
                    "maxbatch",
                    int(params["max_num_samples"] // params["batch_size"]),
                )
            else:
                raise TypeError("max_num_samples must be an int or 'max'")
        else:
            print_and_set(params, "maxbatch", "max")

        if params["start_sample"] is not None:
            if isinstance(params["start_sample"], (int, np.integer)):
                print_and_set(
                    params,
                    "start_batch",
                    int(params["start_sample"] // params["batch_size"]),
                )
            else:
                raise TypeError("start_sample must be an int.")
        else:
            print_and_set(params, "start_batch", 0)

        if params["vol_size"] is not None:
            print_and_set(params, "vmin", -1 * params["vol_size"] / 2)
            print_and_set(params, "vmax", params["vol_size"] / 2)

        if params["heatmap_reg"] and not params["expval"]:
            raise Exception(
                "Heatmap regularization enabled only for AVG networks -- you are using MAX"
            )

        if params["n_rand_views"] == "None":
            print_and_set(params, "n_rand_views", None)

    # There will be strange behavior if using a mirror acquisition system and are cropping images
    if params["mirror"] and params["crop_height"][-1] != params["raw_im_h"]:
        msg = "Note: You are using a mirror acquisition system with image cropping."
        msg = (
            msg
            + " All coordinates will be flipped relative to the raw image height, so ensure that your labels are also in that reference frame."
        )
        warnings.warn(msg)

    # Handle COM network name backwards compatibility
    # if params["net"].lower() == "unet2d_fullbn":
    #     print_and_set(params, "norm_method", "batch")
    # elif params["net"] == "unet2d_fullIN":
    #     print_and_set(params, "norm_method", "layer")

    # if not dannce_net:
    #     print_and_set(params, "net", "unet2d_full")

    return params

def print_and_set(params, varname, value):
    # Should add new values to params in place, no need to return
    params[varname] = value
    print("Setting {} to {}.".format(varname, params[varname]))


def check_config(params, dannce_net, prediction):
    """
    Add parameter checks and restrictions here.
    """
    check_camnames(params)

    if params["exp"] is not None:
        for expdict in params["exp"]:
            check_camnames(expdict)

    if dannce_net:
        # check_net_expval(params)
        check_vmin_vmax(params)

def check_vmin_vmax(params):
    for v in ["vmin", "vmax", "nvox"]:
        if params[v] is None:
            raise Exception(
                "{} not in parameters. Please add it, or use vol_size instead of vmin and vmax".format(
                    v
                )
            )

def check_camnames(camp):
    """
    Raises an exception if camera names contain '_'
    """
    if "camnames" in camp:
        for cam in camp["camnames"]:
            if "_" in cam:
                raise Exception("Camera names cannot contain '_' ")


def check_net_expval(params):
    """
    Raise an exception if the network and expval (i.e. AVG/MAX) are incompatible
    """
    if params["net"] is None:
        raise Exception("net is None. You must set either net or net_type.")
    if params["net_type"] is not None:
        if (
            params["net_type"] == "AVG"
            and "AVG" not in params["net"]
            and "expected" not in params["net"]
        ):
            raise Exception("net_type is set to AVG, but you are using a MAX network")
        if (
            params["net_type"] == "MAX"
            and "MAX" not in params["net"]
            and params["net"] != "unet3d_big"
        ):
            raise Exception("net_type is set to MAX, but you are using a AVG network")

    if (
        params["expval"]
        and "AVG" not in params["net"]
        and "expected" not in params["net"]
    ):
        raise Exception("expval is set to True but you are using a MAX network")
    if (
        not params["expval"]
        and "MAX" not in params["net"]
        and params["net"] != "unet3d_big"
    ):
        raise Exception("expval is set to False but you are using an AVG network")

def copy_config(results_dir, main_config, io_config):
    """
    Copies config files into the results directory, and creates results
        directory if necessary
    """
    print("Saving results to: {}".format(results_dir))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    mconfig = os.path.join(
        results_dir, "copy_main_config_" + main_config.split(os.sep)[-1]
    )
    dconfig = os.path.join(results_dir, "copy_io_config_" + io_config.split(os.sep)[-1])

    shutil.copyfile(main_config, mconfig)
    shutil.copyfile(io_config, dconfig)

def inherit_config(child, parent, keys):
    """
    If a key in keys does not exist in child, assigns the key-value in parent to
        child.
    """
    for key in keys:
        if key not in child.keys():
            child[key] = parent[key]
            print(
                "{} not found in io.yaml file, falling back to main config".format(key)
            )

    return child    

def write_config(results_dir, configdict, message, filename="modelconfig.cfg"):
    """Write a dictionary of k-v pairs to file.

    A much more customizable configuration writer. Accepts a dictionary of
    key-value pairs and just writes them all to file,
    together with a custom message
    """
    f = open(results_dir + filename, "w")
    for key in configdict:
        f.write("{}: {}\n".format(key, configdict[key]))
    f.write("message:" + message)

def read_config(filename):
    """Read configuration file.

    :param filename: Path to configuration file.
    """
    with open(filename) as f:
        params = yaml.safe_load(f)

    return params

def make_paths_safe(params):
    """Given a parameter dictionary, loops through the keys and replaces any \\ or / with os.sep
    to promote OS agnosticism
    """
    for key in params.keys():
        if isinstance(params[key], str):
            params[key] = params[key].replace("/", os.sep)
            params[key] = params[key].replace("\\", os.sep)

    return params

def make_none_safe(pdict):
    if isinstance(pdict, dict):
        for key in pdict:
            pdict[key] = make_none_safe(pdict[key])
    else:
        if (
            pdict is None
            or (isinstance(pdict, list) and None in pdict)
            or (isinstance(pdict, tuple) and None in pdict)
        ):
            return "None"
        else:
            return pdict
    return pdict

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
    base_params = read_config(base_config)
    base_params = make_paths_safe(base_params)
    params = read_config(base_params["io_config"])
    params = make_paths_safe(params)
    params = inherit_config(params, base_params, list(base_params.keys()))
    check_unrecognized_params(params)
    return params
