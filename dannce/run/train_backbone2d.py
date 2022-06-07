import numpy as np
import os
from typing import Dict
import os, psutil
from tqdm import tqdm

import torch
from dannce import config
from dannce.engine.data import serve_data_DANNCE, dataset, generator, processing
from dannce.engine.models.backbone import get_pose_net
from dannce.engine.trainer.backbone_trainer import BackboneTrainer
from dannce.interface import make_folder
from dannce.engine.logging.logger import setup_logging, get_logger

process = psutil.Process(os.getpid())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

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

    custom_model_params = params["custom_model"]

    # Make the training directory if it does not exist.
    make_folder("dannce_train_dir", params)

    # setup logger
    setup_logging(params["dannce_train_dir"])
    logger = get_logger("training.log", verbosity=2)

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
    # logger.info(params)

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

    genfunc = generator.MultiviewImageGenerator

    # Used to initialize arrays for mono, and also in *frommem (the final generator)
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    spec_params = {
        "channel_combo":  params["channel_combo"],
        "expval": params["expval"],
    }

    valid_params = {**base_params, **spec_params}

    # Setup a generator that will read videos and labels
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
    
    train_dataloader = torch.utils.data.DataLoader(
        train_generator, batch_size=params["batch_size"], shuffle=True, 
        # num_workers=valid_batch_size
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_generator, params["batch_size"], shuffle=False, 
        # num_workers=valid_batch_size
    )

    # Build network
    logger.info("Initializing Network...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = get_pose_net(
        num_joints=params["n_channels_out"],
        params=custom_model_params,
        logger=logger
    )

    model = model.to(device)

    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_params, lr=params["lr"])

    logger.info("COMPLETE\n")

    # set up trainer
    trainer = BackboneTrainer(
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