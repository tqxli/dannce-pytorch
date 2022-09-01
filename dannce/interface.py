"""Handle training and prediction for DANNCE and COM networks."""
import os
from typing import Dict
import psutil
import torch

import dannce.config as config
import dannce.engine.inference as inference
from dannce.engine.models.nets import initialize_train, initialize_model, initialize_com_train
from dannce.engine.trainer.dannce_trainer import DannceTrainer
from dannce.engine.trainer.com_trainer import COMTrainer
from dannce.engine.logging.logger import setup_logging, get_logger
from dannce.run_utils import *

process = psutil.Process(os.getpid())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

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
    logger = get_logger(verbosity=2)

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
        device = torch.device("cuda")
    logger.info("***Use {} GPU for training.***".format(params["gpu_id"]))
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # fix random seed if specified
    if params["random_seed"] is not None:
        set_random_seed(params["random_seed"])
        logger.info("***Fix random seed as {}***".format(params["random_seed"]))

    if params["dataset"] == "rat7m":
        dataset_preparer = make_rat7m  
    elif params["dataset"] == "pair":
        dataset_preparer = make_pair
    else:
        dataset_preparer = make_dataset

    train_dataloader, valid_dataloader, n_cams = dataset_preparer(
        params,  
        base_params,
        shared_args,
        shared_args_train,
        shared_args_valid,
        logger
    )

    # Build network
    logger.info("Initializing Network...")
    model, optimizer, lr_scheduler = initialize_train(params, n_cams, device, logger)
    logger.info(model)
    logger.info("COMPLETE\n")

    # set up trainer
    trainer_class = DannceTrainer
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

    # Because CUDA_VISBILE_DEVICES is already set to a single GPU, the gpu_id here should be "0"
    device = "cuda:0"
    params, valid_params = config.setup_predict(params)
    predict_generator, predict_generator_sil, camnames, partition = make_dataset_inference(params, valid_params)

    # model = build_model(params, camnames)
    print("Initializing Network...")
    model = initialize_model(params, len(camnames[0]), device)

    # load predict model
    model.load_state_dict(torch.load(params["dannce_predict_model"])['state_dict'])
    model.eval()

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
    inference.save_results(params, save_data)

def com_train(params: Dict):
    """Train COM network
    Args:
        params (Dict): Parameters dictionary.
    """
    params, train_params, valid_params = config.setup_com_train(params)

    # make the train directory if does not exist
    make_folder("com_train_dir", params)

    # setup logger
    setup_logging(params["com_train_dir"])
    logger = get_logger(verbosity=2) 
    
    assert torch.cuda.is_available(), "No available GPU device."
    params["gpu_id"] = [0]
    device = torch.device("cuda")
    logger.info("***Use {} GPU for training.***".format(params["gpu_id"]))

    # fix random seed if specified
    if params["random_seed"] is not None:
        set_random_seed(params["random_seed"])
        logger.info("***Fix random seed as {}***".format(params["random_seed"]))

    train_dataloader, valid_dataloader = make_data_com(params, train_params, valid_params, logger)

    # Build network
    logger.info("Initializing Network...")
    model, optimizer, lr_scheduler = initialize_com_train(params, device, logger)
    logger.info(model)
    logger.info("COMPLETE\n")

    # set up trainer
    trainer_class = COMTrainer
    trainer = trainer_class(
        params=params,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device,
        logger=logger,
        lr_scheduler=lr_scheduler
    )

    trainer.train()

def com_predict(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    make_folder("com_predict_dir", params)
    setup_logging(params["com_predict_dir"], filename="prediction.log")
    logger = get_logger(verbosity=2) 

    device = "cuda:0"
    params, predict_params = config.setup_com_predict(params)
    predict_generator, params, partition, camera_mats, cameras, datadict = make_dataset_com_inference(params, predict_params)

    print("Initializing Network...")
    model = initialize_com_train(params, device, logger)[0]
    model.load_state_dict(torch.load(params["com_predict_weights"])['state_dict'])
    model.eval()

    # do frame-wise inference
    save_data = {}
    endIdx = np.min(
        [
            params["start_sample"] + params["max_num_samples"],
            len(predict_generator),
        ]
    ) if params["max_num_samples"] != "max" else len(predict_generator)

    save_data = inference.infer_com(
        params["start_sample"],
        endIdx,
        predict_generator,
        params,
        model,
        partition,
        save_data,
        camera_mats,
        cameras,
        device
    )

    filename = "com3d" if params["max_num_samples"] != "max" else "com3d%d" % (params["start_sample"])
    processing.save_COM_checkpoint(
        save_data, params["com_predict_dir"], datadict, cameras, params, file_name=filename
    )

    print("done!")