"""Handle training and prediction for DANNCE and COM networks."""
import numpy as np
import os, time
from typing import Dict
import psutil
import torch

from dannce.engine.data import processing
from dannce.engine.data.processing import savedata_tomat, savedata_expval
import dannce.config as config
from dannce.engine.models.social.nets import SocialXAttn
from dannce.engine.models.nets import initialize_model, initialize_train
from dannce.engine.trainer.posegcn_trainer import GCNTrainer
from dannce.engine.logging.logger import setup_logging, get_logger
from dannce.config import print_and_set
from dannce.interface import make_folder
from dannce.run_utils import *

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

    # handle specific params
    custom_model_params = params["custom_model"]

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
        device = torch.device("cuda")
    logger.info("***Use {} GPU for training.***".format(params["gpu_id"]))
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # fix random seed if specified
    if params["random_seed"] is not None:
        set_random_seed(params["random_seed"])
        logger.info("***Fix random seed as {}***".format(params["random_seed"]))

    train_dataloader, valid_dataloader, n_cams = make_dataset(
        params,  
        base_params,
        shared_args,
        shared_args_train,
        shared_args_valid,
        logger
    )

    # Build network
    logger.info("Initializing Network...")

    # first stage: pose generator    
    params["use_features"] = True
    pose_generator = initialize_train(params, n_cams, 'cpu', logger)[0]

    # second stage: pose refiner
    model = SocialXAttn(pose_generator)
    if 'checkpoint' in custom_model_params.keys():
        model.load_state_dict(torch.load(custom_model_params["checkpoint"])["state_dict"])
    model = model.to(device)
    logger.info(model)

    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)

    lr_scheduler = None
    if params["lr_scheduler"] is not None:
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, params["lr_scheduler"]["type"])
        lr_scheduler = lr_scheduler_class(optimizer=optimizer, **params["lr_scheduler"]["args"], verbose=True)
        logger.info("Using learning rate scheduler.")

    logger.info("COMPLETE\n")

    # set up trainer
    trainer_class = GCNTrainer
    trainer = trainer_class(
        params=params,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device,
        logger=logger,
        visualize_batch=False,
        lr_scheduler=lr_scheduler,
        predict_diff=False,
        multi_stage=False,
        relpose=True,
        dual_sup=False,
    )

    trainer.train()

def predict(params: Dict):
    """Predict with dannce network

    Args:
        params (Dict): Paremeters dictionary.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    make_folder("dannce_predict_dir", params)

    # handle specific params
    custom_model_params = params["custom_model"]

    # Because CUDA_VISBILE_DEVICES is already set to a single GPU, the gpu_id here should be "0"
    device = "cuda:0"
    params, valid_params = config.setup_predict(params)
    predict_generator, predict_generator_sil, camnames, partition = make_dataset_inference(params, valid_params)

    # first stage: pose generator    
    params["use_features"] = True
    pose_generator = initialize_model(params, len(camnames[0]), 'cpu')

    # second stage: pose refiner
    model = SocialXAttn(pose_generator)
    if 'checkpoint' in custom_model_params.keys():
        model.load_state_dict(torch.load(custom_model_params["checkpoint"])["state_dict"])
    model = model.to(device)

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
    
    end_time = time.time()
    save_data, save_data_init = {}, {}
    start_ind = params["start_batch"]
    end_ind = params["maxbatch"]

    for idx, i in enumerate(range(start_ind, end_ind)):
        print("Predicting on batch {}".format(i), flush=True)
        if (i - start_ind) % 10 == 0 and i != start_ind:
            print(i)
            print("10 batches took {} seconds".format(time.time() - end_time))
            end_time = time.time()

        if (i - start_ind) % 1000 == 0 and i != start_ind:
            print("Saving checkpoint at {}th batch".format(i))
            p_n = savedata_expval(
                params["dannce_predict_dir"] + "/save_data_AVG.mat",
                params,
                write=True,
                data=save_data,
                tcoord=False,
                num_markers=(params["n_markers"] // 2),
                pmax=True,
            )
            p_n = savedata_expval(
                params["dannce_predict_dir"] + "/init_save_data_AVG.mat",
                params,
                write=True,
                data=save_data_init,
                tcoord=False,
                num_markers=params["n_markers"],
                pmax=True,
            )

        ims = predict_generator.__getitem__(i)
        vols = torch.from_numpy(ims[0][0]).permute(0, 4, 1, 2, 3)

        model_inputs = [vols.to(device)]
        grid_centers = torch.from_numpy(ims[0][1]).to(device)
        model_inputs.append(grid_centers)

        init_poses, final_poses, heatmaps = model(*model_inputs)
        
        com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1)
        nvox = round(grid_centers.shape[1]**(1/3))
        vsize = (grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()) / nvox
        final_poses = final_poses * vsize + com3d

        probmap = torch.amax(heatmaps, dim=(2, 3, 4)).squeeze(0).detach().cpu().numpy()
        heatmaps = heatmaps.squeeze().detach().cpu().numpy()
        pred = final_poses.detach().cpu().numpy()
        pred_init = init_poses.detach().cpu().numpy()
        for j in range(pred.shape[0]):
            pred_max = probmap[j]
            sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]
            save_data[idx * pred.shape[0] + j] = {
                "pred_max": pred_max,
                "pred_coord": pred[j],
                "sampleID": sampleID,
            }
            save_data_init[idx * pred.shape[0] + j] = {
                "pred_max": pred_max,
                "pred_coord": pred_init[j],
                "sampleID": sampleID,
            }


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
        num_markers=(params["n_markers"] // 2),
        pmax=True,
    )
    
    path =os.path.join(params["dannce_predict_dir"], "init_save_data_AVG.mat")
    p_n = savedata_expval(
        path,
        params,
        write=True,
        data=save_data_init,
        tcoord=False,
        num_markers=params["n_markers"],
        pmax=True,
    ) 