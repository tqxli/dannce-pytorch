"""Handle training and prediction for DANNCE and COM networks."""
import numpy as np
import os, time
from typing import Dict
import psutil
import torch

from dannce.engine.data import processing
from dannce.engine.data.processing import savedata_tomat, savedata_expval
import dannce.config as config
import dannce.engine.models.posegcn.nets as gcn_nets
from dannce.engine.models.nets import initialize_model, initialize_train
from dannce.engine.trainer.posegcn_trainer import GCNTrainer
from dannce.config import print_and_set
from dannce.engine.logging.logger import setup_logging, get_logger
from dannce.interface import make_folder
from dannce.engine import inference
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
    n_instances = custom_model_params["n_instances"]
    params["social_training"] = params["social_training"] or (n_instances > 1)

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
    params["use_features"] = custom_model_params.get("use_features", False)    
    pose_generator = initialize_train(params, n_cams, 'cpu', logger)[0]

    # second stage: pose refiner
    model_class = getattr(gcn_nets, custom_model_params.get("model", "PoseGCN"))
    model = model_class(
        custom_model_params,
        pose_generator,
        n_instances=n_instances,
        n_joints=params["n_channels_out"],
        t_dim=params.get("temporal_chunk_size", 1),
    )
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
        predict_diff=custom_model_params.get("predict_diff", False),
        multi_stage=(custom_model_params.get("model") == "PoseGCN_MultiStage"),
        relpose=custom_model_params.get("relpose", True),
        dual_sup=custom_model_params.get("dual_sup", False),
    )

    trainer.train()

def predict(params):
    device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    make_folder("dannce_predict_dir", params)

    params, valid_params = config.setup_predict(params)

    # handle specific params
    # might be better to load in params in the checkpoint ...
    # n_instances = params["custom_model"]["n_instances"]
    custom_model_params = torch.load(params["dannce_predict_model"])["params"]["custom_model"]
    n_instances = custom_model_params["n_instances"]

    # params["social_training"] = (n_instances > 1)

    # if n_instances > 1:
    #     com_files = params["com_file"].split(',')
    #     assert len(com_files) == 2, "For multi animal model, need multiple comfile inputs."
    #     params["experiment"][0]["com_file"] = com_files[0]
    #     params["experiment"][1] = deepcopy(params["experiment"][0])
    #     params["experiment"][1]["com_file"] = com_files[1] 

        # valid_params["predict_flag"] = False

    predict_generator, predict_generator_sil, camnames, partition = make_dataset_inference(params, valid_params)

    print("Initializing Network...")
    # first stage: pose generator
    params["use_features"] = custom_model_params.get("use_features", False)    
    pose_generator = initialize_model(params, len(camnames[0]), "cpu")   

    # second stage: pose refiner
    model_class = getattr(gcn_nets, custom_model_params.get("model", "PoseGCN"))
    model = model_class(
        custom_model_params,
        pose_generator,
        n_instances=n_instances,
        n_joints=params["n_channels_out"],
        t_dim=params.get("temporal_chunk_size", 1),
    ).to(device)

    if params.get("inference_ttt", None) is not None:
        ttt_params = params["inference_ttt"]
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)
        save_data = inference.inference_ttt(
            predict_generator,
            params,
            model,
            optimizer,
            device,
            partition,
            online=ttt_params.get("online", False),
            niter=ttt_params.get("niter", 20),
            transform=ttt_params.get("transform", False),
            downsample=ttt_params.get("downsample", 1),
            gcn=True,
        )
        inference.save_results(params, save_data)
        return 

    # load predict model
    model.load_state_dict(torch.load(params["dannce_predict_model"])['state_dict'])
    model.eval()

    if n_instances > 1:
        print("Obtaining initial pose estimations.")
        # model = model.pose_generator

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
                num_markers=params["n_markers"],
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
        # replace occluded view
        if params["downscale_occluded_view"]:
            occlusion_scores = ims[0][2]
            occluded_views = (occlusion_scores > 0.5)
            
            vols = vols.reshape(vols.shape[0], -1, 3, *vols.shape[2:]) #[B, 6, 3, H, W, D]

            for instance in range(occluded_views.shape[0]):
                occluded = np.where(occluded_views[instance])[0]
                unoccluded = np.where(~occluded_views[instance])[0]
                for view in occluded:
                    alternative = np.random.choice(unoccluded)
                    vols[instance][view] = vols[instance][alternative]
                    print(f"Replace view {view} with {alternative}")

            vols = vols.reshape(vols.shape[0], -1, *vols.shape[3:])

        model_inputs = [vols.to(device)]
        grid_centers = torch.from_numpy(ims[0][1]).to(device)
        model_inputs.append(grid_centers)

        init_poses, heatmaps, inter_features = model.pose_generator(*model_inputs)

        if not params["social_training"]:
            final_poses = model.inference(init_poses, grid_centers, heatmaps, inter_features) #+ init_poses
        else:
            final_poses = init_poses

        if custom_model_params.get("relpose", True):
            com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1) #[N, 3, 1]
            nvox = round(grid_centers.shape[1]**(1/3))
            vsize = (grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()) / nvox
            final_poses = final_poses * vsize
            if not custom_model_params.get("predict_diff", True):
                final_poses += com3d

        if custom_model_params.get("predict_diff", True):
            final_poses += init_poses

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
        num_markers=params["n_markers"],
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

def predict_multi_animal(params):
    import scipy.io as sio
    from tqdm import tqdm

    # currently, to perform inference over multiple animals
    # first obtain initial pose estimates from each separately, then run refinement w/ GCN
    custom_model_params = params["custom_model"]
    n_instances = custom_model_params["n_instances"]
    device = "cuda:0"

    # load initial predictions
    instance0_dir = params["dannce_predict_dir"].replace('final', 'rat1_init')
    instance1_dir = params["dannce_predict_dir"].replace('final', 'rat2_init')
    input1 = sio.loadmat(os.path.join(instance0_dir, 'save_data_AVG{}.mat'.format(params["start_batch"]))) # [N, 3, n_joints]
    input2 = sio.loadmat(os.path.join(instance1_dir, 'save_data_AVG{}.mat'.format(params["start_batch"])))

    pose_inputs = np.concatenate((input1["pred"], input2["pred"]), axis=0)
    pose_inputs = np.reshape(pose_inputs, (2, -1, *pose_inputs.shape[1:]))
    pose_inputs = np.transpose(pose_inputs, (1, 0, 2, 3))
    pose_inputs = np.reshape(pose_inputs, (-1, *pose_inputs.shape[2:]))

    pose_inputs = torch.tensor(pose_inputs, dtype=torch.float32)

    predict_dataloader = torch.utils.data.DataLoader(
        pose_inputs, batch_size = params["batch_size"], shuffle=False, num_workers=params["batch_size"]
    )

    print("Initializing Network...")
    # first stage: pose generator    
    pose_generator = None

    # if this is a multi animal model, need multi-stage inference ...    

    # second stage: pose refiner
    model_class = getattr(gcn_nets, custom_model_params.get("model", "PoseGCN"))
    model = model_class(
        custom_model_params,
        pose_generator,
        n_instances=n_instances,
        n_joints=params["n_channels_out"],
        t_dim=params.get("temporal_chunk_size", 1),
    ).to(device)
    
    # load predict model
    model.load_state_dict(torch.load(params["dannce_predict_model"])['state_dict'], strict=False)
    model.eval()

    # run inference
    final_poses = []
    pbar = tqdm(predict_dataloader)
    for init_poses in pbar:
        pred = model.inference(init_poses.to(device))

        final_poses.append(pred.detach().cpu().numpy() + init_poses)
    
    final_poses = np.concatenate(final_poses, axis=0)
    final_poses = np.reshape(final_poses, (-1, 2, *final_poses.shape[1:]))
    final_poses = np.transpose(final_poses, (1, 0, 2, 3))

    for i, input in enumerate([input1, input2]):
        input["pred"] = final_poses[i]
        save_data = input
        save_dir = params["dannce_predict_dir"].replace('final', f'rat{i+1}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        path = os.path.join(save_dir, "save_data_AVG{}.mat".format(params["start_batch"])) 
        sio.savemat(path, save_data)

    return