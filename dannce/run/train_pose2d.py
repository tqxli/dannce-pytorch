from tkinter import N
import dannce.config as config
from dannce.engine.data import ops
from dannce.engine.models.pose2d.sleap import SLEAPUNet
from dannce.engine.models.pose2d import pose_net
from dannce.engine.models.pose2d.dlc import DLC
from dannce.run_utils import *
from dannce.engine.logging.logger import setup_logging, get_logger
from dannce.engine.trainer.com_trainer import COMTrainer
from dannce.engine.data.dataset import RAT7MImageDataset
from dannce.engine.data.ops import expected_value_2d, spatial_softmax

import scipy.io as sio
from tqdm import tqdm

def load_data2d_into_mem(params, logger, partition, n_cams, generator, image_size=512, train=True):
    n_samples = len(partition["train_sampleIDs"]) if train else len(partition["valid_sampleIDs"]) 
    message = "Loading training data into memory" if train else "Loading validation data into memory"

    # initialize
    # (n_samples, n_cams, params["chan_num"], image_size, image_size)
    X = []
    y = torch.empty((n_samples, n_cams, 2, params["n_channels_out"]), dtype=torch.float32)
    
    # load data from generator
    for i in tqdm(range(n_samples)):
        # print(i, end='\r')
        rr = generator.__getitem__(i)
        X += rr[0][0][0]
        y[i] = rr[1][1]
    
    # X = X.reshape(-1, *X.shape[2:]) 
    y = y.reshape(-1, *y.shape[2:]) #[n_samples*n_cams, 2, n_joints]
    return X, y


def configure_dataset(custom_params):
    model_type = custom_params.get("type", "SLEAP")
    DLC_flag = (model_type == "dlc")
    use_gt_bbox = custom_params.get("use_gt_bbox", True)
    return_Gaussian=custom_params.get("return_gaussian", True)
    use_original_image = custom_params.get("use_original_image", False)

    if use_original_image:
        image_params = {
            "resize": False,
            "crop": False,
            "resize_to_nearest": False,
            "image_size": 1152 # placeholder
        }
    else:
        image_params = {
            "resize": False if DLC_flag else True,
            "crop_size": custom_params.get("crop_size", 512),
            "image_size": custom_params.get("resize_size", 256),
            "use_gt_bbox": use_gt_bbox,
            "resize_to_nearest": DLC_flag
        }

    if DLC_flag:
        heatmap_size = [image_params["image_size"]//8, image_params["image_size"]//8]
        ds_fac = 8
    else:
        heatmap_size = [image_params["image_size"]//4, image_params["image_size"]//4]
        ds_fac = 4
    
    return image_params, heatmap_size, ds_fac, return_Gaussian

def batched_collate_fn(batch):
    X = torch.cat([item[0] for item in batch], dim=0)
    y = torch.cat([item[1] for item in batch], dim=0)
    return X, y

def collate_fn(batch):
    X = torch.stack([item[0] for item in batch], dim=0)
    y = torch.stack([item[1] for item in batch], dim=0)
    return X, y

def build_model(params, logger=None, train=True):
    model_type = params["custom_model"].get("type", "SLEAP")
    if model_type == "SLEAP":
        model = SLEAPUNet(params["n_channels_in"], params["n_channels_out"])
    elif model_type == "posenet":
        model = pose_net.get_pose_net(params["n_channels_out"], params["custom_model"], logger)
    elif model_type == "dlc":
        model = DLC(params["n_channels_out"])
    else:
        if logger is not None:
            logger.info("Invalid architecture.")
    if not train:
        return model

    if params["train_mode"] == "finetune" and params["dannce_finetune_weights"] is not None:
        logger.info("Loading checkpoint from {}".format(params["dannce_finetune_weights"]))
        state_dict = torch.load(params["dannce_finetune_weights"])["state_dict"]
        ckpt_channel_num = state_dict["final_layer.weight"].shape[0]
        if ckpt_channel_num != params["n_channels_out"]:
            state_dict.pop("final_layer.weight", None)
            state_dict.pop("final_layer.bias", None)
            logger.info("Replacing the last output layer from {} to {}".format(ckpt_channel_num, params["n_channels_out"]))
        model.load_state_dict(state_dict, strict=False)
    
    return model

def train(params):
    params, base_params = config.setup_train(params)[:2]
    custom_params = params["custom_model"]

    # make the train directory if does not exist
    make_folder("dannce_train_dir", params)

    # setup logger
    setup_logging(params["dannce_train_dir"])
    logger = get_logger("training.log", verbosity=2) 
    
    assert torch.cuda.is_available(), "No available GPU device."
    params["gpu_id"] = [0]
    device = torch.device("cuda")
    logger.info("***Use {} GPU for training.***".format(params["gpu_id"]))

    # fix random seed if specified
    if params["random_seed"] is not None:
        set_random_seed(params["random_seed"])
        logger.info("***Fix random seed as {}***".format(params["random_seed"]))

    model = build_model(params, logger)

    model_params = [p for p in model.parameters() if p.requires_grad]
    if "optimizer" in params.keys():
        optimizer_class = getattr(torch.optim, params["optimizer"])
        optimizer = optimizer_class(model_params, lr=params["lr"])
    else:
        optimizer = torch.optim.Adam(model_params, eps=1e-7)
    lr_scheduler = None
    if params["lr_scheduler"] is not None:
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, params["lr_scheduler"]["type"])
        lr_scheduler = lr_scheduler_class(optimizer=optimizer, **params["lr_scheduler"]["args"], verbose=True)
        logger.info("Using learning rate scheduler.")

    model = model.to(device)
    logger.info(model)
    logger.info("COMPLETE\n")

    if params["dataset"] == "rat7m":
        dataset_train = RAT7MImageDataset(train=True, downsample=1)
        dataset_valid = RAT7MImageDataset(train=False, downsample=800)
        logger.info("Train: {} samples".format(len(dataset_train)))
        logger.info("Validation: {} samples".format(len(dataset_valid)))

    elif params["dataset"] == "label3d":
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

        genfunc = generator.MultiviewImageGenerator

        base_params = {
            **base_params,
            "camnames": camnames,
            "vidreaders": vids,
            "chunks": total_chunks,
        }
        spec_params = {
            "channel_combo":  params["channel_combo"],
            "expval": params["expval"],
        }

        valid_params = {**base_params, **spec_params}
        image_params, heatmap_size, ds_fac, return_Gaussian = configure_dataset(custom_params)

        # workaround:
        # must use the same bounding box for neighboring cropped 2D images
        # otherwise temporal loss does not make sense
        if params["use_temporal"]:
            chunk_keys = ["train_chunks", "valid_chunks"]
            sample_keys = ["train_sampleIDs", "valid_sampleIDs"]
            labeled_idx = params["temporal_chunk_size"] // 2
            for ckey, skey in zip(chunk_keys, sample_keys):
                for chunk_idx, chunk in enumerate(partition[ckey]):
                    chunk_sampleIDs = np.array(partition[skey])[chunk]
                    # at least one chunk should have ground truth labels
                    labeled_samp = chunk_sampleIDs[labeled_idx]
                    # replace neighbors' nan labels
                    for sampidx, samp in enumerate(chunk_sampleIDs):
                        if samp == labeled_samp:
                            continue
                        expid = samp.split("_")[0]
                        if np.isnan(datadict[samp]['data'][f'{expid}_Camera1'][:, 0]).sum() < 2:
                            samp_new = samp+"b"
                            print(f"repetitive sampleID in chunks: {samp_new}")
                            samp_new_idx = chunk_idx*params["temporal_chunk_size"] + sampidx
                            partition[skey][samp_new_idx] = samp_new
                            datadict[samp_new] = datadict[labeled_samp]
                            datadict_3d[samp_new] = datadict_3d[labeled_samp]
                            com3d_dict[samp_new] = com3d_dict[labeled_samp]
                        else:
                            datadict[samp] = datadict[labeled_samp]
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
        train_generator = genfunc(**train_gen_params, **valid_params, **image_params)
        valid_generator = genfunc(**valid_gen_params, **valid_params, **image_params)
        
        # load everything into memory
        X_train, y_train = load_data2d_into_mem(params, logger, partition, n_cams, train_generator, train=True, image_size=image_params["image_size"])
        X_valid, y_valid = load_data2d_into_mem(params, logger, partition, n_cams, valid_generator, train=False, image_size=image_params["image_size"])

        args_common = {   
            "return_Gaussian": return_Gaussian,         
            "num_joints": params["n_channels_out"],
            "image_size": [image_params["image_size"]]*2,
            "heatmap_size": heatmap_size,
            "heatmap_type": custom_params.get("heatmap_type", "gaussian"),            
            "ds_fac": ds_fac,
            "sigma": custom_params.get("sigma", 2),
            "return_chunk_size": params.get("temporal_chunk_size", 1),
        }
        dataset_train = dataset.ImageDataset(
            images=X_train,
            labels=y_train,
            train=True,
            augs=custom_params.get("augs", ['hflip', 'vflip', 'randomrot']),
            **args_common
        )
        dataset_valid = dataset.ImageDataset(
            images=X_valid,
            labels=y_valid,
            train=False,
            **args_common
        )
    # sample = dataset_valid[0]
    if params["use_temporal"]:
        valid_batch_size = params["batch_size"] // params["temporal_chunk_size"]
    else:
        valid_batch_size = params["batch_size"]
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=valid_batch_size, shuffle=True,
        collate_fn=batched_collate_fn,
        num_workers=4,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=valid_batch_size, shuffle=False,
        collate_fn=batched_collate_fn,
        num_workers=4
    )

    params["com_train_dir"] = params["dannce_train_dir"]
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
        lr_scheduler=lr_scheduler,
        return_gaussian=return_Gaussian
    )

    trainer.train()

def predict(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]

    make_folder("dannce_predict_dir", params)
    setup_logging(params["dannce_predict_dir"])
    logger = get_logger("training.log", verbosity=2) 
    device = "cuda:0"
    params["n_instances"] = params["n_channels_out"]
    n_cams = len(params["camnames"])
    custom_params = params["custom_model"]

    # used for getting access to ground truth labels
    params["return_full2d"] = True

    if params["dataset"] == "rat7m":    
        # inference over the withheld animal (subject 5)
        dataset_valid = RAT7MImageDataset(train=False, downsample=1) #s5-d1: 10445, s5-d2: 14091
        cameras = dataset_valid.cameras
        expname = 5
        cameras = cameras[5]
        for k, cam in cameras.items():
            cammat = ops.camera_matrix(cam["K"], cam["R"], cam["t"])
            cameras[k]["cammat"] = cammat
        generator_len = dataset_valid.n_samples // n_cams
        endIdx = generator_len if params["max_num_samples"] == "max" else params["start_sample"] + params["max_num_samples"]
        partition = {"valid_sampleIDs": np.arange(params["start_sample"], endIdx)}
    elif params["dataset"] == "label3d":
        params, valid_params = config.setup_predict(params)

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
        
        expname = 0

        # Parameters
        valid_params = {
            **valid_params,
            "camnames": camnames,
            "vidreaders": vids,
            "chunks": params["chunks"],
        }

        # Datasets
        endIdx = np.min(
            [
                params["start_sample"] + params["max_num_samples"],
                len(samples)
            ]
        ) if params["max_num_samples"] != "max" else len(samples)

        valid_inds = np.arange(len(samples))
        partition = {"valid_sampleIDs": samples[valid_inds]}

        for k in tqdm(samples[params["start_sample"]:endIdx]):
            com3d = com3d_dict[k]
            com3d = torch.from_numpy(com3d[np.newaxis, :]).float()
            for camname, cam in cameras[expname].items():
                K = cam["K"]
                R = cam["R"]
                t = cam["t"]
                M = torch.as_tensor(
                    ops.camera_matrix(K, R, t), dtype=torch.float32
                )
                cam["cammat"] = M.numpy()
                com2d = ops.project_to2d(com3d, M, "cpu")[:, :2]
                com2d = ops.distortPoints(com2d, K, np.squeeze(cam["RDistort"]), np.squeeze(cam["TDistort"]), "cpu")

                datadict[k]["data"][camname] = com2d.numpy()

        # TODO: Remove tifdirs arguments, which are deprecated
        tifdirs = []

        # Generators
        genfunc = generator.MultiviewImageGenerator
        valid_gen_params = {
            "list_IDs": partition["valid_sampleIDs"],
            "labels": datadict,
            "labels_3d": datadict_3d,
            "camera_params": cameras,
            "clusterIDs": partition["valid_sampleIDs"],
            "com3d": com3d_dict,
            "tifdirs": tifdirs
        }

        image_params, heatmap_size, ds_fac, return_Gaussian = configure_dataset(custom_params)
        dataset_valid = genfunc(
            **valid_gen_params,
            **valid_params,
            **image_params
        )

        generator_len = len(dataset_valid)
        
        cameras = cameras[expname]
    
    print("Initializing Network...")
    model = build_model(params, logger, train=False)
    model.load_state_dict(torch.load(params["dannce_predict_model"])['state_dict'])
    model.eval()   
    model = model.to(device)

    save_data = {}
    for idx, i in enumerate(tqdm(range(params["start_sample"], endIdx))):
        if params["dataset"] == "rat7m":
            batch, coms = [], []
            for j in range(n_cams):
                batch.append(dataset_valid[i+j*generator_len][0])
                coms.append(dataset_valid.coms[i+j*generator_len])
            batch = torch.stack(batch, dim=0)
        elif params["dataset"] == "label3d":
            data = dataset_valid[i]
            batch = data[0][0][0]
            # batch = batch.reshape(-1, *batch.shape[2:]).float()
            ID = partition["valid_sampleIDs"][idx]
            coms = [np.nanmean(dataset_valid.labels[ID]["data"][f"0_Camera{j}"].round(), axis=1) for j in range(1, 7)]

        pred = []
        heatmap_shapes = [] # this is important for keeping track of sizes

        for view in batch:
            out = model(view.unsqueeze(0).to(device))
            out = out.detach().cpu()
            pred.append(out)
            heatmap_shapes.append(out.shape[-2:])
        # pred = model(batch.to(device))
        # pred = pred.detach().cpu()

        if return_Gaussian:
            pred = [out.numpy()[0] for out in pred]
            n_joints = pred[0].shape[0]
        else:
            pred = [expected_value_2d(spatial_softmax(out)) for out in pred]
            pred = torch.stack(pred, dim=1).numpy()[0] # [n_cams, J, 2]
            n_joints = pred.shape[1]

        sample_id = partition["valid_sampleIDs"][idx]
        save_data[sample_id] = {}
        save_data[sample_id]["triangulation"] = {}

        for n_cam in range(n_cams):
            camname = str(expname)+"_"+params["camnames"][n_cam] if params["dataset"] == "rat7m" else params["camnames"][n_cam]

            save_data[sample_id][params["camnames"][n_cam]] = {
                "COM": np.zeros((params["n_channels_out"], 2)),
            }
            heatmap_shape = heatmap_shapes[n_cam]

            old_image_size = dataset_valid.old_image_shapes[ID][camname]
            # old_image_size = (old_image_size[1], old_image_size[0])
            scales = [old_image_size[0] / heatmap_shape[0], old_image_size[1] / heatmap_shape[1]]

            try:
                bbox = dataset_valid.image_bboxs[ID][camname]
            except:
                # no cropping was done, make the bbox cover the entire image
                bbox = [0, 0, old_image_size[1], old_image_size[0]]
            center_real = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
            # ori_inds = []
            for n_joint in range(n_joints):
                # take the absolute maximum
                if return_Gaussian:
                    ind = (
                        np.array(processing.get_peak_inds(np.squeeze(pred[n_cam][n_joint])))
                    )
                # take the softargmax
                else:
                    ind = pred[n_cam, n_joint]
                    # convert to ij to keep consistency with above
                    ind = ind[::-1]
                
                # scale back to the original image scale
                # this is in ij
                ind[0] = (ind[0] - heatmap_shape[0]//2) * scales[0]
                ind[1] = (ind[1] - heatmap_shape[1]//2) * scales[1]
                ind[0] = ind[0] + center_real[1]
                ind[1] = ind[1] + center_real[0]
                # convert to xy
                ind = ind[::-1]

                # Undistort this COM here.
                pts1 = ind
                pts1 = pts1[np.newaxis, :]
                pts1 = ops.unDistortPoints(
                    pts1,
                    cameras[camname]["K"],
                    cameras[camname]["RDistort"],
                    cameras[camname]["TDistort"],
                    cameras[camname]["R"],
                    cameras[camname]["t"],
                )

                save_data[sample_id][params["camnames"][n_cam]]["COM"][n_joint] = np.squeeze(pts1)
        
        # triangulation
        save_data[sample_id]["joints"] = np.zeros((params["n_channels_out"], 3))
        prefix = str(expname)+"_" if params["dataset"] == "rat7m" else None
        ransac = params.get("ransac", False)
        direct_optimization = params.get("direct_optimization", False)

        # for each joint
        for joint in range(n_joints):
            view_set = set(range(6))
            inlier_set = set()
            # for each camera pair
            for n_cam1 in range(n_cams):
                for n_cam2 in range(n_cam1 + 1, n_cams):
                    camname_1 = str(expname)+"_"+params["camnames"][n_cam1] if params["dataset"] == "rat7m" else params["camnames"][n_cam1]
                    camname_2 = str(expname)+"_"+params["camnames"][n_cam2] if params["dataset"] == "rat7m" else params["camnames"][n_cam2]

                    pts1 = save_data[sample_id][params["camnames"][n_cam1]]["COM"][joint]
                    pts2 = save_data[sample_id][params["camnames"][n_cam2]]["COM"][joint]
                    pts1 = pts1[np.newaxis, :]
                    pts2 = pts2[np.newaxis, :]
                    
                    # triangulate into 3D
                    test3d = ops.triangulate(
                        pts1,
                        pts2,
                        cameras[camname_1]["cammat"],
                        cameras[camname_2]["cammat"],
                    ).squeeze()

                    keypoints_2d = save_data[sample_id]
                    if ransac:
                        # compute the reprojection errors
                        reproj_errs = ops.cal_reprojection_error(
                            test3d, keypoints_2d, joint, cameras, params["camnames"], prefix
                        )

                        # keep the inlier views
                        new_inlier_set = set([n_cam1, n_cam2])
                        for view in view_set:
                            if reproj_errs[view] < 15:
                                new_inlier_set.add(view)

                        if len(new_inlier_set) > len(inlier_set):
                            inlier_set = new_inlier_set
                    else:
                        save_data[sample_id]["triangulation"][
                            "{}_{}".format(
                                params["camnames"][n_cam1], params["camnames"][n_cam2]
                            )
                        ] = test3d
            
            if ransac:
                inlier_set = np.array(sorted(inlier_set))
                inlier_pts = [save_data[sample_id][params["camnames"][view]]["COM"][joint] for view in inlier_set]
                inlier_pts = [pt[np.newaxis, :] for pt in inlier_pts]
                
                if params["dataset"] == "rat7m":
                    inlier_cams = [cameras[prefix+params["camnames"][view]]["cammat"] for view in inlier_set]
                else:
                    inlier_cams = [cameras[params["camnames"][view]]["cammat"] for view in inlier_set]
                
                final = ops.triangulate_multi_instance(inlier_pts, inlier_cams)
                final = np.squeeze(final)

                if direct_optimization:
                    from scipy.optimize import least_squares
                    def residual_function(x):
                        residuals = ops.cal_reprojection_error(
                            x, save_data[sample_id], joint, cameras, np.array(params["camnames"])[inlier_set], prefix
                        )[0]
                        return residuals
                    x0 = final
                    res = least_squares(residual_function, x0, loss="huber", method="trf")
                    final = res.x

            else:
                pairs = [
                    v for v in save_data[sample_id]["triangulation"].values() if len(v) == 3
                ]   
                pairs = np.stack(pairs, axis=1)
                # find final reconstructed points by taking their median
                final = np.nanmedian(pairs, axis=1).squeeze()
            save_data[sample_id]["joints"][joint] = final
    
    pose3d = np.stack([v["joints"] for k, v in save_data.items()], axis=0) #[N, 3, 20]
    pose2d = []
    for k, v in save_data.items():
        pose = [v[cam]["COM"] for cam in params["camnames"]]
        pose2d.append(np.stack(pose, axis=0))
    pose2d = np.stack(pose2d, axis=0)

    sio.savemat(
        os.path.join(params["dannce_predict_dir"], "pred{}.mat".format(params["start_sample"])),
        {
            "pred": pose3d,
            "pose2d": pose2d,
        },
    )
    return