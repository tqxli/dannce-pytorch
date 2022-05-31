import numpy as np
from typing import Dict
import shutil
import os, time
import torch
from copy import deepcopy

from dannce.engine.data import serve_data_DANNCE, dataset, generator, processing
from dannce.interface import make_folder
import dannce.config as config
from dannce.engine.logging.logger import setup_logging, get_logger
from dannce.engine.trainer.motiondannce_trainer import MotionDANNCETrainer
from dannce.engine.models.nets import initialize_train, initialize_model
from dannce.engine.models.motion_discriminator import MotionDiscriminator, TemporalEncoder
from dannce.engine.data.processing import savedata_tomat, savedata_expval

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
    params["use_temporal"] = True
    params["temporal_chunk_size"] = params["batch_size"] * custom_model_params["accumulation_step"]
    params["downsample"] = downsample = custom_model_params["downsample"]

    with_temporal_encoder = "temporal_encoder" in custom_model_params.keys()
    with_motion_discriminator = "motion_discriminator" in custom_model_params.keys()

    # Make the training directory if it does not exist.
    make_folder("dannce_train_dir", params)

    # setup logger
    setup_logging(params["dannce_train_dir"])
    logger = get_logger("training.log", verbosity=2)

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
    cam3_train = True if params["cam3_train"] else False
    tifdirs = []  # Training from single images not yet supported in this demo

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
            #"occlusion": params["downscale_occluded_view"],
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
    
    y_train_aux, y_valid_aux = None, None
    if (not params["use_npy"]) and (params["social_training"]):
        X_train, X_train_grid, y_train, y_train_aux = processing.align_social_data(X_train, X_train_grid, y_train, y_train_aux)
        X_valid, X_valid_grid, y_valid, y_valid_aux = processing.align_social_data(X_valid, X_valid_grid, y_valid, y_valid_aux)
    
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

    train_dataloader = torch.utils.data.DataLoader(
        train_generator, batch_size=1, shuffle=True, collate_fn=serve_data_DANNCE.collate_fn,
        # num_workers=params["batch_size"],
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_generator, batch_size=1, shuffle=False, collate_fn=serve_data_DANNCE.collate_fn,
        # num_workers=params["batch_size"],
    )

    # initialize mocap dataset
    mocap_dataset = dataset.RAT7MSeqDataset(
        downsample=downsample, 
        seqlen=params["temporal_chunk_size"], 
        root=custom_model_params["mocap_root"]
    )
    mocap_dataloader = torch.utils.data.DataLoader(mocap_dataset, 1, shuffle=True)

    # Build network
    logger.info("Initializing Network...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    posenet, _, lr_scheduler = initialize_train(params, n_cams, device, logger)
    
    # temporal encoder
    temporal_encoder = TemporalEncoder(
        input_size=params["n_channels_out"]*3, 
        **custom_model_params["temporal_encoder"]
    ).to(device) if with_temporal_encoder else None

    # motion discriminator
    motion_discriminator = MotionDiscriminator(
        input_size=mocap_dataset.input_shape,
        **custom_model_params["motion_discriminator"]
    ).to(device) if with_motion_discriminator else None

    # use separate optimizer for generator and discriminator
    gen_params = [p for p in posenet.parameters() if p.requires_grad] 
    if with_temporal_encoder:
        gen_params += [p for p in temporal_encoder.parameters() if p.requires_grad]

    gen_optimizer = torch.optim.Adam(gen_params, lr=params["lr"])

    if motion_discriminator is not None:
        disc_params = [p for p in motion_discriminator.parameters() if p.requires_grad]
        disc_optimizer = torch.optim.Adam(disc_params, lr=custom_model_params["optim"]["lr"])
    else:
        disc_optimizer = None

    logger.info("COMPLETE\n")

    # set up trainer
    trainer = MotionDANNCETrainer(
        params=params,
        # model
        model=posenet,
        temporal_encoder=temporal_encoder,
        motion_discriminator=motion_discriminator,
        disc_loss_weight= custom_model_params["disc_loss_weight"],
        # data
        motion_loader=mocap_dataloader,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        # train
        accumulation_step=custom_model_params["accumulation_step"],
        optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        device=device,
        logger=logger,
        visualize_batch=False,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()

def inference(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    make_folder("dannce_predict_dir", params)

    params, valid_params = config.setup_predict(params)

    # handle custom model parameters
    custom_model_params = params["custom_model"]
    accumulation_step = custom_model_params["accumulation_step"]
    params["downsample"] = downsample = custom_model_params["downsample"]
    params["batch_size"] *= accumulation_step

    valid_params["batch_size"] = params["batch_size"]

    if isinstance(params['maxbatch'], (int, np.integer)):
        params["maxbatch"] = int(params["maxbatch"] / accumulation_step)

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
    posenet = initialize_model(params, len(camnames[0]), device)
    # temporal encoder
    temporal_encoder = TemporalEncoder(
        input_size=params["n_channels_out"]*3, 
        **custom_model_params["temporal_encoder"]
    ).to(device)

    # load prediction checkpoint (no discriminator)
    checkpoint = torch.load(params["dannce_predict_model"])
    posenet.load_state_dict(checkpoint['posenet_state_dict'])
    posenet.eval()

    temporal_encoder.load_state_dict(checkpoint['temporal_encoder_state_dict'])
    temporal_encoder.eval()

    if params["maxbatch"] != "max" and params["maxbatch"] > len(predict_generator):
        print(
            "Maxbatch was set to a larger number of matches than exist in the video. Truncating"
        )
        config.print_and_set(params, "maxbatch", len(predict_generator))

    if params["maxbatch"] == "max":
        config.print_and_set(params, "maxbatch", len(predict_generator))

    if params["write_npy"] is not None:
        print("Writing samples to .npy files")
        processing.write_npy(params["write_npy"], predict_generator)
        return
    
    if params["write_visual_hull"] is not None:
        print("Writing visual hull to .npy files")
        processing.write_sil_npy(params["write_visual_hull"], predict_generator_sil)

    # inference
    save_heatmaps=False
    end_time = time.time()
    save_data = {}
    start_ind = params["start_batch"]
    end_ind = params["maxbatch"]

    if save_heatmaps:
        save_path = os.path.join(params["dannce_predict_dir"], "heatmaps")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for idx, i in enumerate(range(start_ind, end_ind)):
        print("Predicting on batch {}".format(i), flush=True)
        if (i - start_ind) % 10 == 0 and i != start_ind:
            print(i)
            print("10 batches took {} seconds".format(time.time() - end_time))
            end_time = time.time()

        if (i - start_ind) % 1000 == 0 and i != start_ind:
            print("Saving checkpoint at {}th batch".format(i))
            if params["expval"]:
                p_n = savedata_expval(
                    params["dannce_predict_dir"] + "save_data_AVG.mat",
                    params,
                    write=True,
                    data=save_data,
                    tcoord=False,
                    num_markers=params["n_markers"],
                    pmax=True,
                )
            else:
                p_n = savedata_tomat(
                    params["dannce_predict_dir"] + "save_data_MAX.mat",
                    params,
                    params["vmin"],
                    params["vmax"],
                    params["nvox"],
                    write=True,
                    data=save_data,
                    num_markers=params["n_markers"],
                    tcoord=False,
                )

        X = predict_generator.__getitem__(i)[0]
        volumes = torch.FloatTensor(X[0]).permute(0, 4, 1, 2, 3).to(device)
        grid_centers = torch.FloatTensor(X[1]).to(device)

        inputs = torch.split(volumes, volumes.shape[0] // 2, dim=0)
        grids = torch.split(grid_centers, volumes.shape[0] // 2, dim=0)

        fake_motion_seq, all_heatmaps = [], []
        for j in range(2):
            # regress 3D poses [BS, 3, N_JOINTS]
            keypoints_3d_pred, heatmaps = posenet(inputs[j], grids[j])

            fake_motion_seq.append(keypoints_3d_pred)
            all_heatmaps.append(heatmaps)

        all_heatmaps = torch.cat(all_heatmaps, dim=0)

        fake_motion_seq = torch.cat(fake_motion_seq, dim=0)
        fake_motion_seq = fake_motion_seq.reshape(1, fake_motion_seq.shape[0], -1)

        fake_motion_seq = temporal_encoder(fake_motion_seq)

        pred = fake_motion_seq.reshape(fake_motion_seq.shape[1], 3, -1)

        # breakpoint()
        if params["expval"]:
            probmap = torch.amax(all_heatmaps, dim=(2, 3, 4)).squeeze(0).detach().cpu().numpy()
            heatmaps = all_heatmaps.squeeze().detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            for j in range(pred.shape[0]):
                pred_max = probmap[j]
                sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]
                save_data[idx * pred.shape[0] + j] = {
                    "pred_max": pred_max,
                    "pred_coord": pred[j],
                    "sampleID": sampleID,
                }
                if save_heatmaps:
                    np.save(os.path.join(save_path, sampleID), heatmaps[j])
    # breakpoint()
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