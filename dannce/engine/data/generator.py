"""Generator module for dannce training.
"""
import os
import numpy as np
from dannce.engine.data import processing, ops
from dannce.engine.data.video import LoadVideoFrame
import warnings
import time
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Dict, Tuple, Text

import tensorflow as tf
import torch
import torchvision
import torchvision.transforms.functional as TF

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MISSING_KEYPOINTS_MSG = (
    "If mirror augmentation is used, the right_keypoints indices and left_keypoints "
    + "indices must be specified as well. "
    + "For the skeleton, ['RHand, 'LHand', 'RFoot', 'LFoot'], "
    + "set right_keypoints: [0, 2] and left_keypoints: [1, 3] in the config file"
)

"""
Notes: 
DataGenerator_3Dconv: returns BATCHED volumes without augmentation.
    Without use of data loaders and custom collate function during inference.

DataGenerator_3Dconv_frommem and its children: return chunked (minimum 1), augmented volumes.
"""

class DataGenerator(torch.utils.data.Dataset):
    """
    Attributes:
        batch_size (int): Batch size to generate
        camnames (List): List of camera names.
        clusterIDs (List): List of sampleIDs
        crop_height (Tuple): (first, last) pixels in image height
        crop_width (tuple): (first, last) pixels in image width
        currvideo (Dict): Contains open video objects
        currvideo_name (Dict): Contains open video object names
        dim_in (Tuple): Input dimension
        dim_out (Tuple): Output dimension
        extension (Text): Video extension
        indexes (np.ndarray): sample indices used for batch generation
        labels (Dict): Label dictionary
        list_IDs (List): List of sampleIDs
        mono (bool): If True, use grayscale image.
        n_channels_in (int): Number of input channels
        n_channels_out (int): Number of output channels
        out_scale (int): Scale of the output gaussians.
        samples_per_cluster (int): Samples per cluster
        shuffle (bool): If True, shuffle the samples.
        vidreaders (Dict): Dict containing video readers.
        predict_flag (bool): If True, use imageio for reading videos, rather than OpenCV
    """

    def __init__(
        self,
        list_IDs: List,
        labels: Dict,
        clusterIDs: List,
        batch_size: int = 32,
        dim_in: Tuple = (32, 32, 32),
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        out_scale: float = 5,
        shuffle: bool = True,
        camnames: List = [],
        crop_width: Tuple = (0, 1024),
        crop_height: Tuple = (20, 1300),
        samples_per_cluster: int = 0,
        vidreaders: Dict = None,
        chunks: int = 3500,
        mono: bool = False,
        mirror: bool = False,
        predict_flag: bool = False,
    ):
        """Initialize Generator.
        """
        self.dim_in = dim_in
        self.dim_out = dim_in
        self.batch_size = batch_size
        self.labels = labels
        self.vidreaders = vidreaders
        self.list_IDs = list_IDs
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        # sigma for the ground truth joint probability map Gaussians
        self.out_scale = out_scale
        self.camnames = camnames
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.clusterIDs = clusterIDs
        self.samples_per_cluster = samples_per_cluster
        self._N_VIDEO_FRAMES = chunks
        self.mono = mono
        self.mirror = mirror
        self.predict_flag = predict_flag

        if self.vidreaders is not None:
            self.extension = (
                "." + list(vidreaders[camnames[0][0]].keys())[0].rsplit(".")[-1]
            )

        assert len(self.list_IDs) == len(self.clusterIDs)

        self.load_frame = LoadVideoFrame(
            self._N_VIDEO_FRAMES, self.vidreaders, self.camnames, self.predict_flag
        )

    def __len__(self) -> int:
        """Denote the number of batches per epoch.

        Returns:
            int: Batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

class DataGenerator_3Dconv(DataGenerator):
    """Update generator class to handle multiple experiments.
    """

    def __init__(
        self,
        list_IDs: List,
        labels: Dict,
        labels_3d: Dict,
        camera_params: Dict,
        clusterIDs: List,
        com3d: Dict,
        tifdirs: List,
        batch_size: int = 32,
        dim_in: Tuple = (32, 32, 32),
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        out_scale: int = 5,
        shuffle: bool = True,
        camnames: List = [],
        crop_width: Tuple = (0, 1024),
        crop_height: Tuple = (20, 1300),
        vmin: int = -100,
        vmax: int = 100,
        nvox: int = 32,
        gpu_id: Text = "0",
        interp: Text = "linear",
        depth: bool = False,
        channel_combo=None,
        mode: Text = "3dprob",
        samples_per_cluster: int = 0,
        immode: Text = "tif",
        rotation: bool = False,
        vidreaders: Dict = None,
        distort: bool = True,
        expval: bool = False,
        multicam: bool = True,
        var_reg: bool = False,
        COM_aug: bool = None,
        crop_im: bool = True,
        norm_im: bool = True,
        chunks: int = 3500,
        mono: bool = False,
        mirror: bool = False,
        predict_flag: bool = False,
        segmentation_model=None,
    ):
        """Initialize data generator.

        Args:
            list_IDs (List): List of sample Ids
            labels (Dict): Dictionary of labels
            labels_3d (Dict): Dictionary of 3d labels.
            camera_params (Dict): Camera parameters dictionary.
            clusterIDs (List): List of sample Ids
            com3d (Dict): Dictionary of com3d data.
            tifdirs (List): Directories of .tifs
            batch_size (int, optional): Batch size to generate
            dim_in (Tuple, optional): Input dimension
            n_channels_in (int, optional): Number of input channels
            n_channels_out (int, optional): Number of output channels
            out_scale (int, optional): Scale of the output gaussians.
            shuffle (bool, optional): If True, shuffle the samples.
            camnames (List, optional): List of camera names.
            crop_width (Tuple, optional): (first, last) pixels in image width
            crop_height (Tuple, optional): (first, last) pixels in image height
            vmin (int, optional): Minimum box dim (relative to the COM)
            vmax (int, optional): Maximum box dim (relative to the COM)
            nvox (int, optional): Number of voxels per box side
            gpu_id (Text, optional): Identity of GPU to use.
            interp (Text, optional): Interpolation method.
            depth (bool): If True, appends voxel depth to sampled image features [DEPRECATED]
            channel_combo (Text): Method for shuffling camera input order
            mode (Text): Toggles output label format to match MAX vs. AVG network requirements.
            samples_per_cluster (int, optional): Samples per cluster
            immode (Text): Toggles using 'video' or 'tif' files as image input [DEPRECATED]
            rotation (bool, optional): If True, use simple rotation augmentation.
            vidreaders (Dict, optional): Dict containing video readers.
            distort (bool, optional): If true, apply camera undistortion.
            expval (bool, optional): If True, process an expected value network (AVG)
            multicam (bool): If True, formats data to work with multiple cameras as input.
            var_reg (bool): If True, adds a variance regularization term to the loss function.
            COM_aug (bool, optional): If True, augment the COM.
            crop_im (bool, optional): If True, crop images.
            norm_im (bool, optional): If True, normalize images.
            chunks (int, optional): Size of chunks when using chunked mp4.
            mono (bool, optional): If True, use grayscale image.
            predict_flag (bool, optional): If True, use imageio for reading videos, rather than OpenCV
        """
        DataGenerator.__init__(
            self,
            list_IDs,
            labels,
            clusterIDs,
            batch_size,
            dim_in,
            n_channels_in,
            n_channels_out,
            out_scale,
            shuffle,
            camnames,
            crop_width,
            crop_height,
            samples_per_cluster,
            vidreaders,
            chunks,
            mono,
            mirror,
            predict_flag,
        )
        self.vmin = vmin
        self.vmax = vmax
        self.nvox = nvox
        self.vsize = (vmax - vmin) / nvox
        self.dim_out_3d = (nvox, nvox, nvox)
        self.labels_3d = labels_3d
        self.camera_params = camera_params
        self.interp = interp
        self.depth = depth
        self.channel_combo = channel_combo
        print(self.channel_combo)
        self.gpu_id = gpu_id
        self.mode = mode
        self.immode = immode
        self.tifdirs = tifdirs
        self.com3d = com3d
        self.rotation = rotation
        self.distort = distort
        self.expval = expval
        self.multicam = multicam
        self.var_reg = var_reg
        self.COM_aug = COM_aug
        self.crop_im = crop_im
        # If saving npy as uint8 rather than training directly, dont normalize
        self.norm_im = norm_im

        self.device = torch.device("cuda:" + self.gpu_id)

        self.threadpool = ThreadPool(len(self.camnames[0]))
        self.segmentation_model = segmentation_model

        ts = time.time()

        for ID in list_IDs:
            experimentID = int(ID.split("_")[0])
            for camname in self.camnames[experimentID]:
                # M only needs to be computed once for each camera
                K = self.camera_params[experimentID][camname]["K"]
                R = self.camera_params[experimentID][camname]["R"]
                t = self.camera_params[experimentID][camname]["t"]
                M = torch.as_tensor(
                    ops.camera_matrix(K, R, t), dtype=torch.float32
                )
                self.camera_params[experimentID][camname]["M"] = M

        print("Init took {} sec.".format(time.time() - ts))

        self.pj_method = self.pj_grid_mirror if self.mirror else self.pj_grid

    def __getitem__(self, index: int):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data X
                (np.ndarray): Input volume y
                (np.ndarray): Target
        """
        # Find list of IDs
        list_IDs_temp = self.list_IDs[index*self.batch_size : (index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def pj_grid(self, X_grid, camname, ID, experimentID):
        """Projects 3D voxel centers and sample images as projected 2D pixel coordinates

        Args:
            X_grid (np.ndarray): 3-D array containing center coordinates of each voxel.
            camname (Text): camera name
            ID (Text): string denoting a sample ID
            experimentID (int): identifier for a video recording session.

        Returns:
            np.ndarray: projected voxel centers, now in 2D pixels
        """
        ts = time.time()
        # Need this copy so that this_y does not change
        this_y = torch.as_tensor(self.labels[ID]["data"][camname], dtype=torch.float32, device=self.device).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0])*float("nan")
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, axis=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, axis=1)

        thisim = self.load_frame.load_vid_frame(
            self.labels[ID]["frames"][camname],
            camname,
            extension=self.extension,
        )[
            self.crop_height[0] : self.crop_height[1],
            self.crop_width[0] : self.crop_width[1],
        ]
        return self.pj_grid_post(
            X_grid, camname, ID, experimentID, com, com_precrop, thisim
        )

    def pj_grid_mirror(self, X_grid, camname, ID, experimentID, thisim):
        this_y = torch.as_tensor(
            self.labels[ID]["data"][camname],
            dtype=torch.float32,
            device=self.device,
        ).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0]) * float("nan")
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, dim=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, dim=1)

        if not self.mirror:
            raise Exception(
                "Trying to project onto mirrored images without mirror being set properly"
            )

        if self.camera_params[experimentID][camname]["m"] == 1:
            passim = thisim[-1::-1].copy()
        elif self.camera_params[experimentID][camname]["m"] == 0:
            passim = thisim.copy()
        else:
            raise Exception("Invalid mirror parameter, m, must be 0 or 1")

        return self.pj_grid_post(
            X_grid, camname, ID, experimentID, com, com_precrop, passim
        )

    def pj_grid_post(self, X_grid, camname, ID, experimentID, com, com_precrop, thisim):
        # separate the porjection and sampling into its own function so that
        # when mirror == True, this can be called directly
        if self.crop_im:
            if torch.all(torch.isnan(com)):
                thisim = torch.zeros(
                    (self.dim_in[1], self.dim_in[0], self.n_channels_in),
                    dtype=torch.uint8,
                    device=self.device,
                )
            else:
                thisim, _ = processing.cropcom(thisim, com, size=self.dim_in[0])
        # print('Frame loading took {} sec.'.format(time.time() - ts))

        if self.segmentation_model is not None:
            input = [torchvision.transforms.functional.to_tensor(thisim.copy()).to(self.device,  dtype=torch.float)]
            prediction = self.segmentation_model(input)[0]
            
            mask = prediction["masks"][0].permute(1, 2, 0).detach().cpu().numpy()
            mask = (mask >= 0.5).astype(np.uint8)
            
            thisim *= mask # return the segmented foreground object

        ts = time.time()
        proj_grid = ops.project_to2d(
            X_grid, self.camera_params[experimentID][camname]["M"], self.device
        )
        # print('Project2d took {} sec.'.format(time.time() - ts))

        ts = time.time()
        if self.distort:
            proj_grid = ops.distortPoints(
                proj_grid[:, :2],
                self.camera_params[experimentID][camname]["K"],
                np.squeeze(self.camera_params[experimentID][camname]["RDistort"]),
                np.squeeze(self.camera_params[experimentID][camname]["TDistort"]),
                self.device,
            )
            proj_grid = proj_grid.transpose(0, 1)
            # print('Distort took {} sec.'.format(time.time() - ts))

        ts = time.time()
        if self.crop_im:
            proj_grid = proj_grid[:, :2] - com_precrop + self.dim_in[0] // 2
            # Now all coordinates should map properly to the image cropped around the COM
        else:
            # Then the only thing we need to correct for is crops at the borders
            proj_grid = proj_grid[:, :2]
            proj_grid[:, 0] = proj_grid[:, 0] - self.crop_width[0]
            proj_grid[:, 1] = proj_grid[:, 1] - self.crop_height[0]

        rgb = ops.sample_grid(thisim, proj_grid, self.device, method=self.interp)
        # print('Sample grid {} sec.'.format(time.time() - ts))

        if (
            ~torch.any(torch.isnan(com_precrop))
            or (self.channel_combo == "avg")
            or not self.crop_im
        ):
            X = rgb.permute(0, 2, 3, 4, 1)

        return X
    
    def _init_vars(self, first_exp):
        X = torch.zeros(
            (
                self.batch_size * len(self.camnames[first_exp]),
                *self.dim_out_3d,
                self.n_channels_in + self.depth,
            ),
            dtype=torch.uint8,
            device=self.device,
        )

        if self.mode == "3dprob":
            y_3d = torch.zeros(
                (self.batch_size, self.n_channels_out, *self.dim_out_3d),
                dtype=torch.float32,
                device=self.device,
            )
        elif self.mode == "coordinates":
            y_3d = torch.zeros(
                (self.batch_size, 3, self.n_channels_out),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise Exception("not a valid generator mode")

        sz = self.dim_out_3d[0] * self.dim_out_3d[1] * self.dim_out_3d[2]
        X_grid = torch.zeros(
            (self.batch_size, sz, 3),
            dtype=torch.float32,
            device=self.device,
        )

        return X, y_3d, X_grid
    
    def _generate_coord_grid(self, this_COM_3d):
        xgrid = torch.arange(
            self.vmin + this_COM_3d[0] + self.vsize / 2,
            this_COM_3d[0] + self.vmax,
            self.vsize,
            dtype=torch.float32,
            device=self.device,
        )
        ygrid = torch.arange(
            self.vmin + this_COM_3d[1] + self.vsize / 2,
            this_COM_3d[1] + self.vmax,
            self.vsize,
            dtype=torch.float32,
            device=self.device,
        )
        zgrid = torch.arange(
            self.vmin + this_COM_3d[2] + self.vsize / 2,
            this_COM_3d[2] + self.vmax,
            self.vsize,
            dtype=torch.float32,
            device=self.device,
        )
        (x_coord_3d, y_coord_3d, z_coord_3d) = torch.meshgrid(
            xgrid, ygrid, zgrid
        )

        grid = torch.stack(
            (
            x_coord_3d.transpose(0, 1).flatten(),
            y_coord_3d.transpose(0, 1).flatten(),
            z_coord_3d.transpose(0, 1).flatten(),
            ),
            dim=1,
        )

        return (x_coord_3d, y_coord_3d, z_coord_3d), grid

    def _generate_targets(self, i, y_3d, this_y_3d, coords_3d):
        if self.mode == "3dprob":
            # generate Gaussian targets
            for j in range(self.n_channels_out):
                y_3d[i, j] = torch.exp(
                    -(
                        (coords_3d[1] - this_y_3d[1, j]) ** 2
                        + (coords_3d[0] - this_y_3d[0, j]) ** 2
                        + (coords_3d[2] - this_y_3d[2, j]) ** 2
                    )
                    / (2 * self.out_scale ** 2)
                )
                # When the voxel grid is coarse, we will likely miss
                # the peak of the probability distribution, as it
                # will lie somewhere in the middle of a large voxel.
                # So here we renormalize to [~, 1]
        
        if self.mode == "coordinates":
            if this_y_3d.shape == y_3d[i].shape:
                y_3d[i] = this_y_3d
            else:
                msg = "Note: ignoring dimension mismatch in 3D labels"
                warnings.warn(msg)
        
        return y_3d

    def _adjust_vol_channels(self, X, y_3d, first_exp, num_cams):
        if self.multicam:
            X = X.reshape(
                (
                    self.batch_size,
                    len(self.camnames[first_exp]),
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    X.shape[4],
                )
            )
            X = X.permute(0, 2, 3, 4, 5, 1)

            if self.channel_combo == "avg":
                X = torch.mean(X, dim=-1)

            # Randomly reorder the cameras fed into the first layer
            elif self.channel_combo == "random":
                X = X[..., torch.randperm(X.shape[-1])]
                X = X.transpose(4, 5).reshape(*X.shape[:4], -1)
            else:
                X = X.transpose(4, 5).reshape(*X.shape[:4], -1)
        else:
            # Then leave the batch_size and num_cams combined
            y_3d = y_3d.repeat(num_cams, 1, 1, 1, 1)
        
        return X, y_3d
 
    def _convert_tensor_to_numpy(self, X, y_3d, X_grid):
        # ts = time.time()
        if torch.is_tensor(X):
            X = X.float().cpu().numpy()
        if torch.is_tensor(y_3d):
            y_3d = y_3d.cpu().numpy()
        if torch.is_tensor(X_grid):
            X_grid = X_grid.cpu().numpy()
        # print('Numpy took {} sec'.format(time.time() - ts))

        return X, y_3d, X_grid

    def _finalize_samples(self, X, y_3d, X_grid):
        if self.var_reg or self.norm_im:
            X = processing.preprocess_3d(X)

        inputs, targets = [X], [y_3d]

        if self.expval:
            inputs.append(X_grid)
        
        if self.var_reg:
            targets.append(torch.zeros((self.batch_size, 1)))
        
        return inputs, targets

    def __data_generation(self, list_IDs_temp):
        """
        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d: Targets
        Raises:
            Exception: Invalid generator mode specified.
        """
        # Initialization
        first_exp = int(self.list_IDs[0].split("_")[0])
        X, y_3d, X_grid = self._init_vars(first_exp)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            num_cams = len(self.camnames[experimentID])

            # For 3D ground truth (keypoints, COM)
            this_y_3d = torch.as_tensor(self.labels_3d[ID],
                dtype=torch.float32,
                device=self.device,
            )
            this_COM_3d = torch.as_tensor(
                self.com3d[ID], 
                dtype=torch.float32, 
                device=self.device
            )

            # Create and project the grid here,
            coords_3d, grid = self._generate_coord_grid(this_COM_3d)
            X_grid[i] = grid
            
            # Generate training targets
            y_3d = self._generate_targets(i, y_3d, this_y_3d, coords_3d)

            # Compute projected images in parallel using multithreading
            # ts = time.time()
            arglist = []
            if self.mirror:
                # Here we only load the video once, and then parallelize the projection
                # and sampling after mirror flipping. For setups that collect views
                # in a single image with the use of mirrors
                loadim = self.load_frame.load_vid_frame(
                    self.labels[ID]["frames"][self.camnames[experimentID][0]],
                    self.camnames[experimentID][0],
                    extension=self.extension,
                )[
                    self.crop_height[0] : self.crop_height[1],
                    self.crop_width[0] : self.crop_width[1],
                ]

            for c in range(num_cams):
                args = [X_grid[i], self.camnames[experimentID][c], ID, experimentID]
                if self.mirror:
                    args.append(loadim)
                
                arglist.append(args)

            result = self.threadpool.starmap(self.pj_method, arglist)

            for c in range(num_cams):
                ic = c + i * num_cams
                X[ic, :, :, :, :] = result[c]
            # print('MP took {} sec.'.format(time.time()-ts))

        # adjust volume channels
        X, y_3d = self._adjust_vol_channels(X, y_3d, first_exp, num_cams)
        
        # 3dprob is required for *training* MAX networks
        if self.mode == "3dprob":
            y_3d = y_3d.permute([0, 2, 3, 4, 1])

        if self.mono and self.n_channels_in == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = X.reshape(*X.shape[:4], len(self.camnames[first_exp]), -1)
            X = X[..., 0] * 0.2125 + X[..., 1] * 0.7154 + X[..., 2] * 0.0721

        # Convert pytorch tensors back to numpy array
        X, y_3d, X_grid = self._convert_tensor_to_numpy(X, y_3d, X_grid)

        return self._finalize_samples(X, y_3d, X_grid)

class DataGenerator_3Dconv_social(DataGenerator_3Dconv):
    def __init__(
        self,
        list_IDs: List,
        labels: Dict,
        labels_3d: Dict,
        camera_params: Dict,
        clusterIDs: List,
        com3d: Dict,
        tifdirs: List,
        n_instances = 2,
        **kwargs
    ):
        DataGenerator_3Dconv.__init__(
            self,
            list_IDs,
            labels,
            labels_3d,
            camera_params,
            clusterIDs,
            com3d,
            tifdirs,
            **kwargs
        )

        self.n_instances = n_instances
        self.batch_size = n_instances
        self.list_IDs = [ID for ID in self.list_IDs if int(ID.split("_")[0]) % n_instances == 0]

    def __getitem__(self, index: int):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data X
                (np.ndarray): Input volume y
                (np.ndarray): Target
        """
        # Find list of IDs
        thisID = self.list_IDs[index]
        experimentID, sampleID = thisID.split("_")

        list_IDs_temp = [str(int(experimentID)+i)+"_"+sampleID for i in range(self.n_instances)]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def proj_grid(self, X_grids, camnames, IDs, experimentIDs, com_3ds):
        """Projects 3D voxel centers and sample images as projected 2D pixel coordinates

        Args:
            X_grid (np.ndarray): 3-D array containing center coordinates of each voxel.
            camname (Text): camera name
            ID (Text): string denoting a sample ID
            experimentID (int): identifier for a video recording session.

        Returns:
            np.ndarray: projected voxel centers, now in 2D pixels
        """
        ts = time.time()

        # Need this copy so that this_y does not change
        thisims, coms, com_precrops = [], [], []

        # only load the frame once for all animals present
        thisim = self.load_frame.load_vid_frame(
            self.labels[IDs[0]]["frames"][camnames[0]],
            camnames[0],
            extension=self.extension,
        )[
            self.crop_height[0] : self.crop_height[1],
            self.crop_width[0] : self.crop_width[1],
        ]
        
        for i in range(self.n_instances):
            this_y = torch.as_tensor(
                self.labels[IDs[i]]["data"][camnames[i]],
                dtype=torch.float32,
                device=self.device,
            ).round()

            if torch.all(torch.isnan(this_y)):
                com_precrop = torch.zeros_like(this_y[:, 0])*float("nan")
            else:
                com_precrop = torch.mean(this_y, axis=1)

            this_y[0, :] = this_y[0, :] - self.crop_width[0]
            this_y[1, :] = this_y[1, :] - self.crop_height[0]
            com = torch.mean(this_y, axis=1)

            coms.append(com)
            com_precrops.append(com_precrop)
            thisims.append(thisim)

        return self.pj_grid_post(
            X_grids, camnames, IDs, experimentIDs, coms, com_precrops, thisims, com_3ds
        )

    def pj_grid_mirror(self, X_grid, camname, ID, experimentID, thisim):
        this_y = torch.as_tensor(
            self.labels[ID]["data"][camname],
            dtype=torch.float32,
            device=self.device,
        ).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0]) * float("nan")
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, dim=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, dim=1)

        if not self.mirror:
            raise Exception(
                "Trying to project onto mirrored images without mirror being set properly"
            )

        if self.camera_params[experimentID][camname]["m"] == 1:
            passim = thisim[-1::-1].copy()
        elif self.camera_params[experimentID][camname]["m"] == 0:
            passim = thisim.copy()
        else:
            raise Exception("Invalid mirror parameter, m, must be 0 or 1")

        return self.pj_grid_post(
            X_grid, camname, ID, experimentID, com, com_precrop, passim
        )

    def visualize_2d(self, im, IDs, camnames, camcoords, bb1, bb2, scores, savedir='./vis_occlusion/2021_07_03_M2_M3'):
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        depths = camcoords[:, 2]
        fname = IDs[0].split("_")[-1] + "_" + camnames[0].split("_")[-1] + ".jpg"
        fig, ax = plt.subplots(1, 1)
        ax.imshow(im)
        ax.set_title("B: {:.2f}, {:.2f} | R: {:.2f}, {:.2f}".format(depths[0], scores[0], depths[1], scores[1]))
        # Create a Rectangle patch
        rect1 = patches.Rectangle((bb1[0], bb1[1]), bb1[2]-bb1[0], bb1[3]-bb1[1], linewidth=1, edgecolor='b', facecolor='none')
        rect2 = patches.Rectangle((bb2[0], bb2[1]), bb2[2]-bb2[0], bb2[3]-bb2[1], linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        fig.savefig(os.path.join(savedir, fname))
        plt.close(fig)
    
    @classmethod
    def apply_mask(self, image, mask, color, alpha=0.3):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
        return image

    def visualize_mask(self, IDs, camnames, im, masks, coms, msg, savedir='./vis_occlusion_masks'):
        colors = [(0, 255, 125), (252, 51, 51)]
        if not os.path.exists(savedir):
            print("Saving to ", savedir)
            os.makedirs(savedir)
        fname = IDs[0].split("_")[-1] + "_" + camnames[0].split("_")[-1] + ".jpg"

        fig, ax = plt.subplots(1, 1)
        for i, mask in enumerate(masks):
            im = self.apply_mask(im, np.squeeze(mask), colors[i])
        ax.imshow(im)

        com1 = coms[0]
        com2 = coms[1]
        ax.scatter(x=com1[0], y=com1[1], color="blue")
        ax.scatter(x=com2[0], y=com2[1], color="red")

        ax.set_title(msg)
        
        fig.savefig(os.path.join(savedir, fname))
        plt.close(fig)

    def pj_grid_post(self, X_grids, camnames, IDs, experimentIDs, coms, com_precrops, thisims, com_3ds):
        # separate the porjection and sampling into its own function so that
        # when mirror == True, this can be called directly
        if self.crop_im:
            for i in range(self.n_instances):
                if torch.all(torch.isnan(coms[i])):
                    thisims[i] = torch.zeros(
                        (self.dim_in[1], self.dim_in[0], self.n_channels_in),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                else:
                    thisims[i] = processing.cropcom(thisims[i], coms[i], size=self.dim_in[0])[0] 
        # print('Frame loading took {} sec.'.format(time.time() - ts))
        
        # convert 3D COMs into camera coordinate frame for occlusion check
        com_3ds_cam = ops.world_to_cam(
            com_3ds.clone(), 
            self.camera_params[experimentIDs[0]][camnames[0]]["M"],
            self.device
        ).detach().cpu().numpy()
        depths = com_3ds_cam[:, 2]
        com_2ds_pix = com_3ds_cam[:, :2] / com_3ds_cam[:, 2:]

        # generate bounding boxes in pixel coordinates
        # this only APPROXIMATES the occlusion between different animals
        w, h = 200, 150
        bb1 = [com_2ds_pix[0, 0]-w, com_2ds_pix[0, 1]-h, com_2ds_pix[0, 0]+w, com_2ds_pix[0, 1]+h]
        bb2 = [com_2ds_pix[1, 0]-w, com_2ds_pix[1, 1]-h, com_2ds_pix[1, 0]+w, com_2ds_pix[1, 1]+h]

        # check depths
        instance_front, instance_back = np.argmin(depths), np.argmax(depths)
        occlusion_scores = np.ones((self.n_instances)) # the foreground animal is not occluded

        # check overlap region
        occlusion_scores[instance_back] = processing.bbox_iou(bb1, bb2)

        # self.visualize_2d(thisims[0], IDs, camnames, com_3ds_cam, bb1, bb2, occlusion_scores)
        
        if self.segmentation_model is not None:
            # if there is no occlusion, 
            # the model should be able to detect `n_instances`` masks 
            # in this specific camera view, with high confidence (score > 0.9)
            # occlusion_flag = (occlusion_scores[instance_front] > 0.8)
            # initialize masks as ones ==> no influence if no masks predicted
            masks = np.ones((self.n_instances, *thisims[0].shape[:2], 1))

            # get mask predictions
            input = [torchvision.transforms.functional.to_tensor(thisims[0].copy()).to(self.device,  dtype=torch.float)]
            prediction = self.segmentation_model(input)[0]
            # filter by confidence scores
            filtering_by_scores = (prediction["scores"] > 0.85)
            raw_masks = prediction["masks"][filtering_by_scores]
            # print(f"{len(raw_masks)} masks detected.")
            if len(raw_masks) > self.n_instances:
                raw_masks = raw_masks[:self.n_instances]

            # adjust dimension, convert to numpy
            masks_unordered = []
            for j in range(len(raw_masks)):
                mask = raw_masks[j].permute(1, 2, 0).detach().cpu().numpy()
                masks_unordered.append((mask >= 0.5).astype(np.uint8))
            masks_unordered = np.stack(masks_unordered)

            # mask matching
            msg = ""
            if len(raw_masks) == 0:
                msg = "No mask predicted."
            
            elif len(raw_masks) < self.n_instances:
                counts = processing.compute_support(com_2ds_pix, masks_unordered[0])
                assignment = np.argmax(counts)
                masks[assignment] = masks_unordered[0]
                
                if assignment == instance_front:
                    msg = "Mask only predicted on the foreground animal."
                    non_assignment =((np.arange(self.n_instances)) != assignment)
                    # masks[non_assignment] = (masks_unordered[0] == 0).astype(np.uint8)
                else:
                    msg = "Mask only predicted on animal behind."

            elif len(raw_masks) == self.n_instances:
                # remove intersected region
                # mask_intersect = processing.mask_intersection(masks_unordered[0], masks_unordered[1])
                # masks_unordered = [mask-mask_intersect for mask in masks_unordered]
            
                counts0 = processing.compute_support(com_2ds_pix, masks_unordered[0])
                counts1 = processing.compute_support(com_2ds_pix, masks_unordered[1])
                
                assignment0 = np.argmax(counts0)
                assignment1 = np.argmax(counts1)

                if assignment0 != assignment1:
                    # perfect matching
                    msg = "Perfect matching."
                    masks[assignment0] = masks_unordered[0]
                    masks[assignment1] = masks_unordered[1]
                elif (assignment0 == instance_front):
                    msg = "Mask ambiguity. Assume the higher confidence mask belongs to the front."
                    masks[instance_front] = masks_unordered[0]
                    masks[instance_back] = masks_unordered[1]
                else:
                    msg = "Mask ambiguity."
            
            # self.visualize_mask(IDs, camnames, thisims[0].copy(), masks, com_2ds_pix, msg)            
            for i, im in enumerate(thisims):
                # thisims[i] = im * masks[i]
                thisims[i] = np.tile(masks[i], (1, 1, 3))

        X = []
        for i in range(self.n_instances):
            # ts = time.time()
            proj_grid = ops.project_to2d(
                X_grids[i], self.camera_params[experimentIDs[i]][camnames[i]]["M"], self.device
            )
            # print('Project2d took {} sec.'.format(time.time() - ts))

            # ts = time.time()
            if self.distort:
                proj_grid = ops.distortPoints(
                    proj_grid[:, :2],
                    self.camera_params[experimentIDs[i]][camnames[i]]["K"],
                    np.squeeze(self.camera_params[experimentIDs[i]][camnames[i]]["RDistort"]),
                    np.squeeze(self.camera_params[experimentIDs[i]][camnames[i]]["TDistort"]),
                    self.device,
                )
                proj_grid = proj_grid.transpose(0, 1)
                # print('Distort took {} sec.'.format(time.time() - ts))

            # ts = time.time()
            if self.crop_im:
                proj_grid = proj_grid[:, :2] - com_precrops[i] + self.dim_in[0] // 2
                # Now all coordinates should map properly to the image cropped around the COM
            else:
                # Then the only thing we need to correct for is crops at the borders
                proj_grid = proj_grid[:, :2]
                proj_grid[:, 0] = proj_grid[:, 0] - self.crop_width[0]
                proj_grid[:, 1] = proj_grid[:, 1] - self.crop_height[0]

            rgb = ops.sample_grid(thisims[i], proj_grid, self.device, method=self.interp)

            if (
                ~torch.any(torch.isnan(com_precrops[i]))
                or (self.channel_combo == "avg")
                or not self.crop_im
            ):
                X.append(rgb.permute(0, 2, 3, 4, 1))

        return X, occlusion_scores

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        X : (n_samples, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d: Targets
                rotangle: Rotation angle
        Raises:
            Exception: Invalid generator mode specified.
        """
        # Initialization
        first_exp = int(self.list_IDs[0].split("_")[0])

        X, y_3d, X_grid = self._init_vars(first_exp)

        com_3ds = torch.zeros(
            (self.batch_size, 3),
            dtype=torch.float32,
            device=self.device
        )
        
        # Generate data
        experimentIDs = []
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            experimentIDs.append(experimentID)

            # For 3D ground truth
            this_y_3d = torch.as_tensor(
                self.labels_3d[ID],
                dtype=torch.float32,
                device=self.device,
            )
            this_COM_3d = torch.as_tensor(
                self.com3d[ID], 
                dtype=torch.float32, 
                device=self.device
            )

            com_3ds[i] = this_COM_3d

            # Create and project the grid here,
            coords_3d, grid = self._generate_coord_grid(this_COM_3d)
            X_grid[i] = grid

            # Generate training targets
            y_3d = self._generate_targets(i, y_3d, this_y_3d, coords_3d)

        # Compute projected images in parallel using multithreading
        ts = time.time()
        arglist = []

        num_cams = len(self.camnames[experimentIDs[0]])
        occlusion_scores = np.zeros((self.batch_size, num_cams, 2), dtype=float)

        for c in range(num_cams):  
            arglist.append([
                X_grid, 
                [self.camnames[experimentID][c] for experimentID in experimentIDs], 
                list_IDs_temp, 
                experimentIDs, 
                com_3ds]
            )
        result = self.threadpool.starmap(self.proj_grid, arglist)
        
        for c in range(num_cams):
            for j in range(self.n_instances):
                ic = c + j * num_cams
                X[ic, ...] = result[c][0][j][0] #[H, W, D, C]
                occlusion_scores[j, c] = result[c][1] # [2]
        # print('MP took {} sec.'.format(time.time()-ts))

        # adjust camera channels
        X, y_3d = self._adjust_vol_channels(X, y_3d, first_exp, num_cams)

        # 3dprob is required for *training* MAX networks
        if self.mode == "3dprob":
            y_3d = y_3d.permute([0, 2, 3, 4, 1])

        if self.mono and self.n_channels_in == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = X.reshape(*X.shape[:4], len(self.camnames[first_exp]), -1)
            X = X[..., 0] * 0.2125 + X[..., 1] * 0.7154 + X[..., 2] * 0.0721

        # Convert pytorch tensors back to numpy array
        X, y_3d, X_grid = self._convert_tensor_to_numpy(X, y_3d, X_grid)

        return self._finalize_samples(X, y_3d, X_grid)

class DataGenerator_3Dconv_frommem(torch.utils.data.Dataset):
    """Generate 3d conv data from memory.

    Attributes:
        augment_brightness (bool): If True, applies brightness augmentation
        augment_continuous_rotation (bool): If True, applies rotation augmentation in increments smaller than 90 degrees
        augment_hue (bool): If True, applies hue augmentation
        batch_size (int): Batch size
        bright_val (float): Brightness augmentation range (-bright_val, bright_val), as fraction of raw image brightness
        chan_num (int): Number of input channels
        data (np.ndarray): Image volumes
        expval (bool): If True, crafts input for an AVG network
        hue_val (float): Hue augmentation range (-hue_val, hue_val), as fraction of raw image hue range
        indexes (np.ndarray): Sample indices used for batch generation
        labels (Dict): Label dictionary
        list_IDs (List): List of sampleIDs
        nvox (int): Number of voxels in each grid dimension
        random (bool): If True, shuffles camera order for each batch
        rotation (bool): If True, applies rotation augmentation in 90 degree increments
        rotation_val (float): Range of angles used for continuous rotation augmentation
        shuffle (bool): If True, shuffle the samples before each epoch
        var_reg (bool): If True, returns input used for variance regularization
        xgrid (np.ndarray): For the AVG network, this contains the 3D grid coordinates
        n_rand_views (int): Number of reviews to sample randomly from the full set
        replace (bool): If True, samples n_rand_views with replacement
        aux_labels (np.ndarray): If not None, contains the 3D MAX training targets for AVG+MAX training.
        temporal_chunk_list (np.ndarray, optional): If not None, contains chunked sampleIDs -- useful when loading in temporally contiguous samples
    """

    def __init__(
        self,
        list_IDs,
        data,
        labels,
        # batch_size,
        rotation=True,
        random=True,
        chan_num=3,
        shuffle=True,
        expval=False,
        xgrid=None,
        var_reg=False,
        nvox=64,
        augment_brightness=True,
        augment_hue=True,
        augment_continuous_rotation=True,
        mirror_augmentation=False,
        right_keypoints=None,
        left_keypoints=None,
        bright_val=0.05,
        hue_val=0.05,
        rotation_val=5,
        replace=True,
        n_rand_views=None,
        heatmap_reg=False,
        heatmap_reg_coeff=0.01,
        aux_labels=None,
        temporal_chunk_list=None,
        occlusion=False,
    ):
        """Initialize data generator.
        """
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels
        self.rotation = rotation
        # self.batch_size = batch_size
        self.random = random
        self.chan_num = chan_num
        self.shuffle = shuffle
        self.expval = expval
        self.augment_hue = augment_hue
        self.augment_continuous_rotation = augment_continuous_rotation
        self.augment_brightness = augment_brightness
        self.mirror_augmentation = mirror_augmentation
        self.right_keypoints = right_keypoints
        self.left_keypoints = left_keypoints
        if self.mirror_augmentation and (
            self.right_keypoints is None or self.left_keypoints is None
        ):
            raise Exception(MISSING_KEYPOINTS_MSG)
        self.var_reg = var_reg
        self.xgrid = xgrid
        self.nvox = nvox
        self.bright_val = bright_val
        self.hue_val = hue_val
        self.rotation_val = rotation_val
        self.n_rand_views = n_rand_views
        self.replace = replace
        self.heatmap_reg = heatmap_reg
        self.heatmap_reg_coeff = heatmap_reg_coeff
        self.aux_labels = aux_labels
        self.temporal_chunk_list = temporal_chunk_list
        self.temporal_chunk_size = 1

        self._update_temporal_batch_size()

        self.occlusion = occlusion
        if self.occlusion:
            assert aux_labels is not None, "Missing aux labels for occlusion training."

    def __len__(self):
        """Denote the number of batches per epoch.

        Returns:
            int: Batches per epoch
        """
        if self.temporal_chunk_list is not None:
           return len(self.temporal_chunk_list)

        return len(self.list_IDs)
    
    def _update_temporal_batch_size(self):
        if self.temporal_chunk_list is not None:
            self.temporal_chunk_size = len(self.temporal_chunk_list[0])
            # self.temporal_batch_size = self.batch_size // self.temporal_chunk_size

    def __getitem__(self, index):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data
                X (np.ndarray): Input volume
                y (np.ndarray): Target
        """
        if self.temporal_chunk_list is not None:
            list_IDs_temp = self.temporal_chunk_list[index]
        else:
            list_IDs_temp = [self.list_IDs[index]]
        X, X_grid, y_3d, aux = self.__data_generation(list_IDs_temp)
        return X, X_grid, y_3d, aux

    def rot90(self, X):
        """Rotate X by 90 degrees CCW.

        Args:
            X (np.ndarray): Image volume or grid

        Returns:
            X (np.ndarray): Rotated image volume or grid
        """
        X = np.transpose(X, [1, 0, 2, 3])
        X = X[:, ::-1, :, :]
        return X

    def mirror(self, X, y_3d, X_grid):
        # Flip the image and x coordinates about the x axis
        X = X[:, ::-1, ...]
        X_grid = X_grid[:, ::-1, ...]

        # Flip the left and right keypoints.
        temp = y_3d[..., self.left_keypoints].copy()
        y_3d[..., self.left_keypoints] = y_3d[..., self.right_keypoints]
        y_3d[..., self.right_keypoints] = temp
        return X, y_3d, X_grid

    def rot180(self, X):
        """Rotate X by 180 degrees.

        Args:
            X (np.ndarray): Image volume or grid

        Returns:
            X (np.ndarray): Rotated image volume or grid
        """
        X = X[::-1, ::-1, :, :]
        return X

    def random_rotate(self, X, y_3d, aux=None):
        """Rotate each sample by 0, 90, 180, or 270 degrees.

        Args:
            X (np.ndarray): Image volumes
            y_3d (np.ndarray): 3D grid coordinates (AVG) or training target volumes (MAX)
            aux (np.ndarray or None): Populated in MAX+AVG mode with the training target volumes
        Returns:
            X (np.ndarray): Rotated image volumes
            y_3d (np.ndarray): Rotated 3D grid coordinates (AVG) or training target volumes (MAX)
        """
        # rotation for all volumes within each chunk must be consistent
        rot = np.random.choice(np.arange(4), 1)
        
        if rot == 0:
            pass
        elif rot == 1:
            # Rotate180
            for j in range(self.temporal_chunk_size):
                X[j], y_3d[j] = self.rot180(X[j]), self.rot180(y_3d[j])
                if aux is not None:
                    aux[j] = self.rot180(aux[j])
        elif rot  == 2:
            # Rotate90
            for j in range(self.temporal_chunk_size):
                X[j], y_3d[j] = self.rot90(X[j]), self.rot90(y_3d[j])
                if aux is not None:
                    aux[j] = self.rot90(aux[j])
        elif rot == 3:
            # Rotate -90/270
            for j in range(self.temporal_chunk_size):
                X[j], y_3d[j] = self.rot180(self.rot90(X[j])), self.rot180(self.rot90(y_3d[j]))
                if aux is not None:
                    aux[j] = self.rot180(self.rot90(aux[j]))
    
        if aux is not None:
            return X, y_3d, aux
        return X, y_3d
    
    def random_continuous_rotation(self, X, y_3d, max_delta=5):
        """Rotates X and y_3d a random amount around z-axis.

        Args:
            X (np.ndarray): input image volume
            y_3d (np.ndarray): 3d target (for MAX network) or voxel center grid (for AVG network)
            max_delta (int, optional): maximum range for rotation angle.

        Returns:
            np.ndarray: rotated image volumes
            np.ndarray: rotated grid coordimates
        """
        # rotangle = np.random.rand() * (2 * max_delta) - max_delta
        # X = torch.as_tensor(X).reshape(*X.shape[:3], -1).permute(0, 3, 1, 2) # dimension [B, D*C, H, W]
        # y_3d = torch.as_tensor(y_3d).reshape(y_3d.shape[:3], -1).permute(0, 3, 1, 2)
        # for i in range(X.shape[0]):
        #     X[i] = TF.affine(X[i], angle=rotangle)
        #     y_3d[i] = TF.affine(y_3d[i], angle=rotangle)

        # X = X.permute(0, 2, 3, 1).reshape(*X.shape[:3], X.shape[2], -1).numpy()
        # y_3d = y_3d.permute(0, 2, 3, 1).reshape(*X.shape[:3], X.shape[2], -1).numpy()

        # return X, y_3d
        rotangle = np.random.rand() * (2 * max_delta) - max_delta
        X = tf.reshape(X, [X.shape[0], X.shape[1], X.shape[2], -1]).numpy()
        y_3d = tf.reshape(y_3d, [y_3d.shape[0], y_3d.shape[1], y_3d.shape[2], -1]).numpy()
        for i in range(X.shape[0]):
            X[i] = tf.keras.preprocessing.image.apply_affine_transform(
                X[i],
                theta=rotangle,
                row_axis=0,
                col_axis=1,
                channel_axis=2,
                fill_mode="nearest",
                cval=0.0,
                order=1,
            )
            y_3d[i] = tf.keras.preprocessing.image.apply_affine_transform(
                y_3d[i],
                theta=rotangle,
                row_axis=0,
                col_axis=1,
                channel_axis=2,
                fill_mode="nearest",
                cval=0.0,
                order=1,
            )

        X = tf.reshape(X, [X.shape[0], X.shape[1], X.shape[2], X.shape[2], -1]).numpy()
        y_3d = tf.reshape(
            y_3d,
            [y_3d.shape[0], y_3d.shape[1], y_3d.shape[2], y_3d.shape[2], -1],
        ).numpy()

        return X, y_3d

    def visualize(self, original, augmented):
        """Plots example image after augmentation

        Args:
            original (np.ndarray): image before augmentation
            augmented (np.ndarray): image after augmentation.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Original image")
        plt.imshow(original)

        plt.subplot(1, 2, 2)
        plt.title("Augmented image")
        plt.imshow(augmented)
        plt.show()
        input("Press Enter to continue...")

    def do_augmentation(self, X, X_grid, y_3d, aux=None):
        """Applies augmentation

        Args:
            X (np.ndarray): image volumes
            X_grid (np.ndarray): 3D grid coordinates
            y_3d (np.ndarray): training targets
            aux (np.ndarray or None): additional target volumes if using MAX+AVG mode

        Returns:
            X (np.ndarray): Augemented image volumes
            X_grid (np.ndarray): 3D grid coordinates
            y_3d (np.ndarray): Training targets
        """
        if self.rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = np.reshape(
                    X_grid,
                    (self.temporal_chunk_size, self.nvox, self.nvox, self.nvox, 3),
                )
                if aux is not None:
                    X, X_grid, aux = self.random_rotate(X.copy(), X_grid.copy(), aux.copy())
                else:
                    X, X_grid = self.random_rotate(X.copy(), X_grid.copy())
                # Need to reshape back to raveled version
                X_grid = np.reshape(X_grid, (self.temporal_chunk_size, -1, 3))
            else:
                X, y_3d = self.random_rotate(X.copy(), y_3d.copy())

        if self.augment_continuous_rotation and aux is None:
            if self.expval:
                # First make X_grid 3d
                X_grid = np.reshape(
                    X_grid,
                    (self.temporal_chunk_size, self.nvox, self.nvox, self.nvox, 3),
                )
                X, X_grid = self.random_continuous_rotation(
                    X.copy(), X_grid.copy(), self.rotation_val
                )
                # Need to reshape back to raveled version
                X_grid = np.reshape(X_grid, (self.temporal_chunk_size, -1, 3))
            else:
                X, y_3d = self.random_continuous_rotation(
                    X.copy(), y_3d.copy(), self.rotation_val
                )

        if self.augment_hue and self.chan_num == 3:
            for n_cam in range(int(X.shape[-1] / self.chan_num)):
                channel_ids = np.arange(
                    n_cam * self.chan_num,
                    n_cam * self.chan_num + self.chan_num,
                )
                X[..., channel_ids] = tf.image.random_hue(
                    X[..., channel_ids], self.hue_val
                )
                #X_temp = torch.as_tensor(X[..., channel_ids]).permute(0, 3, 4, 1, 2)
                #random_hue_val = float(torch.empty(1).uniform_(-self.hue_val, self.hue_val))
                #X_temp = TF.adjust_hue(X_temp, random_hue_val)
                #X[..., channel_ids] = X_temp.permute(0, 3, 4, 1, 2).numpy()

        elif self.augment_hue:
            warnings.warn(
                "Trying to augment hue with an image that is not RGB. Skipping."
            )

        if self.augment_brightness:
            for n_cam in range(int(X.shape[-1] / self.chan_num)):
                channel_ids = np.arange(
                    n_cam * self.chan_num,
                    n_cam * self.chan_num + self.chan_num,
                )
                X[..., channel_ids] = tf.image.random_brightness(
                    X[..., channel_ids], self.bright_val
                )
                # X_temp = torch.as_tensor(X[..., channel_ids]).permute(0, 3, 4, 1, 2)
                # random_bright_val = float(torch.empty(1).uniform_(1-self.bright_val, 1+self.bright_val))
                # X_temp = TF.adjust_brightness(X_temp, random_bright_val)
                #X[..., channel_ids] = X_temp.permute(0, 3, 4, 1, 2).numpy()

        if self.mirror_augmentation and self.expval and aux is None:
            if np.random.rand() > 0.5:
                X_grid = np.reshape(
                    X_grid,
                    (self.temporal_chunk_size, self.nvox, self.nvox, self.nvox, 3),
                )
                # Flip the image and the symmetric keypoints
                X, y_3d, X_grid = self.mirror(X.copy(), y_3d.copy(), X_grid.copy())
                X_grid = np.reshape(X_grid, (self.temporal_chunk_size, -1, 3))
        else:
            pass
            ##TODO: implement mirror augmentation for max and avg+max modes

        if self.occlusion:
            if np.random.rand() > 0.5:
                occlusion_idx = np.random.choice(self.__len__())
                rand_cam = np.random.choice(int(X.shape[-1] // self.chan_num)-1)
                foreground_obj = self.aux_labels[occlusion_idx:(occlusion_idx+1), :, :, :, rand_cam*3:(rand_cam+1)*3]
                occluded_area = (foreground_obj != -1)
                X[..., rand_cam*3:(rand_cam+1)*3][occluded_area] = foreground_obj[occluded_area]
                
        return X, X_grid, y_3d, aux

    def do_random(self, X):
        """Randomly re-order camera views

        Args:
            X (np.ndarray): image volumes

        Returns:
            X (np.ndarray): Shuffled image volumes
        """

        if self.random:
            X = X.reshape((*X.shape[:4], self.chan_num, -1), order="F")
            X = X[..., np.random.permutation(X.shape[-1])]
            X = X.reshape((*X.shape[:4], -1), order="F")

        if self.n_rand_views is not None:
            # Select a set of cameras randomly with replacement.
            X = X.reshape((*X.shape[:4], self.chan_num, -1), order="F")
            if self.replace:
                X = X[..., np.random.randint(X.shape[-1], size=(self.n_rand_views,))]
            else:
                if not self.random:
                    raise Exception(
                        "For replace=False for n_rand_views, random must be turned on"
                    ) 
                X = X[..., :self.n_rand_views]
            X = X.reshape((*X.shape[:4], -1), order="F")

        return X

    def get_max_gt_ind(self, X_grid, y_3d):
        """Uses the gt label position to find the index of the voxel corresponding to it.
        Used for heatmap regularization.
        """

        diff = np.sum(
            (X_grid[:, :, :, np.newaxis] - y_3d[:, np.newaxis, :, :]) ** 2, axis=2
        )
        inds = np.argmin(diff, axis=1)
        grid_d = int(np.round(X_grid.shape[1] ** (1 / 3)))
        inds = np.unravel_index(inds, (grid_d, grid_d, grid_d))
        return np.stack(inds, axis=1)
    
    def _convert_numpy_to_tensor(self, X, X_grid, y_3d, aux):
        if X_grid is not None:
            X_grid = torch.from_numpy(X_grid)
        if aux is not None:
            aux = torch.from_numpy(aux).permute(0, 4, 1, 2, 3)

        return torch.from_numpy(X).permute(0, 4, 1, 2, 3), X_grid, torch.from_numpy(y_3d), aux

    def __data_generation(self, list_IDs_temp):
        """
        X : (chunk_size, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Chunked training samples
                X: Input volumes
                y_3d: Targets
        Raises:
            Exception: For replace=False for n_rand_views, random must be turned on.
        """
        # Initialization
        X = np.zeros((self.temporal_chunk_size, *self.data.shape[1:]))
        y_3d = np.zeros((self.temporal_chunk_size, *self.labels.shape[1:]))

        # Only used for AVG mode
        if self.expval:
            X_grid = np.zeros((self.temporal_chunk_size, *self.xgrid.shape[1:]))
        else:
            X_grid = None

        # Only used for AVG+MAX mode
        if (not self.occlusion) and (self.aux_labels is not None):
            aux = np.zeros((self.temporal_chunk_size, *self.aux_labels.shape[1:]))
        else:
            aux = None

        for i, ID in enumerate(list_IDs_temp):
            X[i] = self.data[ID].copy()
            y_3d[i] = self.labels[ID]
            if self.expval:
                X_grid[i] = self.xgrid[ID]
            if aux is not None:
                aux[i] = self.aux_labels[ID]
        X, X_grid, y_3d, aux = self.do_augmentation(X, X_grid, y_3d, aux)
        # Randomly re-order, if desired
        X = self.do_random(X)
        
        return self._convert_numpy_to_tensor(X, X_grid, y_3d, aux)
    
    def compute_avg_bone_length(self):
        return

class DataGenerator_3Dconv_npy(DataGenerator_3Dconv_frommem):
    """Generates 3d conv data from npy files.

    Attributes:
    augment_brightness (bool): If True, applies brightness augmentation
    augment_continuous_rotation (bool): If True, applies rotation augmentation in increments smaller than 90 degrees
    augment_hue (bool): If True, applies hue augmentation
    batch_size (int): Batch size
    bright_val (float): Brightness augmentation range (-bright_val, bright_val), as fraction of raw image brightness
    chan_num (int): Number of input channels
    labels_3d (Dict): training targets
    expval (bool): If True, crafts input for an AVG network
    hue_val (float): Hue augmentation range (-hue_val, hue_val), as fraction of raw image hue range
    indexes (np.ndarray): Sample indices used for batch generation
    list_IDs (List): List of sampleIDs
    nvox (int): Number of voxels in each grid dimension
    random (bool): If True, shuffles camera order for each batch
    rotation (bool): If True, applies rotation augmentation in 90 degree increments
    rotation_val (float): Range of angles used for continuous rotation augmentation
    shuffle (bool): If True, shuffle the samples before each epoch
    var_reg (bool): If True, returns input used for variance regularization
    n_rand_views (int): Number of reviews to sample randomly from the full set
    replace (bool): If True, samples n_rand_views with replacement
    imdir (Text): Name of image volume npy subfolder
    griddir (Text): Name of grid volumw npy subfolder
    mono (bool): If True, return monochrome image volumes
    sigma (float): For MAX network, size of target Gaussian (mm)
    cam1 (bool): If True, prepares input for training a single camea network
    prefeat (bool): If True, prepares input for a network performing volume feature extraction before fusion
    npydir (Dict): path to each npy volume folder for each recording (i.e. experiment)
    """

    def __init__(
        self,
        list_IDs,
        labels_3d,
        npydir,
        # batch_size,
        imdir="image_volumes",
        griddir="grid_volumes",
        aux=False,
        auxdir="visual_hulls",
        prefeat=False,        
        mono=False,
        cam1=False,    
        sigma=10,
        pairs=None,
        **kwargs
    ):
        """Generates 3d conv data from npy files.

        Args:
            list_IDs (List): List of sampleIDs
            labels_3d (Dict): training targets
            npydir (Dict): path to each npy volume folder for each recording (i.e. experiment)
            batch_size (int): Batch size
            imdir (Text, optional): Name of image volume npy subfolder
            griddir (Text, optional): Name of grid volumw npy subfolder
            mono (bool, optional): If True, return monochrome image volumes
            cam1 (bool, optional): If True, prepares input for training a single camea network
            prefeat (bool, optional): If True, prepares input for a network performing volume feature extraction before fusion
            sigma (float, optional): For MAX network, size of target Gaussian (mm)
        """
        super(DataGenerator_3Dconv_npy, self).__init__(
            list_IDs=list_IDs,
            data=None,
            labels=None,
            **kwargs
        )
        self.labels_3d = labels_3d
        self.npydir = npydir
        self.griddir = griddir
        self.imdir = imdir
        self.mono = mono
        self.cam1 = cam1
        #self.replace = replace
        self.prefeat = prefeat
        self.sigma = sigma
        self.auxdir = auxdir
        self.aux = aux

        self.pairs = pairs
        if self.pairs is not None:
            self.temporal_chunk_size = len(self.pairs[0])

    def __len__(self):
        if self.temporal_chunk_list is not None:
           return len(self.temporal_chunk_list)
        
        if self.pairs is not None:
            return len(self.pairs)

        return len(self.list_IDs)


    def __getitem__(self, index):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data
                X (np.ndarray): Input volume
                y (np.ndarray): Target
        """
        if self.temporal_chunk_list is not None:
            list_IDs_temp = self.temporal_chunk_list[index]
        elif self.pairs is not None:
            list_IDs_temp = self.pairs[index]
        else:
            list_IDs_temp = [self.list_IDs[index]]
        # Generate data
        X, X_grid, y_3d, aux = self.__data_generation(list_IDs_temp)
        return X, X_grid, y_3d, aux

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        X : (n_samples, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d or y_3d_max: Targets
        Raises:
            Exception: For replace=False for n_rand_views, random must be turned on.
        """

        # Initialization

        X = []
        y_3d = []
        X_grid = []
        aux = []

        for i, ID in enumerate(list_IDs_temp):
            # Need to look up the experiment ID to get the correct directory
            IDkey = ID.split("_")
            eID = int(IDkey[0])
            sID = IDkey[1]

            X.append(
                np.load(
                    os.path.join(self.npydir[eID], self.imdir, "0_" + sID + ".npy")
                ).astype("float32")
            )

            y_3d.append(self.labels_3d[ID])
            X_grid.append(
                np.load(
                    os.path.join(self.npydir[eID], self.griddir, "0_" + sID + ".npy")
                )
            )

            if self.aux:
                aux.append(
                    np.load(os.path.join(self.npydir[eID], self.auxdir, "0_" + sID + ".npy")
                ).astype("float32")
                )

        X = np.stack(X)
        y_3d = np.stack(y_3d)

        X_grid = np.stack(X_grid)
        aux = np.stack(aux) if len(aux) != 0 else None

        if not self.expval:
            y_3d_max = np.zeros(
                (self.temporal_chunk_size, self.nvox, self.nvox, self.nvox, y_3d.shape[-1])
            )

        if not self.expval:
            X_grid = np.reshape(X_grid, (-1, self.nvox, self.nvox, self.nvox, 3))
            for gridi in range(X_grid.shape[0]):
                x_coord_3d = X_grid[gridi, :, :, :, 0]
                y_coord_3d = X_grid[gridi, :, :, :, 1]
                z_coord_3d = X_grid[gridi, :, :, :, 2]
                for j in range(y_3d_max.shape[-1]):
                    y_3d_max[gridi, :, :, :, j] = np.exp(
                        -(
                            (y_coord_3d - y_3d[gridi, 1, j]) ** 2
                            + (x_coord_3d - y_3d[gridi, 0, j]) ** 2
                            + (z_coord_3d - y_3d[gridi, 2, j]) ** 2
                        )
                        / (2 * self.sigma ** 2)
                    )

        if self.mono and self.chan_num == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = np.reshape(
                X,
                (
                    X.shape[0],
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    self.chan_num,
                    -1,
                ),
                order="F",
            )
            X = (
                X[:, :, :, :, 0] * 0.2125
                + X[:, :, :, :, 1] * 0.7154
                + X[:, :, :, :, 2] * 0.0721
            )

        ncam = int(X.shape[-1] // self.chan_num)

        X, X_grid, y_3d, aux = self.do_augmentation(X, X_grid, y_3d, aux)

        # Randomly re-order, if desired
        X = self.do_random(X)

        if self.cam1:
            # collapse the cameras to the batch dimensions.
            X = np.reshape(
                X,
                (X.shape[0], X.shape[1], X.shape[2], X.shape[3], self.chan_num, -1),
                order="F",
            )
            X = np.transpose(X, [0, 5, 1, 2, 3, 4])
            X = np.reshape(X, (-1, X.shape[2], X.shape[3], X.shape[4], X.shape[5]))
            if self.expval:
                y_3d = np.tile(y_3d, [ncam, 1, 1])
                X_grid = np.tile(X_grid, [ncam, 1, 1])
            else:
                y_3d = np.tile(y_3d, [ncam, 1, 1, 1, 1])

        X = processing.preprocess_3d(X) 

        return self._convert_numpy_to_tensor(X, X_grid, y_3d, aux)

class DataGenerator_Social(DataGenerator_3Dconv_frommem):
    def __init__(
        self, 

        **kwargs):
        # assume now
        # data: [n_samples, 2, 64, 64, 64, 18] array
        # labels: [n_samples, 2, 3, 22] array
        # x_grid: [n_samples, 2, 64**3, 3]

        super().__init__(**kwargs)
        self.temporal_chunk_size = self.data.shape[1]

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index]
        X, X_grid, y_3d, aux = self.__data_generation(list_IDs_temp)
        return X, X_grid, y_3d, aux
    
    def __len__(self):
        return self.data.shape[0]
    
    def __data_generation(self, ID):
        X = self.data[ID].copy()
        y_3d = self.labels[ID]

        if self.expval:
            X_grid = self.xgrid[ID]
        else:
            X_grid = None

        if self.aux_labels is not None:
            aux = self.aux_labels[ID]
        else:
            aux = None

        X, X_grid, y_3d, aux = self.do_augmentation(X, X_grid, y_3d, aux)
        # Randomly re-order, if desired
        X = self.do_random(X)
        
        # shape of outputs:
        # X: [temporal_chunk_size, H, W, D, n_cam*chan_num]

        return self._convert_numpy_to_tensor(X, X_grid, y_3d, aux)
