import os
import numpy as np
from dannce.engine.data import processing
import warnings

import torch
import torchvision.transforms.functional as TF

MISSING_KEYPOINTS_MSG = (
    "If mirror augmentation is used, the right_keypoints indices and left_keypoints "
    + "indices must be specified as well. "
    + "For the skeleton, ['RHand, 'LHand', 'RFoot', 'LFoot'], "
    + "set right_keypoints: [0, 2] and left_keypoints: [1, 3] in the config file"
)

class PoseDatasetFromMem(torch.utils.data.Dataset):
    """Generate 3d conv data from memory.

    Attributes:
        augment_brightness (bool): If True, applies brightness augmentation
        augment_continuous_rotation (bool): If True, applies rotation augmentation in increments smaller than 90 degrees
        augment_hue (bool): If True, applies hue augmentation
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
        pairs=None
    ):
        """Initialize data generator.
        """
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels
        self.rotation = rotation
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

        self.occlusion = occlusion
        # if self.occlusion:
        #     assert aux_labels is not None, "Missing aux labels for occlusion training."
        
        self.pairs = pairs
        if self.pairs is not None:
            self.temporal_chunk_size = len(self.pairs[0])
        
        self._update_temporal_batch_size()

    def __len__(self):
        """Denote the number of batches per epoch.

        Returns:
            int: Batches per epoch
        """
        if self.temporal_chunk_list is not None:
           return len(self.temporal_chunk_list)
        
        if self.pairs is not None:
            return len(self.pairs)

        return len(self.list_IDs)
    
    def _update_temporal_batch_size(self):
        self.temporal_chunk_size = 1
        if self.temporal_chunk_list is not None:
            self.temporal_chunk_size = len(self.temporal_chunk_list[0])

        if self.pairs is not None:
            self.temporal_chunk_size = len(self.pairs[0])

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
            list_IDs_temp = index
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
        # torchvision.transforms.functional.affine - input: [..., H, W]
        rotangle = np.random.rand() * (2 * max_delta) - max_delta
        X = torch.from_numpy(X).reshape(*X.shape[:3], -1).permute(0, 3, 1, 2) # dimension [B, D*C, H, W]
        y_3d = torch.from_numpy(y_3d).reshape(y_3d.shape[:3], -1).permute(0, 3, 1, 2)
        for i in range(X.shape[0]):
            X[i] = TF.affine(X[i], angle=rotangle)
            y_3d[i] = TF.affine(y_3d[i], angle=rotangle)

        X = X.permute(0, 2, 3, 1).reshape(*X.shape[:3], X.shape[2], -1).numpy()
        y_3d = y_3d.permute(0, 2, 3, 1).reshape(*X.shape[:3], X.shape[2], -1).numpy()

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
            # [???, 1 or 3, H, W]
            for n_cam in range(int(X.shape[-1] / self.chan_num)):
                channel_ids = np.arange(
                    n_cam * self.chan_num,
                    n_cam * self.chan_num + self.chan_num,
                )
                X_temp = torch.from_numpy(X[..., channel_ids]).permute(0, 3, 4, 1, 2) #[bs, D, 3, H, W]
                random_hue_val = float(torch.empty(1).uniform_(-self.hue_val, self.hue_val))
                X_temp = TF.adjust_hue(X_temp, random_hue_val)
                X[..., channel_ids] = X_temp.permute(0, 3, 4, 1, 2).numpy()

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
                X_temp = torch.as_tensor(X[..., channel_ids]).permute(0, 3, 4, 1, 2)
                random_bright_val = float(torch.empty(1).uniform_(1-self.bright_val, 1+self.bright_val))
                X_temp = TF.adjust_brightness(X_temp, random_bright_val)
                X[..., channel_ids] = X_temp.permute(0, 3, 4, 1, 2).numpy()

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

        # if self.occlusion:
        #     if np.random.rand() > 0.5:
        #         occlusion_idx = np.random.choice(self.__len__())
        #         rand_cam = np.random.choice(int(X.shape[-1] // self.chan_num)-1)
        #         foreground_obj = self.aux_labels[occlusion_idx:(occlusion_idx+1), :, :, :, rand_cam*3:(rand_cam+1)*3]
        #         occluded_area = (foreground_obj != -1)
        #         X[..., rand_cam*3:(rand_cam+1)*3][occluded_area] = foreground_obj[occluded_area]
                
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
        if self.pairs is None:
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
        else:
            ID = list_IDs_temp
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
        
        return self._convert_numpy_to_tensor(X, X_grid, y_3d, aux)
    
    def compute_avg_bone_length(self):
        return

class PoseDatasetNPY(PoseDatasetFromMem):
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
        super(PoseDatasetNPY, self).__init__(
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

    def _downscale_occluded_views(self, X, occlusion_scores):
        """
        X: [H, W, D, n_cams*3]
        occlusion_scores: [n_cams]
        """
        occluded_X = np.reshape(X.copy(), (*X.shape[:3], occlusion_scores.shape[-1], -1))
        occlusion_scores = occlusion_scores[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

        occluded_X *= occlusion_scores

        occluded_X = np.reshape(occluded_X, (*X.shape[:3], -1))

        return occluded_X

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

            vol = np.load(
                os.path.join(self.npydir[eID], self.imdir, "0_" + sID + ".npy")
                ).astype("float32")

            if self.occlusion:
                occlusion_scores = np.load(os.path.join(self.npydir[eID], 'occlusion_scores', "0_" + sID + ".npy")).astype("float32")
                vol = self._downscale_occluded_views(vol, occlusion_scores)
            X.append(vol)

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
    
class ChunkedKeypointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        labels_3d,
    ):
        self.labels_3d = labels_3d

    def __len__(self):
        return 
    
    def __getitem__(self,index):
        return