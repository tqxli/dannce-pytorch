import time
from copy import deepcopy
from easydict import EasyDict as edict
# from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dannce.engine.data.ops as ops
from dannce.engine.models.backbone import get_pose_net
from dannce.engine.models.nets import DANNCE

class FeatureDANNCE(nn.Module):
    def __init__(
        self, 
        n_cams,
        output_channels, 
        input_shape, 
        bottleneck_channels=6,
        norm_method='layer', 
        residual=False, 
        norm_upsampling=False,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.input_shape = input_shape

        self.backbone = self._init_backbone()
        self.process_features = nn.Conv2d(
            self.backbone_cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1],
            bottleneck_channels, 
            1)
        self.posenet = DANNCE(bottleneck_channels*n_cams, output_channels, input_shape, norm_method, residual, norm_upsampling)      

        # self.threadpool = ThreadPool(n_cams)

    def forward(self, images, grid_coords, cameras):
        """
        images: [BS, 6, 3, H, W]
        grid_coords: [BS, nvox**3, 3]
        cameras: [BS, 6]
        """
        bs, n_cams = images.shape[:2]
        image_shape = images.shape[-2:]
            
        # extract features
        images = images.view(-1, *images.shape[2:])
        # start = time.time()
        features, heatmaps = self.backbone(images)
        # print('backbone: ', time.time()-start)
        # print('feature shape: ', features.shape)
        
        # reduce channel numbers for passing into pose regresssion network
        # start = time.time()
        features = self.process_features(features)
        feature_shape = features.shape[-2:]
        # print('reduce channel: ', time.time()-start)

        # update camera matrices
        # start = time.time()
        new_cameras = self._update_cameras(cameras, image_shape, feature_shape)

        # feature unprojection
        features = features.reshape(bs, n_cams, *features.shape[1:])
        
        volumes = []
        for idx in range(bs):
            batch_vols = []
            for j, cam in enumerate(new_cameras[idx]):
                batch_vols.append(self._unproject_heatmaps(features[idx, j].permute(1, 2, 0), cam, grid_coords[idx]))
            #arglist = [[feat.permute(1, 2, 0), cam, grid_coords[idx]] for feat, cam in zip(features[idx], new_cameras[idx])]
            #batch_vols = self.threadpool.starmap(self._unproject_heatmaps, arglist)
            volumes.append(torch.stack(batch_vols, dim=0))
        # print('feature unprojection: ', time.time()-start)        
        
        volumes = torch.stack(volumes, dim=0) #[BS, n_cams, C, H, W, D]
        volumes = volumes.reshape(bs, -1, *volumes.shape[3:]) #[BS, n_cams*C, H, W, D]

        start = time.time()
        coords, joint_heatmaps = self.posenet(volumes, grid_coords)
        # print('pose regression: ', time.time()-start)

        return coords, joint_heatmaps

    def _init_backbone(self):
        backbone_cfg = edict()
        backbone_cfg.POSE_RESNET = edict()
        backbone_cfg.POSE_RESNET.NUM_LAYERS = 50
        backbone_cfg.POSE_RESNET.DECONV_WITH_BIAS = False
        backbone_cfg.POSE_RESNET.NUM_DECONV_LAYERS = 3
        backbone_cfg.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 128]
        backbone_cfg.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
        backbone_cfg.POSE_RESNET.FINAL_CONV_KERNEL = 1

        backbone_cfg.NETWORK = edict()
        backbone_cfg.NETWORK.PRETRAINED = ''
        backbone_cfg.NETWORK.PRETRAINED_BACKBONE = ''
        backbone_cfg.NETWORK.NUM_JOINTS = self.output_channels
        self.backbone_cfg = backbone_cfg

        return get_backbone(cfg=backbone_cfg, is_train=True)
    
    def _update_cameras(self, cameras, image_shape, heatmap_shape):
        # new_cameras = deepcopy(cameras)
        for batch in cameras:
            for cam in batch:
                cam.update_after_resize(image_shape, heatmap_shape)
        return cameras

    def _unproject_heatmaps(self, feature, cam, grid_coords):
        proj_grid = ops.project_to2d(grid_coords, cam.camera_matrix(), feature.device)
        proj_grid = ops.distortPoints(proj_grid[:, :2], cam.K, cam.rdist, cam.tdist, device=feature.device)
        proj_grid = proj_grid.permute(1, 0)[:, :2]

        vol = ops.sample_grid(feature, proj_grid, device=feature.device)
        return vol.squeeze() # [C, H, W, D]

"""TEST"""
if __name__ == "__main__":
    from dannce.engine.data.io import load_com, load_camera_params, load_labels, load_sync
    from dannce.engine.data.processing import cropcom
    import imageio
    import os
    device = "cuda:0"

    bs, n_cams, c, w, h = 1, 6, 3, 512, 512
    nvox = 80

    vmin, vmax = -60, 60
    vsize = (vmax - vmin) / nvox
    idx = 10
    exp = '/media/mynewdrive/datasets/dannce/demo/markerless_mouse_1'
    label3d = os.path.join(exp, 'label3d_dannce.mat')

    # grab com3d, com2d, camera params
    com2ds, frames = [], []
        
    camparams = load_camera_params(label3d)
    cameras = [ops.Camera(c['r'], c['t'], c['K'], c['TDistort'], c['RDistort']) for c in camparams]

    labels = load_labels(label3d)
    pose3d = np.reshape(labels[0]['data_3d'][idx], (1, -1, 3))
    
    for label in labels:
        com2d = label['data_2d'][idx]
        frame_idx = label['data_frame'][0, idx]

        com2ds.append(com2d)
        frames.append(frame_idx)
    
    sampleID = labels[0]['data_sampleID'][0, idx]
    sync = load_sync(label3d)[0]['data_sampleID'][:, 0]
    com_idx = np.where(sync == sampleID)[0]
    com3d = load_com(label3d)['com3d'][com_idx][0]
    
    # grab images
    vids = [imageio.get_reader(os.path.join(exp, 'videos', f'Camera{c+1}', '0.mp4')) for c in range(n_cams)]
    images = [vid.get_data(frame) for (vid, frame) in zip(vids, frames)]
       
    for i, im in enumerate(images):
        im, cropdim = cropcom(im, com2ds[i], size=512)

        bbox = (cropdim[0], cropdim[2], cropdim[1], cropdim[3])
        cameras[i].update_after_crop(bbox)

        images[i] = im
    
    images = torch.tensor(np.stack(images)).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    xgrid = torch.arange(
        vmin + com3d[0] + vsize / 2,
        com3d[0] + vmax,
        vsize,
        dtype=torch.float32,
    )
    ygrid = torch.arange(
        vmin + com3d[1] + vsize / 2,
        com3d[1] + vmax,
        vsize,
        dtype=torch.float32,
    )
    zgrid = torch.arange(
        vmin + com3d[2] + vsize / 2,
        com3d[2] + vmax,
        vsize,
        dtype=torch.float32,
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

    grid = grid.unsqueeze(0).to(device)

    """create model"""
    start = time.time()
    model = FeatureDANNCE(n_cams=n_cams, output_channels=22, input_shape=nvox, bottleneck_channels=6)
    model = model.to(device)
    model.train()
    print('model initialization: ', time.time()-start)

    start = time.time()
    output = model(images, grid, [cameras])
    print(f"ONE EPOCH: {time.time()-start}")
    
    print('Coords: ', output[0].shape)
    print('Heatmaps: ', output[1].shape)
    
    loss = F.mse_loss(output[0], torch.from_numpy(pose3d).permute(0, 2, 1).float().to(device))
    print('MSE loss: ', loss)

    loss.backward()

    # check whether grad flows correctly
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(name, param.grad.sum())
    #     else:
    #         print(name, param.grad)