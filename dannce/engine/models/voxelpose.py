import time
from copy import deepcopy
from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dannce.engine.data.ops as ops
from dannce.engine.models.pose2d.pose_net import get_pose_net
from dannce.engine.models.nets import DANNCE
from dannce.engine.data.ops import spatial_softmax, expected_value_3d

class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )
    
    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)

    
class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2

        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x            


class V2VNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(V2VNet, self).__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Res3DBlock(16, 32),
        )

        self.encoder_decoder = EncoderDecoder()

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class VoxelPose(nn.Module):
    def __init__(self, output_channels, params, logger):
        super().__init__()

        self.output_channels = output_channels
        self.backbone = get_pose_net(output_channels, params["custom_model"], logger)
        self.v2v = V2VNet(output_channels, output_channels)

    def _unproject_heatmaps(self, feature, cam, grid_coords):
        proj_grid = ops.project_to2d(grid_coords, cam.camera_matrix(), feature.device)
        proj_grid = ops.distortPoints(proj_grid[:, :2], cam.K, cam.rdist, cam.tdist, device=feature.device)
        proj_grid = proj_grid.permute(1, 0)[:, :2]

        vol = ops.sample_grid(feature, proj_grid, device=feature.device)
        return vol.squeeze() # [C, H, W, D]

    def forward(self, images, grid_coords, cameras):
        bs, n_cams = images.shape[:2]
            
        # extract heatmaps from 2D backbone
        images = images.view(-1, *images.shape[2:])
        heatmaps2d = self.backbone(images)
        heatmaps2d = heatmaps2d.reshape(bs, n_cams, *heatmaps2d.shape[1:]) #[B, J, C, H, W]
        
        # feature unprojection
        # start = time.time()
        volumes = []
        for idx in range(bs):
            batch_vols = []
            for j, cam in enumerate(cameras[idx]):
                batch_vols.append(
                    self._unproject_heatmaps(
                        heatmaps2d[idx, j].permute(1, 2, 0), #[H, W, C]
                        cam, 
                        grid_coords[idx]
                    )
                )
            volumes.append(torch.stack(batch_vols, dim=0))
        # end = time.time()
        # print("Feature volume unprojection takes {}".format(end-start))
        volumes = torch.stack(volumes, dim=0) #[BS, n_cams, C, H, W, D]
        volumes = volumes.mean(1) #[BS, C, H, W, D]

        # process by the 3D v2v
        heatmaps3d = self.v2v(volumes)

        softmax_heatmaps = spatial_softmax(heatmaps3d)
        coords = expected_value_3d(softmax_heatmaps, grid_coords)

        return heatmaps2d, coords


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
        # backbone_cfg = edict()
        # backbone_cfg.POSE_RESNET = edict()
        # backbone_cfg.POSE_RESNET.NUM_LAYERS = 50
        # backbone_cfg.POSE_RESNET.DECONV_WITH_BIAS = False
        # backbone_cfg.POSE_RESNET.NUM_DECONV_LAYERS = 3
        # backbone_cfg.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 128]
        # backbone_cfg.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
        # backbone_cfg.POSE_RESNET.FINAL_CONV_KERNEL = 1

        # backbone_cfg.NETWORK = edict()
        # backbone_cfg.NETWORK.PRETRAINED = ''
        # backbone_cfg.NETWORK.PRETRAINED_BACKBONE = ''
        # backbone_cfg.NETWORK.NUM_JOINTS = self.output_channels
        # self.backbone_cfg = backbone_cfg

        # return get_backbone(cfg=backbone_cfg, is_train=True)
        params = {"architecture": "pose_resnet"}
        return get_pose_net(self.output_channels, params)
    
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