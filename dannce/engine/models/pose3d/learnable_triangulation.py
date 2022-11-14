import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dannce.engine.models.pose3d.voxelpose import Res3DBlock, Basic3DBlock, Pool3DBlock, Upsample3DBlock
from dannce.engine.models.pose2d.pose_net import get_pose_net
import dannce.engine.data.ops as ops

class EncoderDecorder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128)
        self.encoder_pool4 = Pool3DBlock(2)
        self.encoder_res4 = Res3DBlock(128, 128)
        self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlock(128, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res5 = Res3DBlock(128, 128)
        self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res4 = Res3DBlock(128, 128)
        self.decoder_upsample4 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res3 = Res3DBlock(128, 128)
        self.decoder_upsample3 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)
        self.skip_res3 = Res3DBlock(128, 128)
        self.skip_res4 = Res3DBlock(128, 128)
        self.skip_res5 = Res3DBlock(128, 128)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)
        skip_x4 = self.skip_res4(x)
        x = self.encoder_pool4(x)
        x = self.encoder_res4(x)
        skip_x5 = self.skip_res5(x)
        x = self.encoder_pool5(x)
        x = self.encoder_res5(x)

        x = self.mid_res(x)

        x = self.decoder_res5(x)
        x = self.decoder_upsample5(x)
        x = x + skip_x5
        x = self.decoder_res4(x)
        x = self.decoder_upsample4(x)
        x = x + skip_x4
        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )

        self.encoder_decoder = EncoderDecorder()

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1),
            Basic3DBlock(32, 32, 1),
        )

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)

class VolumetricTriangulationNet(nn.Module):
    def __init__(self, output_channels, params, logger, training=False):
        super().__init__()

        self.backbone = get_pose_net(output_channels, params["custom_model"], logger)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.output_channels = output_channels
        self.volume_net = V2VModel(32, self.output_channels)   

        self.training = training
    
    def set_train(self):
        self.training = True
    
    def set_eval(self):
        self.training = False

    def _unproject_heatmaps(self, feature, cam, grid_coords):
        proj_grid = ops.project_to2d(grid_coords, cam.camera_matrix(), feature.device)
        proj_grid = ops.distortPoints(proj_grid[:, :2], cam.K, cam.rdist, cam.tdist, device=feature.device)
        proj_grid = proj_grid.permute(1, 0) # transpose here since output from distortPoints is channel first

        vol = ops.sample_grid(feature, proj_grid, device=feature.device)
        return vol.squeeze() # [C, H, W, D]
    
    def get_rotation_matrix(self, axis, theta):
        """Returns the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def rotate_coord_volume(self, coord_volume, theta, axis):
        shape = coord_volume.shape
        device = coord_volume.device

        rot = self.get_rotation_matrix(axis, theta)
        rot = torch.from_numpy(rot).type(torch.float).to(device)

        coord_volume = coord_volume.view(-1, 3)
        coord_volume = rot.mm(coord_volume.t()).t()

        coord_volume = coord_volume.view(*shape)

        return coord_volume
    
    def forward(self, images, grid_coords, cameras):
        bs, n_cams = images.shape[:2]
            
        # extract heatmaps from 2D backbone
        images = images.view(-1, *images.shape[2:])
        _, features = self.backbone(images)
        features = features.reshape(bs, n_cams, *features.shape[1:]) #[B, n_cams, 256, 96, 96]

        # process features before unprojecting, reduce channel from 256 to 32
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(bs, n_cams, *features.shape[1:]) #[B, n_cams, 32, 96, 96]
        
        # feature unprojection
        # start = time.time()
        volumes = []
        for idx in range(bs):
            batch_vols = []
            grids = grid_coords[idx].clone()

            # rotation augmentation during training
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            center = torch.mean(grids, 0)
            axis = [0, 0, 1]

            grids = grids - center
            grids = self.rotate_coord_volume(grids, theta, axis)
            grids = grids + center

            # unprojection
            for j, cam in enumerate(cameras[idx]):
                batch_vols.append(
                    self._unproject_heatmaps(
                        features[idx, j].permute(1, 2, 0), #[H, W, C]
                        cam, 
                        grids
                    )
                )
            batch_vols = torch.stack(batch_vols, dim=0)
            volumes.append(batch_vols)
        # end = time.time()
        # print("Feature volume unprojection takes {}".format(end-start))
        volumes = torch.stack(volumes, dim=0) #[BS, n_cams, 32, 64, 64, 64]

        # softmax aggregation across views
        volumes_softmin = volumes.clone()
        volumes_softmin = volumes_softmin.view(bs, n_cams, -1)
        volumes_softmin = nn.functional.softmax(volumes_softmin, dim=1) #[bs, -1]
        volumes_softmin = volumes_softmin.view(bs, n_cams, *volumes.shape[2:])

        volumes = (volumes * volumes_softmin).sum(1) #[bs, 32, 64, 64, 64]

        # process by the 3D v2v
        heatmaps3d = self.volume_net(volumes)

        softmax_heatmaps = ops.spatial_softmax(heatmaps3d)
        coords = ops.expected_value_3d(softmax_heatmaps, grid_coords)

        return heatmaps3d, coords   