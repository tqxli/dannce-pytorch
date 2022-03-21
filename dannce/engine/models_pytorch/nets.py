import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .blocks import *
from dannce.engine.ops import spatial_softmax_torch, expected_value_3d_torch

class EncoderDecorder_DANNCE(nn.Module):
    """
    3D UNet class for 3D pose estimation.
    """
    def __init__(self, normalization, input_shape, residual=False):
        super().__init__()
        conv_block = Res3DBlock if residual else Basic3DBlock

        self.encoder_res1 = conv_block(64, 64, normalization, [input_shape]*3)
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res2 = conv_block(64, 128, normalization, [input_shape//2]*3)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res3 = conv_block(128, 256, normalization, [input_shape//4]*3)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res4 = conv_block(256, 512, normalization, [input_shape//8]*3)

        self.decoder_res3 = conv_block(512, 256, normalization, [input_shape//4]*3)
        self.decoder_upsample3 = Upsample3DBlock(512, 256, 2, 2, normalization, [input_shape//4]*3)
        self.decoder_res2 = conv_block(256, 128, normalization, [input_shape//2]*3)
        self.decoder_upsample2 = Upsample3DBlock(256, 128, 2, 2, normalization, [input_shape//2]*3)
        self.decoder_res1 = conv_block(128, 64, normalization, [input_shape]*3)
        self.decoder_upsample1 = Upsample3DBlock(128, 64, 2, 2, normalization, [input_shape]*3)

    def forward(self, x):
        # encoder
        x = self.encoder_res1(x)
        skip_x1 = x
        x = self.encoder_pool1(x)

        x = self.encoder_res2(x)    
        skip_x2 = x    
        x = self.encoder_pool2(x)

        x = self.encoder_res3(x)
        skip_x3 = x
        x = self.encoder_pool3(x)

        x = self.encoder_res4(x)

        # decoder with skip connections
        x = self.decoder_upsample3(x)
        x = self.decoder_res3(torch.cat([x, skip_x3], dim=1))

        x = self.decoder_upsample2(x)
        x = self.decoder_res2(torch.cat([x, skip_x2], dim=1))

        x = self.decoder_upsample1(x)
        x = self.decoder_res1(torch.cat([x, skip_x1], dim=1))

        return x

class DANNCE(nn.Module):
    def __init__(self, input_channels, output_channels, input_shape, return_coords=True, norm_method='layer', residual=False):
        super().__init__()
        # torch Layer Norm requires explicit input shape for initialization
        # self.normalization = NORMALIZATION_MODES[norm_method]
        if residual:
            self.front_layers = Res3DBlock(input_channels, 64, norm_method, input_shape=[input_shape]*3)
        else:
            self.front_layers = Basic3DBlock(input_channels, 64, norm_method, input_shape=[input_shape]*3)

        self.encoder_decoder = EncoderDecorder_DANNCE(norm_method, input_shape, residual)
        self.output_layer = nn.Conv3d(64, output_channels, kernel_size=1, stride=1, padding=0)
        
        # self._initialize_weights()

        self.return_coords = return_coords
        self.n_joints = output_channels

    def forward(self, volumes, grid_centers):
        """
        volumes: Tensor [batch_size, C, H, W, D]
        grid_centers: [batch_size, nvox**3, 3]
        """
        volumes = self.front_layers(volumes)
        volumes = self.encoder_decoder(volumes)
        volumes = self.output_layer(volumes)

        heatmaps = spatial_softmax_torch(volumes)
        # if self.return_coords:
        coords = expected_value_3d_torch(heatmaps, grid_centers)

        return coords, torch.amax(heatmaps, dim=(2, 3, 4)).squeeze(0)

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