import torch
import torch.nn as nn

class FreiPose(nn.Module):
    def __init__(self, out_channels=22):
        # 2d CNN for initial keypoint predictions
        self.posenet2d = nn.ModuleList()
        layers_per_block = [2, 2, 4, 4, 2]
        out_chan_list = [3, 64, 128, 256, 512, 512]
        pool_list = [True, True, True, False, False]

        for block_id, (layer_num, pool) in enumerate(zip(layers_per_block, pool_list)):
            block = []
            for layer_id in range(layer_num):
                layer = [
                    nn.Conv2d(out_chan_list[block_id], out_chan_list[block_id+1], kernel_size=3, stride=1),
                    nn.ReLU(inplace=True),
                ]
                block += layer
            
            if pool:
                block += [nn.MaxPool2d()]
            
            self.posenet2d.append(block)

        self.encoding = nn.Sequential(
            [nn.Conv2d(out_chan_list[-1], 128, kernel_size=3, stride=1), nn.ReLU(inplace=True)]
        )

        self.out = nn.Sequential([
            nn.Conv2d(128, 512, kernel_size=1, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channels, 1, 1)
        ])

        # 3D encoder
        self.encoder_3d = nn.ModuleList()
        enc_chan_list = [128*6, 32, 64, 64, 64]
        for i in range(len(enc_chan_list)):
            block = nn.Sequential([
                nn.Conv3d(enc_chan_list[i], enc_chan_list[i+1], kernel_size=1),
                nn.Conv3d(enc_chan_list[i+1], enc_chan_list[i+1], kernel_size=3, stride=2)
            ])
            self.encoder_3d.append(block)
        
        bottleneck = nn.Sequential([
            nn.Conv3d(enc_chan_list[-1], 64, kernel_size=1),
            nn.ReLU(inplace=True)
        ])
        self.encoder_3d.append(bottleneck)

        # 3D decoder
        self.decoder_3d = nn.Sequential()
        kernels = [16, 8, 4]
        dec_chan_list = [64, 32, 32, 32]
        for i, kernel in enumerate(kernels):
            block = nn.Sequential([
                nn.ConvTranspose3d(dec_chan_list[i], dec_chan_list[i+1], kernel_size=4, stride=2),
                nn.Conv3d(enc_chan_list[-i], dec_chan_list[i], kernel_size=1),
                nn.Conv3d(2*dec_chan_list[i], dec_chan_list[i], 3)
            ])
            self.decoder.append(block)

        final = nn.ConvTranspose3d(dec_chan_list[-1], 64, 4, stride=2)
        self.decoder_3d.append(final)

        self.output_layer = nn.Conv3d(64, out_channels, 1)

        #     x, scorevol = self._dec3D_stop(x, skips.pop(), scorevol, chan, num_chan, kernel, is_training)
        #     scorevolumes.append(scorevol)

        # # final decoder step
        # x = slim.conv3d_transpose(x, 64, kernel_size=[4, 4, 4], trainable=is_training, stride=2, activation_fn=tf.nn.relu)
        # scorevol_delta = slim.conv3d(x, num_chan, kernel_size=[1, 1, 1], trainable=is_training, activation_fn=None)
        # scorevol = scorevol_delta
        # scorevolumes.append(scorevol)


        # refinement
