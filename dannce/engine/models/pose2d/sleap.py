import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EncoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, apply_pooling=True, return_skip=True):
        super().__init__()
    
        self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
                nn.ReLU(True)])
        self.conv2 = nn.Sequential(*[
                nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
                nn.ReLU(True)])
        
        self.apply_pooling = apply_pooling
        if apply_pooling:
            self.pool = nn.MaxPool2d(2)
        
        self.return_skip = return_skip

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x 
        if self.apply_pooling:
            x = self.pool(x)
        
        if self.return_skip:
            return x, skip
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size-1)//2), output_padding=0)
        self.conv = EncoderBlock(in_planes, out_planes, apply_pooling=False, return_skip=False)

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.conv(torch.cat((x, skip), dim=1))

        return x

class SLEAPUNet(nn.Module):
    def __init__(self, input_channels, output_channels, 
            stem_stride=None, max_stride=16, output_stride=2, filters=24, filters_rate=2, 
            middle_block=True,
    ):
        super().__init__()

        stem_blocks = 0
        if stem_stride is not None:
            stem_blocks = np.log2(stem_stride).astype(int)
        down_blocks = np.log2(max_stride).astype(int) - stem_blocks
        up_blocks = np.log2(max_stride / output_stride).astype(int)

        # construct layers
        encoder = []
        in_filters = input_channels
        for block in range(down_blocks):
            block_filters = int(
                filters * (filters_rate ** (block + stem_blocks))
            )
            encoder.append(EncoderBlock(in_filters, block_filters, apply_pooling=True))
            in_filters = block_filters
        
        if middle_block:
            block_filters = int(
                    filters
                    * (filters_rate ** (down_blocks + stem_blocks))
            )
            encoder.append(EncoderBlock(in_filters, block_filters, apply_pooling=False, return_skip=False))

        self.encoder = nn.ModuleList(encoder)

        decoder = []
        for block in range(up_blocks):
            block_filters_in = int(
                filters
                * (
                    filters_rate
                    ** (down_blocks + stem_blocks - 1 - block)
                )
            )
            block_filters_out = block_filters_in
            decoder.append(DecoderBlock(
                block_filters_in*2, block_filters_out
            ))

        self.decoder = nn.ModuleList(decoder)

        self.skip_index = np.arange(down_blocks)[::-1][::(len(encoder) // up_blocks)]

        self.output_layer = nn.Conv2d(block_filters_out, output_channels, kernel_size=1, stride=output_stride, padding=0)

    def forward(self, x):
        skips = []
        # encoder
        for i, layer in enumerate(self.encoder[:-1]):
            x, skip = layer(x)
            skips.append(skip)
        
        x = self.encoder[-1](x)

        # decoder
        for i, layer in enumerate(self.decoder):
            x = layer(x, skips[-i-1])

        x = self.output_layer(x)

        return x

if __name__ == "__main__":
    model = SLEAPUNet(3, 23)

    input = torch.randn(4, 3, 256, 256)
    output = model(input)
    print(output.shape)