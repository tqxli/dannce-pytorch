import torch
import torch.nn as nn
from torchvision.models import resnet50

class DLC(nn.Module):
    """
    DLC uses a similar architecture to DeeperCut,
    that a ResNet backbone (no global avg pool, no linear) followed by a conv2d transpose layer
    """
    def __init__(self, num_joints=22):
        super().__init__()

        # ResNet50 with output stride=16
        backbone = resnet50(replace_stride_with_dilation=[False, False, True], pretrained=True)
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        self.backbone = backbone

        # "same" padding imposed here
        # refer to DLC/layers.py, the 'prediction layer' is indeed one single 2D transpose conv layer
        # there is no intermediate supervision from conv3 as indicated by the paper & DeeperCut
        # when using the default setting
        self.deconvolution_layer = nn.ConvTranspose2d(
            in_channels=2048, out_channels=num_joints,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
    
    def forward(self, x):
        features = self.backbone(x) # [B, 2048, w//16, h//16]

        heatmaps = self.deconvolution_layer(features) #[B, 2048, w//8, h//8]

        return heatmaps

if __name__ == "__main__":
    dlc = DLC()
    device = "cuda:0"

    x = torch.zeros(16, 3, 400, 400).to(device)
    dlc = dlc.to(device)
    prediction = dlc(x)
    print(prediction.shape)
