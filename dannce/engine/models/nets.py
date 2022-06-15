import torch
import torch.nn as nn
import torch.nn.functional as F

from dannce.engine.models.blocks import *
from dannce.engine.data.ops import spatial_softmax, expected_value_3d

class EncoderDecorder_DANNCE(nn.Module):
    """
    3D UNet class for 3D pose estimation.
    """
    def __init__(self, in_channels, normalization, input_shape, residual=False, norm_upsampling=False):
        super().__init__()
        conv_block = Res3DBlock if residual else Basic3DBlock
        deconv_block = Upsample3DBlock if norm_upsampling else BasicUpSample3DBlock

        self.encoder_res1 = conv_block(in_channels, 64, normalization, [input_shape]*3)
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res2 = conv_block(64, 128, normalization, [input_shape//2]*3)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res3 = conv_block(128, 256, normalization, [input_shape//4]*3)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res4 = conv_block(256, 512, normalization, [input_shape//8]*3)

        self.decoder_res3 = conv_block(512, 256, normalization, [input_shape//4]*3)
        self.decoder_upsample3 = deconv_block(512, 256, 2, 2, normalization, [input_shape//4]*3)
        self.decoder_res2 = conv_block(256, 128, normalization, [input_shape//2]*3)
        self.decoder_upsample2 = deconv_block(256, 128, 2, 2, normalization, [input_shape//2]*3)
        self.decoder_res1 = conv_block(128, 64, normalization, [input_shape]*3)
        self.decoder_upsample1 = deconv_block(128, 64, 2, 2, normalization, [input_shape]*3)

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
    def __init__(self, input_channels, output_channels, input_shape, norm_method='layer', residual=False, norm_upsampling=False):
        super().__init__()

        self.encoder_decoder = EncoderDecorder_DANNCE(input_channels, norm_method, input_shape, residual, norm_upsampling)
        self.output_layer = nn.Conv3d(64, output_channels, kernel_size=1, stride=1, padding=0)
        
        self._initialize_weights()
        self.n_joints = output_channels

    def forward(self, volumes, grid_centers):
        """
        volumes: Tensor [batch_size, C, H, W, D]
        grid_centers: [batch_size, nvox**3, 3]
        """
        volumes = self.encoder_decoder(volumes)
        heatmaps = self.output_layer(volumes)

        if grid_centers is not None:
            softmax_heatmaps = spatial_softmax(heatmaps)
            coords = expected_value_3d(softmax_heatmaps, grid_centers)
        else:
            coords = None

        return coords, heatmaps
        
        # torch.amax(heatmaps, dim=(2, 3, 4)).squeeze(0) # torch amax returns max values, not position

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


def initialize_model(params, n_cams, device):
    """
    Initialize DANNCE model with params and move to GPU.
    """
    model_params = {
        "input_channels": (params["chan_num"] + params["depth"]) * n_cams,
        "output_channels": params["n_channels_out"],
        "norm_method": params["norm_method"],
        "input_shape": params["nvox"]
    }

    if params["net_type"] == "dannce":
        model_params = {**model_params, "residual": False, "norm_upsampling": False}
    elif params["net_type"] == "semi-v2v":
        model_params = {**model_params, "residual": False, "norm_upsampling": True}
    elif params["net_type"] == "v2v":
        model_params = {**model_params, "residual": True, "norm_upsampling": True}
    elif params["net_type"] == "autoencoder":
        model_params["input_channels"] = model_params["input_channels"] - 3
        model_params["output_channels"] = 3
        model_params = {**model_params, "residual": True, "norm_upsampling": True}

    model = DANNCE(**model_params)

    # model = model.to(device)
    if params["multi_gpu_train"]:
        model = nn.parallel.DataParallel(model, device_ids=params["gpu_id"])
    
    model.to(device)

    return model

def initialize_train(params, n_cams, device, logger):
    """
    Initialize model, load pretrained checkpoints if needed.
    """
    params["start_epoch"] = 1

    if params["train_mode"] == "new":
        logger.info("*** Traininig from scratch. ***")
        model = initialize_model(params, n_cams, device)
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)

    elif params["train_mode"] == "finetune":
        logger.info("*** Finetuning from {}. ***".format(params["dannce_finetune_weights"]))
        checkpoints = torch.load(params["dannce_finetune_weights"])
        model = initialize_model(params, n_cams, device)

        state_dict = checkpoints["state_dict"]
        # replace final output layer if do not match with the checkpoint
        ckpt_channel_num = state_dict["output_layer.weight"].shape[0]
        if ckpt_channel_num != params["n_channels_out"]:
            state_dict.pop("output_layer.weight", None)
            state_dict.pop("output_layer.bias", None)

        model.load_state_dict(state_dict, strict=False)

        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)
    
    elif params["train_mode"] == "continued":
        logger.info("*** Resume training from {}. ***".format(params["dannce_finetune_weights"]))
        checkpoints = torch.load(params["dannce_finetune_weights"])
        
        # ensure the same architecture
        model = initialize_model(checkpoints["params"], n_cams, device)
        model.load_state_dict(checkpoints["state_dict"], strict=True)

        model_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(model_params)
        optimizer.load_state_dict(checkpoints["optimizer"])

        # specify the start epoch
        params["start_epoch"] = checkpoints["epoch"]
    
    lr_scheduler = None
    if params["lr_scheduler"] is not None:
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, params["lr_scheduler"]["type"])
        lr_scheduler = lr_scheduler_class(optimizer=optimizer, **params["lr_scheduler"]["args"], verbose=True)
        logger.info("Using learning rate scheduler.")
    
    return model, optimizer, lr_scheduler

if __name__ == "__main__":
    model_params = {
        "input_channels": 18,
        "output_channels": 23,
        "norm_method": 'batch',
        "input_shape": 80
    }
    model_params = {**model_params, "residual": False, "norm_upsampling": False}
    model = DANNCE(**model_params)

    input_shape = [128, 80, 8] # encoder-decoder downsamples for 3 times which force input dimension to be divisble by 2**3 = 8
    inputs = torch.randn(1, 18, *input_shape)
    (x_coord, y_coord, z_coord) = torch.meshgrid(torch.arange(input_shape[0]), torch.arange(input_shape[1]), torch.arange(input_shape[2]))
    grid_centers = torch.stack((x_coord, y_coord, z_coord), axis=0).unsqueeze(0)
    grid_centers = grid_centers.reshape(*grid_centers.shape[:2], -1)

    _, heatmaps = model(inputs, grid_centers)
    print(heatmaps.shape)