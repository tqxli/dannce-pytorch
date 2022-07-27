import torch
import torch.nn as nn
import torch.nn.functional as F

from dannce.engine.models.transformer.transformer import build_transformer
from dannce.engine.models.transformer.position_encoding import PositionEmbeddingSine
from dannce.engine.models.transformer.layer import MLP
from dannce.engine.data import ops

class SocialTransformer(nn.Module):
    def __init__(self, posenet, transformer, pos_embedding):
        super().__init__()

        self.posenet = posenet
        self.transformer = transformer
        self.pos_embedding = pos_embedding
        self.query_embed = nn.Embedding(23, 256)

        # MLP for converting multiscale features into 256-D features
        self.linear = MLP(input_dim=224, hidden_dim=[1024, 512, 256], output_dim=256-24, num_layers=4, is_activation_last=True)

        # predict
        self.linear_pose = MLP(256, 256, 3, 3)

    def _sample_features(self, grids, feature_pyramid):
        f3 = F.grid_sample(feature_pyramid[0], grids, align_corners=True).squeeze(-1).squeeze(-1)
        f2 = F.grid_sample(feature_pyramid[1], grids, align_corners=True).squeeze(-1).squeeze(-1)
        f1 = F.grid_sample(feature_pyramid[2], grids, align_corners=True).squeeze(-1).squeeze(-1)
        f = torch.cat((f3, f2, f1), dim=1).permute(0, 2, 1) # [N, 46, 224]

        return f
    
    def _sample_pos(self, grids, pos_embed):
        return F.grid_sample(pos_embed, grids, align_corners=True).squeeze().permute(0, 2, 1)

    def forward(self, volumes, grid_centers):
        # initial pose generation from encoder-decoder
        _, heatmaps, feature_pyramid = self.posenet(volumes, grid_centers)

        # locate 
        grids = ops.max_coord_3d(heatmaps).unsqueeze(2).unsqueeze(2) #[N, 46, 1, 1, 3]

        masks = grids.new_ones(*grids.shape[:2])

        # get position embedding
        pos_embed = self.pos_embedding(volumes, volumes.new_ones(volumes.shape[0], *volumes.shape[2:]).bool()) #[N, embed*3, H, W, D]
        positions = self._sample_pos(grids, pos_embed) # [N, 46, embed*3]
        
        # get multiscale features
        multiscale_features = self._sample_features(grids, feature_pyramid) # [N, 46, 224]

        # get input sequences
        input_seq = self.linear(multiscale_features) #[N, 46, 256]
        input_seq = torch.cat([input_seq, positions], dim=-1).permute(0, 2, 1) #[N, 448, 46]
        positions = torch.zeros_like(input_seq).to(input_seq.device)

        transformer_out, memory = self.transformer(
            src=input_seq, mask=torch.logical_not(masks),
            query_embed=self.query_embed.weight,
            pos_embed=positions,
        )
    
        # get poses
        pose = self.linear_pose(transformer_out)

        return pose, heatmaps
        
def build_model(posenet, args):
    transformer = build_transformer(args)
    pos_embedding = PositionEmbeddingSine(8, normalize=True)

    return SocialTransformer(posenet, transformer, pos_embedding)

        