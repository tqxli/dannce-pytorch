import torch
import torch.nn as nn
import torch.nn.functional as F

from dannce.engine.models.transformer.layer import MLP
from dannce.engine.data import ops

class XIA(nn.Module):
    def __init__(self, embed_dim=256, nb_h=8, dropout=0.1):
        super(XIA, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nb_h, dropout=dropout)

        self.fc = nn.Sequential(nn.LayerNorm(embed_dim),
                            nn.Linear(embed_dim,embed_dim),
                            nn.ReLU(),
                            nn.Linear(embed_dim,embed_dim),
                            nn.LayerNorm(embed_dim))

    def forward(self, k1, k2):
        # return k1_new
        query = k2.permute(2,0,1) # could possibly replace w/ learnt embedding
        key = k1.permute(2,0,1)
        value = k1.permute(2,0,1)
        k1=k1.permute(2,0,1)

        k = self.self_attn(query, key, value=value)[0]
        k1 = k1+k
        k1 = self.fc(k1)
        return k1.permute(1,2,0)

class SocialXAttn(nn.Module):
    def __init__(self, posenet, n_kpts=23, in_features=227, d_model=256):
        super().__init__()
        self.posenet = posenet

        self.n_kpts = n_kpts
        self.in_features = in_features

        self.ffn1 = MLP(input_dim=in_features, hidden_dim=d_model, output_dim=d_model, num_layers=2, is_activation_last=True)
        self.ffn2 = MLP(input_dim=in_features, hidden_dim=d_model, output_dim=d_model, num_layers=2, is_activation_last=True)

        self.XIA = XIA(embed_dim=d_model, nb_h=8, dropout=0.1)

        self.linear_pose = MLP(input_dim=d_model, hidden_dim=[1024, 512, 256], output_dim=3, num_layers=4, is_activation_last=False)

        # self._freeze_stages()

    def _freeze_stages(self):
        for name, param in self.posenet.named_parameters():
            param.requires_grad = False

    def forward(self, volumes, grid_centers):
        # initial pose generation from encoder-decoder
        init_poses, heatmaps, feature_pyramid = self.posenet(volumes, grid_centers)

        com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1) #[N, 3, 1]
        nvox = round(grid_centers.shape[1]**(1/3))
        vsize = (grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()) / nvox
        x = (init_poses - com3d) / vsize

        coord_grids = ops.max_coord_3d(heatmaps).unsqueeze(2).unsqueeze(2)

        f3 = F.grid_sample(feature_pyramid[0], coord_grids, align_corners=True).squeeze(-1).squeeze(-1)
        f2 = F.grid_sample(feature_pyramid[1], coord_grids, align_corners=True).squeeze(-1).squeeze(-1)
        f1 = F.grid_sample(feature_pyramid[2], coord_grids, align_corners=True).squeeze(-1).squeeze(-1)

        f = torch.cat((f3, f2, f1), dim=1) #[B, 224, 46]

        x = torch.cat((f, x), dim=1).permute(0, 2, 1) #[B, 46, 224+3]

        src_primary, src_secondary = x[:, :self.n_kpts, :], x[:, self.n_kpts:, :]

        k_primary = self.ffn1(src_primary).permute(0, 2, 1) #[B, 256, 23]
        k_secondary = self.ffn2(src_secondary).permute(0, 2, 1)

        cross_feats = self.XIA(k_primary, k_secondary) #[B, 256, 23]

        final_poses = self.linear_pose(cross_feats.permute(0, 2, 1)) #[B, 23, 3]
        final_poses = final_poses.permute(0, 2, 1)

        return init_poses, final_poses, heatmaps



