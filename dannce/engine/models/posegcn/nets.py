from functools import reduce
import torch
import torch.nn as nn

from dannce.engine.models.posegcn.gcn_blocks import _GraphConv, _ResGraphConv_Attention, _ResGraphConv, SemGraphConv
from dannce.engine.models.posegcn.non_local import _GraphNonLocal
from dannce.engine.models.posegcn.utils import *

# NODES_GROUP = [[1, 2], [0, 3], [5, 6], [7, 11], [8, 9], [9, 10], [12, 13], [15, 19], [16, 17], [17, 18], [10, 12], [12, 13]]
NODES_GROUP = [[i] for i in range(23)]
TEMPORAL_FLOW = np.array([0, 4, 9, 13, 17, 21]) # restrict the flows along temporal dimension 

class PoseGCN(nn.Module):
    def __init__(self, 
            pose_generator, 
            input_dim=3, 
            hid_dim=128, 
            n_layers=4, 
            n_instances=1,
            t_dim=1,
            t_flow=TEMPORAL_FLOW,
            non_local=False,
            nodes_group=NODES_GROUP,
            base_block='sem',
            norm_type='batch',
            dropout=None,
        ):
        super(PoseGCN, self).__init__()
        self.pose_generator = pose_generator
        self.n_instances = n_instances
        self.adj = adj = build_adj_mx_from_edges(social=(n_instances > 1), t_dim=t_dim, t_flow=t_flow)
        self.t_dim = t_dim

        # construct GCN
        self.gconv_input = [_GraphConv(adj, input_dim, hid_dim, dropout, base_block=base_block, norm_type=norm_type)]

        gconv_layers = []
        if not non_local:
            for i in range(n_layers):
                gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, dropout, base_block=base_block, norm_type=norm_type))
        else:
            group_size = len(nodes_group[0])
            # assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            self.gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(n_layers):
                gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, dropout, base_block=base_block, norm_type=norm_type))
                gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*self.gconv_input)
        self.gconv_layers = nn.Sequential(*gconv_layers)

        self.gconv_output = SemGraphConv(hid_dim, input_dim, adj)

    def forward(self, volumes, grid_centers):
        # pose generation
        init_poses, heatmaps = self.pose_generator(volumes, grid_centers)
        
        x = init_poses.transpose(2, 1) #[B, 23, 3]
        x = x.reshape(init_poses.shape[0] // self.n_instances, -1, 3) #[n, 46, 3] or [n, 23, 3]
        x = x.reshape(-1, self.t_dim * x.shape[1], x.shape[2]) #[n, t_dim*23, 3]

        x = self.gconv_input(x)
        x = self.gconv_layers(x)
        x = self.gconv_output(x)

        final_poses = x.reshape(init_poses.shape[0], -1, 3).transpose(2, 1) + init_poses
        return final_poses, heatmaps

if __name__ == "__main__":
    model = PoseGCN()

    input = torch.randn(5, 23, 3)
    print("Input: ", input.shape)
    output = model(input)
    print("Output: ", output.shape)