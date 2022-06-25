from functools import reduce
import torch
import torch.nn as nn

from dannce.engine.models.posegcn.gcn_blocks import _GraphConv, _ResGraphConv_Attention, _ResGraphConv, SemGraphConv, MLP
from dannce.engine.models.posegcn.non_local import _GraphNonLocal
from dannce.engine.models.posegcn.utils import *

# NODES_GROUP = [[1, 2], [0, 3], [5, 6], [7, 11], [8, 9], [9, 10], [12, 13], [15, 19], [16, 17], [17, 18], [10, 12], [12, 13]]
NODES_GROUP = [[i] for i in range(23)]
TEMPORAL_FLOW = np.array([0, 4, 9, 13, 17, 21]) # restrict the flows along temporal dimension 

class PoseGCN(nn.Module):
    def __init__(self, 
            model_params,
            pose_generator, 
            n_instances=1,
            n_joints=23,
            t_dim=1,
        ):
        super(PoseGCN, self).__init__()

        # primary pose estimator
        self.pose_generator = pose_generator

        # GCN architecture
        self.input_dim = input_dim = model_params.get("input_dim", 3)
        self.hid_dim = hid_dim = model_params.get("hidden_dim", 128)
        self.n_layers = n_layers = model_params.get("n_layers", 3)
        self.non_local = model_params.get("non_local", False)
        self.base_block = base_block = model_params.get("base_block", "sem")
        self.norm_type = norm_type = model_params.get("norm_type", "batch")
        self.dropout = dropout = model_params.get("dropout", None)
        
        # skeletal graph construction 
        self.n_instances = n_instances
        self.n_joints = n_joints # num of nodes = n_instance * n_joints
        self.t_dim = t_dim 
        self.t_flow = t_flow = model_params.get("t_flow", TEMPORAL_FLOW)
        inter_social = model_params.get("inter_social", False)
        self.social = (n_instances > 1) and inter_social
        # adjacency matrix
        self.adj = adj = build_adj_mx_from_edges(social=self.social, t_dim=t_dim, t_flow=t_flow)
        # nonlocal joint groups
        self.nodes_group = nodes_group = model_params.get("nodes_group", NODES_GROUP)

        # construct GCN layers

        # self.gconv_input = _GraphConv(adj, input_dim, hid_dim, dropout, base_block=base_block, norm_type=norm_type)
        self.gconv_input = SemGraphConv(input_dim, hid_dim, adj)
        
        gconv_layers = []
        if not self.non_local:
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
        
        self.gconv_layers = nn.Sequential(*gconv_layers)

        self.gconv_output = SemGraphConv(hid_dim, input_dim, adj)

        self.aggre = model_params.get("aggre", None)
        #if self.aggre == "mlp":
        #    self.aggre_layer = MLP(2*n_joints*input_dim, n_joints*input_dim)
        
        self.use_residual = model_params.get("use_residual", True)

    def forward(self, volumes, grid_centers):
        # initial pose generation
        init_poses, heatmaps = self.pose_generator(volumes, grid_centers)
        
        x = init_poses.transpose(2, 1).contiguous() #[B, 23, 3]
        
        if self.social:
            # whether jointly optimize both sets of keypoints
            x = x.reshape(init_poses.shape[0] // self.n_instances, -1, 3).contiguous() #[n, 46, 3] or [n, 23, 3]
        else:
            # treat separately
            x = x.reshape(init_poses.shape[0] * self.n_instances, -1, 3).contiguous() #[x*2, 23, 3]

        # if inputs are across time
        x = x.reshape(-1, self.t_dim * x.shape[1], x.shape[2]).contiguous() #[n, t_dim*23, 3]
        
        x = self.gconv_input(x)
        x = self.gconv_layers(x)
        x = self.gconv_output(x)
        
        x = x.reshape(init_poses.shape[0], -1, 3).transpose(2, 1).contiguous() #[n, 3, 23]

        if self.aggre is not None:
            x = self.aggre_layer(torch.cat((x, init_poses.transpose(2, 1)), dim=-1).reshape(init_poses.shape[0], -1).contiguous())
            x = x.reshape(init_poses.shape[0], -1, 3).contiguous().transpose(2, 1).contiguous()
        
        final_poses = x
        if self.use_residual:
            final_poses += init_poses

        # print("Mean Euclidean correction:", torch.norm(correction, dim=1).mean())
        # final_poses = init_poses
        # return final_poses, heatmaps
        return init_poses, final_poses, heatmaps
    
    def inference(self, init_poses):
        x = init_poses.transpose(2, 1).contiguous() #[B, 23, 3]
        x = x.reshape(init_poses.shape[0] // self.n_instances, -1, 3).contiguous() #[n, 46, 3] or [n, 23, 3]
        x = x.reshape(-1, self.t_dim * x.shape[1], x.shape[2]).contiguous() #[n, t_dim*23, 3]

        x = self.gconv_input(x)
        x = self.gconv_layers(x)
        x = self.gconv_output(x)

        correction = x.reshape(init_poses.shape[0], -1, 3).transpose(2, 1).contiguous()
        final_poses = correction + init_poses

        # print("Mean Euclidean correction:", torch.norm(correction, dim=1).mean())

        return final_poses

if __name__ == "__main__":
    model = PoseGCN()

    input = torch.randn(5, 23, 3)
    print("Input: ", input.shape)
    output = model(input)
    print("Output: ", output.shape)