from networks.extractors.lidar.pointnet2 import MLP
from utils.multiview_warping_and_projection import project_pc

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch

from networks.common.trunc_norm_init import trunc_normal_


class Projector(nn.Module):
    def __init__(self, in_chans, out_chans, learn_zero_encoding=True, activation_cls=None, linear_mapping=True):
        super().__init__()

        mlp = [MLP([in_chans, in_chans // 4, out_chans])]

        if linear_mapping:
            mlp.append(nn.ReLU(inplace=True))
            mlp.append(nn.Linear(out_chans, out_chans))

        self.mlp = nn.Sequential(*mlp)

        self.zero_encoding = None
        if learn_zero_encoding:
            self.zero_encoding = Parameter(torch.zeros(1, out_chans))
            trunc_normal_(self.zero_encoding, std=.02)

        self.activation = None
        if activation_cls is not None:
            self.activation = activation_cls(inplace=True)

    def forward(self, cam, pc_input, B, H, W, return_idxs=False):
        pc_features, pc_pos, pc_batch = pc_input

        # get given lidar features into a new fusion feature space
        pc_features = self.mlp(pc_features)

        projected_lidar_features, uv_map, batch_idx = project_pc(cam, pc_features, pc_pos, pc_batch, B, H, W)

        if self.zero_encoding is not None:
            no_lidar = projected_lidar_features.new_ones(B,H,W)
            no_lidar[batch_idx, uv_map[:, 1], uv_map[:, 0]] = 0.
            no_lidar = torch.nonzero(no_lidar)
            projected_lidar_features[no_lidar[:,0], :, no_lidar[:,1], no_lidar[:,2]] = self.zero_encoding

        out = projected_lidar_features

        if self.activation is not None:
            out = self.activation(out)

        if return_idxs:
            return out, uv_map, batch_idx

        return out
