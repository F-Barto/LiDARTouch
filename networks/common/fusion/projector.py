from networks.extractors.lidar.pointnet2 import MLP
from utils.multiview_warping_and_projection import project_pc

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch

from networks.common.trunc_norm_init import trunc_normal_


class Projector(nn.Module):
    def __init__(self, in_chans, out_chans, learn_zero_encoding=True, activation_cls=None):
        super().__init__()

        self.mlp = MLP([in_chans, in_chans // 4, out_chans])

        self.zero_encoding = None
        if learn_zero_encoding:
            self.zero_encoding = Parameter(torch.zeros(1, out_chans))
            trunc_normal_(self.zero_encoding, std=.02)

        self.norm = nn.BatchNorm2d(out_chans)

        self.activation = None
        if activation_cls is not None:
            self.activation = activation_cls(inplace=True)

    def forward(self, cam, pc_input, B, H, W):
        pc_features, pc_pos, pc_batch = pc_input

        # get given lidar features into a new fusion feature space
        pc_features = self.mlp(pc_features)

        projected_lidar_features, uv_map, batch_idx = project_pc(cam, pc_features, pc_pos, pc_batch, B, H, W)

        if self.zero_encoding is not None:
            zeros = self.zero_encoding.expand(batch_idx.size(0), -1)
            projected_lidar_features[batch_idx, :, uv_map[:, 1], uv_map[:, 0]] = zeros

        out = self.norm(projected_lidar_features)

        if self.activation is not None:
            out = self.activation(out)

        return out
