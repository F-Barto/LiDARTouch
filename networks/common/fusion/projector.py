from networks.extractors.lidar.pointnet2 import MLP
from utils.multiview_warping_and_projection import project_pc

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch


class Projector(nn.Module):
    def __init__(self, in_chans, out_chans, learn_zero_encoding=True, activation_cls=None):
        super().__init__()

        self.mlp = MLP([in_chans, in_chans // 4])
        self.lin = nn.Linear(in_chans // 4, out_chans)
        self.activation = activation_cls(inplace=True)

        self.zero_encoding = None
        if learn_zero_encoding:
            self.zero_encoding = Parameter(torch.Tensor(out_chans))
            nn.init.uniform_(self.zero_encoding)

    def forward(self, cam, pc_input, B, downsampled_H, downsampled_W):
        pc_features, pc_pos, pc_batch = pc_input

        # get given lidar features into a new fusion feature space
        pc_features = self.lin(self.mlp(pc_features))

        projected_lidar_features, uv_map, batch_idx = project_pc(cam, pc_features, pc_pos, pc_batch, B,
                                              downsampled_H, downsampled_W)

        if self.zero_encoding is not None:
            zeros = self.zero_encoding.unsqueeze(0).expand(batch_idx.size(0), -1)
            projected_lidar_features[batch_idx, :, uv_map[:, 1], uv_map[:, 0]] = zeros

        return self.activation(projected_lidar_features)