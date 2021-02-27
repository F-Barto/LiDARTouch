from networks.common.fusion.channels_equalizer import ModalitiesEqualizer
from networks.common.fusion.fusion_base import FusionBase

from networks.common.basic_blocks import conv1x1

from networks.extractors.lidar.pointnet2 import MLP

from utils.camera import Camera
from utils.multiview_warping_and_projection import project_pc

import torch.nn as nn
import torch

class ProjectSqueezeFuse(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, activation_cls=None, reduction=16):
        super().__init__()

        self.activation = None
        if lidar_in_chans is not None and image_in_chans is not None and activation_cls is not None:
            self.setup_module(lidar_in_chans, image_in_chans, activation_cls, reduction=reduction)

    def setup_module(self, lidar_in_chans, image_in_chans, activation_cls, reduction=16):
        self.activation = activation_cls(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.local_mlp = MLP([lidar_in_chans + 3, lidar_in_chans // 4, image_in_chans])
        self.global_mlp = MLP([lidar_in_chans, lidar_in_chans // 4, image_in_chans])

        self.conv = nn.Sequential(
            conv1x1(image_in_chans, image_in_chans, bias=False),
            nn.BatchNorm2d(image_in_chans),
            self.activation
        )

        self.conv_out = nn.Sequential(
            conv1x1(image_in_chans*2, image_in_chans, bias=False),
            nn.BatchNorm2d(image_in_chans),
            self.activation
        )

        fc_in_chans = image_in_chans * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_in_chans, fc_in_chans // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fc_in_chans // reduction, fc_in_chans, bias=False),
            nn.Sigmoid()
        )

    @property
    def require_chans(self):
        return True

    @property
    def require_activation(self):
        return True

    def forward(self, image_features, pc_local, pc_global_features, intrinsics, scale_factor):
        B, _, downsampled_H, downsampled_W = image_features.shape

        # get given lidar features into a new fusion feature space
        pc_local_features, pc_local_pos, pc_local_batch = pc_local
        pc_local_features = self.local_mlp(torch.cat([pc_local_features, pc_local_pos], dim=1))

        pc_global_features = self.global_mlp(pc_global_features)

        # project 3D-to-2D
        cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(image_features.device)
        projected_lidar_features = project_pc(cam, pc_local_features, pc_local_pos, pc_local_batch, B,
                                              downsampled_H, downsampled_W)

        # get given image features into a new fusion feature space
        image_features = self.conv(image_features)

        # computes channels weights
        b, c, _, _ = image_features.size()
        gap_image_features = self.gap(image_features).view(b, c)
        weights = torch.cat([gap_image_features, pc_global_features], dim=1)
        weights = self.fc(weights).view(b, c*2, 1, 1)

        out = torch.cat([image_features, projected_lidar_features], dim=1)
        out = out * weights.expand_as(out)

        # C*2 -> C
        out = self.conv_out(out)

        return out


