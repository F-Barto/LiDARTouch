from networks.common.fusion.fusion_base import FusionBase

from networks.common.basic_blocks import conv1x1

from networks.extractors.lidar.pointnet2 import MLP

from utils.camera import Camera
import torch.nn as nn
import torch

from .projector import Projector


class ProjectSqueezeFuse(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, activation_cls=None, reduction=16, **kwargs):
        super().__init__()

        self.activation = None
        if lidar_in_chans is not None and image_in_chans is not None and activation_cls is not None:
            self.setup_module(lidar_in_chans, image_in_chans, activation_cls, reduction=reduction, **kwargs)

    def setup_module(self, lidar_in_chans, image_in_chans, activation_cls, reduction=16, **kwargs):
        self.activation = activation_cls(inplace=True)

        self.projector = Projector(lidar_in_chans, image_in_chans, activation_cls=activation_cls, **kwargs)

        self.img_conv = nn.Sequential(
            conv1x1(image_in_chans, image_in_chans, bias=False),
            nn.BatchNorm2d(image_in_chans),
            self.activation
        )


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = MLP([lidar_in_chans, lidar_in_chans // 4, image_in_chans])

        fc_in_chans = image_in_chans * 2
        self.squeeze_excite_fc = nn.Sequential(
            nn.Linear(fc_in_chans, fc_in_chans // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fc_in_chans // reduction, fc_in_chans, bias=False),
            nn.Sigmoid()
        )

        self.conv_out = nn.Sequential(
            conv1x1(image_in_chans * 2, image_in_chans, bias=False),
            nn.BatchNorm2d(image_in_chans),
            self.activation
        )

    @property
    def require_chans(self):
        return True

    @property
    def require_activation(self):
        return True

    def forward(self, image_features, pc_local, pc_global_features, intrinsics, scale_factor):
        B, _, downsampled_H, downsampled_W = image_features.shape

        pc_global_features = self.global_mlp(pc_global_features)

        # project 3D-to-2D
        cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(image_features.device)
        projected_lidar_features = self.projector(cam, pc_local, B, downsampled_H, downsampled_W)

        # get given image features into a new fusion feature space
        image_features = self.img_conv(image_features)

        # computes channels weights
        b, c, _, _ = image_features.size()
        gap_image_features = self.gap(image_features).view(b, c)
        weights = torch.cat([gap_image_features, pc_global_features], dim=1)
        weights = self.squeeze_excite_fc(weights).view(b, c*2, 1, 1)

        out = torch.cat([image_features, projected_lidar_features], dim=1)
        out = out * weights.expand_as(out)

        # C*2 -> C
        out = self.conv_out(out)

        return out


