from networks.common.fusion.channels_equalizer import ModalitiesEqualizer
from networks.common.fusion.fusion_base import FusionBase
from networks.common.basic_blocks import conv1x1

from utils.camera import Camera

from .projector import Projector

import torch
import torch.nn as nn
import einops as eop


class SpatialAttentionFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, activation_cls=None, nb_partitions=16, **kwargs):
        super().__init__()

        self.activation = None
        self.equalizer = None
        if lidar_in_chans is not None and image_in_chans is not None and activation_cls is not None:
            self.setup_module(lidar_in_chans, image_in_chans, activation_cls, nb_partitions=nb_partitions, **kwargs)

    def setup_module(self, lidar_in_chans, image_in_chans, activation_cls, nb_partitions=16, **kwargs):
        self.activation = activation_cls(inplace=True)

        self.projector = Projector(lidar_in_chans, image_in_chans, activation_cls=activation_cls, **kwargs)

        self.conv1x1 = conv1x1(image_in_chans, nb_partitions + 1, bias=True)
        self.softmax = nn.Softmax(dim=2)

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

    def forward(self, image_features, pc_input, intrinsics, scale_factor):
        B, _, downsampled_H, downsampled_W = image_features.shape

        # project 3D-to-2D
        cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(image_features.device)
        lidar_features = self.projector(cam, pc_input, B, downsampled_H, downsampled_W)

        spatial_partitions = self.conv1x1(image_features)

        b, _, h, w = image_features.shape

        # reshape tensor into big matrix and transpose if needed -> BxCxHxW into BxCxHW or BxHWxC
        lidar_features = eop.rearrange(lidar_features, 'b c h w -> b (h w) c')
        image_features = eop.rearrange(image_features, 'b c h w -> b (h w) c')

        spatial_partitions = eop.rearrange(spatial_partitions, 'b n h w -> b n (h w)')
        spatial_partitions = self.softmax(spatial_partitions)

        lidar_spatial_partitions = spatial_partitions[:, 1:, :]
        lidar_tokens = lidar_spatial_partitions @ lidar_features
        lidar_spatial_partitions = eop.rearrange(lidar_spatial_partitions, 'b n hw -> b hw n')
        lidar_tokenized_partitions = lidar_spatial_partitions @ lidar_tokens

        image_spatial_partitions = spatial_partitions[:, :1, :]
        image_tokens = image_spatial_partitions @ image_features
        image_spatial_partitions = eop.rearrange(image_spatial_partitions, 'b n hw -> b hw n')
        image_tokenized_partitions = image_spatial_partitions @ image_tokens

        tokenized_partitions = image_tokenized_partitions + lidar_tokenized_partitions

        output = eop.rearrange(tokenized_partitions, 'b (h w) c -> b c h w', h=h, w=w)
        image_features = eop.rearrange(image_features, 'b (h w) c -> b c h w', h=h, w=w)
        output = torch.cat([output, image_features], dim=1)
        output = self.conv_out(output)

        return output


