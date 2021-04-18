from networks.common.fusion.fusion_base import FusionBase
from networks.common.basic_blocks import conv1x1, conv3x3

from utils.camera import Camera

from .projector import Projector

import torch
import torch.nn as nn
import einops as eop


class SpatialAttentionFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, activation_cls=None, **kwargs):
        super().__init__()

        self.activation = None
        if lidar_in_chans is not None and image_in_chans is not None and activation_cls is not None:
            self.setup_module(lidar_in_chans, image_in_chans, activation_cls, **kwargs)

    def setup_module(self, lidar_in_chans, image_in_chans, activation_cls, nb_partitions=16, learn_zero_encoding=False,
                     add_image_partition=True, end_is_concat=True):
        self.activation = activation_cls(inplace=True)

        self.learn_zero_encoding = learn_zero_encoding
        if learn_zero_encoding:
            print('WARNING: learn_zero_encoding is set to True for SpatialAttentionFusion module')
        self.projector = Projector(lidar_in_chans, image_in_chans, activation_cls=None,
                                   learn_zero_encoding=learn_zero_encoding)

        self.img_fusion_mapping = nn.Sequential(
            nn.BatchNorm2d(image_in_chans),
            self.activation,
            conv3x3(image_in_chans, image_in_chans, bias=False),
        )

        self.add_image_partition = add_image_partition
        if add_image_partition:
            nb_partitions += 1
        self.conv1x1_partition = conv1x1(image_in_chans, nb_partitions, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.end_is_concat = end_is_concat
        if self.end_is_concat:
            self.concat_conv = nn.Sequential(
                conv3x3(image_in_chans * 2, image_in_chans, bias=False),
            )

        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(image_in_chans),
            self.activation,
            conv3x3(image_in_chans, image_in_chans, bias=False),
            nn.BatchNorm2d(image_in_chans),
            self.activation
        )

    @property
    def require_chans(self):
        return True

    @property
    def require_activation(self):
        return True

    def forward(self, image_input, pc_input, intrinsics, scale_factor):

        image_features = self.img_fusion_mapping(image_input)

        B, _, downsampled_H, downsampled_W = image_features.shape

        # project 3D-to-2D
        cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(image_features.device)
        lidar_features = self.projector(cam, pc_input, B, downsampled_H, downsampled_W)

        b, _, h, w = image_features.shape

        spatial_partitions = self.conv1x1_partition(image_features)
        # reshape tensor into big matrix and transpose if needed -> BxCxHxW into BxCxHW or BxHWxC
        spatial_partitions = eop.rearrange(spatial_partitions, 'b n h w -> b n (h w)')
        spatial_partitions = self.softmax(spatial_partitions)

        lidar_features = eop.rearrange(lidar_features, 'b c h w -> b (h w) c')
        nb_nonzero_by_batch = (lidar_features.sum(dim=-1) > 0).sum(dim=1).unsqueeze(-1).unsqueeze(-1) # bx1x1
        image_features = eop.rearrange(image_features, 'b c h w -> b (h w) c')

        if self.add_image_partition:
            lidar_spatial_partitions = spatial_partitions[:, 1:, :]
            lidar_tokens = lidar_spatial_partitions @ lidar_features
            if not self.learn_zero_encoding:
                lidar_tokens = lidar_tokens / nb_nonzero_by_batch
            else:
                lidar_tokens = lidar_tokens / (h*w)
            lidar_spatial_partitions = eop.rearrange(lidar_spatial_partitions, 'b n hw -> b hw n')
            lidar_tokenized_partitions = lidar_spatial_partitions @ lidar_tokens

            image_spatial_partitions = spatial_partitions[:, :1, :]
            image_tokens = (image_spatial_partitions @ image_features) / (h*w)
            image_spatial_partitions = eop.rearrange(image_spatial_partitions, 'b n hw -> b hw n')
            image_tokenized_partitions = image_spatial_partitions @ image_tokens

            tokenized_partitions = image_tokenized_partitions + lidar_tokenized_partitions
        else:
            lidar_tokens = spatial_partitions @ lidar_features
            if not self.learn_zero_encoding:
                lidar_tokens = lidar_tokens / nb_nonzero_by_batch
            else:
                lidar_tokens = lidar_tokens / (h * w)
            spatial_partitions = eop.rearrange(spatial_partitions, 'b n hw -> b hw n')
            tokenized_partitions = spatial_partitions @ lidar_tokens

        output = eop.rearrange(tokenized_partitions, 'b (h w) c -> b c h w', h=h, w=w)

        if self.end_is_concat:
            output = torch.cat([output, image_input], dim=1)
            output = self.concat_conv(output)

        output = output + image_input
        output = self.conv_out(output)

        return output