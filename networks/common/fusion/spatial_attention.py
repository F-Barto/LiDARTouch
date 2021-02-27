from networks.common.fusion.channels_equalizer import ModalitiesEqualizer
from networks.common.fusion.fusion_base import FusionBase

import einops as eop
from networks.common.basic_blocks import conv1x1
import torch.nn as nn

class SpatialAttentionFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, activation_cls=None, nb_partitions=16):
        super().__init__()

        self.activation = None
        self.equalizer = None
        if lidar_in_chans is not None and image_in_chans is not None and activation_cls is not None:
            self.setup_module(lidar_in_chans, image_in_chans, activation_cls)

        in_chans = max(lidar_in_chans, image_in_chans)

        self.conv1x1 = conv1x1(in_chans, nb_partitions + 1, bias=True)
        self.softmax = nn.Softmax(dim=2)

    def setup_module(self, lidar_in_chans, image_in_chans, activation_cls):
            self.activation = activation_cls(inplace=True)
            self.equalizer = ModalitiesEqualizer(lidar_in_chans, image_in_chans, activation_cls)

    @property
    def require_chans(self):
        return True

    @property
    def require_activation(self):
        return True

    def forward(self, image_features, lidar_features):

        image_features, lidar_features = self.equalizer(image_features, lidar_features)

        spatial_partitions = self.conv1x1(image_features)

        _, _, h, w = image_features.shape

        # reshape tensor into big matrix and transpose if needed -> BxCxHxW into BxCxHW or BxHWxC
        image_features = eop.rearrange(image_features, 'b c h w -> b (h w) c')
        lidar_features = eop.rearrange(lidar_features, 'b c h w -> b (h w) c')

        spatial_partitions = eop.rearrange(spatial_partitions, 'b n h w -> b n (h w)')
        spatial_partitions = self.softmax(spatial_partitions)

        tokens = spatial_partitions[:, :-1, :] @ lidar_features
        spatial_partitions = eop.rearrange(spatial_partitions, 'b n hw -> b hw n')

        weighted_tokens = spatial_partitions[:, :, :-1] @ tokens

        output = image_features * weighted_tokens + image_features
        output = eop.rearrange(output, 'b (h w) c -> b c h w', h=h, w=w)

        return output


