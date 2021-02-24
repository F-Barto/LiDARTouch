import torch.nn as nn
import torch

class ConcatFusion(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, image_features, lidar_features):

        c = torch.cat([image_features, lidar_features], dim=1)

        return c

    def setup_module(self, **kwargs):
        pass

    @property
    def require_chans(self):
        return False

    @property
    def require_activation(self):
        return False