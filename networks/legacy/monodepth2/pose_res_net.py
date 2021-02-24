# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from networks.legacy.monodepth2.layers.resnet_encoder import ResnetEncoder
from networks.legacy.monodepth2.layers.pose_decoder import PoseDecoder
from networks.legacy.monodepth2.layers.common import get_activation

########################################################################################################################

class PoseResNet(nn.Module):
    """
    Pose network based on the ResNet architecture.
    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, num_layers=18, activation='relu', input_channels=3):
        super().__init__()

        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        activation_cls = get_activation(activation)

        self.encoder = ResnetEncoder(num_layers=num_layers, activation=activation_cls, num_input_images=2,
                                     input_channels=input_channels)
        self.decoder = PoseDecoder(num_ch_enc=self.encoder.num_ch_enc, num_input_features=1,
                                   activation=activation_cls, num_frames_to_predict_for=2)

    def forward(self, target_image, ref_imgs):
        """
        Runs the network and returns predicted poses
        (1 for each reference image).
        """
        outputs = []
        for i, ref_img in enumerate(ref_imgs):
            inputs = torch.cat([target_image, ref_img], 1)
            axisangle, translation = self.decoder([self.encoder(inputs)])
            outputs.append(torch.cat([translation[:, 0], axisangle[:, 0]], 2))
        pose = torch.cat(outputs, 1)
        return pose
