import torch
import torch.nn as nn

from networks.common.basic_blocks import get_activation

from networks.common.resnet_base import build_model
from networks.extractors.image.resnet import ResNetExtractor
from networks.decoders.pose_decoder import PoseDecoder

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
    def __init__(self, version=18, input_channels=3, activation='relu', **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)
        self.encoder = build_model(ResNetExtractor, activation_cls, version=version, input_channels=input_channels*2)

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
