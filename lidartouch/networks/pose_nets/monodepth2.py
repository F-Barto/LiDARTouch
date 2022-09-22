import torch
import torch.nn as nn

from lidartouch.networks.layers.basic_blocks import get_activation

from lidartouch.networks.layers.resnet_base import build_network
from lidartouch.networks.extractors.image.resnet import ResNetExtractor
from lidartouch.networks.decoders.pose_decoder import PoseDecoder

########################################################################################################################

class PoseResNet(nn.Module):
    """
    Pose network based on the ResNet architecture.
    Args:
        version :  18, 34 or 50
    """
    def __init__(self, version:int=18, input_channels:int=3, activation:str='relu', **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)
        self.encoder = build_network(ResNetExtractor, activation_cls, version=version, input_channels=input_channels * 2)

        self.decoder = PoseDecoder(num_ch_enc=self.encoder.num_ch_enc, num_input_features=1,
                                   activation=activation_cls, num_frames_to_predict_for=2)

    def forward(self, target_image, ref_imgs, **kwargs):
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
