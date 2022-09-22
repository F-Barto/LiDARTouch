import torch
import torch.nn as nn

from lidartouch.networks.layers.basic_blocks import get_activation

from lidartouch.networks.layers.resnet_base import build_network
from lidartouch.networks.extractors.image.resnet import ResNetExtractor
from lidartouch.networks.extractors.lidar.resnet import LiDARResNetExtractor
from lidartouch.networks.decoders.pose_decoder import PoseDecoder

########################################################################################################################

class PoseResNetLIDAR(nn.Module):
    """
    Pose network based on the ResNet architecture.
    Args:
        version :  18, 34 or 50
    """
    def __init__(self, version:int=18, input_channels:int=3, activation:str='relu', **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)
        self.encoder = build_network(ResNetExtractor, activation_cls, version=version, input_channels=input_channels * 2)
        self.encoder_lidar = build_network(LiDARResNetExtractor, activation_cls, version=version, input_channels= 1,
                                           inv_input_depth=True)

        num_ch_enc = [img_chans + lidar_chans for (img_chans, lidar_chans) in
                      zip(self.encoder.num_ch_enc, self.encoder_lidar.num_ch_enc)]

        self.decoder = PoseDecoder(num_ch_enc=num_ch_enc, num_input_features=1,
                                   activation=activation_cls, num_frames_to_predict_for=2)

    def forward(self, target_image, ref_imgs, sparse_depth):
        """
        Runs the network and returns predicted poses
        (1 for each reference image).
        """
        outputs = []
        for i, ref_img in enumerate(ref_imgs):
            inputs = torch.cat([target_image, ref_img], 1)
            img_features = self.encoder(inputs)
            lidar_features = self.encoder_lidar(sparse_depth)

            features = []
            for f_i, f_l in zip(img_features, lidar_features):
                features.append(torch.cat([f_i, f_l], dim=1))

            axisangle, translation = self.decoder([features])
            outputs.append(torch.cat([translation[:, 0], axisangle[:, 0]], 2))
        pose = torch.cat(outputs, 1)
        return pose







