"""
Multi-scale Fusion and Multi-scale Prediction w/ same nb of resolutions levels from each modality
"""

import torch
import torch.nn as nn

from networks.nets.net_base import NetworkBase

from networks.common.basic_blocks import get_activation

from networks.extractors.extractors_utils import select_image_extractor

from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder

from networks.extractors.lidar.pointnet2 import PointNet2MRGExtractor

from networks.common.fusion.spatial_attention import SpatialAttentionFusion

from utils.camera import Camera
from utils.multiview_warping_and_projection import project_pc


########################################################################################################################


class SpatialAttention(NetworkBase):
    """
    Inverse depth network based on the ResNet architecture.
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
    def __init__(self, image_extractor_name, image_extractor_hparams, lidar_extractor_hparams,
                 decoder_hparams, activation, fusion_hparams, **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)

        # keeping the name `encoder` for image encoder so that we can use pre-trained weights from TRI for monodepth2 and packnet
        self.encoder = select_image_extractor(image_extractor_name, activation=activation_cls,
                                              **image_extractor_hparams)

        self.lidar_encoder = PointNet2MRGExtractor(**lidar_extractor_hparams)

        num_ch_enc = self.encoder.num_ch_enc
        lidar_ch_enc = self.lidar_encoder.out_chans

        self.fusion = SpatialAttentionFusion(lidar_in_chans=lidar_ch_enc, image_in_chans=num_ch_enc[-1],
                                             activation_cls=activation_cls, **fusion_hparams)

        self.decoder = MultiscalePredictionDecoder(num_ch_enc=num_ch_enc, activation=activation_cls,
                                                   **decoder_hparams)
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)
        '''


    @property
    def require_lidar_input(self):
        return False

    @property
    def require_pc_input(self):
        return True

    @property
    def require_image_input(self):
        return True

    @property
    def require_intrinsics(self):
        return True


    def forward(self, image_input, pc_input, intrinsics):

        cam_features = self.encoder(image_input)
        lidar_features = self.lidar_encoder(pc_input.x, pc_input.pos, pc_input.batch)

        # Generate camera
        B, _, H, W = image_input.shape
        _, _, downsampled_H, downsampled_W = cam_features[-1].shape
        scale_factor = downsampled_W / float(W)
        cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(image_input.device)

        projected_lidar_features = project_pc(cam, lidar_features, pc_input.pos, pc_input.batch, B,
                                              downsampled_H, downsampled_W)

        fused_features = self.fusion(cam_features[-1], projected_lidar_features)
        
        #project lidar feature in image plane
        outputs = self.decoder(cam_features[:-1] + [fused_features])

        return outputs

