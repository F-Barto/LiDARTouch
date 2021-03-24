"""
Multi-scale Fusion and Multi-scale Prediction w/ same nb of resolutions levels from each modality
"""


import torch.nn as nn

from networks.nets.net_base import NetworkBase

from networks.common.basic_blocks import get_activation

from networks.extractors.hybrid.infused_pvt import pvt
from networks.extractors.lidar.pointnet2 import PointNet2MRGExtractor

from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder


########################################################################################################################

class InfusedPVTDepthNet(NetworkBase):
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
    def __init__(self, pvt_hparams, lidar_extractor_hparams, decoder_hparams, activation, **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)

        # keeping the name `encoder` for image encoder so that we can use pre-trained weights from TRI for monodepth2 and packnet
        self.encoder = pvt(**pvt_hparams)

        self.num_ch_enc = self.encoder.num_ch_enc
        self.lidar_ch_enc = self.num_ch_enc[-1]

        if 'head' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('head')
        if 'out_chans' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('out_chans')
        self.lidar_encoder = PointNet2MRGExtractor(self.lidar_ch_enc, head='local', **lidar_extractor_hparams)

        self.decoder = MultiscalePredictionDecoder(chans_enc=self.num_ch_enc, activation=activation_cls,
                                                   **decoder_hparams)

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
        pc_features, pos, batch = pc_input.x, pc_input.pos, pc_input.batch
        pc_features = self.lidar_encoder(pc_features, pos, batch)

        pc_input = pc_features, pos, batch
        features = self.encoder(image_input, pc_input, intrinsics)

        outputs = self.decoder(features)

        return outputs