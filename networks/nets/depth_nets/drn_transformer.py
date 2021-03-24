"""
Multi-scale Fusion and Multi-scale Prediction w/ same nb of resolutions levels from each modality
"""


import torch.nn as nn

from networks.nets.net_base import NetworkBase

from networks.common.basic_blocks import get_activation

from networks.extractors.extractors_utils import select_image_extractor
from networks.extractors.lidar.pointnet2 import PointNet2MRGExtractor

from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder

from networks.common.fusion.fusion_transformer import TransformerFusion

########################################################################################################################

class DRNTransformer(NetworkBase):
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

        self.num_ch_enc = self.encoder.num_ch_enc
        self.lidar_ch_enc = self.num_ch_enc[-1]

        if 'head' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('head')
        if 'out_chans' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('out_chans')
        self.lidar_encoder = PointNet2MRGExtractor(self.lidar_ch_enc, head='local', **lidar_extractor_hparams)

        self.fusion = TransformerFusion(self.lidar_ch_enc, self.num_ch_enc[-1], **fusion_hparams)

        chans_enc = [*self.num_ch_enc, self.num_ch_enc[-1]]

        self.decoder = MultiscalePredictionDecoder(chans_enc=chans_enc, activation=activation_cls,
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
        _, _, H, W = image_input.shape
        features, pos, batch = pc_input.x, pc_input.pos, pc_input.batch

        image_features = self.encoder(image_input)
        local_pc_features = self.lidar_encoder(features, pos, batch)
        pc_local = local_pc_features, pos, batch

        _, _, downsampled_H, downsampled_W = image_features[-1].shape
        scale_factor = downsampled_W / float(W)

        fused_feature = self.fusion(image_features[-1], pc_local, intrinsics, scale_factor)

        image_features.append(fused_feature)
        outputs = self.decoder(image_features)

        return outputs