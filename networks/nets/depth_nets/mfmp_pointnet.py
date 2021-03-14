"""
Multi-scale Fusion and Multi-scale Prediction w/ same nb of resolutions levels from each modality
"""


import torch.nn as nn

from networks.nets.net_base import NetworkBase

from networks.common.basic_blocks import get_activation

from networks.extractors.extractors_utils import select_image_extractor
from networks.extractors.lidar.pointnet2 import PointNet2MRGExtractor

from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder

from networks.common.fusion.fusion_utils import select_fusion_module

########################################################################################################################


class MFMPPointNet(NetworkBase):
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
                 decoder_hparams, activation, fusion_name='project_squeeze_fuse', fusion_hparams=None, **kwargs):
        super().__init__(**kwargs)

        assert fusion_name in ['project_squeeze_fuse']


        activation_cls = get_activation(activation)
        if fusion_hparams is None:
            fusion_hparams = {}

        # keeping the name `encoder` for image encoder so that we can use pre-trained weights from TRI for monodepth2 and packnet
        self.encoder = select_image_extractor(image_extractor_name, activation=activation_cls,
                                              **image_extractor_hparams)

        self.num_ch_enc = self.encoder.num_ch_enc
        self.lidar_ch_enc = self.num_ch_enc[-1]

        if 'head' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('head')
        if 'out_chans' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('out_chans')
        self.lidar_encoder = PointNet2MRGExtractor(self.lidar_ch_enc, head='local+global', **lidar_extractor_hparams)

        self.fusion_name = fusion_name
        self.fusion_module = select_fusion_module(self.fusion_name)

        # at each resblock fuse with guidance the features of both encoders
        self.fusions = nn.ModuleDict()
        for i in range(len(self.num_ch_enc)):

             fusion_module = self.fusion_module()
             args = []
             if fusion_module.require_chans:
                 args += [self.lidar_ch_enc, self.num_ch_enc[i]]
             if fusion_module.require_activation:
                 args.append(activation_cls)
             fusion_module.setup_module(*args, **fusion_hparams)

             self.fusions[f"{self.fusion_name}_{i}"] = fusion_module

        if 'scales' not in decoder_hparams:
            decoder_hparams['scales'] = len(self.num_ch_enc)-1

        self.decoder = MultiscalePredictionDecoder(num_ch_enc=self.num_ch_enc, activation=activation_cls,
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
        global_pc_features, local_pc_features = self.lidar_encoder(features, pos, batch)
        pc_local = local_pc_features, pos, batch

        self.fused_features = []
        for i in range(len(self.num_ch_enc)):

            _, _, downsampled_H, downsampled_W = image_features[i].shape
            scale_factor = downsampled_W / float(W)

            fused_feature = self.fusions[f"{self.fusion_name}_{i}"](image_features[i], pc_local, global_pc_features,
                                                                    intrinsics, scale_factor)
            self.fused_features.append(fused_feature)

        outputs = self.decoder(self.fused_features)

        return outputs