"""
Multi-scale Fusion and Multi-scale Prediction w/ same nb of resolutions levels from each modality
"""


import torch.nn as nn

from lidartouch.networks.net_base import NetworkBase
from lidartouch.networks.layers.basic_blocks import get_activation
from lidartouch.networks.extractors.extractors_utils import select_image_extractor, select_lidar_extractor
from lidartouch.networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder
from lidartouch.networks.layers.fusion.fusion_utils import select_fusion_module, get_ch_post_fusion

from lidartouch.utils.identifiers import IMAGE, SPARSE_DEPTH


########################################################################################################################


class Monodepth2LiDAR(NetworkBase):
    """
    Inverse depth network based on the ResNet architecture.
    Args:
        image_extractor_name: ipsum
        image_extractor_hparams: ipsum
        lidar_extractor_name: ipsum
        lidar_extractor_hparams: ipsum
        decoder_hparams: ipsum
        fusion_name: ipsum
        activation: ipsum
        fusion_hparams: ipsum
    """
    def __init__(self, image_extractor_name: str, image_extractor_hparams: dict, lidar_extractor_name: str,
                 lidar_extractor_hparams:dict, decoder_hparams: dict, fusion_name: str, activation: str,
                 fusion_hparams: dict=None, **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)
        if fusion_hparams is None:
            fusion_hparams = {}

        # keeping the name `encoder` for image encoder so that we can use pre-trained weights from TRI for monodepth2 and packnet
        self.encoder = select_image_extractor(image_extractor_name, activation=activation_cls,
                                              **image_extractor_hparams)

        self.lidar_encoder = select_lidar_extractor(lidar_extractor_name, activation=activation_cls,
                                                    **lidar_extractor_hparams)

        self.fusion_name = fusion_name
        self.fusion_module = select_fusion_module(self.fusion_name)

        self.num_ch_enc = self.encoder.num_ch_enc
        self.lidar_ch_enc = self.lidar_encoder.num_ch_enc

        assert len(self.num_ch_enc) == len(self.lidar_ch_enc)

        self.num_ch_skips = get_ch_post_fusion(self.fusion_name, self.lidar_ch_enc, self.num_ch_enc)

        # at each resblock fuse with guidance the features of both encoders
        self.fusions = nn.ModuleDict()
        for i in range(len(self.num_ch_enc)):

             fusion_module = self.fusion_module()
             args = []
             if fusion_module.require_chans:
                 args += [self.num_ch_enc[i], self.lidar_ch_enc[i]]
             if fusion_module.require_activation:
                 args.append(activation_cls)
             fusion_module.setup_module(*args, **fusion_hparams)

             self.fusions[f"{self.fusion_name}_{i}"] = fusion_module

        if 'scales' not in decoder_hparams:
            decoder_hparams['scales'] = len(self.num_ch_enc)-1

        self.decoder = MultiscalePredictionDecoder(chans_enc=self.num_ch_skips, activation=activation_cls,
                                                   **decoder_hparams)

    @property
    def required_inputs(self):
        return [IMAGE, SPARSE_DEPTH]



    def forward(self, camera_image, sparse_depth):

        cam_features = self.encoder(camera_image)
        lidar_features = self.lidar_encoder(sparse_depth)

        self.fused_features = []
        for i in range(len(self.num_ch_enc)):

            fused_feature = self.fusions[f"{self.fusion_name}_{i}"](cam_features[i], lidar_features[i])
            self.fused_features.append(fused_feature)

        outputs = self.decoder(self.fused_features)

        return outputs

