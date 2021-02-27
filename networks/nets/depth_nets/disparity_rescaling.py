"""
Multi-scale Fusion and Multi-scale Prediction w/ same nb of resolutions levels from each modality
"""

import torch
import torch.nn as nn

from networks.nets.net_base import NetworkBase

from networks.common.basic_blocks import get_activation

from networks.extractors.extractors_utils import select_image_extractor, select_lidar_extractor

from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder


########################################################################################################################


class DispRescaling(NetworkBase):
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
    def __init__(self, image_extractor_name, image_extractor_hparams, lidar_extractor_name, lidar_extractor_hparams,
                 decoder_hparams, activation, **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)

        # keeping the name `encoder` for image encoder so that we can use pre-trained weights from TRI for monodepth2 and packnet
        self.encoder = select_image_extractor(image_extractor_name, activation=activation_cls,
                                              **image_extractor_hparams)

        self.lidar_encoder = select_lidar_extractor(lidar_extractor_name, **lidar_extractor_hparams)

        self.num_ch_enc = self.encoder.num_ch_enc
        self.lidar_ch_enc = self.lidar_encoder.out_chans

        # at each prediction rescale
        self.params_gens = nn.ModuleDict()
        for i in range(len(self.num_ch_enc)-1):
            # one factor by channel
            self.params_gens[f"gammas_{i}"] = nn.Linear(self.lidar_ch_enc, 1)
            self.params_gens[f"betas_{i}"] = nn.Linear(self.lidar_ch_enc, 1)

        if 'scales' not in decoder_hparams:
            decoder_hparams['scales'] = len(self.num_ch_enc)-1

        self.decoder = MultiscalePredictionDecoder(num_ch_enc=self.num_ch_enc, activation=activation_cls,
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


    def forward(self, image_input, pc_input):

        cam_features = self.encoder(image_input)
        lidar_features = self.lidar_encoder(pc_input.x, pc_input.pos, pc_input.batch)

        self.gammas = []
        self.betas = []
        for i in range(len(self.num_ch_enc)-1):
            _, _, height, width = cam_features[i].data.shape
            height = height * 2
            width = width * 2

            gammas = self.params_gens[f"gammas_{i}"](lidar_features)
            betas = self.params_gens[f"betas_{i}"](lidar_features)

            # extend the betas and gammas of each channel across the height and width of feature map
            betas_expanded = torch.stack([betas] * height, dim=2)
            betas_expanded = torch.stack([betas_expanded] * width, dim=3)

            gammas_expanded = torch.stack([gammas] * height, dim=2)
            gammas_expanded = torch.stack([gammas_expanded] * width, dim=3)

            self.gammas.append(gammas_expanded)
            self.betas.append(betas_expanded)

        outputs = self.decoder(cam_features, gammas=self.gammas, betas=self.betas)

        return outputs

