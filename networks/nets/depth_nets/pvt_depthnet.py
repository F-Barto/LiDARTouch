from networks.common.basic_blocks import get_activation

from networks.nets.net_base import NetworkBase
from networks.extractors.extractors_utils import select_image_extractor
from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder



class PVTDepthNet(NetworkBase):
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
    def __init__(self, decoder_options, encoder_name, encoder_options, activation='relu', **kwargs):
        super().__init__(**kwargs)

        assert encoder_name in ['pvt_tiny','pvt_relemb_tiny', 'pvt_relemb'], encoder_name
        self.encoder = select_image_extractor(encoder_name, **encoder_options)

        activation_cls = get_activation(activation)
        self.decoder = MultiscalePredictionDecoder(chans_enc=self.encoder.num_ch_enc, activation=activation_cls,
                                                   **decoder_options)

    @property
    def require_lidar_input(self):
        return False

    @property
    def require_image_input(self):
        return True

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.encoder(x)
        outputs = self.decoder(x)

        return outputs