from lidartouch.networks.layers.basic_blocks import get_activation

from lidartouch.networks.net_base import NetworkBase
from lidartouch.networks.layers.resnet_base import build_network
from lidartouch.networks.extractors.image.resnet import ResNetExtractor
from lidartouch.networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder


from lidartouch.utils.identifiers import IMAGE

class DepthNetMonodepth2(NetworkBase):
    """
    Inverse depth network based on the ResNet architecture.
    Args:
        decoder_options: ipsum
        encoder_options: ipsum
        activation: ipsum
    """
    def __init__(self, decoder_options: dict = {}, encoder_options:dict = {} , activation: str = 'relu', **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)
        self.encoder = build_network(ResNetExtractor, activation_cls, **encoder_options)

        self.decoder = MultiscalePredictionDecoder(chans_enc=self.encoder.num_ch_enc, activation=activation_cls,
                                                   **decoder_options)

    @property
    def required_inputs(self):
        return [IMAGE]

    def forward(self, camera_image):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.encoder(camera_image)
        outputs = self.decoder(x)

        return outputs