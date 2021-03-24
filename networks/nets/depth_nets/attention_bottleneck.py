from networks.common.basic_blocks import get_activation

from networks.nets.net_base import NetworkBase
from networks.common.resnet_base import build_model
from networks.extractors.image.resnet import ResNetExtractor
from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder
from networks.common.vision_transformer import TransformerBlock, gen_pos_idxs
from networks.extractors.extractors_utils import select_image_extractor

import torch.nn as nn
from functools import partial
import einops as eop

class DepthNetAttentionBottleneck(NetworkBase):
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
    def __init__(self, image_extractor_name, image_extractor_hparams, decoder_hparams,
                 transformer_hparams, activation='relu', **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)
        self.encoder = select_image_extractor(image_extractor_name, activation=activation_cls,
                                              **image_extractor_hparams)


        chans = self.encoder.num_ch_enc[-1]
        self.pos_embed = nn.Linear(2, chans)

        self.transformer = TransformerBlock(dim=chans, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            act_layer= nn.GELU, **transformer_hparams)

        self.decoder = MultiscalePredictionDecoder(chans_enc=self.encoder.num_ch_enc, activation=activation_cls,
                                                   **decoder_hparams)

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

        other_features = x[:-1]
        bottleneck = x[-1]
        B, _, H, W = bottleneck.shape
        patchs_idxs = gen_pos_idxs(H, W, bottleneck.device, bottleneck.dtype, normalized=True)
        pos_emb = self.pos_embed(patchs_idxs)

        bottleneck = eop.rearrange(bottleneck, 'b c h w -> b (h w) c')
        bottleneck = bottleneck + pos_emb
        bottleneck = self.transformer(bottleneck, H, W)
        bottleneck = eop.rearrange(bottleneck, 'b (h w) c -> b c h w', h=H, w=W)

        x = other_features + [bottleneck]

        outputs = self.decoder(x)

        return outputs