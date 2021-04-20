"""
Multi-scale Fusion and Multi-scale Prediction w/ same nb of resolutions levels from each modality
"""


import torch.nn as nn
import einops as eop

from networks.nets.net_base import NetworkBase

from networks.common.basic_blocks import get_activation, conv1x1

from networks.extractors.extractors_utils import select_image_extractor

from networks.extractors.lidar.pointnet2 import PointNet2MRGExtractor

from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder

from networks.common.transformer_utils import Transformer, gen_pos_idxs, fourier_encode
from networks.common.fusion.slot_attention import SlotAttentionFusion, OldSlotAttentionFusion

from networks.common.trunc_norm_init import trunc_normal_

########################################################################################################################

class DRNSlot(NetworkBase):
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
                 decoder_hparams, activation, self_attention_hparams, fusion_hparams, **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)

        self.encoder = select_image_extractor(image_extractor_name, activation=activation_cls,
                                              **image_extractor_hparams)

        chans_enc = self.encoder.num_ch_enc
        last_in_chans = chans_enc[-1]

        if 'head' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('head')
        if 'out_chans' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('out_chans')
        self.lidar_encoder = PointNet2MRGExtractor(last_in_chans, head='local', **lidar_extractor_hparams)

        #self.pos_embed = nn.Linear(2 * ((4 * 2) + 1), last_in_chans)
        self.pos_embed = conv1x1(2 * ((4 * 2) + 1), last_in_chans, stride=1, bias=True)

        self.self_attention = Transformer(depth=1, dim=last_in_chans, **self_attention_hparams)
        self.slot_attention = SlotAttentionFusion(last_in_chans, last_in_chans, **fusion_hparams)

        self.decoder = MultiscalePredictionDecoder(chans_enc=chans_enc, activation=activation_cls,
                                                   **decoder_hparams)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

        # pre-compute LiDAR point cloud features
        features, pos, batch = pc_input.x, pc_input.pos, pc_input.batch
        pc_features = self.lidar_encoder(features, pos, batch)
        pc_features = pc_features, pos, batch

        # image features with self-attention
        B, _, H, W = image_input.shape

        image_features = self.encoder(image_input)
        other_features = image_features[:-1]

        bottleneck = image_features[-1]
        _, _, dH, dW = bottleneck.shape
        # patchs_idxs = gen_pos_idxs(dH, dW, bottleneck.device, bottleneck.dtype, normalized=True)
        # pos_emb = self.pos_embed(patchs_idxs)
        pos_emb = fourier_encode(dH, dW, bottleneck.device, bottleneck.dtype, 5, num_bands=4, base=2)
        pos_emb = eop.rearrange(pos_emb, '... n d -> ... (n d)')
        pos_emb = eop.repeat(pos_emb, '... -> b ...', b=B)
        pos_emb = eop.rearrange(pos_emb, 'b h w c -> b c h w')
        pos_emb = self.pos_embed(pos_emb)
        pos_emb = eop.rearrange(pos_emb, 'b c h w -> b (h w) c')

        bottleneck = eop.rearrange(bottleneck, 'b c h w -> b (h w) c')
        bottleneck = bottleneck + pos_emb
        bottleneck = self.self_attention(bottleneck, dH, dW, context=None)
        bottleneck = eop.rearrange(bottleneck, 'b (h w) c -> b c h w', h=dH, w=dW)

        # slot attention based fusion
        scale_factor = dW / float(W)
        bottleneck = self.slot_attention(bottleneck, pc_features, intrinsics, scale_factor, pos_emb=pos_emb)

        # add fused bottleneck to the feature pyramid
        image_features = other_features + [bottleneck]

        outputs = self.decoder(image_features)

        return outputs