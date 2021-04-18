from networks.common.basic_blocks import get_activation

from networks.nets.net_base import NetworkBase
from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder
from networks.common.vision_transformer import TransformerBlock, gen_pos_idxs
from networks.common.trunc_norm_init import trunc_normal_
from networks.common.fusion.spatial_attention import SpatialAttentionFusion

from networks.extractors.lidar.pointnet2 import PointNet2MRGExtractor


from networks.common.resnet_base import build_model
from networks.extractors.hybrid.earlyfusion_dilated_resnet_C import DRN_C

import torch.nn as nn
from functools import partial
import einops as eop

class DepthNetSAFBottleneck(NetworkBase):
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
    def __init__(self, image_extractor_hparams, lidar_extractor_hparams, decoder_hparams,
                 transformer_hparams, saf_hparams, activation='elu', **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)
        self.encoder = build_model(DRN_C, activation=activation_cls, **image_extractor_hparams)


        chans = self.encoder.num_ch_enc[-1]

        if 'head' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('head')
        if 'out_chans' in lidar_extractor_hparams:
            lidar_extractor_hparams.pop('out_chans')
        self.lidar_encoder = PointNet2MRGExtractor(chans, head='local', **lidar_extractor_hparams)

        self.pos_embed = nn.Linear(2, chans)
        self.transformer = TransformerBlock(dim=chans, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            act_layer= nn.GELU, **transformer_hparams)

        self.saf = SpatialAttentionFusion(lidar_in_chans=chans, image_in_chans=chans, activation_cls=activation_cls,
                                          **saf_hparams)

        self.decoder = MultiscalePredictionDecoder(chans_enc=self.encoder.num_ch_enc, activation=activation_cls,
                                                   **decoder_hparams)

        self.transformer.apply(self._transformer_init_weights)
        self.saf.apply(self._saf_init_weights)

    def _transformer_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _saf_init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            self._transformer_init_weights(m)

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
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        _,_,H,W = image_input.shape

        features, pos, batch = pc_input.x, pc_input.pos, pc_input.batch
        local_pc_features = self.lidar_encoder(features, pos, batch)
        pc_local = local_pc_features, pos, batch

        x = self.encoder(image_input, pc_local, intrinsics)

        other_features = x[:-1]
        bottleneck = x[-1]

        # self-attention
        _, _, bH, bW = bottleneck.shape
        patchs_idxs = gen_pos_idxs(bH, bW, bottleneck.device, bottleneck.dtype, normalized=True)
        pos_emb = self.pos_embed(patchs_idxs)

        bottleneck = eop.rearrange(bottleneck, 'b c h w -> b (h w) c')
        bottleneck = bottleneck + pos_emb
        bottleneck = self.transformer(bottleneck, bH, bW)
        bottleneck = eop.rearrange(bottleneck, 'b (h w) c -> b c h w', h=bH, w=bW)

        # fusion
        scale_factor = bW / float(W)
        bottleneck = self.saf(bottleneck, pc_local, intrinsics, scale_factor)

        x = other_features + [bottleneck]

        outputs = self.decoder(x)

        return outputs