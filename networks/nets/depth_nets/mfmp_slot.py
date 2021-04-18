"""
Multi-scale Fusion and Multi-scale Prediction w/ same nb of resolutions levels from each modality
"""


import torch.nn as nn
import einops as eop

from networks.nets.net_base import NetworkBase

from networks.common.basic_blocks import get_activation

from networks.common.resnet_base import build_model
from networks.extractors.image.dilated_resnet_C import DRN_C

from networks.extractors.extractors_utils import select_lidar_extractor

from networks.decoders.iterative_multiscale_decoder import MultiscalePredictionDecoder

from networks.common.fusion.fusion_utils import select_fusion_module, get_ch_post_fusion

from networks.common.transformer_utils import Transformer, gen_pos_idxs
from networks.common.fusion.slot_attention import SlotAttentionFusion

########################################################################################################################


class MFMPSlotDepthNet(NetworkBase):
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
    def __init__(self, image_extractor_hparams, lidar_extractor_name, lidar_extractor_hparams,
                 decoder_hparams, fusion_name, activation, self_attention_hparams, slot_fusion_hparams,
                 fusion_hparams=None, **kwargs):
        super().__init__(**kwargs)

        activation_cls = get_activation(activation)
        if fusion_hparams is None:
            fusion_hparams = {}

        # keeping the name `encoder` for image encoder so that we can use pre-trained weights from TRI for monodepth2 and packnet
        self.encoder = build_model(DRN_C, activation=activation_cls, **image_extractor_hparams)

        self.lidar_encoder = select_lidar_extractor(lidar_extractor_name, activation=activation_cls,
                                                    **lidar_extractor_hparams)

        self.fusion_name = fusion_name
        self.fusion_module = select_fusion_module(self.fusion_name)

        self.num_ch_enc = self.encoder.num_ch_enc
        self.lidar_ch_enc = self.lidar_encoder.num_ch_enc



        assert len(self.num_ch_enc) == len(self.lidar_ch_enc)

        self.num_ch_skips = get_ch_post_fusion(self.fusion_name, self.lidar_ch_enc, self.num_ch_enc)
        self.num_ch_skips[-1] = self.num_ch_enc[-1]

        # at each resblock fuse with guidance the features of both encoders
        self.fusions = nn.ModuleDict()
        for i in range(len(self.num_ch_enc)-1):

             fusion_module = self.fusion_module()
             args = []
             if fusion_module.require_chans:
                 args += [self.num_ch_enc[i], self.lidar_ch_enc[i]]
             if fusion_module.require_activation:
                 args.append(activation_cls)
             fusion_module.setup_module(*args, **fusion_hparams)

             self.fusions[f"{self.fusion_name}_{i}"] = fusion_module

        last_in_chans = self.num_ch_enc[-1]
        self.pos_embed = nn.Linear(2, last_in_chans)

        self.self_attention = Transformer(depth=1, dim=last_in_chans, **self_attention_hparams)
        self.slot_attention = SlotAttentionFusion(last_in_chans, last_in_chans, **slot_fusion_hparams)

        self.decoder = MultiscalePredictionDecoder(chans_enc=self.num_ch_skips, activation=activation_cls,
                                                   **decoder_hparams)


    @property
    def require_lidar_input(self):
        return True

    @property
    def require_image_input(self):
        return True


    def forward(self, image_input, lidar_input):

        cam_features = self.encoder(image_input)
        lidar_features = self.lidar_encoder(lidar_input)


        self.fused_features = []
        for i in range(len(self.num_ch_enc)-1):

            fused_feature = self.fusions[f"{self.fusion_name}_{i}"](cam_features[i], lidar_features[i])
            self.fused_features.append(fused_feature)

        # image features with self-attention
        bottleneck = cam_features[-1]
        _, _, dH, dW = bottleneck.shape
        patchs_idxs = gen_pos_idxs(dH, dW, bottleneck.device, bottleneck.dtype, normalized=True)
        pos_emb = self.pos_embed(patchs_idxs)

        bottleneck = eop.rearrange(bottleneck, 'b c h w -> b (h w) c')
        bottleneck = bottleneck + pos_emb
        bottleneck = self.self_attention(bottleneck, dH, dW, context=None)
        bottleneck = eop.rearrange(bottleneck, 'b (h w) c -> b c h w', h=dH, w=dW)

        # slot attention based fusion
        bottleneck = self.slot_attention(bottleneck, lidar_features=lidar_features[-1], pos_emb=pos_emb)

        # add fused bottleneck to the feature pyramid
        self.fused_features.append(bottleneck)

        outputs = self.decoder(self.fused_features)

        return outputs

