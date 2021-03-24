# adapted from https://github.com/whai362/PVT/blob/main/pvt.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

import numpy as np
from functools import partial

from networks.common.fusion.projector import Projector
from networks.common.vision_transformer import TransformerBlock, PatchEmbed, gen_pos_idxs
from networks.common.trunc_norm_init import trunc_normal_
from networks.common.resnet_blocks import BasicBlock
from networks.common.basic_blocks import conv7x7, conv1x1

from utils.camera import Camera


class InfusedPyramidVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6], sr_ratios=[8, 4, 2],
                 learn_zero_encoding=True):

        super().__init__()
        self.depths = depths

        self.num_ch_enc = np.array(embed_dims)

        self.inplanes = 16
        self.layer0 = nn.Sequential(
            conv7x7(in_chans, self.inplanes, stride=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ELU(inplace=True),
            self._make_layer(BasicBlock, 16, 1, nn.ELU),
            self._make_layer(BasicBlock, embed_dims[0], 1, nn.ELU, stride=2)
        )

        self.layer0.apply(self._init_head_weights)

        # patch_embed
        self.patch_embed = nn.ModuleDict()
        for i in range(3):
            self.patch_embed[f'image_embed_{i}'] = PatchEmbed(patch_size=2, in_chans=embed_dims[i],
                                                              embed_dim=embed_dims[i+1])
            self.patch_embed[f'lidar_embed_{i}'] = PatchEmbed(patch_size=2, in_chans=embed_dims[i+1],
                                                              embed_dim=embed_dims[i+1])

        # pos_embed
        self.pos_embed = nn.ModuleDict()
        for i in range(3):
            self.pos_embed[f'pos_embed_{i}'] = nn.Linear(2, embed_dims[i+1])
            self.pos_embed[f'pos_drop_{i}'] = nn.Dropout(p=drop_rate)

        # drop
        self.drop = nn.ModuleDict()
        for i in range(3):
            self.drop[f'drop_{i}'] = nn.Dropout(p=drop_rate)

        # modality_embed
        self.modality_embed = nn.ParameterDict()
        for i in range(3):
            emb = nn.Parameter(torch.zeros(1, 1, embed_dims[i+1]))
            trunc_normal_(emb, std=.02)
            self.modality_embed[f'image_embed_{i}'] = emb
            emb = nn.Parameter(torch.zeros(1, 1, embed_dims[i+1]))
            trunc_normal_(emb, std=.02)
            self.modality_embed[f'lidar_embed_{i}'] = emb

        # projectors
        self.projectors = nn.ModuleDict()
        for i in range(3):
            self.projectors[f'projector_{i}'] = Projector(self.num_ch_enc[-1], embed_dims[i+1],
                                                          learn_zero_encoding=learn_zero_encoding)

        # transformers blocks
        self.transformers = nn.ModuleDict()
        for i in range(3):
            self.transformers[f"image_block_{i}"] = nn.ModuleList([TransformerBlock(
                dim=embed_dims[i+1], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,  norm_layer=norm_layer,
                sr_ratio=sr_ratios[i]) for j in range(depths[i])])

            self.transformers[f"cross_block_{i}"] = TransformerBlock(
                dim=embed_dims[i+1], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,  norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])


        # init weights
        self.apply(self._init_weights)

    def _make_layer(self, block, planes, blocks, activation, stride=1, ):
        norm_layer = nn.BatchNorm2d

        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, activation, stride=stride, downsample=downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation))

        return nn.Sequential(*layers)

    def _init_head_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)



    def forward(self, image_input, pc_input, intrinsics):

        features = []
        B, _, H, W = image_input.shape

        # 7x7 conv head
        x = self.layer0(image_input)
        features.append(x)

        _, _, d_H, d_W = features[-1].shape
        for i in range(3):
            scale_factor = d_W / float(W)
            cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(x.device)
            lidar_features = self.projectors[f'projector_{i}'](cam, pc_input, B, d_H, d_W)
            lidar_patch_emb, (d_H, d_W) = self.patch_embed[f'lidar_embed_{i}'](lidar_features)
            patchs_idxs = gen_pos_idxs(d_H, d_W, x.device, x.dtype, normalized=True)
            pos_emb = self.pos_embed[f'pos_embed_{i}'](patchs_idxs)
            lidar_patch_emb = lidar_patch_emb + pos_emb + self.modality_embed[f'lidar_embed_{i}']

            image_patch_emb, (d_H, d_W) = self.patch_embed[f'image_embed_{i}'](features[-1])
            image_patch_emb = image_patch_emb + pos_emb + self.modality_embed[f'image_embed_{i}']

            image_patch_emb = self.drop[f'drop_{i}'](image_patch_emb)
            for blk in self.transformers[f"image_block_{i}"]:
                image_patch_emb = blk(image_patch_emb, d_H, d_W)
            output = image_patch_emb

            self.transformers[f"cross_block_{i}"](output, d_H, d_W, x_kv=lidar_patch_emb)
            output = output.reshape(B, d_H, d_W, -1).permute(0, 3, 1, 2).contiguous()

            features.append(output)

        return features

def pvt(**kwargs):
    model = InfusedPyramidVisionTransformer(qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def pvt_tiny(**kwargs):
    model = pvt(embed_dims=[32, 64, 128, 512], num_heads=[1, 2, 6], mlp_ratios=[8, 8, 4], depths=[2, 2, 2],
                sr_ratios=[8, 4, 2], **kwargs)
    return model
