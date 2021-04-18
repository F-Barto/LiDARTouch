# adapted from https://github.com/whai362/PVT/blob/main/pvt.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

import numpy as np
from functools import partial

from networks.common.vision_transformer import TransformerBlock, PatchEmbed, gen_pos_idxs
from networks.common.trunc_norm_init import trunc_normal_


class PyramidVisionTransformer(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        self.num_ch_enc = np.array(embed_dims)

        # patch_embed
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])

        # pos_embed
        #self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_embed1 = nn.Linear(2,embed_dims[0])
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        #self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_embed2 = nn.Linear(2, embed_dims[1])
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        #self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_embed3 = nn.Linear(2, embed_dims[2])
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        #self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, embed_dims[3]))
        self.pos_embed4 = nn.Linear(2, embed_dims[3])
        self.pos_drop4 = nn.Dropout(p=drop_rate)

        # transformer encoder
        self.block1 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,  norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        self.block2 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        self.block3 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        self.block4 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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



    def forward(self, x):

        features = []
        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        patchs_idxs = gen_pos_idxs(H, W, x.device, x.dtype, normalized=True)
        x = x + self.pos_embed1(patchs_idxs)
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        features.append(x)

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        patchs_idxs = gen_pos_idxs(H, W, x.device, x.dtype, normalized=True)
        x = x + self.pos_embed2(patchs_idxs)
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        features.append(x)

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        patchs_idxs = gen_pos_idxs(H, W, x.device, x.dtype, normalized=True)
        x = x + self.pos_embed3(patchs_idxs)
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        features.append(x)

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        patchs_idxs = gen_pos_idxs(H, W, x.device, x.dtype, normalized=True)
        x = x + self.pos_embed4(patchs_idxs)
        x = self.pos_drop4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        features.append(x)

        return features

def pvt(**kwargs):
    model = PyramidVisionTransformer(qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def pvt_tiny(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model

def pvt_small(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model

def pvt_medium(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model


def pvt_large(**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model