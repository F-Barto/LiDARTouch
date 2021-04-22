from networks.common.fusion.fusion_base import FusionBase
from networks.common.transformer_utils import PreNorm, FeedForward, WeightedAttention, SlotAttention, WeightedTransformer

from utils.camera import Camera

from .projector import Projector
from .block import Block

import torch
import torch.nn as nn
import einops as eop


class SlotAttentionFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, **kwargs):
        super().__init__()

        self.activation = None
        if lidar_in_chans is not None and image_in_chans is not None:
            self.setup_module(lidar_in_chans, image_in_chans, **kwargs)

    def setup_module(self, lidar_in_chans, image_in_chans, learn_zero_encoding=False,
                     num_slots=32, slot_temp=1., slot_dim=128, iters=3, mlp_ratio=4., lidar_dropout=0.5,
                     lidar_attn_mask=True):

        self.lidar_attn_mask = lidar_attn_mask

        self.in_mapping = PreNorm(image_in_chans, nn.Linear(image_in_chans, slot_dim))
        self.slot_image_attention = SlotAttention(num_slots, slot_dim, iters=iters,mlp_ratio=mlp_ratio,
                                                  return_attn=True, temp=slot_temp)

        self.learn_zero_encoding = learn_zero_encoding
        if learn_zero_encoding:
            print('WARNING: learn_zero_encoding is set to True for SlotAttentionFusion module')
        self.projector = Projector(lidar_in_chans, lidar_in_chans, activation_cls=None,
                                   learn_zero_encoding=learn_zero_encoding)
        self.lidar_in_mapping = PreNorm(lidar_in_chans, nn.Linear(lidar_in_chans, slot_dim))

        self.slots_lidar_transformer = WeightedTransformer(1, slot_dim, mlp_ratio=mlp_ratio, dropout=lidar_dropout,
                                                           temp=slot_temp)

        self.inputs_to_slots_attn = PreNorm(slot_dim, WeightedAttention(slot_dim, softmax_dim=2, weighted_mean_dim=1))
        self.attn_ff = PreNorm(slot_dim, FeedForward(slot_dim, int(slot_dim * mlp_ratio)))

        self.out_mapping = PreNorm(slot_dim, nn.Linear(slot_dim, image_in_chans))


    @property
    def require_chans(self):
        return True

    def forward(self, image_input, pc_input=None, intrinsics=None, scale_factor=None, pos_emb=None,
                lidar_features=None):

        B, _, H, W = image_input.shape

        image_features = eop.rearrange(image_input, 'b c h w -> b (h w) c')
        if pos_emb is not None:
            image_features = image_features + pos_emb
        image_features = self.in_mapping(image_features)

        slots, slot_attn = self.slot_image_attention(image_features)

        # project 3D-to-2D
        if lidar_features is None and pc_input is not None:
            cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(image_features.device)
            lidar_features = self.projector(cam, pc_input, B, H, W)
        else:
            assert lidar_features is not None
        lidar_features = eop.rearrange(lidar_features, 'b c h w -> b (h w) c')
        if pos_emb is not None:
            lidar_features = lidar_features + pos_emb
        lidar_features = self.lidar_in_mapping(lidar_features)

        lidar_mask = None
        if self.lidar_attn_mask:
            lidar_mask = lidar_features.sum(dim=-1, keepdim=True) > 0.
        slots = self.slots_lidar_transformer(slots, context=lidar_features, mask=lidar_mask)

        # integrate fused features as residual in image features
        image_features = self.inputs_to_slots_attn(image_features, context=slots) + image_features
        image_features = self.attn_ff(image_features) + image_features

        output = self.out_mapping(image_features)
        output = eop.rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)

        output = output + image_input

        return output



class OldSlotAttentionFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, **kwargs):
        super().__init__()

        self.activation = None
        if lidar_in_chans is not None and image_in_chans is not None:
            self.setup_module(lidar_in_chans, image_in_chans, **kwargs)

    def setup_module(self, lidar_in_chans, image_in_chans, learn_zero_encoding=False,
                     num_slots=32, slot_dim=128, iters=3, mlp_ratio=4.,
                     block_mm_ratio=4., block_chunks=8, block_rank=10):

        self.in_mapping = PreNorm(image_in_chans, nn.Linear(image_in_chans, slot_dim))
        self.slot_image_attention = SlotAttention(num_slots, slot_dim, iters=iters,
                                                  mlp_ratio=mlp_ratio, return_attn=True)

        self.learn_zero_encoding = learn_zero_encoding
        if learn_zero_encoding:
            print('WARNING: learn_zero_encoding is set to True for SlotAttentionFusion module')
        self.projector = Projector(lidar_in_chans, lidar_in_chans, activation_cls=None,
                                   learn_zero_encoding=learn_zero_encoding)
        self.lidar_in_mapping = PreNorm(lidar_in_chans, nn.Linear(lidar_in_chans, slot_dim))

        self.block_fusion = Block([slot_dim]*2, slot_dim, mm_dim=int(slot_dim * block_mm_ratio),
                                  add_first_unimodal=True, chunks=block_chunks, rank=block_rank)

        self.inputs_to_slots_attn = PreNorm(slot_dim, WeightedAttention(slot_dim, softmax_dim =2, weighted_mean_dim=1))
        self.attn_ff = PreNorm(slot_dim, FeedForward(slot_dim, int(slot_dim * mlp_ratio)))

        self.out_mapping = PreNorm(slot_dim, nn.Linear(slot_dim, image_in_chans))

    @property
    def require_chans(self):
        return True

    def forward(self, image_input, pc_input=None, intrinsics=None, scale_factor=None, pos_emb=None,
                lidar_features=None):

        B, _, H, W = image_input.shape

        image_features = eop.rearrange(image_input, 'b c h w -> b (h w) c')
        if pos_emb is not None:
            image_features = image_features + pos_emb
        image_features = self.in_mapping(image_features)

        image_slots, slot_attn = self.slot_image_attention(image_features)

        # project 3D-to-2D
        if lidar_features is None and pc_input is not None:
            cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(image_features.device)
            lidar_features = self.projector(cam, pc_input, B, H, W)
        else:
            assert lidar_features is not None
        lidar_features = eop.rearrange(lidar_features, 'b c h w -> b (h w) c')
        if pos_emb is not None:
            lidar_features = lidar_features + pos_emb
        lidar_features = self.lidar_in_mapping(lidar_features)
        # nb_nonzero_by_batch = (lidar_features.sum(dim=-1) > 0).sum(dim=1).unsqueeze(-1).unsqueeze(-1)  # bx1x1

        # generate LiDAR slots
        lidar_slots = torch.einsum('bij,bjd->bid', slot_attn, lidar_features)  # [B,M,N] x [B,N,D] -> [B,M,D]

        # fuse LiDAR and Image slots with BLOCK
        image_slots = eop.rearrange(image_slots, 'b n d -> (b n) d')
        lidar_slots = eop.rearrange(lidar_slots, 'b n d -> (b n) d')
        fused_slots = self.block_fusion([image_slots, lidar_slots])
        fused_slots = eop.rearrange(fused_slots, '(b n) d -> b n d', b=B)

        # integrate fused features as residual in image features
        image_features = self.inputs_to_slots_attn(image_features, context=fused_slots) + image_features
        image_features = self.attn_ff(image_features) + image_features

        output = self.out_mapping(image_features)
        output = eop.rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)

        output = output + image_input

        return output


