from networks.common.fusion.channels_equalizer import ModalitiesEqualizer
from networks.common.fusion.fusion_base import FusionBase
from networks.common.basic_blocks import conv1x1

from utils.camera import Camera

from .projector import Projector

import torch
import torch.nn as nn
import einops as eop


from networks.common.vision_transformer import TransformerBlock, PatchEmbed, gen_pos_idxs
from networks.common.trunc_norm_init import trunc_normal_


class TransformerFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, **kwargs):
        super().__init__()

        if lidar_in_chans is not None and image_in_chans is not None:
            self.setup_module(lidar_in_chans, image_in_chans, **kwargs)

    def setup_module(self, lidar_in_chans, image_in_chans, learn_zero_encoding=True, drop_rate=0., depth=1,
                     patch_size=16, **kwargs):

        self.projector = Projector(lidar_in_chans, image_in_chans, learn_zero_encoding=learn_zero_encoding)

        chans = image_in_chans
        self.patch_size = patch_size
        self.image_patch_embed = PatchEmbed(patch_size=patch_size, in_chans=chans, embed_dim=chans)
        self.lidar_patch_embed = PatchEmbed(patch_size=patch_size, in_chans=chans, embed_dim=chans)

        # transformer encoder
        self.image_block = nn.ModuleList([
            TransformerBlock(dim=chans, drop=drop_rate, **kwargs)
            for i in range(depth)])
        self.cross_block = nn.ModuleList([
            TransformerBlock(dim=chans, drop=drop_rate, **kwargs)
            for i in range(depth)])

        # modality_embed
        self.image_embed = nn.Parameter(torch.zeros(1, 1, chans))
        self.lidar_embed = nn.Parameter(torch.zeros(1, 1, chans))

        self.pos_embed = nn.Linear(2, chans)
        self.pos_drop = nn.Dropout(p=drop_rate)

        trunc_normal_(self.image_embed, std=.02)
        trunc_normal_(self.lidar_embed, std=.02)
        trunc_normal_(self.pos_embed.weight, std=.02)

        if self.pos_embed.bias is not None:
            nn.init.constant_(self.pos_embed.bias, 0)


    @property
    def require_chans(self):
        return True

    def forward(self, image_features, pc_input, intrinsics, scale_factor):
        B, _, H, W = image_features.shape

        # project 3D-to-2D
        cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(image_features.device)
        lidar_features = self.projector(cam, pc_input, B, H, W)

        patchs_idxs = gen_pos_idxs(H // self.patch_size, W // self.patch_size,
                                   image_features.device, image_features.dtype,
                                   normalized=True)
        pos_emb = self.pos_embed(patchs_idxs)

        image_patch_emb, (H, W) = self.image_patch_embed(image_features)
        image_patch_emb = image_patch_emb + pos_emb + self.image_embed
        image_patch_emb = self.pos_drop(image_patch_emb)
        for blk in self.image_block:
            image_patch_emb = blk(image_patch_emb, H, W)
        #image_patch_emb = image_patch_emb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        output=image_patch_emb

        lidar_patch_emb, (H, W) = self.lidar_patch_embed(lidar_features)
        lidar_patch_emb = lidar_patch_emb + pos_emb + self.lidar_embed
        lidar_patch_emb = self.pos_drop(lidar_patch_emb)
        for blk in self.cross_block:
            output = blk(output, H, W, x_kv=lidar_patch_emb)
        output = output.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return output


