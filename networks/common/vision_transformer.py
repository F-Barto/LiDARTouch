# adapted from https://github.com/whai362/PVT/blob/main/pvt.py


import torch
import torch.nn as nn
from functools import lru_cache


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, x_kv=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if x_kv is None:
            x_kv = x

        if self.sr_ratio > 1:
            x_kv = x_kv.permute(0, 2, 1).reshape(B, C, H, W)
            x_kv = self.sr(x_kv).reshape(B, C, -1).permute(0, 2, 1)
            x_kv = self.norm(x_kv)
            kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AddAndNorm(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)

    def forward(self, x, out):
        x = self.norm(x + out)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.add_norm1 = AddAndNorm(dim, norm_layer)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.add_norm2 = AddAndNorm(dim, norm_layer)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, x_kv=None):
        attn = self.attn(x, H, W, x_kv=x_kv)
        x = self.add_norm1(x, attn)
        mlp = self.mlp(x)
        x = self.add_norm2(x, mlp)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"img_size {(H, W)} should be divided by patch_size {self.patch_size}."

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


@lru_cache(maxsize=None)
def gen_pos_idxs(H, W, device, dtype, normalized=True):
    """
    Create patch indexes with a specific resolution
    Parameters
    ----------
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        output type
    device : torch.device
        output device
    normalized : bool
        True if indexes are normalized between -1 and 1
    Returns
    -------
    xs : torch.Tensor [1,1,W]
        Meshgrid in dimension x
    ys : torch.Tensor [1,H,1]
        Meshgrid in dimension y
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])

    """ example of meshgrid for H,W = 3,4
            >>> xs
            tensor([[0., 1., 2., 3.],
                    [0., 1., 2., 3.],
                    [0., 1., 2., 3.]])
            >>> ys
            tensor([[0., 0., 0., 0.],
                    [1., 1., 1., 1.],
                    [2., 2., 2., 2.]])
    """

    ys, xs = ys.flatten().unsqueeze(-1), xs.flatten().unsqueeze(-1)
    patch_idxs = torch.cat([ys, xs], dim=-1)

    """ example of patch_idxs for H,W = 3,4
        >>> patch_idxs
        tensor([[0., 0.],
                [0., 1.],
                [0., 2.],
                [0., 3.],
                [1., 0.],
                [1., 1.],
                [1., 2.],
                [1., 3.],
                [2., 0.],
                [2., 1.],
                [2., 2.],
                [2., 3.]])
    """

    return patch_idxs.unsqueeze(0) # output is dim 1x(HxW)x2