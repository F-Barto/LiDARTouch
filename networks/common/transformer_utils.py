"""
adapted from
https://github.com/whai362/PVT/blob/main/pvt.py
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cait.py
https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention_experimental.py
https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py


if you are new to this type of implem check https://theaisummer.com/einsum-attention/
"""


from math import pi, log
import torch
from torch import nn
from torch.nn import init
from functools import lru_cache

import einops as eop

# classes
class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class NormBlock(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        self.normed_fn = LayerScale(dim, PreNorm(dim, fn), depth=depth)

    def forward(self, x, *args, **kwargs):
        return self.normed_fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, dropout=0.):
        out_dim = out_dim if out_dim is not None else dim
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        print('dim', dim)
        self.q = nn.Linear(dim, inner_dim, bias=False)

        # spatial reduction https://arxiv.org/pdf/2102.12122.pdf
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.kv = nn.Linear(dim, inner_dim * 2, bias=False)

        project_out = not (num_heads == 1 and inner_dim == dim)
        self.project_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, H, W, context=None, mask=None):
        q = eop.rearrange(self.q(x), 'b n (h d) -> b h n d', h=self.num_heads)  # [B,H,N,D] or [B,num_heads,N,dim]

        if context is None:
            context = x

        if self.sr_ratio > 1:
            context = eop.rearrange(context, 'b (h w) c -> b c h w', h=H, w=W)
            context = eop.rearrange(self.sr(context), 'b c h w -> b (h w) c')
            context = self.norm(context)

        kv = self.kv(context).chunk(2, dim=-1)  # 2*[B,M,HxD]
        k, v = map(lambda t: eop.rearrange(t, 'b m (h d) -> b h m d', h=self.num_heads), kv)  # 2*[B,H,M,D]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [B,H,N,M]


        if mask is not None:
            # mask is a binary mask either bool (True/False) or float (0./1.)
            # assumes mask is True or 1. for elements we want to have attention for and False or 0. otherwise
            mask = eop.rearrange(mask, 'b ... -> b (...)')
            mask = eop.repeat(mask, 'b m -> b () () m') # Bx1x1xM
            # Softmax includes an exponential, since e^0 = 1, we set the mask to a large negative value instead of 0
            log_mask = torch.log(mask.float()) # torch.log(0) = -inf
            dots = dots * mask + log_mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B,H,N,M] x [B,H,M,D] -> [B,H,N,D]
        out = eop.rearrange(out, 'b h n d -> b n (h d)')
        out = self.project_out(out)  # reduce the heads into one head

        return out

class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads, dim_head, mlp_ratio=4., dropout=0., sr_ratio=1):
        super().__init__()
        self.layers = nn.ModuleList([])

        for d in range(1, depth+1):
            attention = Attention(dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout, sr_ratio=sr_ratio)
            ff = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)
            self.layers.append(nn.ModuleList([
                NormBlock(dim, attention, depth=d),
                NormBlock(dim, ff, depth=d)
            ]))
    def forward(self, x, H, W, context=None, mask=None):
        for attn, ff in self.layers:
            x = attn(x, H, W, context=context, mask=mask) + x
            x = ff(x) + x
        return x


class WeightedAttention(nn.Module):
    def __init__(self, dim, eps=1e-8, softmax_dim=1, weighted_mean_dim=2, return_attn=False):
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.eps = eps
        self.scale = dim ** -0.5
        self.softmax_dim = softmax_dim
        self.weighted_mean_dim = weighted_mean_dim

        self.return_attn = return_attn

    def forward(self, inputs, context, mask=None):

        inputs = self.norm_input(inputs)
        context = self.norm_context(context)

        q = self.to_q(inputs)
        k = self.to_k(context)
        v = self.to_v(context)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=self.softmax_dim) + self.eps

        if mask is not None:
            # mask is a binary mask either bool (True/False) or float (0./1.)
            # assumes mask is True or 1. for elements we want to have attention for and False or 0. otherwise
            mask = eop.rearrange(mask, 'b ... -> b (...)')
            mask = eop.repeat(mask, 'b m -> b () m') # Bx1xM

            # masking explanation:
            # let N be the number of slots and M the number of pixels
            # in out slot attention inputs->slots we create a dot product map 'dots' of shape NxM
            # we "assign" each pixel to a slot by making the slots competes through a Softmax on the N dimension
            # this way no input pixel is ignored as sum(dots[:,i])=1 for all i because of the softmax
            # Here we do the masking post-softmax, because the softmax is on slot dim if we set 0 or -inf
            # for the pixel column prior to softmax we will have either a uniform distrib or a nan output
            # so we set 0 to the pixel column post softmax so that the pixel does not influence the Value computation
            attn = dots * mask

        mean_attn = attn / (attn.sum(dim=self.weighted_mean_dim, keepdim=True) + self.eps)

        updates = torch.einsum('bjd,bij->bid', v, mean_attn)

        if self.return_attn:
            return updates, attn
        return updates

class WeightedTransformer(nn.Module):
    def __init__(self, depth, dim, softmax_dim=1, weighted_mean_dim=2, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for d in range(1, depth+1):
            attention = WeightedAttention(dim, eps=1e-6, softmax_dim=softmax_dim, weighted_mean_dim=weighted_mean_dim)
            ff = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attention),
                PreNorm(dim, ff)
            ]))
    def forward(self, x, context=None, mask=None):
        for attn, ff in self.layers:
            x = attn(x, context=context, mask=mask) + x
            x = ff(x) + x
        return x


class GatedResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.fn = fn
    def forward(self, *args):
        inputs = args[0]
        b, _, d = inputs.shape

        out = self.fn(*args)

        attn = None
        if isinstance(out, tuple):
            updates, attn = out
        else:
            updates = out

        updated_inputs = self.gru(
            updates.reshape(-1, d),
            inputs.reshape(-1, d)
        )

        out = updated_inputs.reshape(b, -1, d)

        if attn is not None:
            return out, attn
        return out

class SlotAttentionExperimental(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters

        self.norm_inputs = nn.LayerNorm(dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.slots_to_inputs_attn = GatedResidual(dim, WeightedAttention(dim, eps=eps))
        self.slots_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

        self.inputs_to_slots_attn = GatedResidual(dim, WeightedAttention(dim, eps=eps,
                                                                         softmax_dim =2, weighted_mean_dim=1))
        self.inputs_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_inputs(inputs)

        for _ in range(self.iters):
            slots = self.slots_to_inputs_attn(slots, inputs)
            slots = self.slots_ff(slots)

            inputs = self.inputs_to_slots_attn(inputs, slots)
            inputs = self.inputs_ff(inputs)

        return slots, inputs

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, mlp_ratio=4., return_attn=False):

        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.return_attn = return_attn

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.slots_to_inputs_attn = GatedResidual(dim, WeightedAttention(dim, eps=eps, return_attn=return_attn))
        self.ff = PreNorm(dim, FeedForward(dim, int(dim * mlp_ratio)))

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        for _ in range(self.iters):
            slots, attn = self.slots_to_inputs_attn(slots, inputs)
            slots = self.ff(slots) + slots

        if self.return_attn:
            return slots, attn
        return slots

@lru_cache(maxsize=None)
def fourier_encode(H, W, device, dtype, max_freq, num_bands=4, base=2):
    # maximum frequency, hyperparameter depending on how fine the data is
    # number of freq bands, with original value (2 * K + 1)

    axis = (H,W)
    axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
    pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1).unsqueeze(-1)
    orig_pos = pos

    scales = torch.logspace(1., log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(pos.shape) - 1)), Ellipsis)]

    pos = pos * scales * pi
    pos = torch.cat([pos.sin(), pos.cos()], dim=-1)
    pos = torch.cat((pos, orig_pos), dim=-1)
    return pos # pos_dim = 2 * ((num_freq_bands * 2) + 1)

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