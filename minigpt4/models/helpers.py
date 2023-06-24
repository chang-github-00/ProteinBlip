"""
Taken from https://github.com/lucidrains/flamingo-pytorch
"""

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

def LinearMap(in_dim, out_dim):
    return nn.Linear(in_dim, out_dim)


def default(val, d):
    return val if exists(val) else d

class PerceiverAttention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        """
        Args:
            x (torch.Tensor): features
                shape  (batch_size, num_query, query_dim)
            context (torch.Tensor): latent features
                shape ### to-be-added
        """
        h = self.heads

        q = self.to_q(x) # (b, L, D)
        context = default(context, x)    
        k, v = self.to_kv(context).chunk(2, dim = -1) # (b, M, D)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale # (b, L, M)

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)
    
    
class PerceiverAdapter(nn.Module):
    def __init__(
        self,
        dim, # query_dim
        context_dim=None,
        depth=2,
        dim_head=64,
        heads=8,
        num_latents=64,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim)) # latent vectors, shape (n, D)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(query_dim=query_dim, dim_head=dim_head, heads = heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input features
                shape (b, L, D)
        Returns:
            shape (b, n, D) where n is self.num_latents
        """
        b, L, D = x.shape[:3]

        # blocks
        latents = repeat(self.latents, "n d -> b n d", b=b)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)