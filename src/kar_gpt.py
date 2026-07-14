import torch
from torch import nn, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm

from einops import rearrange, reduce, einsum
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

# KAR = Kimi Attention Residuals (Kimi Team 2025)
# implements Full AttnRes (eq. 1-4): the fixed residual accumulation h_l = Σ v_i
# is replaced with softmax attention over the embedding + all preceding layer outputs

# ein notation

# b - batch
# n - sequence
# h - heads
# l - logits
# y - laYer outputs attended over (depth axis)
# d - feature dimension

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention residual op

class AttentionResidual(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        # the learned pseudo-query w_l - a plain vector, decoupled from the layer's forward computation
        # must be zero-initialized: attention starts uniform, so training begins from an
        # equal-weight average of past outputs (§5, prevents early training volatility)

        self.pseudo_query = nn.Parameter(torch.zeros(dim))

        # keys are RMSNorm'ed so layers with large-magnitude outputs
        # cannot dominate the attention weights (eq. 2)

        self.norm = RMSNorm(dim)

        # analysis hook: when store_attn is set, forward keeps the depth-attention
        # weights α_{i→l} averaged over batch and sequence (fig. 8 style plots)

        self.store_attn = False
        self.stored_attn = None

    def forward(
        self,
        values # Float['y b n d'] - token embedding v_0 + all preceding layer outputs v_i
    ):
        keys = self.norm(values)

        sim = einsum(self.pseudo_query, keys, 'd, y b n d -> y b n')

        attn = sim.softmax(dim = 0)

        if self.store_attn:
            self.stored_attn = reduce(attn.detach(), 'y b n -> y', 'mean')

        return einsum(attn, values, 'y b n, y b n d -> b n d')

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        self.heads = heads
        dim_inner = heads * dim_head

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_k = nn.Linear(dim, dim_inner, bias = False)
        self.to_v = nn.Linear(dim, dim_inner, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x
    ):

        x = self.norm(x)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        # relative positions

        q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

        # attention branch

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        out = self.merge_heads(out)

        return self.to_out(out)

# feedforward

def FeedForward(dim, expansion_factor = 4.):
    dim_hidden = int(dim * expansion_factor)

    return nn.Sequential(
        RMSNorm(dim),
        Linear(dim, dim_hidden),
        nn.GELU(),
        Linear(dim_hidden, dim)
    )

# classes

class KARGPT(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_expansion_factor = 4.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        # each self-attention and each MLP counts as its own layer (§2 notation),
        # so every sublayer gets its own pseudo-query / AttnRes op
        # (the very first op sees only v_0, where softmax over one element is 1 -
        # an identity with an untrained pseudo-query, kept for uniformity)

        layers = []
        for _ in range(depth):
            attn_res = AttentionResidual(dim)
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads)

            ff_res = AttentionResidual(dim)
            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

            layers.append(ModuleList([attn_res, attn, ff_res, ff]))

        self.layers = ModuleList(layers)

        # the output head aggregates over all layer outputs as well (fig. 1b)

        self.final_attn_res = AttentionResidual(dim)

        self.norm = RMSNorm(dim)
        self.to_logits = Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        ids,
        return_loss = False
    ):
        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        tokens = self.token_emb(ids)

        # v_0 is the token embedding (eq. 3)

        values = [tokens]

        for attn_res, attn, ff_res, ff in self.layers:

            # h_l = Σ α_{i→l} · v_i (eq. 4), then v_l = f_l(h_l) - the layer output
            # alone is stored, there is no additive residual anymore

            h = attn_res(stack(values))
            values.append(attn(h))

            h = ff_res(stack(values))
            values.append(ff(h))

        pooled = self.final_attn_res(stack(values))

        embed = self.norm(pooled)

        logits = self.to_logits(embed)

        if not return_loss:
            return logits

        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
