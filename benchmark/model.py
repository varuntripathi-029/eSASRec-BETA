"""SASRec-family model with a swappable layer design.

The layer type is one of the four component slots studied by eSASRec:

    postln - vanilla Transformer layer, normalisation AFTER each sub-layer
    preln  - normalisation BEFORE each sub-layer
    ligr   - pre-norm, and each attention / feed-forward output is multiplied by
             a learned gate: sigmoid(linear projection of the normalised input).

LiGR is implemented from the eSASRec paper's description ("each multi-head
attention and the feed-forward layer is gated with a linear projection and
sigmoid activation"). There is no official reference code here, so treat this
as our interpretation and check it against the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_TYPES = ("postln", "preln", "ligr")
LAYER_NAMES = {"postln": "Post-LN", "preln": "Pre-LN", "ligr": "LiGR"}


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, allowed):
        b, l, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        shape = (b, l, self.n_heads, d // self.n_heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        # allowed: bool (B, 1, L, L), True = may attend
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, l, d))


class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, layer_type, dim, n_heads, dropout):
        super().__init__()
        self.layer_type = layer_type
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.ffn = FeedForward(dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        if layer_type == "ligr":
            self.gate_attn = nn.Linear(dim, dim)
            self.gate_ffn = nn.Linear(dim, dim)

    def forward(self, x, allowed):
        if self.layer_type == "postln":
            x = self.norm1(x + self.drop(self.attn(x, allowed)))
            x = self.norm2(x + self.drop(self.ffn(x)))
        elif self.layer_type == "preln":
            x = x + self.drop(self.attn(self.norm1(x), allowed))
            x = x + self.drop(self.ffn(self.norm2(x)))
        else:  # ligr
            h = self.norm1(x)
            x = x + self.drop(torch.sigmoid(self.gate_attn(h)) * self.attn(h, allowed))
            h = self.norm2(x)
            x = x + self.drop(torch.sigmoid(self.gate_ffn(h)) * self.ffn(h))
        return x


class SASRec(nn.Module):
    def __init__(self, num_items, layer_type, dim=128, max_len=200, n_layers=2,
                 n_heads=2, dropout=0.2):
        super().__init__()
        if layer_type not in LAYER_TYPES:
            raise ValueError(f"layer_type must be one of {LAYER_TYPES}")
        self.num_items = num_items
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(layer_type, dim, n_heads, dropout) for _ in range(n_layers))
        # Pre-norm stacks need a final norm; post-norm already ends normalised.
        self.final_norm = nn.Identity() if layer_type == "postln" else nn.LayerNorm(dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, seq):
        """seq: (B, L) item ids, left-padded with 0  ->  (B, L, dim) hidden states."""
        b, l = seq.shape
        real = seq != 0
        keep = real.unsqueeze(-1)
        pos = torch.arange(l, device=seq.device)
        x = self.emb_drop(self.item_emb(seq) + self.pos_emb(pos)) * keep

        # May attend: causal AND the key is a real item. The diagonal is always
        # allowed so an all-padding query row still attends to something - a fully
        # masked row makes softmax return NaN, which is the bug Beta 0.1 hit.
        causal = torch.ones(l, l, dtype=torch.bool, device=seq.device).tril()
        allowed = (causal & real[:, None, None, :]) | torch.eye(
            l, dtype=torch.bool, device=seq.device)

        for block in self.blocks:
            x = block(x, allowed) * keep
        return self.final_norm(x)
