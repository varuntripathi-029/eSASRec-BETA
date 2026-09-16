"""Two demo models.

MeanPoolRec       - the simplest thing that works (guide section 5).
TinyTransformerRec- embeddings + positional + causal self-attention, scored by
                    dot product against the shared item embedding table
                    (guide section 6). This is deliberately SASRec-shaped so
                    that step 2 of the project is an extension, not a rewrite.

Both expose the same interface:
    forward(seq) -> logits over all items, shape (batch, num_items + 1)
"""

import torch
import torch.nn as nn


class MeanPoolRec(nn.Module):
    def __init__(self, num_items, dim=64, pad_idx=0):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=pad_idx)
        self.out = nn.Linear(dim, dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[pad_idx].zero_()

    def forward(self, seq):
        mask = (seq != 0).unsqueeze(-1).float()          # (B, L, 1)
        emb = self.item_emb(seq) * mask                  # zero out padding
        pooled = emb.sum(1) / mask.sum(1).clamp(min=1.0)  # mean over real items
        h = self.out(pooled)
        return h @ self.item_emb.weight.t()


class TinyTransformerRec(nn.Module):
    def __init__(self, num_items, dim=64, max_len=20, n_layers=2, n_heads=2,
                 dropout=0.2, pad_idx=0):
        super().__init__()
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[pad_idx].zero_()

    def forward(self, seq):
        b, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(b, length)

        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.norm(self.dropout(x))

        # Zero out padded positions, as the official SASRec implementation does.
        #
        # We deliberately do NOT pass src_key_padding_mask. Sequences are
        # left-padded, so an all-padding query row would attend to nothing, and
        # softmax over a fully-masked row yields NaN. With more than one layer
        # those NaNs then propagate to every position (a NaN key poisons the
        # attention scores before masking is applied), silently zeroing the
        # output for every short history. Masking the values instead keeps the
        # computation finite.
        keep = (seq != 0).unsqueeze(-1).float()
        x = x * keep

        # Causal mask: position t may only attend to <= t (as in SASRec).
        causal = torch.triu(
            torch.ones(length, length, device=seq.device, dtype=torch.bool), diagonal=1
        )

        x = self.encoder(x, mask=causal)
        x = x * keep

        h = x[:, -1, :]  # representation at the most recent position
        return h @ self.item_emb.weight.t()


def build_model(name, num_items, max_len, dim=64, **kwargs):
    if name == "meanpool":
        return MeanPoolRec(num_items, dim=dim)
    if name == "transformer":
        return TinyTransformerRec(num_items, dim=dim, max_len=max_len, **kwargs)
    raise ValueError(f"unknown model: {name}")
