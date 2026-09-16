"""The loss slot: four ways to measure how wrong a prediction is.

    bce             - SASRec's original: one random negative per position
    gbce            - gSASRec (RecSys 2023): BCE with k negatives and a
                      calibrated positive term that corrects overconfidence
    sampled_softmax - eSASRec's choice: softmax over the true item plus k
                      sampled negatives
    full_softmax    - softmax over the entire catalogue

Negatives for gbce and sampled_softmax are drawn PER USER SEQUENCE (k items for
each user in the batch, shared across that user's positions). An earlier version
shared a single set of k negatives across the whole batch; with ~30,000 positions
and only ~24 updates per epoch, most items then almost never received a negative
signal and both losses barely learned. Per-user negatives give k x batch draws
per update for a few tens of MB.
"""

import torch
import torch.nn.functional as F

LOSSES = ("bce", "gbce", "sampled_softmax", "full_softmax")
LOSS_NAMES = {"bce": "BCE", "gbce": "gBCE", "sampled_softmax": "Sampled softmax",
              "full_softmax": "Full softmax"}


def gbce_beta(n_neg, num_items, t):
    """gSASRec's calibration exponent. alpha is the negative sampling rate."""
    alpha = n_neg / (num_items - 1)
    return alpha * (t * (1 - 1 / alpha) + 1 / alpha)


def compute_loss(name, H, tgt, mask, item_emb, num_items, n_neg=256, gbce_t=0.75):
    """H: (B, L, dim) hidden states; tgt: (B, L) next-item ids (0 = padding);
    mask: (B, L) bool, True where there is a real target."""
    E = item_emb.weight
    h = H[mask]                      # (N, dim)
    pos = tgt[mask]                  # (N,)

    if name == "full_softmax":
        # Score every real item (column 0 is padding, so it is excluded) and
        # shift targets down by one to match.
        logits = (h @ E[1:].t()).float()
        return F.cross_entropy(logits, pos - 1)

    pos_score = (h * E[pos]).sum(-1).float()

    if name == "bce":
        neg = torch.randint(1, num_items + 1, pos.shape, device=pos.device)
        neg_score = (h * E[neg]).sum(-1).float()
        real_negative = (neg != pos).float()   # a sampled "negative" can be the target
        # -log sigmoid(s) = softplus(-s);  -log(1 - sigmoid(s)) = softplus(s)
        return (F.softplus(-pos_score) + real_negative * F.softplus(neg_score)).mean()

    # k negatives per user: (B, k). Scores for every position of that user: (B, L, k).
    B = H.shape[0]
    neg = torch.randint(1, num_items + 1, (B, n_neg), device=H.device)
    neg_scores = torch.bmm(H, E[neg].transpose(1, 2))[mask].float()          # (N, k)
    accidental_hit = (neg[:, None, :] == tgt[:, :, None])[mask]               # (N, k)

    if name == "sampled_softmax":
        neg_scores = neg_scores.masked_fill(accidental_hit, float("-inf"))
        logits = torch.cat([pos_score[:, None], neg_scores], dim=1)
        target = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        return F.cross_entropy(logits, target)

    if name == "gbce":
        beta = gbce_beta(n_neg, num_items, gbce_t)
        neg_term = F.softplus(neg_scores).masked_fill(accidental_hit, 0.0).sum(1)
        return (beta * F.softplus(-pos_score) + neg_term).mean()

    raise ValueError(f"unknown loss {name!r}; expected one of {LOSSES}")
