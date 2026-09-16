"""Accuracy, coverage, cost and Pareto analysis.

Accuracy uses the same protocol as Beta 0.1 (and the eSASRec paper's academic
benchmark): leave-one-out, ranked against the FULL catalogue, items the user has
already seen are masked, ties broken pessimistically.
"""

import time

import torch

AUTOCAST = dict(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def score_last(model, ctx):
    """Scores for every item, from the most recent position of each context."""
    with torch.autocast(**AUTOCAST):
        h = model.encode(ctx)[:, -1, :]
        return (h @ model.item_emb.weight.t()).float()


@torch.no_grad()
def evaluate(model, contexts, targets, seen, k=10, batch=512):
    """contexts (U, L), targets (U,), seen (U, V+1) bool - all on the GPU."""
    was_training = model.training
    model.eval()
    n_items = seen.shape[1] - 1
    hits = ndcg = mrr = 0.0
    recommended = torch.zeros(n_items + 1, dtype=torch.bool, device=contexts.device)

    for s in range(0, len(contexts), batch):
        scores = score_last(model, contexts[s:s + batch])
        if not torch.isfinite(scores).all():
            raise RuntimeError("non-finite scores during evaluation - the metric "
                               "would be silently inflated")
        scores[:, 0] = float("-inf")
        scores[seen[s:s + batch]] = float("-inf")

        true = scores.gather(1, targets[s:s + batch, None])
        rank = (scores > true).sum(1) + (scores == true).sum(1) - 1   # 0 = top
        top = rank < k
        hits += top.sum().item()
        ndcg += (1.0 / torch.log2(rank[top].float() + 2.0)).sum().item()
        mrr += (1.0 / (rank.float() + 1.0)).sum().item()
        recommended[scores.topk(k, dim=1).indices.flatten()] = True

    if was_training:
        model.train()
    n = len(contexts)
    return {"recall": hits / n, "ndcg": ndcg / n, "mrr": mrr / n,
            "coverage": recommended[1:].sum().item() / n_items}


@torch.no_grad()
def measure_speed(model, contexts, n_queries=100, repeats=7, batch=512, k=10):
    """Latency = one request on its own. Throughput = users/second in batches.

    Single-request latency on a laptop swings with CPU and GPU power states: in one
    full run, two models with an identical inference path measured 1.6 ms and 4.7 ms.
    So latency is taken as the MEDIAN of several repeated passes, which ignores a
    lucky or unlucky pass instead of averaging it in.
    """
    was_training = model.training
    model.eval()
    for i in range(30):                                   # warm up kernels
        score_last(model, contexts[i:i + 1]).topk(k)
    torch.cuda.synchronize()

    passes = []
    for r in range(repeats):
        t0 = time.perf_counter()
        for i in range(n_queries):
            score_last(model, contexts[i:i + 1]).topk(k)
        torch.cuda.synchronize()
        passes.append((time.perf_counter() - t0) / n_queries * 1000)
    latency_ms = sorted(passes)[len(passes) // 2]

    t0 = time.perf_counter()
    for s in range(0, len(contexts), batch):
        score_last(model, contexts[s:s + batch]).topk(k)
    torch.cuda.synchronize()
    throughput = len(contexts) / (time.perf_counter() - t0)

    if was_training:
        model.train()
    return latency_ms, throughput


def pareto_front(points, maximize):
    """Indices of non-dominated points.

    A point is dominated if another point is at least as good on every metric
    and strictly better on at least one.
    """
    front = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            at_least = all((qv >= pv) if mx else (qv <= pv)
                           for qv, pv, mx in zip(q, p, maximize))
            strictly = any((qv > pv) if mx else (qv < pv)
                           for qv, pv, mx in zip(q, p, maximize))
            if at_least and strictly:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front
