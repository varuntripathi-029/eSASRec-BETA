"""Train a demo model and evaluate with full-catalogue ranking.

Note on evaluation: we rank the held-out item against the ENTIRE catalogue,
not against a sample of negatives. Krichene & Rendle (KDD 2020) showed sampled
metrics do not reliably preserve which model is better, so we avoid them from
day one. On ML-1M (~3.4k items) full ranking is cheap.

Usage:
    python train.py --model transformer --epochs 5
"""

import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn

import data as data_mod
from model import build_model


def evaluate(model, contexts, targets, seen, device, k=10, batch_size=256):
    """Recall@k and NDCG@k with previously-seen items masked out."""
    model.eval()
    hits, ndcg, n = 0.0, 0.0, 0
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            ctx = torch.from_numpy(contexts[start:start + batch_size]).to(device)
            tgt = torch.from_numpy(targets[start:start + batch_size]).to(device)
            logits = model(ctx)

            # Guard against a silent failure mode: if the model emits NaN or a
            # constant row, every item ties with the true item, rank becomes 0
            # and the batch scores a perfect hit. That inflates the metric
            # instead of erroring, so check explicitly rather than trusting it.
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "Non-finite logits during evaluation - the metric would be "
                    "silently inflated. Check padding/masking in the model."
                )

            logits[:, 0] = -float("inf")  # never recommend padding

            # Mask items the user already interacted with.
            for row, idx in enumerate(range(start, min(start + batch_size, len(contexts)))):
                already = seen[idx]
                if already:
                    logits[row, already] = -float("inf")

            # Rank of the true item = how many items score strictly higher,
            # plus every item it ties with. Ties are broken pessimistically on
            # purpose: an optimistic tie-break would let a degenerate model
            # that scores everything equally record a perfect hit.
            true_score = logits.gather(1, tgt.unsqueeze(1))
            rank = (logits > true_score).sum(1) + (logits == true_score).sum(1) - 1

            in_topk = rank < k
            hits += in_topk.sum().item()
            ndcg += (1.0 / torch.log2(rank[in_topk].float() + 2.0)).sum().item()
            n += len(tgt)
    return hits / n, ndcg / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="transformer", choices=["meanpool", "transformer"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None, help="checkpoint path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    print("Loading MovieLens-1M ...")
    sequences, titles, num_items = data_mod.load_sequences()
    train_seqs, valid_items, test_items = data_mod.split(sequences)
    print(f"users={len(sequences)}  items={num_items}")

    ctx_train, y_train = data_mod.make_training_pairs(train_seqs, args.max_len)
    print(f"training pairs: {len(ctx_train):,}")

    # Validation predicts the 2nd-to-last item from the training history.
    ctx_valid = data_mod.make_eval_contexts(train_seqs, args.max_len)
    # Test predicts the last item, and may legitimately see the validation item.
    ctx_test = data_mod.make_eval_contexts(train_seqs, args.max_len, extra=valid_items)

    seen_valid = [set(s) for s in train_seqs]
    seen_test = [set(s) | {v} for s, v in zip(train_seqs, valid_items)]
    seen_valid = [sorted(s) for s in seen_valid]
    seen_test = [sorted(s) for s in seen_test]

    kwargs = {}
    if args.model == "transformer":
        kwargs = dict(n_layers=args.n_layers, n_heads=args.n_heads)
    model = build_model(args.model, num_items, args.max_len, dim=args.dim, **kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}  parameters={n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    ctx_t = torch.from_numpy(ctx_train)
    y_t = torch.from_numpy(y_train)
    n = len(ctx_t)

    history = []
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total, nb = 0.0, 0
        t0 = time.time()
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb = ctx_t[idx].to(device)
            yb = y_t[idx].to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            nb += 1
        epoch_time = time.time() - t0

        recall, ndcg = evaluate(model, ctx_valid, np.array(valid_items), seen_valid, device)
        print(f"epoch {epoch}/{args.epochs}  loss={total/nb:.4f}  "
              f"valid Recall@10={recall:.4f}  NDCG@10={ndcg:.4f}  ({epoch_time:.1f}s)")
        history.append(dict(epoch=epoch, loss=total / nb, recall=recall, ndcg=ndcg,
                            seconds=epoch_time))

    total_train = time.time() - train_start
    recall, ndcg = evaluate(model, ctx_test, np.array(test_items), seen_test, device)
    print(f"\nTEST  Recall@10={recall:.4f}  NDCG@10={ndcg:.4f}")

    # Efficiency numbers (guide section 15) - recorded from the start.
    #
    # Two DIFFERENT things are measured here, and conflating them is a common
    # way to report a latency that nobody actually experiences:
    #   * batched  - amortised cost per user at batch 256. This is throughput,
    #                the right number for "how long does an offline eval take".
    #   * batch-1  - the latency of one request on its own. This is what a user
    #                waiting for a recommendation actually feels, and it is
    #                roughly an order of magnitude slower per query.
    peak_mem = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0

    n_batched = 1000
    t0 = time.time()
    _ = evaluate(model, ctx_test[:n_batched], np.array(test_items[:n_batched]),
                 seen_test[:n_batched], device)
    ms_batched = (time.time() - t0) / n_batched * 1000

    model.eval()
    single = torch.from_numpy(ctx_test[:1]).to(device)
    with torch.no_grad():
        for _ in range(10):          # warm up CUDA kernels first
            model(single)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(200):
            model(torch.from_numpy(ctx_test[i:i + 1]).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms_batch1 = (time.time() - t0) / 200 * 1000

    stats = dict(model=args.model, params=n_params, train_seconds=total_train,
                 peak_gpu_mb=peak_mem,
                 ms_per_user_batched=ms_batched, ms_per_query_batch1=ms_batch1,
                 test_recall_at_10=recall, test_ndcg_at_10=ndcg, history=history,
                 config=vars(args))
    print(f"params={n_params:,}  train={total_train:.1f}s  peak GPU={peak_mem:.0f}MB")
    print(f"latency: {ms_batch1:.2f} ms/query (batch-1)  |  "
          f"{ms_batched:.3f} ms/user (batched, throughput)")

    out = args.out or f"checkpoint_{args.model}.pt"
    torch.save(dict(state_dict=model.state_dict(), num_items=num_items,
                    config=vars(args)), out)
    with open(f"results_{args.model}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"saved {out} and results_{args.model}.json")


if __name__ == "__main__":
    main()
