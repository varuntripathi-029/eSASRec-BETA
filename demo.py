"""Interactive demo: history in -> Top-K recommendations out.

Shows the central point of sequential recommendation: change the history,
and the predictions change.

Usage:
    python demo.py                      # runs the scripted demo
    python demo.py --user 42            # a specific MovieLens user
    python demo.py --interactive        # type your own movie ids
"""

import argparse

import torch

import data as data_mod
from model import build_model


def load(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    kwargs = {}
    if cfg["model"] == "transformer":
        kwargs = dict(n_layers=cfg["n_layers"], n_heads=cfg["n_heads"])
    model = build_model(cfg["model"], ckpt["num_items"], cfg["max_len"],
                        dim=cfg["dim"], **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def recommend(model, history, max_len, device, k=5, exclude_seen=True):
    ctx = history[-max_len:]
    padded = [0] * (max_len - len(ctx)) + ctx
    x = torch.tensor([padded], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[0]
    logits[0] = -float("inf")
    if exclude_seen:
        for item in set(history):
            logits[item] = -float("inf")
    scores, items = torch.topk(logits, k)
    return items.tolist(), scores.tolist()


def name(titles, item):
    return titles.get(item, f"item {item}")


def show(model, titles, history, max_len, device, k=5, label="Listening history"):
    print(f"\n{label}:")
    for i, item in enumerate(history[-10:], 1):
        print(f"  {i}. {name(titles, item)}")
    print("\nPredicting next item ...\n")
    items, scores = recommend(model, history, max_len, device, k)
    print(f"Top {k} recommendations:")
    for rank, (item, score) in enumerate(zip(items, scores), 1):
        print(f"  {rank}. {name(titles, item):<45} Score: {score:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint_transformer.pt")
    ap.add_argument("--user", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load(args.checkpoint, device)
    max_len = cfg["max_len"]

    sequences, titles, _ = data_mod.load_sequences()

    print("=" * 60)
    print("Sequential Recommendation - Beta 0.1")
    print("=" * 60)
    print(f"Model: {cfg['model']}   device: {device}")

    if args.interactive:
        print("\nEnter item ids separated by spaces (blank line to quit).")
        while True:
            raw = input("\nhistory> ").strip()
            if not raw:
                break
            try:
                history = [int(t) for t in raw.split()]
            except ValueError:
                print("Please enter integers only.")
                continue
            show(model, titles, history, max_len, device, args.k, "Your history")
        return

    # Scripted demo: two different users -> two different recommendation sets.
    idx_a = args.user if args.user is not None else 0
    history_a = sequences[idx_a][:-2][-8:]
    show(model, titles, history_a, max_len, device, args.k,
         f"User {idx_a} history")

    print("\n" + "-" * 60)
    print("Now change the history entirely:")
    print("-" * 60)

    idx_b = (idx_a + 1234) % len(sequences)
    history_b = sequences[idx_b][:-2][-8:]
    show(model, titles, history_b, max_len, device, args.k,
         f"User {idx_b} history")

    print("\n" + "=" * 60)
    print("Different interaction sequences produce different predictions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
