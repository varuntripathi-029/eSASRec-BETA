"""Local web frontend for the sequential recommendation demo.

Deliberately uses only the Python standard library for the server, so there is
nothing extra to install and nothing to break on a teammate's machine.

Usage:
    python app.py                 # http://localhost:8000
    python app.py --port 8080
    python app.py --checkpoint checkpoint_meanpool.pt
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import data as data_mod
from model import build_model

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

STATE = {}  # model, cfg, titles, sequences, device


def dataset_stats(sequences, titles, popularity):
    """Summary statistics + distributions for the Dataset tab."""
    lengths = sorted(len(s) for s in sequences)
    n_ratings = sum(lengths)

    # Rating values and timestamps are dropped by load_sequences(), so read the
    # raw file once for the value distribution and the time span.
    rating_dist, first_ts, last_ts = {}, None, None
    path = os.path.join("data", "ml-1m", "ratings.dat")
    if os.path.exists(path):
        with open(path, encoding="latin-1") as f:
            for line in f:
                parts = line.split("::")
                if len(parts) < 4:
                    continue
                value = int(parts[2])
                ts = int(parts[3])
                rating_dist[value] = rating_dist.get(value, 0) + 1
                first_ts = ts if first_ts is None or ts < first_ts else first_ts
                last_ts = ts if last_ts is None or ts > last_ts else last_ts

    bins = [(20, 49), (50, 99), (100, 199), (200, 499), (500, 999), (1000, 10**9)]
    labels = ["20-49", "50-99", "100-199", "200-499", "500-999", "1000+"]
    hist = []
    for (lo, hi), label in zip(bins, labels):
        hist.append({"label": label,
                     "count": sum(1 for n in lengths if lo <= n <= hi)})

    mid = len(lengths) // 2
    top = sorted(popularity, key=lambda i: -popularity[i])[:200]

    return {
        "n_ratings": n_ratings,
        "n_users": len(sequences),
        "n_items": len(titles) or len(popularity),
        "density_pct": round(100 * n_ratings / (len(sequences) * max(len(popularity), 1)), 4),
        "seq_len": {
            "min": lengths[0], "max": lengths[-1],
            "median": lengths[mid],
            "mean": round(n_ratings / len(sequences), 1),
        },
        "rating_dist": [{"label": str(v), "count": rating_dist.get(v, 0)}
                        for v in sorted(rating_dist)],
        "length_hist": hist,
        "first_ts": first_ts, "last_ts": last_ts,
        "top_items": [{"id": i, "title": titles.get(i, f"item {i}"),
                       "count": popularity[i]} for i in top],
    }


def load_metrics():
    """Read results_*.json produced by train.py."""
    out = {}
    for name in ("transformer", "meanpool"):
        path = f"results_{name}.json"
        if os.path.exists(path):
            with open(path) as f:
                out[name] = json.load(f)
    return out


def load_everything(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    kwargs = {}
    if cfg["model"] == "transformer":
        kwargs = dict(n_layers=cfg["n_layers"], n_heads=cfg["n_heads"])
    model = build_model(cfg["model"], ckpt["num_items"], cfg["max_len"],
                        dim=cfg["dim"], **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sequences, titles, num_items = data_mod.load_sequences()

    # Popularity is used only to order search results sensibly.
    popularity = {}
    for seq in sequences:
        for item in seq:
            popularity[item] = popularity.get(item, 0) + 1

    STATE.update(model=model, cfg=cfg, titles=titles, sequences=sequences,
                 device=device, num_items=num_items, popularity=popularity,
                 checkpoint=os.path.basename(checkpoint))
    STATE["dataset"] = dataset_stats(sequences, titles, popularity)
    return STATE


def recommend(history, k=10):
    model, cfg, device = STATE["model"], STATE["cfg"], STATE["device"]
    max_len = cfg["max_len"]

    ctx = [i for i in history if 0 < i <= STATE["num_items"]][-max_len:]
    if not ctx:
        return [], 0.0
    padded = [0] * (max_len - len(ctx)) + ctx
    x = torch.tensor([padded], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logits[0] = -float("inf")
    for item in set(ctx):
        logits[item] = -float("inf")

    scores, items = torch.topk(logits, k)
    out = [
        {"id": int(i), "title": STATE["titles"].get(int(i), f"item {int(i)}"),
         "score": round(float(s), 3)}
        for i, s in zip(items.tolist(), scores.tolist())
    ]
    return out, round(elapsed_ms, 2)


def evaluate_user(user, k=10):
    """Run the model on one real dataset user under the test protocol.

    Mirrors train.py exactly: the model sees the training history plus the
    validation item, and must predict the held-out LAST item. Everything the
    user has already rated is masked, ties are broken pessimistically.
    """
    model, cfg, device, titles = STATE["model"], STATE["cfg"], STATE["device"], STATE["titles"]
    max_len = cfg["max_len"]
    seq = STATE["sequences"][user]
    history, target = seq[:-1], seq[-1]

    ctx = history[-max_len:]
    x = torch.tensor([[0] * (max_len - len(ctx)) + ctx], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not torch.isfinite(logits).all():
        raise RuntimeError(f"non-finite logits for user {user}")

    logits[0] = -float("inf")
    logits[sorted(set(history))] = -float("inf")

    true_score = logits[target]
    rank = int((logits > true_score).sum() + (logits == true_score).sum() - 1)

    scores, items = torch.topk(logits, k)
    name = lambda i: titles.get(int(i), f"item {int(i)}")
    return {
        "user": user,
        "history_len": len(history),
        "context_tail": [name(i) for i in ctx[-5:]],
        "target": {"id": int(target), "title": name(target)},
        "rank": rank + 1,                      # 1 = top of the list
        "hit": rank < k,
        "k": k,
        "top": [{"id": int(i), "title": name(i), "score": round(float(s), 3),
                 "is_target": int(i) == int(target)}
                for i, s in zip(items.tolist(), scores.tolist())],
        "inference_ms": round(elapsed_ms, 2),
    }


def search(query, limit=20):
    query = query.strip().lower()
    titles, popularity = STATE["titles"], STATE["popularity"]
    if not query:
        ranked = sorted(titles, key=lambda i: -popularity.get(i, 0))[:limit]
    else:
        hits = [i for i, t in titles.items() if query in t.lower()]
        ranked = sorted(hits, key=lambda i: -popularity.get(i, 0))[:limit]
    return [{"id": i, "title": titles[i], "count": popularity.get(i, 0)} for i in ranked]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console readable
        pass

    def _send(self, code, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            index = os.path.join(STATIC_DIR, "index.html")
            with open(index, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")

        if path == "/api/info":
            cfg = STATE["cfg"]
            return self._send(200, {
                "model": cfg["model"],
                "checkpoint": STATE["checkpoint"],
                "device": str(STATE["device"]),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "num_items": STATE["num_items"],
                "max_len": cfg["max_len"],
                "num_users": len(STATE["sequences"]),
            })

        if path == "/api/search":
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query).get("q", [""])[0]
            return self._send(200, {"results": search(q)})

        if path == "/api/metrics":
            return self._send(200, {"metrics": load_metrics()})

        if path == "/api/dataset":
            return self._send(200, STATE["dataset"])

        if path == "/api/dataset/users":
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            offset = max(0, int(qs.get("offset", ["0"])[0]))
            limit = max(1, min(int(qs.get("limit", ["25"])[0]), 100))
            titles = STATE["titles"]
            rows = []
            for idx in range(offset, min(offset + limit, len(STATE["sequences"]))):
                seq = STATE["sequences"][idx]
                rows.append({
                    "user": idx,
                    "length": len(seq),
                    "first": titles.get(seq[0], f"item {seq[0]}"),
                    "last": titles.get(seq[-1], f"item {seq[-1]}"),
                    "preview": [titles.get(i, f"item {i}") for i in seq[:6]],
                })
            return self._send(200, {"rows": rows, "total": len(STATE["sequences"]),
                                    "offset": offset, "limit": limit})

        if path == "/api/random_users":
            import random
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            n = max(1, min(int(qs.get("n", ["10"])[0]), 200))
            titles = STATE["titles"]
            # Every user qualifies: MovieLens-1M guarantees >= 20 ratings, and the
            # test protocol needs only a history plus one held-out item. Guard
            # anyway so a different dataset can't produce an unusable record.
            eligible = [i for i, s in enumerate(STATE["sequences"]) if len(s) >= 3]
            picks = random.sample(eligible, min(n, len(eligible)))
            rows = []
            for u in picks:
                seq = STATE["sequences"][u]
                rows.append({
                    "user": u,
                    "history_len": len(seq) - 1,
                    "recent": [titles.get(i, f"item {i}") for i in seq[-4:-1]],
                })
            return self._send(200, {"users": rows, "eligible": len(eligible)})

        if path == "/api/sample":
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            idx = int(qs.get("user", ["0"])[0]) % len(STATE["sequences"])
            hist = STATE["sequences"][idx][:-2][-8:]
            return self._send(200, {
                "user": idx,
                "history": [{"id": i, "title": STATE["titles"].get(i, f"item {i}")}
                            for i in hist],
            })

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/api/recommend", "/api/evaluate_user"):
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})

        if self.path == "/api/evaluate_user":
            try:
                user = int(payload["user"])
            except (KeyError, TypeError, ValueError):
                return self._send(400, {"error": "user must be an integer id"})
            if not 0 <= user < len(STATE["sequences"]):
                return self._send(400, {"error": f"user must be between 0 and {len(STATE['sequences']) - 1}"})
            k = max(1, min(int(payload.get("k", 10)), 50))
            return self._send(200, evaluate_user(user, k))

        history = [int(i) for i in payload.get("history", [])]
        k = max(1, min(int(payload.get("k", 10)), 50))
        results, ms = recommend(history, k)
        return self._send(200, {"recommendations": results, "inference_ms": ms,
                                "history_used": history[-STATE["cfg"]["max_len"]:]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint_transformer.pt")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"{args.checkpoint} not found. Train a model first:\n"
            f"    python train.py --model transformer --epochs 5"
        )

    print(f"Loading {args.checkpoint} ...")
    load_everything(args.checkpoint)
    print(f"Model: {STATE['cfg']['model']}   device: {STATE['device']}")

    # The first CUDA forward pass pays kernel-loading cost (~150 ms here), which
    # would otherwise show up as the latency of the user's first request. Warm
    # up now so the number the UI reports is the steady-state one.
    _, warm_ms = recommend([1, 2, 3], k=5)
    _, warm_ms = recommend([1, 2, 3], k=5)
    print(f"Warm-up done (steady-state inference ~{warm_ms} ms)")
    print(f"\n  ->  http://localhost:{args.port}\n")
    print("Press Ctrl+C to stop.")

    # allow_reuse_address is True by default, and on Windows that lets a second
    # server bind a port another process is already serving. Both then appear to
    # start fine while only the first receives requests - which looks exactly
    # like "my code changes did nothing". Fail loudly instead.
    class SingleBindServer(ThreadingHTTPServer):
        allow_reuse_address = False

    try:
        server = SingleBindServer(("0.0.0.0", args.port), Handler)
    except OSError:
        raise SystemExit(
            f"Port {args.port} is already in use - another server is running.\n"
            f"Stop it, or start this one with --port {args.port + 1}."
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
