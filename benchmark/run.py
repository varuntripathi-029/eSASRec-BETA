"""eSASRec-style component benchmark: 3 layer designs x 4 losses.

Every combination gets the same training-time budget on the same GPU, then is
scored with the paper's metrics (NDCG@10, Recall@10, Coverage@10, Pareto
frontiers) plus the cost metrics the paper does not report (peak GPU memory,
training time, latency, throughput).

    python -m benchmark.run                          # full benchmark
    python -m benchmark.run --report-only            # re-print the last results
    python -m benchmark.run --only preln:full_softmax --budget 8   # quick trial

Run from the project root so the dataset in data/ is found.
"""

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

import data as data_mod
from .evaluate import AUTOCAST, evaluate, measure_speed, pareto_front
from .losses import LOSSES, LOSS_NAMES, compute_loss, gbce_beta
from .model import LAYER_NAMES, LAYER_TYPES, SASRec

RESULTS_DIR = "results"
RESULTS_JSON = os.path.join(RESULTS_DIR, "benchmark_results.json")
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TIME_TARGET_S = 600
VRAM_FLOOR_MB, VRAM_CAP_MB = 2048, 4096


# --------------------------- helpers ---------------------------

def rule(char="-", width=100):
    print(char * width)


def section(title):
    print()
    rule("=")
    print(f" {title}")
    rule("=")


def table(headers, rows, align=None):
    align = align or ["<"] * len(headers)
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:{a}{w}}}" for a, w in zip(align, widths))
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print(fmt.format(*r))


def gpu_stats():
    """Temperature / power / memory from nvidia-smi. Fields a laptop hides come back None."""
    fields = ["temperature.gpu", "power.draw", "power.limit", "memory.used"]
    keys = ["temp_c", "power_w", "power_limit_w", "smi_mem_mb"]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split(",")
    except Exception:
        return dict.fromkeys(keys)
    stats = {}
    for k, v in zip(keys, out):
        try:
            stats[k] = float(v.strip())
        except ValueError:
            stats[k] = None
    return stats


def fmt(v, spec, missing="n/a"):
    return missing if v is None else format(v, spec)


def label(r):
    return f"{LAYER_NAMES[r['layer']]} + {LOSS_NAMES[r['loss']]}"


# --------------------------- data ---------------------------

def build_tensors(max_len, device):
    seqs, _titles, n_items = data_mod.load_sequences()
    train, valid, test = data_mod.split(seqs)
    n_users, L = len(seqs), max_len

    tr_in = np.zeros((n_users, L), np.int64)
    tr_tgt = np.zeros((n_users, L), np.int64)
    val_ctx = np.zeros((n_users, L), np.int64)
    test_ctx = np.zeros((n_users, L), np.int64)
    seen_val = np.zeros((n_users, n_items + 1), bool)
    seen_test = np.zeros((n_users, n_items + 1), bool)
    n_targets = 0

    for u, s in enumerate(train):
        # Shifted sequence: position t is trained to predict item t+1.
        window = s[-(L + 1):]
        x, y = window[:-1], window[1:]
        tr_in[u, L - len(x):] = x
        tr_tgt[u, L - len(y):] = y
        n_targets += len(y)

        v = s[-L:]                       # validation: predict valid[u]
        val_ctx[u, L - len(v):] = v
        t = (s + [valid[u]])[-L:]        # test: predict test[u], may see valid[u]
        test_ctx[u, L - len(t):] = t

        seen_val[u, s] = True
        seen_test[u, s] = True
        seen_test[u, valid[u]] = True

    gpu = lambda a: torch.from_numpy(a).to(device)   # noqa: E731
    return dict(
        n_users=n_users, n_items=n_items, n_ratings=sum(len(x) for x in seqs),
        n_train_targets=n_targets,
        train_in=gpu(tr_in), train_tgt=gpu(tr_tgt),
        val_ctx=gpu(val_ctx), val_tgt=gpu(np.array(valid, dtype=np.int64)),
        test_ctx=gpu(test_ctx), test_tgt=gpu(np.array(test, dtype=np.int64)),
        seen_val=gpu(seen_val), seen_test=gpu(seen_test),
    )


# --------------------------- training ---------------------------

def make_model(layer, T, args, device):
    return SASRec(T["n_items"], layer, dim=args.dim, max_len=args.max_len,
                  n_layers=args.layers, n_heads=args.heads, dropout=args.dropout).to(device)


def train_step(model, opt, loss_name, seq, tgt, T, args):
    mask = tgt != 0
    with torch.autocast(**AUTOCAST):
        H = model.encode(seq)
        loss = compute_loss(loss_name, H, tgt, mask, model.item_emb, T["n_items"],
                            n_neg=args.neg, gbce_t=args.gbce_t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss


def warm_up(T, args, device):
    """Load CUDA kernels for every shape before timing, so run #1 isn't penalised."""
    idx = torch.arange(min(args.batch, T["n_users"]), device=device)
    for layer in ("postln", "ligr"):
        model = make_model(layer, T, args, device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for loss_name in LOSSES:
            train_step(model, opt, loss_name, T["train_in"][idx], T["train_tgt"][idx], T, args)
        evaluate(model, T["val_ctx"][:512], T["val_tgt"][:512], T["seen_val"][:512])
        del model, opt
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def train_one(layer, loss_name, T, args, device, run_no, n_runs):
    torch.manual_seed(args.seed)
    model = make_model(layer, T, args, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gpu_before = gpu_stats()

    print(f"\n[{run_no:2d}/{n_runs}] {LAYER_NAMES[layer]} + {LOSS_NAMES[loss_name]}"
          f"   ({n_params:,} parameters, budget {args.budget:.0f} s)")

    curve, best_ndcg, best_state, best_epoch = [], -1.0, None, 0
    train_seconds, epoch, max_smi = 0.0, 0, gpu_before.get("smi_mem_mb") or 0.0
    n_users = T["n_users"]
    model.train()

    while train_seconds < args.budget:
        epoch += 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        perm = torch.randperm(n_users, device=device)
        loss_sum, steps = torch.zeros((), device=device), 0
        for s in range(0, n_users, args.batch):
            idx = perm[s:s + args.batch]
            loss = train_step(model, opt, loss_name, T["train_in"][idx], T["train_tgt"][idx], T, args)
            loss_sum += loss.detach()        # no .item() per step: that forces a GPU sync
            steps += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        train_seconds += dt
        mean_loss = (loss_sum / steps).item()

        last = train_seconds >= args.budget
        if epoch % args.eval_every == 0 or last:
            v = evaluate(model, T["val_ctx"], T["val_tgt"], T["seen_val"])
            curve.append(dict(epoch=epoch, train_seconds=round(train_seconds, 2),
                              loss=mean_loss, val_ndcg=v["ndcg"], val_recall=v["recall"]))
            improved = v["ndcg"] > best_ndcg
            if improved:
                best_ndcg, best_epoch = v["ndcg"], epoch
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            smi = gpu_stats().get("smi_mem_mb")
            max_smi = max(max_smi, smi or 0.0)
            print(f"   epoch {epoch:3d} | {train_seconds:5.1f} s | loss {mean_loss:8.4f} | "
                  f"val NDCG@10 {v['ndcg']:.4f}  Recall@10 {v['recall']:.4f}"
                  f"{'  *best' if improved else ''}")

    peak_vram = torch.cuda.max_memory_allocated() / 2**20
    model.load_state_dict(best_state)
    test = evaluate(model, T["test_ctx"], T["test_tgt"], T["seen_test"])
    latency_ms, throughput = measure_speed(model, T["test_ctx"])
    gpu_after = gpu_stats()

    row = dict(
        status="ok", layer=layer, loss=loss_name, params=n_params,
        recall=test["recall"], ndcg=test["ndcg"], mrr=test["mrr"], coverage=test["coverage"],
        best_val_ndcg=best_ndcg, epochs=epoch, best_epoch=best_epoch,
        train_seconds=train_seconds, sec_per_epoch=train_seconds / epoch,
        peak_vram_mb=peak_vram, peak_smi_mb=max_smi,
        latency_ms=latency_ms, throughput=throughput,
        temp_start_c=gpu_before.get("temp_c"), temp_end_c=gpu_after.get("temp_c"),
        power_w=gpu_after.get("power_w"), power_limit_w=gpu_after.get("power_limit_w"),
    )
    print(f"   TEST  NDCG@10 {row['ndcg']:.4f}  Recall@10 {row['recall']:.4f}  "
          f"Coverage@10 {row['coverage']:.4f} | peak VRAM {peak_vram:,.0f} MB | "
          f"{row['sec_per_epoch']:.2f} s/epoch | latency {latency_ms:.2f} ms")
    del model, opt, best_state
    torch.cuda.empty_cache()
    return row, curve


# --------------------------- analysis + report ---------------------------

def compute_fronts(ok):
    pts = lambda keys: [tuple(r[k] for k in keys) for r in ok]   # noqa: E731
    return {
        "ndcg_coverage": pareto_front(pts(["ndcg", "coverage"]), (True, True)),
        "ndcg_vram": pareto_front(pts(["ndcg", "peak_vram_mb"]), (True, False)),
        "ndcg_latency": pareto_front(pts(["ndcg", "latency_ms"]), (True, False)),
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def print_report(result):
    cfg, env, rows = result["config"], result["env"], result["rows"]
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    fronts = compute_fronts(ok)
    on = lambda key, r: ok.index(r) in fronts[key]                  # noqa: E731

    section("BENCHMARK SETUP")
    print(f" GPU            : {env['gpu']}  ({env['gpu_total_mb']:,.0f} MB total)")
    print(f" CPU            : {env.get('cpu', 'not recorded')} | "
          f"{env.get('cpu_pinning', 'not pinned (older run)')}")
    print(f" Software       : Python {env['python']} | PyTorch {env['torch']} | CUDA {env['cuda']}")
    print(f" Dataset        : MovieLens-1M | {result['data']['n_ratings']:,} ratings | "
          f"{result['data']['n_users']:,} users | {result['data']['n_items']:,} items")
    print(f" Training data  : {result['data']['n_train_targets']:,} next-item targets "
          f"(shifted sequence, history up to {cfg['max_len']})")
    print(f" Model          : {cfg['layers']} layers | {cfg['heads']} heads | dim {cfg['dim']} | "
          f"dropout {cfg['dropout']} | bf16 mixed precision")
    print(f" Training       : Adam lr {cfg['lr']} | batch {cfg['batch']} users | "
          f"{cfg['budget']:.0f} s budget per combination | seed {cfg['seed']}")
    print(f" Negatives      : {cfg['neg']} uniform per user sequence | gBCE t={cfg['gbce_t']} "
          f"(beta={gbce_beta(cfg['neg'], result['data']['n_items'], cfg['gbce_t']):.3f})")
    print(f" Evaluation     : leave-one-out | full-catalogue ranking | seen items masked | "
          f"best validation epoch reported on test")
    print(f" Memory cap     : {VRAM_CAP_MB:,} MB enforced by PyTorch")

    section("1. ACCURACY - test set, ranked against all items   (* = on accuracy/coverage Pareto frontier)")
    ranked = sorted(ok, key=lambda r: -r["ndcg"])
    table(["#", "Layer", "Loss", "NDCG@10", "Recall@10", "MRR", "Coverage@10", "Epochs", "Best ep.", ""],
          [[i + 1, LAYER_NAMES[r["layer"]], LOSS_NAMES[r["loss"]], f"{r['ndcg']:.4f}",
            f"{r['recall']:.4f}", f"{r['mrr']:.4f}", f"{r['coverage']:.4f}",
            r["epochs"], r["best_epoch"], "*" if on("ndcg_coverage", r) else ""]
           for i, r in enumerate(ranked)],
          ["<", "<", "<", ">", ">", ">", ">", ">", ">", "<"])

    section("2. COST - memory, time and speed   (* = on accuracy/memory Pareto frontier)")
    table(["Layer", "Loss", "Params", "Peak VRAM MB", "nvidia-smi MB", "Train s", "s/epoch",
           "Latency ms", "Users/s", "GPU temp C", "Power W", ""],
          [[LAYER_NAMES[r["layer"]], LOSS_NAMES[r["loss"]], f"{r['params']:,}",
            f"{r['peak_vram_mb']:,.0f}", fmt(r["peak_smi_mb"], ",.0f"),
            f"{r['train_seconds']:.1f}", f"{r['sec_per_epoch']:.2f}", f"{r['latency_ms']:.2f}",
            f"{r['throughput']:,.0f}",
            f"{fmt(r['temp_start_c'], '.0f')}->{fmt(r['temp_end_c'], '.0f')}",
            fmt(r["power_w"], ".0f"), "*" if on("ndcg_vram", r) else ""]
           for r in sorted(ok, key=lambda r: (LAYER_TYPES.index(r["layer"]), LOSSES.index(r["loss"])))],
          ["<", "<", ">", ">", ">", ">", ">", ">", ">", ">", ">", "<"])
    print(" Peak VRAM = PyTorch's own allocations. nvidia-smi also counts the CUDA context and")
    print(" other processes, so it is always higher.")

    section("3. SLOT ABLATION - each slot averaged over the other slot")
    print(" By LOSS (mean of the 3 layer designs)")
    table(["Loss", "NDCG@10", "Recall@10", "Coverage@10", "Peak VRAM MB", "s/epoch", "Latency ms"],
          [[LOSS_NAMES[s]] + [fmt(mean([r[k] for r in ok if r["loss"] == s]), f)
                              for k, f in (("ndcg", ".4f"), ("recall", ".4f"), ("coverage", ".4f"),
                                           ("peak_vram_mb", ",.0f"), ("sec_per_epoch", ".2f"),
                                           ("latency_ms", ".2f"))]
           for s in LOSSES],
          ["<", ">", ">", ">", ">", ">", ">"])
    print("\n By LAYER DESIGN (mean of the 4 losses)")
    table(["Layer", "NDCG@10", "Recall@10", "Coverage@10", "Peak VRAM MB", "s/epoch", "Latency ms"],
          [[LAYER_NAMES[l]] + [fmt(mean([r[k] for r in ok if r["layer"] == l]), f)
                               for k, f in (("ndcg", ".4f"), ("recall", ".4f"), ("coverage", ".4f"),
                                            ("peak_vram_mb", ",.0f"), ("sec_per_epoch", ".2f"),
                                            ("latency_ms", ".2f"))]
           for l in LAYER_TYPES],
          ["<", ">", ">", ">", ">", ">", ">"])

    section("4. PARETO FRONTIERS - combinations no other combination beats on both measures")
    for key, title in (("ndcg_coverage", "Accuracy (NDCG@10) vs Coverage@10   - the paper's view"),
                       ("ndcg_vram", "Accuracy (NDCG@10) vs peak GPU memory - ours"),
                       ("ndcg_latency", "Accuracy (NDCG@10) vs latency       - ours")):
        members = sorted((ok[i] for i in fronts[key]), key=lambda r: -r["ndcg"])
        print(f" {title}")
        for r in members:
            extra = {"ndcg_coverage": f"coverage {r['coverage']:.4f}",
                     "ndcg_vram": f"{r['peak_vram_mb']:,.0f} MB",
                     "ndcg_latency": f"{r['latency_ms']:.2f} ms"}[key]
            print(f"   - {label(r):<34} NDCG@10 {r['ndcg']:.4f}   {extra}")
        print()

    section("5. KEY FINDINGS - computed from the numbers above")
    if ok:
        best = max(ok, key=lambda r: r["ndcg"])
        cov = max(ok, key=lambda r: r["coverage"])
        light = min(ok, key=lambda r: r["peak_vram_mb"])
        fast = min(ok, key=lambda r: r["latency_ms"])
        print(f" - Most accurate          : {label(best)} - NDCG@10 {best['ndcg']:.4f}, "
              f"Recall@10 {best['recall']:.4f}")
        print(f" - Widest coverage        : {label(cov)} - Coverage@10 {cov['coverage']:.4f} "
              f"(the most accurate reaches {best['coverage']:.4f})")
        print(f" - Least memory           : {label(light)} - {light['peak_vram_mb']:,.0f} MB, "
              f"{light['peak_vram_mb'] / best['peak_vram_mb'] * 100:.0f}% of the most accurate's memory, "
              f"at {light['ndcg'] / best['ndcg'] * 100:.0f}% of its NDCG@10")
        print(f" - Fastest single request : {label(fast)} - {fast['latency_ms']:.2f} ms")

        by_loss = {s: mean([r["ndcg"] for r in ok if r["loss"] == s]) for s in LOSSES}
        by_layer = {l: mean([r["ndcg"] for r in ok if r["layer"] == l]) for l in LAYER_TYPES}
        top_loss = max((s for s in LOSSES if by_loss[s] is not None), key=lambda s: by_loss[s])
        top_layer = max((l for l in LAYER_TYPES if by_layer[l] is not None), key=lambda l: by_layer[l])
        print(f" - Best loss on average   : {LOSS_NAMES[top_loss]} (mean NDCG@10 {by_loss[top_loss]:.4f})")
        print(f" - Best layer on average  : {LAYER_NAMES[top_layer]} (mean NDCG@10 {by_layer[top_layer]:.4f})")

        vram = {s: mean([r["peak_vram_mb"] for r in ok if r["loss"] == s]) for s in LOSSES}
        if by_loss.get("full_softmax") and by_loss.get("sampled_softmax"):
            print(f" - Full vs sampled softmax: full uses {vram['full_softmax'] / vram['sampled_softmax']:.1f}x "
                  f"the memory; sampled reaches {by_loss['sampled_softmax'] / by_loss['full_softmax'] * 100:.0f}% "
                  f"of full's NDCG@10")
        if by_loss.get("bce") and by_loss.get("sampled_softmax"):
            print(f" - BCE vs sampled softmax : BCE reaches {by_loss['bce'] / by_loss['sampled_softmax'] * 100:.0f}% "
                  f"of sampled softmax's NDCG@10 (paper: sampled softmax beats BCE)")
        if by_layer.get("ligr") and by_layer.get("postln"):
            change = (by_layer["ligr"] / by_layer["postln"] - 1) * 100
            print(f" - LiGR vs Post-LN        : {change:+.1f}% mean NDCG@10")
        collapsed = [r for r in ok if r["coverage"] < 0.10]
        if collapsed:
            print(f" ! WARNING: {len(collapsed)} runs recommend under 10% of the catalogue "
                  f"({', '.join(label(r) for r in collapsed)}).")
            print("   That is a training failure (the model fell back to popular items), not a fair")
            print("   measurement of the component. Do not quote these runs as evidence it is worse.")
        under = [r for r in ok if r["best_epoch"] == r["epochs"]]
        if under:
            print(f" - Still improving at the end of the budget: {len(under)} of {len(ok)} runs "
                  f"({', '.join(label(r) for r in under)}) - more time would likely raise their scores")

    section("6. TARGET CHECKS")
    total = result["timing"]["total_seconds"]
    heaviest = max((r["peak_vram_mb"] for r in ok), default=0)
    over_cap = [r for r in ok if r["peak_vram_mb"] > VRAM_CAP_MB]
    verdict = lambda ok_: "PASS" if ok_ else "MISS"    # noqa: E731
    print(f" [{verdict(total <= TIME_TARGET_S)}] Total time {total / 60:.1f} min "
          f"(target <= {TIME_TARGET_S / 60:.0f} min: data {result['timing']['data_seconds']:.0f} s, "
          f"warm-up {result['timing']['warmup_seconds']:.0f} s, runs {result['timing']['runs_seconds']:.0f} s, "
          f"charts {result['timing']['plot_seconds']:.0f} s)")
    print(f" [{verdict(VRAM_FLOOR_MB <= heaviest <= VRAM_CAP_MB)}] Heaviest run uses {heaviest:,.0f} MB "
          f"(target 2,048-4,096 MB)")
    print(f" [{verdict(not over_cap)}] Every run within the 4,096 MB cap"
          + ("" if not over_cap else f" - over: {', '.join(label(r) for r in over_cap)}"))
    print(f" [{verdict(not failed)}] All {len(rows)} combinations completed"
          + ("" if not failed else f" - failed: {', '.join(label(r) + ' (' + r['status'] + ')' for r in failed)}"))
    lighter = [r for r in ok if r["peak_vram_mb"] < VRAM_FLOOR_MB]
    if lighter:
        print(f" [INFO] {len(lighter)} cheaper runs use under 2 GB by design (natural memory use) - "
              f"that saving is a result, not a failure")

    section("7. READ BEFORE QUOTING THESE NUMBERS")
    print(" - One seed per combination: no error bars. Differences of a few thousandths of NDCG may be noise.")
    print(" - Equal time budget, not equal epochs: slower combinations get fewer epochs. That is the")
    print("   intended consumer-hardware question, but it is not the paper's protocol.")
    print(" - LiGR and gBCE are implemented from the papers' descriptions, not from official code.")
    print("   gBCE collapses here with t=0.75 and t=0 and with 32 or 256 negatives, while sampled")
    print("   softmax works using the same negatives - so the formula and scoring are not the cause.")
    print("   The likely cause is summed-negative BCE dynamics under a short budget. Unresolved.")
    print(" - Latency = median of 7 passes of 100 single requests, with the process pinned to")
    print("   performance cores. Unpinned, the same model measured ~1.6 ms or ~4.5 ms depending on")
    print("   which core type Windows chose. Latency on another machine depends on its CPU too.")
    print(" - Not comparable to the eSASRec paper's numbers: different split, model size and training length.")

    section("8. FILES")
    for p in result.get("files", []):
        print(f"   {p}")
    print()


# --------------------------- main ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=32.0, help="training seconds per combination")
    ap.add_argument("--batch", type=int, default=256, help="users per batch")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--neg", type=int, default=256, help="negatives for gBCE / sampled softmax")
    ap.add_argument("--gbce-t", type=float, default=0.75)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default="", help="comma list like preln:full_softmax,ligr:bce")
    ap.add_argument("--report-only", action="store_true", help="re-print saved results, no training")
    ap.add_argument("--plots", action="store_true", help="with --report-only: redraw the PNG charts")
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args()


def main():
    # Print progress line by line even when output is piped to a log file (tee),
    # otherwise the log stays empty until the run ends.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    if args.report_only:
        if not os.path.exists(RESULTS_JSON):
            sys.exit(f"No saved results at {RESULTS_JSON}. Run: python -m benchmark.run")
        with open(RESULTS_JSON) as f:
            saved = json.load(f)
        if args.plots:
            from .plots import make_all
            ok = [r for r in saved["rows"] if r["status"] == "ok"]
            charts = make_all(saved["rows"], saved["curves"], compute_fronts(ok), PLOTS_DIR)
            saved["files"] = [p for p in saved.get("files", []) if not p.endswith(".png")] + charts
            print(f"Redrew {len(charts)} charts in {PLOTS_DIR}")
        print_report(saved)
        return

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not found. This benchmark measures GPU memory and speed, so it needs one.")
    if not os.path.isdir("data"):
        sys.exit("Run this from the project root (the folder containing data/): python -m benchmark.run")

    wall0 = time.perf_counter()
    from .hardware import cpu_name, pin_to_performance_cores
    cpu_pinning = pin_to_performance_cores()     # consistent timing on P-core/E-core CPUs
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_MB / total_mb))

    combos = [(l, s) for l in LAYER_TYPES for s in LOSSES]
    if args.only:
        wanted = {tuple(x.split(":")) for x in args.only.split(",")}
        combos = [c for c in combos if c in wanted]
        if not combos:
            sys.exit(f"--only matched nothing. Layers: {LAYER_TYPES}. Losses: {LOSSES}.")
    # Laptops heat up over a long run and slow down. Shuffling the order spreads that
    # effect across layers and losses instead of always penalising the last ones.
    random.Random(args.seed).shuffle(combos)

    print(f"eSASRec-style component benchmark | {len(combos)} combinations | "
          f"{args.budget:.0f} s each | {props.name}")
    print("Run order: " + ", ".join(f"{LAYER_NAMES[l]}+{LOSS_NAMES[s]}" for l, s in combos))

    t0 = time.perf_counter()
    T = build_tensors(args.max_len, device)
    data_s = time.perf_counter() - t0
    print(f"Data ready in {data_s:.1f} s")

    t0 = time.perf_counter()
    warm_up(T, args, device)
    warm_s = time.perf_counter() - t0
    print(f"Warm-up done in {warm_s:.1f} s")

    rows, curves = [], {}
    t0 = time.perf_counter()
    for n, (layer, loss_name) in enumerate(combos, 1):
        try:
            row, curve = train_one(layer, loss_name, T, args, device, n, len(combos))
            curves[f"{layer}|{loss_name}"] = curve
        except torch.cuda.OutOfMemoryError:
            print(f"   OUT OF MEMORY under the {VRAM_CAP_MB:,} MB cap - recorded and skipped")
            row = dict(status="out_of_memory", layer=layer, loss=loss_name)
            torch.cuda.empty_cache()
        rows.append(row)
    runs_s = time.perf_counter() - t0

    # A partial --only trial must never overwrite the full benchmark's results.
    out_dir = os.path.join(RESULTS_DIR, "trial") if args.only else RESULTS_DIR
    json_path = os.path.join(out_dir, "benchmark_results.json")
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    result = dict(
        config=vars(args),
        env=dict(gpu=props.name, gpu_total_mb=total_mb, python=platform.python_version(),
                 torch=torch.__version__, cuda=torch.version.cuda,
                 cpu=cpu_name(), cpu_pinning=cpu_pinning),
        data={k: T[k] for k in ("n_users", "n_items", "n_ratings", "n_train_targets")},
        rows=rows, curves=curves,
        timing=dict(data_seconds=data_s, warmup_seconds=warm_s, runs_seconds=runs_s,
                    plot_seconds=0.0, total_seconds=time.perf_counter() - wall0),
        files=[json_path, csv_path],
    )

    def save():
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    # Save the measurements BEFORE drawing charts, so a charting error can never
    # throw away minutes of training.
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    save()

    t0 = time.perf_counter()
    if not args.no_plots:
        try:
            from .plots import make_all
            ok = [r for r in rows if r["status"] == "ok"]
            result["files"] += make_all(rows, curves, compute_fronts(ok), plots_dir)
        except Exception as exc:        # results are already on disk
            print(f"\nCharts failed ({exc.__class__.__name__}: {exc}). Results are saved; "
                  f"retry charts with: python -m benchmark.run --report-only --plots")
    result["timing"]["plot_seconds"] = time.perf_counter() - t0
    result["timing"]["total_seconds"] = time.perf_counter() - wall0
    save()

    print_report(result)


if __name__ == "__main__":
    main()
