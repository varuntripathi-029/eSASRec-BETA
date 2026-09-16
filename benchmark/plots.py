"""PNG charts for the benchmark.

Colour encodes the layer design (3 hues, validated for colour-blind separation);
marker shape encodes the loss, so identity never depends on colour alone.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .losses import LOSSES, LOSS_NAMES  # noqa: E402
from .model import LAYER_TYPES, LAYER_NAMES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e3e0"
LAYER_COLOR = {"postln": "#2a78d6", "preln": "#eb6834", "ligr": "#1baf7a"}
LOSS_MARKER = {"bce": "o", "gbce": "s", "sampled_softmax": "^", "full_softmax": "D"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def _short(r):
    return f"{LAYER_NAMES[r['layer']]} + {LOSS_NAMES[r['loss']]}"


def _legend_handles():
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=LAYER_COLOR[l], marker="o", linestyle="", markersize=8,
                      label=LAYER_NAMES[l]) for l in LAYER_TYPES]
    handles.append(Line2D([], [], linestyle="", label=""))
    handles += [Line2D([], [], color=MUTED, marker=LOSS_MARKER[s], linestyle="", markersize=8,
                       label=LOSS_NAMES[s]) for s in LOSSES]
    return handles


def _pareto_scatter(rows, front_idx, x_key, x_label, title, path, x_maximize):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for r in rows:
        ax.scatter(r[x_key], r["ndcg"], s=90, color=LAYER_COLOR[r["layer"]],
                   marker=LOSS_MARKER[r["loss"]], edgecolor=SURFACE, linewidth=2, zorder=3)

    front = sorted((rows[i] for i in front_idx), key=lambda r: r[x_key])
    if len(front) > 1:
        ax.plot([r[x_key] for r in front], [r["ndcg"] for r in front],
                color=MUTED, linewidth=1.5, linestyle="--", zorder=2, label="Pareto frontier")
    # Direct labels on frontier points only. Neighbouring frontier points can sit
    # almost on top of each other, so alternate labels above and below the point.
    for i, r in enumerate(front):
        ax.annotate(_short(r), (r[x_key], r["ndcg"]), textcoords="offset points",
                    xytext=(8, 7) if i % 2 == 0 else (8, -14), fontsize=8.5, color=INK)

    ax.set_xlabel(x_label)
    ax.set_ylabel("NDCG@10 (test)")
    better = "right" if x_maximize else "left"
    ax.set_title(f"{title}\nFrontier = no other combination is better on both "
                 f"(up is better, {better} is better)", loc="left")
    handles = _legend_handles()
    if len(front) > 1:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color=MUTED, linestyle="--", label="Pareto frontier"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _grouped_bars(ax, rows, key, fmt):
    ok = {(r["layer"], r["loss"]): r for r in rows}
    width = 0.26
    for li, layer in enumerate(LAYER_TYPES):
        xs, ys = [], []
        for si, loss in enumerate(LOSSES):
            r = ok.get((layer, loss))
            if r is None:
                continue
            xs.append(si + (li - 1) * width)
            ys.append(r[key])
        bars = ax.bar(xs, ys, width=width - 0.02, color=LAYER_COLOR[layer],
                      label=LAYER_NAMES[layer], zorder=3)
        for b, y in zip(bars, ys):
            # Vertical labels: three bars per group are too narrow for horizontal numbers.
            ax.annotate(fmt(y), (b.get_x() + b.get_width() / 2, y), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom", rotation=90,
                        fontsize=7.5, color=MUTED)
    top = max(r[key] for r in rows)
    ax.set_ylim(0, top * 1.22)          # headroom for the vertical labels
    ax.set_xticks(range(len(LOSSES)))
    ax.set_xticklabels([LOSS_NAMES[s] for s in LOSSES])
    ax.grid(axis="x", visible=False)


def make_all(rows, curves, fronts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ok = [r for r in rows if r["status"] == "ok"]
    paths = []
    if not ok:
        return paths

    # 1-3: Pareto scatters (the paper's accuracy-coverage view, plus our cost views)
    specs = [
        ("coverage", "Coverage@10 (share of catalogue ever recommended)",
         "Accuracy vs coverage", "pareto_ndcg_vs_coverage.png", True, "ndcg_coverage"),
        ("peak_vram_mb", "Peak GPU memory during training (MB)",
         "Accuracy vs GPU memory", "pareto_ndcg_vs_vram.png", False, "ndcg_vram"),
        ("latency_ms", "Latency for one request (ms)",
         "Accuracy vs latency", "pareto_ndcg_vs_latency.png", False, "ndcg_latency"),
    ]
    for x_key, x_label, title, fname, x_max, front_key in specs:
        path = os.path.join(out_dir, fname)
        _pareto_scatter(ok, fronts[front_key], x_key, x_label, title, path, x_max)
        paths.append(path)

    # 4: validation curves - small multiples, one panel per loss, 3 layers each
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharey=True)
    for ax, loss in zip(axes.flat, LOSSES):
        for layer in LAYER_TYPES:
            c = curves.get(f"{layer}|{loss}")
            if not c:
                continue
            ax.plot([p["train_seconds"] for p in c], [p["val_ndcg"] for p in c],
                    color=LAYER_COLOR[layer], linewidth=2, marker="o", markersize=4,
                    label=LAYER_NAMES[layer])
        ax.set_title(LOSS_NAMES[loss], loc="left")
        ax.set_xlabel("Training time (s)")
    for ax in axes[:, 0]:               # shared y axis: label the left column only
        ax.set_ylabel("Validation NDCG@10")
    axes.flat[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Validation accuracy over the equal training-time budget",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "validation_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 5: slot ablation - accuracy by loss and layer
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _grouped_bars(axes[0], ok, "ndcg", lambda v: f"{v:.3f}")
    axes[0].set_title("Test NDCG@10 by loss and layer", loc="left")
    axes[0].set_ylabel("NDCG@10")
    _grouped_bars(axes[1], ok, "coverage", lambda v: f"{v:.2f}")
    axes[1].set_title("Test Coverage@10 by loss and layer", loc="left")
    axes[1].set_ylabel("Coverage@10")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "slot_ablation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 6: cost - two separate panels, never a dual axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _grouped_bars(axes[0], ok, "peak_vram_mb", lambda v: f"{v:.0f}")
    axes[0].set_title("Peak GPU memory during training (MB)", loc="left")
    axes[0].set_ylabel("MB")
    axes[0].set_ylim(0, 4096 * 1.08)   # keep the cap on the chart so headroom is visible
    axes[0].axhline(4096, color=MUTED, linestyle=":", linewidth=1.2)
    axes[0].annotate("4 GB cap", (3.45, 4096), textcoords="offset points", xytext=(0, 4),
                     ha="right", fontsize=8, color=MUTED)
    _grouped_bars(axes[1], ok, "sec_per_epoch", lambda v: f"{v:.2f}")
    axes[1].set_title("Training time per epoch (s)", loc="left")
    axes[1].set_ylabel("seconds")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "cost_memory_and_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    return paths
