# Sequential Recommendation — Beta 0.1

This is **Beta 0.1** (the guide's Part A demo), not the project. Per the implementation guide's rule 20:
the demo only proves we understand the sequential recommendation problem and can
make a model predict the next item. SASRec and eSASRec come later.

## What it does

```
user history -> item embeddings -> sequence model -> scores over catalogue -> Top-K
```

Two models are included:

| Model | What it is | Purpose |
|---|---|---|
| `meanpool` | embedding + mean pooling + linear | simplest thing that works (guide §5) |
| `transformer` | item + positional embeddings, causal self-attention, dot-product scoring (guide §6) | deliberately SASRec-shaped, so step 2 is an extension not a rewrite |

## Setup

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy pandas
```

Note: this machine runs Python 3.14, which has no cu124 wheels — use **cu126**.

## Run

```bash
python train.py --model transformer --epochs 5
```

```bash
python demo.py
```

MovieLens-1M downloads automatically on first run (~6 MB) into `data/`.

## Web frontend

```bash
python app.py
```

Then open <http://localhost:8000>. Search films, build a history in order, and
predict the next item. `--checkpoint checkpoint_meanpool.pt` serves the weaker
model instead, which is a good side-by-side for a demo.

Uses **only the Python standard library** for the server — nothing extra to
install, nothing to break on a teammate's machine. The model is warmed up at
startup so the latency shown in the UI is steady-state (~2-4 ms) rather than the
~150 ms first CUDA call.

### Observed popularity bias (worth reporting)

The demo model is not uniformly good, and the failure is instructive:

| History | Top recommendations | Verdict |
|---|---|---|
| Aladdin, Beauty and the Beast, Tarzan, Antz, Hercules … | Prince of Egypt, Rescuers Down Under, Anastasia, Mulan | genuinely on-target |
| Wes Craven's New Nightmare, Freddy's Dead, Pet Sematary II, Children of the Corn III … | Star Wars IV, American Beauty, Braveheart, Schindler's List | mostly just *popular* |

For the niche horror history the model largely falls back to globally popular
items. This is classic popularity bias, and it is exactly what the guide's §13
beyond-accuracy metrics (diversity, novelty, serendipity) are designed to
expose — Recall@10 alone would never reveal it. The demo therefore already
motivates that part of the project rather than treating it as an add-on.

## Evaluation protocol

- **Leave-one-out split**: last item = test, second-to-last = validation, rest = training.
- **Full-catalogue ranking.** We rank the held-out item against every item, *not*
  against sampled negatives. Krichene & Rendle (KDD 2020) showed sampled metrics
  do not reliably preserve which model is better — so we avoid them from day one.
  This matters because the examiner-facing weakness of recsys work is evaluation,
  not modelling.
- **Seen-item masking**: items already in the user's history are excluded.
- Metrics: Recall@10, NDCG@10.

## Efficiency numbers

`train.py` records parameters, training time, peak GPU memory and inference
ms/user into `results_<model>.json`. This feeds the guide's §15 efficiency study —
recorded from the start rather than bolted on at the end.

## Dataset

**MovieLens 1M** — GroupLens Research, University of Minnesota.
1,000,209 ratings · 6,040 users · 3,706 rated movies · 2000–2003.

- Dataset page: <https://grouplens.org/datasets/movielens/1m/>
- Direct download: <https://files.grouplens.org/datasets/movielens/ml-1m.zip>
- README & terms: <https://files.grouplens.org/datasets/movielens/ml-1m-README.txt>

**Cite as:** F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. *ACM Transactions on Interactive Intelligent Systems* 5(4),
Article 19. <https://doi.org/10.1145/2827872>

Terms: research use only; acknowledge the dataset in publications; no commercial
or revenue-bearing use without permission from GroupLens.

## Results (MovieLens-1M, 5 epochs, RTX 4050 6 GB)

Full-catalogue ranking, leave-one-out test split, seen items masked.

| Model | Recall@10 | NDCG@10 | Params | Train time | Peak GPU | Inference |
|---|---|---|---|---|---|---|
| `meanpool` | 0.1588 | 0.0853 | 241k | 37 s | 34 MB | 0.07 ms/user |
| `transformer` | **0.2969** | **0.1732** | 339k | 134 s | 85 MB | 0.07 ms/user |

The transformer is **~1.9× better** for 3.6× the training time and 40% more
parameters. That gap is the whole premise of sequential recommendation stated as
a measurement: mean pooling discards interaction *order*, self-attention keeps
it. Same data, same catalogue, same evaluation — only the ordering assumption
differs.

Do not compare these to the numbers printed in the SASRec paper. Those use
sampled negatives (rank against 100 random items); these rank against all 3,706.
Full-catalogue numbers are always much lower and are not comparable.

## Two bugs found while building this (worth keeping in the report)

1. **NaN from fully-padded rows.** Sequences are left-padded, so an all-padding
   query row attends to nothing and softmax returns NaN — and with 2 layers the
   NaN spread to *every* position, zeroing the output for every short history.
   Fixed by masking values rather than keys (what the official SASRec does).
2. **A silently inflated metric.** Rank was computed as "how many items score
   strictly higher". With all-zero logits that gives rank 0 — a perfect hit. So
   bug 1 was *raising* the reported score instead of crashing. Now ties are
   broken pessimistically and non-finite logits raise.

Bug 2 is the interesting one for the report: the failure made results look
*better*, which is exactly the kind of error that survives into published tables.
It is also a concrete reason to distrust recsys numbers without an error check.

## Component benchmark (eSASRec-style)

A separate benchmark in `benchmark/` that mixes and matches SASRec components the
way the eSASRec paper does, and adds the cost measurements the paper doesn't report.
Beta 0.1 above is untouched and still runs.

```bash
python -m benchmark.run
```

Run from the project root. Takes about 8–9 minutes on an RTX 4050 Laptop GPU.

```bash
python -m benchmark.run --report-only
```

Re-prints the last results instantly, without retraining. Add `--plots` to redraw
the charts from saved results.

```bash
python -m benchmark.run --only preln:full_softmax,ligr:bce --budget 10
```

A quick trial on selected combinations.

### What it tests

**3 layer designs × 4 losses = 12 combinations**, each given the **same training-time
budget** (32 s) on the same GPU.

| Slot | Options |
|---|---|
| Layer design | Post-LN (vanilla) · Pre-LN · LiGR (pre-norm + sigmoid-gated attention and feed-forward) |
| Loss | BCE (1 negative) · gBCE (256 negatives, t = 0.75) · Sampled softmax (256 negatives) · Full softmax |
| Objective | Shifted sequence — every position predicts the next item (fixed for all runs) |

Model: 2 layers, 2 heads, dim 192, history up to 200, bf16 mixed precision, Adam 1e-3,
batch 256 users. Memory is capped at 4 GB by PyTorch; the heaviest combination
(full softmax) uses about 2.5 GB, and cheaper losses use less by design.

### What it reports

- **Accuracy** (paper's metrics): NDCG@10, Recall@10, MRR, Coverage@10 — leave-one-out,
  full-catalogue ranking, seen items masked, best validation epoch evaluated on test.
- **Cost** (not in the paper): peak GPU memory, training time, time per epoch,
  latency for one request, throughput, GPU temperature and power.
- **Slot ablation**: each slot averaged over the other.
- **Pareto frontiers**: accuracy vs coverage (paper's view), accuracy vs memory,
  accuracy vs latency.
- **Target checks**: total time ≤ 10 min, heaviest run within 2–4 GB, nothing over the cap.

### Outputs

| File | Contents |
|---|---|
| `results/benchmark_results.json` | Everything: config, per-run metrics, training curves, timing |
| `results/benchmark_results.csv` | One row per combination, for spreadsheets |
| `results/plots/*.png` | 6 charts: three Pareto scatters, validation curves, slot ablation, memory and time |

### Design decisions worth knowing

- **Why training is ~12× faster per epoch than Beta 0.1.** Beta 0.1 splits every history
  into 982,089 separate windows and takes 3,837 tiny GPU steps per epoch; the GPU sat about
  50% idle. The shifted-sequence objective trains every position of a history in one pass,
  so an epoch is 24 large steps. Using more memory is a side effect, not the cause: batch 128
  and batch 256 took the same time per epoch while using 1.7 GB vs 3.4 GB.
- **Negatives are drawn per user, not per batch.** An earlier version shared 256 negatives
  across the whole batch (~30,000 positions). Most items then almost never got a negative
  signal, and sampled softmax scored NDCG@10 0.004. Drawing 256 per user raised it to 0.074,
  within 4% of full softmax at about half the memory.
- **Equal time, not equal epochs.** Slower combinations get fewer epochs. That answers "what
  does each combination achieve on this hardware in the same time?" — the consumer-hardware
  question — but it is not the paper's protocol.
- **The process is pinned to performance cores.** This laptop's i7-12650H has 6 performance
  cores and 4 efficiency cores. A single request mostly measures how fast the CPU hands work
  to the GPU, so the same model measured ~1.6 ms on performance cores and ~3.2 ms on
  efficiency cores; unpinned, one benchmark run swung between 1.6 and 4.5 ms for identical
  models. `benchmark/hardware.py` asks Windows which cores are which and pins to the fast
  ones. On a CPU without core types it changes nothing.
- **gBCE collapses in this setup — treat it as unresolved, not as a result.** It recommends
  only ~5% of the catalogue whether calibration is on (t = 0.75) or off (t = 0), and with 32
  or 256 negatives. Sampled softmax uses the same negatives and works, so the scoring code
  and formula are not the cause; summed-negative BCE dynamics under a short budget are the
  likely one.
- **LiGR and gBCE are our own implementations** from the papers' descriptions; there is no
  official code for them here.
- **One seed per combination**, so there are no error bars yet.

## Where this goes next

1. **(done)** demo pipeline
2. SASRec proper — shifted-sequence objective, more layers, tuned
3. eSASRec — LiGR layers + sampled softmax
4. BERT4Rec comparison
5. ablations, then the SCE loss experiment

## Files

| File | Purpose |
|---|---|
| `data.py` | MovieLens download, sequence building, leave-one-out split |
| `model.py` | MeanPoolRec and TinyTransformerRec |
| `train.py` | training loop + full-catalogue evaluation + efficiency stats |
| `demo.py` | CLI demo, including `--interactive` mode |
