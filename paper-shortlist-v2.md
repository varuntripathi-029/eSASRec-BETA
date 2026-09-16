# Deep-Dive Round 2 — Corrections, Novelty Audit, and a Revised Recommendation

**Date:** 2026-09-16. Supersedes the novelty and headroom claims in `paper-shortlist.md`.
This round: ~25 additional searches, 12 full-text/HTML fetches. Conventions unchanged: **[Published]** = in the cited source, **[Proposed]** = hypothesis, **[Unverified]** = could not confirm.

---

## PART 1 — Four corrections to Round 1

### Correction 1: PatchCore's "99.6%" is an ensemble number
Round 1 said PatchCore hits 99.6% AUROC on MVTec AD and concluded "accuracy is ceilinged — don't promise accuracy gains."

The paper's own configuration table says 99.6% requires an **ensemble of DenseNet-201 + ResNeXt-101 + WRN-101 at 320px**. The **default single model (WideResNet-50, 224px, 1% coreset) is 99.0%**, at ~0.17 s/image ([ar5iv full text](https://ar5iv.labs.arxiv.org/html/2106.08265)). The ensemble is both the expensive *and* the accurate configuration — which strengthens rather than weakens an efficiency project.

### Correction 2: The accuracy ceiling exists only on MVTec AD
The saturation story collapses the moment you leave one dataset:

| Benchmark | Best reported | Source |
|---|---|---|
| MVTec AD | 99.1% AUROC (EfficientAD) | [comparison study](https://www.sciencedirect.com/science/article/pii/S0166361524000794) |
| VisA | 98.1% | same |
| MVTec LOCO (logical) | 90.7% (EfficientAD); SALAD ICCV'25 reports 96.1% | [SALAD](https://openaccess.thecvf.com/content/ICCV2025/papers/Fucka_SALAD_--_Semantics-Aware_Logical_Anomaly_Detection_ICCV_2025_paper.pdf) |
| AutoVI (real automotive) | 88.4% (PatchCore) | same comparison study |
| **MVTec AD 2** | **SOTA below 60% average AU-PRO** | [arXiv 2503.21622](https://arxiv.org/abs/2503.21622) |

Round 1's "don't promise accuracy gains" advice was wrong in general and right only for MVTec AD.

### Correction 3: Blueprint A was largely already published
Round 1 proposed adaptive per-query pooling/pruning for ColPali and called that the novel part. It is not novel:

- **HPC-ColPali** ([arXiv 2506.21601](https://arxiv.org/html/2506.21601v1)) does **attention-guided pruning decided at query time**, plus K-Means quantization and binary encoding: **32x compression (57x with binary), query latency 120 ms to 60 ms, under 2% nDCG@10 loss**. That is Round 1's Blueprint A, already built.
- **DocPruner** ([arXiv 2509.23883](https://arxiv.org/pdf/2509.23883)) — adaptive patch-level embedding pruning for visual document retrieval.
- **Structural Anchor Pruning** ([arXiv 2601.20107](https://arxiv.org/pdf/2601.20107)) — 751 MB to 75 MB index, 7.9x MaxSim speedup, over 90% nDCG@5 retained.
- **Hybrid-Vector Retrieval** ([arXiv 2510.22215](https://arxiv.org/pdf/2510.22215)) — the single-vector-then-multi-vector cascade, i.e. Round 1's "stage-0 gate".
- Plus prune-then-merge, token merging, NanoVDR distillation, Nemotron ColEmbed V2, MM-Matryoshka.

Also: **ViDoRe V1 is saturated** (top models above 90 nDCG@5), so efficiency work measured on V1 is optimising a solved task ([ViDoRe V2](https://arxiv.org/abs/2505.17166)).

What HPC-ColPali explicitly leaves open: *learned* adaptive pruning policies (theirs is an attention heuristic), hierarchical PQ, streaming codebook updates. A real but narrow gap.

### Correction 4: Blueprint B is equally crowded
"Calibrated, cost-constrained routing with escalation" is the 2026 literature, not a gap:
- **RASER** ([2606.02488](https://arxiv.org/pdf/2606.02488)) — recoverability-aware selective escalation with an explicit cost-budget fraction and confidence threshold.
- **UCCI** ([2605.18796](https://arxiv.org/html/2605.18796)) — calibrated uncertainty for cost-optimal cascade routing; 31% cost cut at micro-F1 0.91 on a 75K-query production workload.
- **When Should Active RAG Retrieve?** ([2607.24010](https://arxiv.org/html/2607.24010v1)) — budget-aware evaluation of utility, calibration and cost.
- **Cost-Aware Query Routing in RAG** ([2606.02581](https://arxiv.org/html/2606.02581v1)) — retrieval-depth tradeoffs.

For the reranking variant: **AcuRank** (NeurIPS 2025, [2505.18512](https://arxiv.org/abs/2505.18512)) already does uncertainty-aware adaptive computation with a Bayesian TrueSkill model.

**Audit summary:** the "make a saturated system cheaper and faster" template is thoroughly occupied in both retrieval and anomaly detection — see also **CPR** (TIP 2024, cascade patch retrieval at 113 FPS, under 1 ms simplified, [code](https://github.com/flyinghu123/CPR)). Round 1 under-searched this and I should have caught it before recommending.

---

## PART 2 — Where the headroom actually is: MVTec AD 2 and the VAND challenge

### The benchmark
- **Paper:** *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*, [arXiv 2503.21622](https://arxiv.org/abs/2503.21622), published in **IJCV** ([Springer](https://link.springer.com/article/10.1007/s11263-026-02743-0)).
- **Content:** 8 scenarios, 8000+ high-resolution images, ~30.4 GB; transparent and overlapping objects, dark-field and backlight illumination, extremely small defects. **CC BY-NC-SA 4.0**, registration required ([dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)).
- **Honest evaluation:** private test-set ground truth is held server-side; you submit anomaly maps to [benchmark.mvtec.com](https://benchmark.mvtec.com/). You cannot accidentally or deliberately overfit the test set — a real credibility advantage in a viva.
- **Headroom:** SOTA below 60% average AU-PRO. In VAND 3.0 the winner scored **SegF1 53.81%** (stable lighting) and **51.43%** (mixed lighting); 2nd 51.00/46.52; 3rd 50.43/44.49 ([challenge report](https://arxiv.org/html/2509.17615v1)).

### What the organisers say is still unsolved — this is the opening
From the VAND 3.0 challenge report:
1. **Robustness to lighting shift** — winners still lose roughly 4–12% relative under changed illumination.
2. **Compute and deployability** — *no submission reported runtime or memory consumption at all*, despite the organisers calling these crucial for on-site deployment. Winners used large backbones at 448x448 or higher with tiling.
3. **Threshold selection** — described as indispensable in practice yet rarely addressed academically.
4. **Unified models** — most entries used different architectures per scenario rather than one versatile model.
5. **Rationale for logical anomalies** — explanation beyond a heat-map is unexplored.

Points 2 and 3 are *exactly* your three evaluation criteria plus the one operational knob that decides business value. The field has left them on the table.

### Why the industry case is the strongest of anything in either round
The chain is unusually tight, and the evidence is specific rather than "a company could use AI":
- AOI systems generate **20–80% false calls** (no-fault-found), forcing manual verification ([industry case study](https://intelliarts.com/blog/case-study-of-reducing-false-alarms-in-the-automated-optical-inspection-system/)).
- **False rejects cost 2–5% of production value** ([MarshallAI](https://marshallai.com/false-rejects-manufacturing-qa/)).
- **Missing parts are ~60% of assembly anomalies**, then misplaced 30%, extra 5%, foreign objects 5% ([assembly inspection overview](https://roboflow.com/ai/assembly-inspection)).

Chain: *lighting or appearance shift changes the anomaly score distribution, a fixed threshold mis-fires, false rejects rise, good parts are scrapped and re-inspected by hand.* Threshold calibration under distribution shift is the exact mechanism producing the cost. Note these three are vendor/industry sources, not peer-reviewed — cite them as industry evidence and lead with the MVTec/VAND academic framing.

### Challenge cadence (timing matters for a capstone)
VAND runs annually: VAND @ CVPR 2023, 3.0 @ CVPR 2025, **4.0 @ CVPR 2026** with an Industrial track on MVTec AD 2 (SegF1) and a Retail track on Kaputt2 ([site](https://sites.google.com/view/vand4-cvpr2026/challenge), [repo](https://github.com/cvpr-vand/vand-2026)). The 2026 window ran 1 April – 14 May. A VAND 5.0 at CVPR 2027 is **[Unverified]** but follows the pattern; if it runs, a spring 2027 window fits a final-year timeline and even a mid-table placing is external validation. Do not *depend* on it — treat it as upside.

---

## PART 3 — Revised blueprint: "Deployable anomaly detection under distribution shift"

**Base papers.** MVTec AD 2 (IJCV 2026) as benchmark; PatchCore (CVPR 2022) and EfficientAD (WACV 2024) as reproducible baselines; **Dinomaly** (CVPR 2025, [code](https://github.com/guojiajeremy/Dinomaly), 99.6/98.7/89.3 AUROC on MVTec-AD/VisA/Real-IAD) as the modern strong baseline.

**Limitation attacked.** Two, both named by the organisers: methods degrade under lighting shift, and nobody reports the runtime/memory cost of the accuracy they buy — so nobody knows which methods are deployable.

**The modification (three parts, mid-to-high substance):**
1. **Shift-robust scoring.** Test-time normalisation / distribution alignment of the anomaly score under illumination change, rather than retraining per lighting condition. Measured by the relative gap between stable and mixed-lighting test sets — the challenge's own robustness measure.
2. **Self-calibrating threshold.** A threshold policy that adapts to the current score distribution (quantile tracking, or conformal control targeting a fixed false-reject rate) instead of a fixed cut. This is where the technical work maps directly onto the 2–5% false-reject cost.
3. **A deployability frontier nobody has published.** Report SegF1/AU-PRO *jointly with* ms/image, peak VRAM and model size on a single laptop-class GPU, across resolutions and backbones. The organisers state no submission reported this. A careful accuracy-cost-latency frontier on MVTec AD 2 is a genuine contribution requiring no new theory.
4. *(Optional GenAI arm, bounded)* the VAND Category-2 setting — few-shot logical anomalies on MVTec LOCO with VLMs (winners FastLogSAD 93.61, UniVAD++ 92.77 F1) — plus a VLM turning a heat-map into a defect rationale, addressing the organisers' "no rationale beyond heat-maps" gap. Evaluate agreement with ground-truth defect types; do not grade it on vibes.

**Why substantial:** not a re-implementation. It needs a robustness mechanism, a calibration policy, a measurement harness the field currently lacks, and interpretation of a three-way trade-off — while every accuracy claim is checked by a server you cannot game.

**AI-assist fit (your sweet spot):** Claude/Codex can write the anomalib datamodule wiring, submission packaging for the eval server, latency/VRAM instrumentation, quantile/conformal threshold code, augmentation pipelines and plotting. Yours: choosing the robustness mechanism, defining operating points, deciding what "deployable" means numerically, and explaining why a method that wins on AUROC loses on the frontier.

**Hardware, with an honest flag.** anomalib supports MVTec AD 2, MVTec LOCO, VisA, PatchCore, EfficientAD and Dinomaly ([anomalib](https://github.com/openvinotoolkit/anomalib)). PatchCore and EfficientAD are comfortable on 6 GB. **Dinomaly's paper used an RTX 3090 (24 GB)** with a frozen DINOv2 ViT-B/14 encoder and trainable bottleneck plus decoder; 6 GB will need reduced resolution, gradient checkpointing, or the training-free route (VAND 4th place, SuperAD, was training-free with DINOv2). Treat "Dinomaly trains on 6 GB" as **[Unverified]** until you test it in week one. High-resolution tiling at 448px and above is where VRAM will actually bite.

**Risks.** (i) 136+ leaderboard submissions means this is *open*, not *uncontested* — your edge is the efficiency and threshold axis, not raw accuracy. (ii) Registration plus non-commercial licence: fine for college, state it in the report. (iii) Private-server evaluation limits submission attempts — build a local validation protocol first. (iv) If the VLM arm balloons, cut it; the core stands without it.

---

## PART 4 — Revised ranking

| Rank | Candidate | Why it moved |
|---|---|---|
| **1** | **MVTec AD 2 / VAND robustness + deployability** | New. Largest verified headroom (under 60% AU-PRO), organiser-stated open problems matching your exact three metrics, honest external evaluation, tightest industry chain, best hardware fit. |
| **2** | PatchCore / EfficientAD efficiency-accuracy frontier | Was #3. Still solid, but the cascade idea is taken (CPR); best used as the baseline layer *inside* #1. |
| **3** | ColPali | Was #1. **Downgraded** — the proposed modification is already published several times over. Viable only if reframed as *accuracy on ViDoRe V2* (SOTA 59–65) rather than efficiency on saturated V1. |
| **4** | Adaptive-RAG | Was #2. **Downgraded** — RASER/UCCI/Active-RAG already occupy calibrated cost-aware routing. |

**Unchanged advice:** spend 2–3 days reproducing one published number before committing. For #1 that means getting PatchCore or EfficientAD running through anomalib on MVTec AD 2 and producing a valid server submission. If that pipeline works in week one, the project is de-risked.
