# Research-Paper Shortlist for a B.Tech Capstone (3–4 students, RTX 3060–4060 / 4050 6 GB)

**Date:** 2026-09-16
**Constraints honoured:** Deep Learning primary, mild/moderate GenAI secondary, mid-to-high substantial modification required, evaluation on *accuracy + cost + search/inference time*, strong industry case, AI-assisted implementation must be in the "sweet spot".

**Conventions used below**
- **[Published]** = number reported in the cited paper/repo. **[Proposed]** = our hypothesis, not a result.
- Anything I could not verify with a source is marked **[Unverified]**.
- Ratings are defined explicitly where used; no unexplained 1–5 stars.

---

## PART 0 — What I rejected, and why

| Rejected | Reason |
|---|---|
| **Self-RAG** (Asai et al., ICLR 2024 oral) | Method *is* excellent, but reproduction means training a Llama-2-7B critic + 7B/13B generator on 150K reflection-token instances ([paper](https://arxiv.org/abs/2310.11511), [code](https://github.com/AkariAsai/self-rag)). Not trainable on a 6 GB card; you'd be reduced to running released checkpoints = reproduction without modification. |
| **Microsoft GraphRAG as a base** | Indexing is LLM-token-bound; Microsoft's own cost guidance and third-party reports put full graph indexing in the tens of dollars per million tokens ([MS cost note](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/graphrag-costs-explained-what-you-need-to-know/4207978)). Repeated ablations would be unaffordable for a student team. Interesting as a *baseline to beat*, not a base. |
| **RankGPT used directly with a GPT-4-class API** | Turns into an API-wrapper project and the cost metric becomes "whatever OpenAI charges". Its *distillation* half is usable (see Candidate 7). |
| **Any "LLM chatbot over documents"** | Explicitly excluded by your brief and by me. |
| **Very new 2026 arXiv-only routing/pruning papers** | Several showed up in search (e.g. RAGRouter-Bench, CRISP, prune-then-merge variants). Too fresh, code maturity unverified, and they occupy exactly the modification space you want to own. Treat them as *related work you must cite*, not as a base. |

---

## PART 1 — The 11 serious candidates

### 1. ColBERTv2 + PLAID — late-interaction retrieval
- **Papers:** Santhanam et al., *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*, NAACL 2022 ([ACL](https://aclanthology.org/2022.naacl-main.272/), [arXiv](https://arxiv.org/abs/2112.01488)); Santhanam et al., *PLAID: An Efficient Engine for Late Interaction Retrieval*, CIKM 2022 ([ACM](https://dl.acm.org/doi/10.1145/3511808.3557325), [arXiv](https://arxiv.org/abs/2205.09707)).
- **Code:** [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT); modern trainable stack: [lightonai/pylate](https://github.com/lightonai/pylate), [AnswerDotAI/RAGatouille](https://github.com/AnswerDotAI/RAGatouille). Checkpoint: [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0).
- **Data:** MS MARCO, BEIR ([beir-cellar/beir](https://github.com/beir-cellar/beir)), LoTTE ([mteb/LoTTE](https://huggingface.co/datasets/mteb/LoTTE)).
- **Published results:** ColBERTv2 cuts late-interaction storage 6–10× vs ColBERT; PLAID cuts search latency 2.5–7× on GPU and 9–45× on CPU vs vanilla ColBERTv2.
- **Real limitation:** multi-vector indexes are still an order of magnitude larger than single-vector ones, and latency is corpus-size sensitive.
- **Risk:** *this space is crowded*. WARP (SIGIR'25, [arXiv](https://arxiv.org/abs/2501.17788), [code](https://github.com/jlscheerer/xtr-warp)) already gives 3× over PLAID, and token pooling ([Clavié et al.](https://arxiv.org/abs/2409.14683)) already gives 50% footprint reduction. Your modification must be *composed* differently, not re-derived.

### 2. ColPali — vision-language document retrieval ⭐
- **Paper:** Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models*, ICLR 2025 ([arXiv](https://arxiv.org/abs/2407.01449), [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf)).
- **Code/models:** [illuin-tech/colpali](https://github.com/illuin-tech/colpali), [huggingface.co/vidore](https://huggingface.co/vidore) incl. small variants [colSmol-256M](https://huggingface.co/vidore/colSmol-256M) and [colSmol-500M](https://huggingface.co/vidore/colSmol-500M).
- **Published results:** backbone PaliGemma-3B + LoRA(r=α=32), 128-d projection; **81.3 avg nDCG@5 on ViDoRe** vs ~65.5 (Unstructured+BM25) and 67.0 (captioning + BGE-M3); ~30 ms/query on an L4; **256 KB/page** index at fp16 vs ~8.6 KB for BGE-M3. Authors explicitly flag index size and lack of multi-vector DB support as open problems. colSmol-256M scores 80.1 on ViDoRe V1 but the model card itself warns of overfitting to V1 and weaker V2 generalisation.
- **Why it fits you:** accuracy, storage cost and latency are all first-class, *stated by the authors*, and the GenAI component (a VLM) is intrinsic rather than bolted on.

### 3. Adaptive-RAG — query-complexity routing ⭐
- **Paper:** Jeong et al., NAACL 2024 ([ACL](https://aclanthology.org/2024.naacl-long.389/), [arXiv](https://arxiv.org/abs/2403.14403)). **Code:** [starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG).
- **Published results:** T5-Large (770M) classifier routing to no-retrieval / single-step / multi-step; smaller 60M and 223M classifiers perform comparably. With FLAN-T5-XL: EM 37.17 vs 34.83 (single-step) and 39.00 (multi-step); **2.17 steps/query vs 4.69**, **3.60 s/query vs 8.81 s**. Oracle routing = 45.00 EM, i.e. a **~7.8 EM gap left on the table by the classifier** — the paper says so itself.
- **Reproduction footprint:** Elasticsearch 7.10.2 retriever server + LLM server, datasets subsampled to 500 per set; single-hop SQuAD/NQ/TriviaQA, multi-hop MuSiQue/HotpotQA/2WikiMultiHopQA.
- **Real limitation (author-acknowledged):** silver labels come from model predictions + dataset bias; classifier is uncalibrated and non-cost-aware; routing is a hard 3-way decision with no confidence/abstention and no budget control.

### 4. LLMLingua-2 — task-agnostic prompt compression
- **Paper:** Pan et al., Findings of ACL 2024 ([ACL](https://aclanthology.org/2024.findings-acl.57/), [arXiv](https://arxiv.org/abs/2403.12968)). **Code:** [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua).
- **Published results:** token-classification compressor on XLM-RoBERTa-large / mBERT; 2×–5× compression, 3×–6× faster than LLMLingua, **1.6×–2.9× end-to-end latency reduction**; evaluated on MeetingBank, LongBench, ZeroScrolls, GSM8K, BBH.
- **Fit:** superb *component* for a cost-reduction story. Weak as a standalone capstone (the core is a token classifier) — best combined with Candidate 3.

### 5. PatchCore (+ EfficientAD) — industrial visual anomaly detection ⭐
- **Papers:** Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022 ([CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html), [arXiv](https://arxiv.org/abs/2106.08265)); Batzner et al., *EfficientAD*, WACV 2024 ([CVF](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html)).
- **Code:** [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection); both models are first-class in [anomalib](https://github.com/openvinotoolkit/anomalib) with a built-in benchmarking harness.
- **Data:** [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) (CC BY-NC-SA 4.0 — academic use only), [VisA](https://github.com/amazon-science/spot-diff) (CC BY 4.0, commercial-friendly).
- **Published results:** PatchCore reaches up to **99.6% image-level AUROC on MVTec AD**; EfficientAD reports ~2 ms latency / ~600 images/s and was evaluated over 32 datasets across three collections.
- **Real limitation:** PatchCore's accuracy comes from a nearest-neighbour memory bank whose size scales with images × resolution × feature dim; coreset subsampling trades accuracy for memory/latency. EfficientAD exists precisely because of that gap — which makes the accuracy-vs-latency frontier a *legitimate, published* research axis rather than one you invented.

### 6. SPLADE / SPLADE++ — learned sparse retrieval
- **Papers:** Formal et al., SIGIR 2021/2022 ([SPLADE v2 arXiv](https://arxiv.org/abs/2109.10086)). **Code:** [naver/splade](https://github.com/naver/splade).
- **Published:** sparse expansion into an inverted index; with query-specific regularisation and disjoint encoders, latency "on par with BM25 under the same computing constraints".
- **Fit:** excellent *hybrid partner* for Candidate 1/2 (cheap first stage). Thin as a standalone base.

### 7. RankGPT + permutation/instruction distillation — reranking
- **Paper:** Sun et al., EMNLP 2023 **Outstanding Paper** ([ACL](https://aclanthology.org/2023.emnlp-main.923/)). **Code:** [sunnweiwei/RankGPT](https://github.com/sunnweiwei/RankGPT).
- **Published:** distilling ChatGPT's ranking ability into a **440M model that beats a 3B supervised model on BEIR**.
- **Fit:** this is the honest way to get "GenAI" into a reranker without an API bill at inference time. Strong as a *stage* inside a bigger system.

### 8. Cross-architecture KD for efficient ranking (Margin-MSE)
- **Code/models:** [sebastian-hofstaetter/neural-ranking-kd](https://github.com/sebastian-hofstaetter/neural-ranking-kd), [matchmaker](https://github.com/sebastian-hofstaetter/matchmaker), HF checkpoints e.g. [distilbert-dot-margin_mse-T2-msmarco](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco).
- **Fit:** the canonical "teacher→student under a latency budget" recipe. Very reproducible; modification depth is medium, not high.

### 9. Matryoshka Representation Learning — adaptive-dimension retrieval
- **Paper:** Kusupati et al., NeurIPS 2022 ([PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf)). **Code:** [RAIVNLab/MRL](https://github.com/RAIVNLab/MRL).
- **Published:** nested embeddings enabling *adaptive retrieval*; matched full-dim top-1 accuracy with **128× fewer MFLOPs/query** on ImageNet-1K, up to 14× real-world retrieval speedups.
- **Fit:** the cleanest published mechanism for "shrink first, refine later". Ideal **ingredient** for candidates 1/2, weak alone (it is a loss function).

### 10. Shopping Queries Dataset (ESCI) — product search ranking
- **Paper/data:** Reddy et al. 2022 ([arXiv](https://arxiv.org/abs/2206.06588), [amazon-science/esci-data](https://github.com/amazon-science/esci-data)); ~130K queries, 2.6M human-labelled query–product pairs (E/S/C/I), EN/ES/JP; used in a KDD Cup '22 track. Image-enriched extension SQID ([arXiv](https://arxiv.org/abs/2405.15190)).
- **Fit:** the strongest *dataset-backed industry case* (real Amazon relevance labels). But it is a benchmark paper, not a method paper — you'd have to pair it with a method from 1/6/7/8. Acceptable, slightly weaker "paper → limitation" narrative.

### 11. Semantic caching for LLM serving
- **Paper/code:** *GPT Semantic Cache* ([arXiv 2411.05276](https://arxiv.org/abs/2411.05276)); [zilliztech/GPTCache](https://github.com/zilliztech/GPTCache).
- **Published:** up to 68.8% API-call reduction, cache-hit 61.6–68.8%.
- **Fit:** a *module*, not a capstone. Include it as one arm of a cost-reduction ablation.

---

## PART 2 — Comparison matrix

**Rating key:** H = high / strong / low-risk-good, M = medium, L = low. "Modification depth" = how much genuine architecture+experiment work a mid-to-high modification demands. "AI-assist feasibility" = H means Claude/Codex can carry most of the code while design decisions stay yours (the sweet spot); L means either trivially promptable or beyond assistant reach.

| Criterion | 2. ColPali | 3. Adaptive-RAG | 5. PatchCore/EffAD | 1. ColBERTv2/PLAID | 10. ESCI product search |
|---|---|---|---|---|---|
| DL relevance | H (VLM, late interaction) | M (routing + LLM orchestration) | H (CV repr. learning, kNN) | H (neural IR) | M–H (ranking models) |
| GenAI compatibility | H (VLM is the model) | H (LLM generators) | M (VLM only for explanations) | L–M | M |
| Hardware feasibility (6 GB) | M — use ColSmol-256M/500M; 3B training on 6 GB is doubtful **[Unverified]** | M — needs Elasticsearch + a served LLM; FLAN-T5-XL is 3B | **H** — WideResNet-scale backbones, no LLM training | H | H |
| Dataset accessibility | H (ViDoRe open on HF) | H (all public) | H (MVTec AD non-commercial; VisA CC BY 4.0) | H | H (signup-free GitHub) |
| Code availability | H (official + HF) | H (official) | H (official + anomalib) | H (official + PyLate) | M (data only) |
| Reproduction difficulty | M | M–H (most moving parts) | **L–M (easiest)** | M | M |
| Modification depth available | H | H | H | M–H | M |
| AI-assist feasibility | **H** | H | H | M (engine internals are C/CUDA-ish) | H |
| Accuracy improvement potential | M–H | H (7.8 EM oracle gap) | M (already 99.6 AUROC — ceiling) | L–M | M |
| Cost reduction potential | **H** (256 KB/page) | H (tokens + steps) | M (memory bank) | M | M |
| Latency improvement potential | H | H | **H** | M (WARP got there first) | M |
| Industry significance | H (doc processing) | H (support/enterprise QA) | **H (QC, hardest evidence)** | M–H | H |
| Evidence for industry need | H | H | H | M | **H (real Amazon labels)** |
| Experiment measurability | H | H | **H** | H | H |
| Presentation clarity | H (visual demo) | M (pipeline-heavy) | **H (defect heatmaps)** | M | M |
| Overall project risk | M | M–H | **L–M** | M–H (novelty crowding) | M |

---

## PART 3 — Deep blueprints (top 3)

### BLUEPRINT A — "Budget-aware cascaded visual document retrieval" (base: ColPali)

**Industry problem.** Enterprises retrieve from PDFs — invoices, spec sheets, filings, manuals. Manual AP invoice handling costs an industry-average **$12.88/invoice** vs **$2.78** best-in-class automated, and cycle time 17.4 days vs 3.1 days ([Ardent Partners metrics summary](https://www.apexanalytix.com/resources/blog/ardent-partners-key-ap-metrics-2025/)). More broadly, McKinsey's widely cited figure is **1.8 hours/day per knowledge worker spent searching for information** ([summary of survey statistics](https://cottrillresearch.com/various-survey-statistics-workers-spend-too-much-time-searching-for-information/)). Traditional pipelines lose the visual layer (tables, charts, layout) at the OCR step.

**Baseline.** ColPali (or ColSmol) multi-vector retrieval over page images; ViDoRe evaluation; 81.3 nDCG@5 **[Published]**.

**Limitation (author-stated).** 256 KB/page index at fp16 (~30× BGE-M3), and every query pays full late-interaction cost over every candidate page, regardless of how easy the query is.

**Our modification (mid-to-high).** A **three-part adaptive cascade**, not a single trick:
1. **Stage-0 cheap gate:** a pooled single-vector (mean-pooled patch embedding, or an MRL-style nested 32/64-d prefix trained with a Matryoshka loss) prunes the corpus to top-K.
2. **Adaptive per-query budget:** a small controller predicts, from stage-0 score margins/entropy, *how many* candidates go to full late interaction and *at what patch-pooling ratio*. This is the genuinely new part: existing work (token pooling, CRISP) compresses the index **statically at indexing time**; we make the compression level a **per-query runtime decision**.
3. **Layout-aware pooling:** cluster patches by visual saliency rather than uniform pooling so that white-space patches collapse hard and table/figure regions stay fine-grained.
4. *(Optional GenAI arm)* a small VLM generates a page-level synthetic caption used only as a sparse-hybrid signal, measured for whether it earns its indexing cost.

**Why substantial.** It changes the retrieval *scoring path*, adds a learned controller, requires a new training objective for nested/pooled representations, and forces a Pareto-frontier analysis — not a hyperparameter sweep.

**Where Claude/Codex genuinely help:** wiring `colpali-engine`/PyLate, writing the pooling + clustering code, building the index-size/latency instrumentation, the ablation runner, the Streamlit/Gradio demo, and debugging tensor shapes and dtype/VRAM issues.
**What you must do yourselves:** decide the controller's features and label definition; choose the accuracy/latency operating points; decide what "budget" means; interpret Pareto curves; defend why layout-aware pooling beats uniform pooling.

**Experiments.** Baseline ColPali → +static token pooling (reproduce Clavié et al. as a *second baseline*, honestly cited) → +MRL prefix → +adaptive budget → full system. Ablate each.
**Metrics.** Accuracy: nDCG@5 / Recall@1,5 on ViDoRe. Cost: KB/page index, peak VRAM, indexing GPU-seconds/page. Latency: p50/p95 query latency at fixed corpus sizes (1K/10K/50K pages).
**Hardware.** ColSmol-256M/500M for training-in-the-loop on 6 GB; ColPali-3B inference in 4-bit for a "big model" comparison row.
**Risks.** (i) The static-pooling baseline may already capture most of the gain — mitigate by making *adaptivity* the claim, with per-query latency variance as the headline. (ii) ViDoRe V1 saturation/overfitting (model card warns) — evaluate on ViDoRe V2 too. (iii) VRAM on 3B training — treat as unverified until you test.

---

### BLUEPRINT B — "Cost-calibrated adaptive RAG" (base: Adaptive-RAG)

**Industry problem.** Support/enterprise QA at scale. Published vendor and analyst benchmarks put human-handled contacts in the ~$5–15 range vs well under $2 for automated ones (figures vary by source and are vendor-influenced — cite with caution: [LiveChatAI benchmark roundup](https://livechatai.com/blog/customer-support-cost-benchmarks)). The technical bottleneck is that most RAG stacks run the *same* expensive multi-step pipeline for trivial and hard questions alike — exactly what Adaptive-RAG measured: 4.69 steps and 8.81 s/query for always-multi-step.

**Limitation.** The classifier is trained on silver labels, is uncalibrated, is not cost-aware, and leaves ~7.8 EM to oracle routing **[Published, author-acknowledged]**.

**Our modification.** Replace hard 3-way classification with a **calibrated, budget-constrained controller**:
1. Confidence-calibrated router (temperature scaling / conformal prediction) with an **abstain-and-escalate** path instead of a hard argmax.
2. **Explicit cost model** (tokens, retrieval calls, wall-clock) and a constrained objective: maximise EM subject to a token budget — producing an accuracy-vs-cost frontier rather than one point.
3. **Cheap-first escalation:** answer, self-check, escalate only if the check fails — turning routing from a prediction problem into a sequential-decision one.
4. **Cost arms to ablate:** LLMLingua-2 context compression and a semantic cache, each measured separately for token savings vs EM loss.

**Where Claude/Codex help:** Elasticsearch/index setup, the FlashRAG or Adaptive-RAG harness ([FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) has 36 pre-processed datasets and is the faster path to clean baselines), calibration code, cost accounting, plotting.
**What you must do:** define the cost model, choose the escalation criterion, decide the budget levels, and defend calibration choices.
**Metrics.** EM/F1 on MuSiQue/HotpotQA/2Wiki/NQ/TriviaQA/SQuAD; tokens/query and steps/query; s/query p50/p95.
**Hardware note.** Use a 7–8B instruct model in 4-bit or Flan-T5-Large locally; keep one small API-model run only as a reference row.
**Risks.** Most moving parts of the three blueprints; an LLM served locally on 6 GB is the bottleneck; results are sensitive to the retriever corpus build (plan ~2 weeks just for infrastructure).

---

### BLUEPRINT C — "Accuracy-preserving, latency-bounded industrial inspection" (base: PatchCore, benchmarked against EfficientAD)

**Industry problem.** Cost of poor quality, scrap/rework and warranty are real line items; human visual inspection is inconsistent. (Treat vendor blog percentages with suspicion; the *defensible* citation for this project is the academic framing plus the MVTec/VisA benchmarks themselves. Cite EfficientAD's own motivation: production-rate-driven runtime budgets.)

**Limitation.** PatchCore's memory bank scales with images × resolution × feature dim; coreset subsampling trades AUROC for memory and latency, and inference is a kNN search.

**Our modification.** A **two-stage, uncertainty-gated inspector**:
1. Stage-1 fast screen (EfficientAD-style student-teacher or a heavily quantised/pruned feature extractor) classifies obvious-normal parts at millisecond latency.
2. Stage-2 PatchCore memory bank runs **only** on uncertain parts, with the memory bank stored in an ANN/PQ index (FAISS IVF-PQ) instead of exhaustive kNN — a real accuracy/memory/latency trade study.
3. **Uncertainty estimation** (score-margin or ensemble/MC) defines the gate; the gate threshold becomes a tunable operating point, letting you report *throughput at fixed AUROC* rather than AUROC alone.
4. *(Mild GenAI arm)* a small VLM turns the anomaly heat-map into a defect description/triage label — evaluated on whether it agrees with ground-truth defect categories, not just "it looks nice".

**Why substantial.** Cascade design + ANN-compressed memory bank + calibrated gating + a throughput-constrained evaluation protocol is genuine systems-ML engineering, and every claim is measurable on a laptop GPU.

**Where Claude/Codex help:** anomalib model/datamodule subclassing, FAISS index integration, quantisation/ONNX-OpenVINO export, the benchmark harness, heat-map visualisation and demo UI.
**What you must do:** design the gate and its threshold policy, decide PQ parameters vs AUROC loss, choose the per-category operating points, interpret failure cases.
**Metrics.** Accuracy: image AUROC + pixel AUROC/PRO on MVTec AD and VisA. Cost: memory-bank MB, peak VRAM, model size. Latency: ms/image and images/s at batch 1 (the industrially meaningful setting), p95 included.
**Hardware.** The most comfortable fit of the three: no LLM training, backbones are WideResNet/EfficientNet-scale, anomalib handles experiment management.
**Risks.** PatchCore is already at 99.6 AUROC, so **do not promise accuracy gains** — frame the contribution as *equal accuracy at a fraction of memory/latency*, plus gains on the harder VisA/MVTec-LOCO settings. MVTec AD is CC BY-NC-SA: fine for a college project, mention the licence in your report.

---

## PART 4 — Trade-offs, phases, and my read

- **A (ColPali)** — best balance: strongest GenAI content, authors themselves name the limitation you'd attack, and the demo (drop a PDF, get the right page) presents brilliantly. Highest "is this novel enough" pressure, because the multi-vector-compression field is moving fast.
- **B (Adaptive-RAG)** — strongest cost story and the largest published headroom (oracle gap), but the heaviest infrastructure and the most ways for a 4-person team to lose a month to plumbing.
- **C (PatchCore)** — lowest execution risk, cleanest hardware fit, most legible industry narrative, most viva-friendly visuals. Accuracy ceiling means the contribution must be efficiency-framed; GenAI is a side dish, not the main course.

**Phases (ranges, not promises):** paper understanding 1–2 wks → baseline reproduction 2–4 wks (B at the high end) → modification design 1–2 wks → implementation 3–6 wks → debugging overlaps throughout → experiments/ablations 3–4 wks (budget for reruns) → optimisation 1–2 wks → demo 1–2 wks → writing 2 wks.

**Non-negotiable early check:** before committing, spend 2–3 days reproducing *one* baseline number from the chosen paper on your own 6 GB machine. If you cannot reproduce a single published number in 3 days, that candidate is wrong for you regardless of how good it looks here.
