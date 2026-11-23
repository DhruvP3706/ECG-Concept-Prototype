# LightECGNet upgrade — audit and implementation plan (no training run, no fold-10 waveform access)

> **Errata (Phase 1, 2026-11-20):** the paper figures were later inspected (`docs/PAPER_VS_V2_ARCHITECTURE.md`); the bf16 "55 records/s" figure below was a memory-spill artefact — see `results/phase1_infrastructure/hw_benchmark.json` (FP32 ≈ 318 rec/s at batch 32–64 is fastest). Other conclusions unchanged.

Status legend: **[V]** verified in this audit (code read, file inspected, or measured); **[I]** inferred; **[U]** unresolved / cannot be determined from the available material; **[P]** proposed.
Date 2026-11-20. Reference checkpoint `models/pretrained/lightv2_ft_fold0.pt` SHA-256 `7779719e…f79f72` re-verified at the end of this audit [V]; nothing was written to `models/`, no training job was launched, no PTB-XL fold-10 waveform was opened [V].
Audit measurements that touched waveforms used **training-role records only** (300 PTB-XL fold 1–8 records, 300 PhysioNet train-role records) and a GPU micro-benchmark on random tensors.

## 0. Which paper was read
`C:\Users\lenovo\Downloads\ECG_paper.pdf` (modified today; "An Integrated Framework for ECG Image Digitization, Lightweight Arrhythmia Classification, and Faithfulness-Validated Explainability in 12-Lead ECGs", 16 pages, template authors/affiliations still placeholders, i.e. an unpublished draft) [V]. Its numbers (macro-AUROC 0.963, accuracy 0.977, recall 0.551) match your brief. An **older version** with the same title is also in Downloads (accuracy 0.974, AUROC 0.962 reported) [V]; the newer one was used. No PDF tool exists in the environment, so I extracted the text with a small stdlib script (scratchpad, not part of the repo). **The figures (architecture diagrams, Fig. 2–5) could not be rendered; the architecture was taken from the text and tables only.** Please confirm that Fig. 2–5 do not contradict the text below.

## 1. Differences: paper LightECGNet (text) vs repository LightECGNetV2 (`src/ecg_concept/lightecgnetv2.py`)
| # | Component | Paper text | LightECGNetV2 (recovered from the original notebook; strict-loads all 4 checkpoints) | Note |
|---|---|---|---|---|
| 1 | Input | (B,12,5000) | same | match |
| 2 | Lead attention | GAP over time → FC bottleneck 6 → sigmoid weights on the raw input; **144 trainable parameters** | Linear(12→6)+ReLU+Linear(6→12)+sigmoid, **with biases = 162 params** | 144 = weights only (2×72). Either bias-free or the paper counts weights only [U]; hidden activation not stated in the paper |
| 3 | Stem | 3 parallel DS branches, kernels 3/7/15, 16/16/32 channels, concat 64, 1×1 conv + PReLU, avg-pool 2 | same (each branch dw+pw+BN+ReLU; proj 1×1+BN+PReLU) | BN placement not in paper |
| 4 | TCN stages | 3 stages, 64/128/256 ch, 6 blocks, dilations (1,2),(4,8),(1,2), avg-pool between stages 2500→1250→625 | same | match |
| 5 | TCN block | "two dilated Conv1d layers, BN+ReLU after the convolutions, SE, dropout, residual" | **one dilated depthwise conv (k=13,13,9) + BN + ReLU, then a 1×1 pointwise conv + BN + ReLU**, SE(r=8, min 4), dropout, residual (1×1+BN skip when channels change), final ReLU | the second layer in V2 is not dilated; kernel sizes and skip form are not in the paper [U] |
| 6 | SE | reduction 8 | reduction 8 (hidden ≥4) | match |
| 7 | Dropout | head 0.5; block rates not given | head **0.4**; blocks 0.1/0.2/0.3 | rates [U] except head |
| 8 | Pooling / head | GAP+GMP → 512 → 256 (ReLU, dropout) → 45 | concat[max\|mean] → 512 → 256 → 45 | match (order irrelevant) |
| 9 | Parameters | not stated directly; "≈10× fewer than reference", "≈32× fewer than 22 M-parameter SENet/ResNet-34", "2.7 MB" ⇒ **≈0.68 M** | **376,108 (1.50 MB fp32)** | 1.8× discrepancy, unexplained [U]. A variant with two DS layers per block gives 558,124 (my count), neither reaches 0.68 M |
| 10 | Rare-class fine-tuning, ensemble | Phase 1.5 fine-tune; 3-fold ensemble | only `ft_fold0` is fine-tuned in the repo; `p1_fold0/1/2` are the 3 CV models | repo has no fine-tuned fold 1/2 [V] |
| 11 | Model selection | macro-AUROC every 5 epochs, patience 15, ≤120 epochs | notebook: best epoch by **validation macro-F1** with per-class thresholds; patience 15 (fold 0) / 10, 120 (fold 0) / 80 epochs | recipe mismatch [V from handover §8.1] |
| 12 | Split | "stratified 9-fold CV, 3-fold ensemble on held-out test" | `RandomState(42).permutation` → 10 equal index folds, **not stratified**; folds 0–8 CV, fold 9 test | the 13,344 pooled-calibration samples = 3 × 4,448 = our folds 0–2 ⇒ the paper's "released implementation" is very likely this notebook [I] |
| 13 | TTA | 5 passes (not specified) | none in the repo inference code | [V] |
| 14 | Accuracy 0.977 | macro-average of per-class binary accuracy | not the exact-match accuracy; inflated by true negatives | not usable for model selection |

Existing measured evidence on the historically inspected test fold (`folds[9]`, 4,447): `ft_fold0` macro-AUROC 0.9628, 3-model ensemble 0.9626, macro-F1 0.385 vs the paper's 0.963 / 0.395 (handover §9.4) [V]. **The AUROC claim is already reproduced with the released checkpoints; what has never been reproduced is training the architecture and recipe from scratch.**

## 2. Parameter counts and tensor shapes (V2 measured)
Total 376,108 trainable: lead attention 162; stem 260 + 308 + 628 + 4,225; TCN 6,280 + 6,280 + 22,096 + 22,800 + 84,640 + 85,536; head 142,893. Shapes (B=2): stem output (B,64,2500) → after stage 1 (B,64,1250) → stage 2 (B,128,625) → stage 3 feature map (B,256,625) → pooled (B,512) → hidden (B,256) → logits (B,45) [V]. Proposed dual-head: shared backbone 233,215 parameters + heads of 131,328 + 257·C parameters (142,893 for C=45, 132,613 for C=5) — to be asserted by shape tests.

## 3. Dataset readiness
**PhysioNet (45 classes)** [V from manifests]: 45,152 records, 44,476 with ≥1 class. Prospective roles (Step 3.7): train 35,568, validation 4,437, test_reserved 4,447, excluded 24. Label cardinality in train: 1 label 22,247; 2: 8,109; 3: 3,345; 4+: 1,867. Lowest train support: VPE 9, 2AVB1 25, AVRT 25, RAH 29, VEB 42, PRIE 46. Validation roles have 0 positives for AVRT and ≤9 positives for 11 classes; test_reserved has ≤9 positives for 12 classes — per-class AUROC/F1 there are very noisy and macro-averages are dominated by that noise. No patient IDs.
**PTB-XL v1.0.3, superclass benchmark** [V from manifests]: 21,799 records / 18,869 patients; train (folds 1–8) 17,418 records, 16,289 eligible (1,129 without a positive diagnostic superclass); validation (fold 9) 2,183 / 2,034 eligible; fold 10: 2,198 / 2,050 eligible (labels exist in the manifest; waveforms untouched). Train support: NORM 7,542, MI 3,307, STTC 4,062, CD 3,900, HYP 1,820; mean label cardinality ≈1.3. Patient overlap between train/val/test = 0 [V]; 1,883 of 16,965 train+val patients have >1 record (max 10) ⇒ record-level bootstrap/early stopping on shuffled records would leak patients; patient-grouped inner CV is required. No exact-duplicate `.dat` files among the 19,601 train+val records [V]; fold-10 hashes were deliberately not computed (the manifest does not contain them).
**Hardware** [V]: GTX 1650, **4 GB**, 6 cores / 12 threads, 7.7 GB RAM (≈2.6 GB free), 39 GB free disk, PyTorch 2.12 + CUDA 12.6; the paper used an RTX 4060 8 GB.

## 4. Preprocessing compatibility
- **Equivalent acquisition format** [V]: both datasets are 10 s, 500 Hz, 12 leads, 5,000 samples, int16, **1000 counts/mV**, baseline 0. Empirically (300 random train records each): median per-lead SD 154 counts in both; per-lead medians similar (PTB-XL I–V6 126, 134, 109, 119, 95, 103, 157, 242, 239, 231, 210, 176; PhysioNet 112, 136, 93, 114, 76, 103, 149, 248, 282, 293, 266, 214); peak |x| median ≈ 950–980. Flat leads: PhysioNet 39/3,600 (1.1 %), PTB-XL 1/3,600. So amplitude scales are comparable, but **device, population, filtering history and label process differ** (Chinese hospitals, GE-MUSE-type vs German Schiller records; labels from different pipelines).
- **Current V2 pipeline** (`preprocessing.py`): pad/crop 5000 → per-lead 0.5–40 Hz Butterworth(3) filtfilt with **int16 truncation** → per-lead z-score. It was already applied unchanged to PTB-XL (variant A_exact selected). Implications: (a) per-lead z-score discards absolute voltage and between-lead amplitude ratios (relevant to voltage-based findings such as LVH and to amplitude interventions in XAI); (b) flat leads become all-zero channels; (c) the int16 truncation is a 1-count artefact kept only for compatibility.
- **Proposal [P]:** one common pipeline for all new models = the exact V2 pipeline (bit-identical test against `preprocess`), with documented ablations later (float path, global per-record scaling, lead-dropout on flat leads). Cache preprocessed signals as float16 memory-mapped arrays (z-scored values have ~3 digits of precision needs; verify max abs diff vs float32 < 1e-3 before use).
- **100 Hz PTB-XL:** not recommended. The recovered kernels (3–15, dilations up to 8, 13-tap TCN) are tuned for 500 Hz; up- or down-sampling changes the receptive field in milliseconds and would confound dataset effects. Only a controlled ablation if speed becomes critical.
- Lead order: PhysioNet positional I…V6; PTB-XL reordered by name (verified by the existing reader) [V].

## 5. Label compatibility and mapping options
The two label spaces are not equivalent: PhysioNet = 45 multi-label **rhythm + morphology** statements; PTB-XL = 5 **diagnostic superclasses** (likelihood ≥50, unknowns negative). Shared concepts only by analogy: conduction (1AVB/2AVB/3AVB/AVB, RBBB, LFBBB, IVB, WPW ↔ CD); hypertrophy (LVH, RVH, RAH ↔ HYP); ST/T (STTC, TWC, TWO, STE, STDD, STTU ↔ STTC); infarct-like (MISW — which merges MI/MIBW/MIFW/MILW in the notebook mapping — and AQW ↔ MI); NORM has no PhysioNet analogue (SR is a rhythm label). Options: **(1) separate heads only (recommended)**; (2) evaluation-time weak mappings (e.g. "any CD-like PhysioNet logit") used only for cross-dataset explanation consistency, never for training; (3) a joint taxonomy — rejected (semantic mismatch, label noise). Any mapping must be fixed before looking at results and reviewed by a cardiology-literate reviewer.

## 6. Historical split exposure and leakage risks
| Item | Status |
|---|---|
| PhysioNet `folds[9]` (test_reserved, 4,447) | **Exposed**: full evaluation in Step 3.5; backbone choice `ft_fold0` over `p1_fold0` partly test-informed. Not an untouched test set. |
| PhysioNet fold 0 (validation, 4,437) | Used by the original training for epoch/threshold/weak-class selection; later by all heads' early stopping ⇒ exposed |
| PhysioNet folds 1–8 (train 35,568) | Used for training the existing weights |
| Duplicates | 87 exact byte-identical pairs, 17 cross the train/test boundary, 43 within train (32 label-conflicting); manifests exist; effect on headline metrics ≤0.001 |
| Patient overlap (PhysioNet) | Cannot be assessed; random record-level folds ⇒ optimistic estimates likely |
| PTB-XL fold 9 | Used for baseline head/fine-tune selection, concept experiments, thresholds ⇒ **exposed validation reference only** |
| PTB-XL fold 10 | Labels in manifest, waveform guard in code (in-process only; persistent release log not implemented), 0 waveforms read |
| Cross-dataset | Different institutions; exact overlap impossible, near-duplicate check not done; Challenge-2021 copy in `data/raw/ptb-xl` is a PTB-XL derivative and must never be mixed in |
| Existing checkpoints as pretraining | `lightv2_ft_fold0.pt` saw PhysioNet folds 1–8; if reused for PTB-XL work, PTB-XL conclusions are unaffected by PhysioNet test exposure, but PhysioNet test numbers of any derived model inherit the exposure |

## 7. Feasible clean validation protocols [P]
**PTB-XL (Exp B/C/D):** inner patient-grouped 4-fold CV over folds 1–8 (use the official `strat_fold` split as groups: folds {1,2},{3,4},{5,6},{7,8} held out in turn) for all hyperparameter, augmentation, loss-weight, threshold and early-stopping decisions; fold 9 is reported once per frozen finalist as a labelled "historically exposed reference"; fold 10 untouched until a frozen protocol, a persistent one-shot release log and your explicit approval. Patient-clustered bootstrap (B=5000, seed fixed) for all PTB-XL intervals (needs implementation and a unit test). Thresholds from out-of-fold predictions of the training CV only.
**PhysioNet (Exp A/C/D):** two explicit tracks.
- **A1 historical reproduction:** the paper's protocol (3 CV models validated on folds 0,1,2, 3-model ensemble, pooled-validation thresholds, evaluation on `folds[9]`) with fresh weights and a recipe selected before looking at `folds[9]`; reported as "reproduction under a historically exposed split".
- **A2 duplicate-aware robustness:** training role 35,568 only, inner duplicate-group-aware 3-fold CV for all decisions, then validation role (4,437) and test_reserved (4,447) reported as exposed references with the Step 3.7 cohorts B/C sensitivity.
There is **no untouched PhysioNet test data**; all PhysioNet claims are "estimates under a non-patient-aware, historically exposed protocol".
**Both:** hyperparameter search budget declared in advance (≤20 configurations per dataset on the inner folds), ≥3 seeds for finalists (5 if time allows), mean ± SD plus paired patient/record bootstrap on the same predictions; models compared with identical metric code (macro/micro AUROC & AP, per-class with support, macro/micro-F1 at inner-CV thresholds, sensitivity/specificity, ECE and calibration slope per class, classes with <10 positives flagged and excluded from macro-AUROC, not set to zero).

## 8. Proposed implementations [P] (all new files; V2 and its checkpoints stay immutable)
- `src/ecg_concept/lightecgnet_paper.py`: configurable class `LightECGNet(num_classes, la_bias, head_dropout, block_variant ∈ {v2, two_ds, two_dilated_std}, kernels, dropouts, dataset_norm)`; default = V2-equivalent; a unit test loads `lightv2_ft_fold0.pt` into the `v2` variant and asserts bit-identical logits to `LightECGNetV2`. Variant list documents every deviation from §1.
- Dual-head model: shared backbone (lead attention → stem → TCN → GAP+GMP) + `Head45`, `Head5`; options: dataset-specific BatchNorm (DSBN) in stem/TCN, dataset-specific lead attention, partial sharing (stage 3 split). Eval-mode forward must equal the single-head model when the other head is ignored.
- Training: `scripts/train_*.py` with config JSON, seeds, checkpoint SHA-256 logging, GPU memory/runtime logging; masked BCE per dataset (loss only on the batch's own dataset; `pos_weight` per head from the training role only, cap 50); samplers: alternating, 50/50 balanced, √-size; loss weighting: fixed equal per-dataset mean loss (default), optional uncertainty weighting; gradient-conflict diagnostics (cosine similarity of the two tasks' gradients on shared parameters, logged every N steps; PCGrad only as a labelled ablation).
- Hardware-driven deviations (declared): batch 256 does not fit 4 GB (measured peak ≈4.3 GB) ⇒ micro-batch 64 with 4× accumulation (BatchNorm statistics differ from true batch 256); bf16 is emulated on this GPU (measured 55 rec/s) ⇒ fp32 or fp16-autocast + GradScaler.

## 9. Experiments A–D [P]
- **A** (PhysioNet from scratch): A0 = V2 architecture + paper recipe as in the notebook (mixup 0.2, aug, AdamW 1e-3/1e-4, warmup 5, cosine → 1e-5, BCE pos_weight ≤50, grad-clip 1); A1 = same with the paper's stated selection (macro-AUROC, patience 15); A2 = paper-described variants (bias-free LA, head dropout 0.5, two-DS block). Compare each against the paper's reported values *and* the existing checkpoints on identical data. Rare-class fine-tuning, 3-fold ensemble, validation-calibrated thresholds and 5-pass TTA are added stepwise (ablation table).
- **B** (PTB-XL, 5 classes): same backbone, new 512→256→5 head; from scratch vs paper-faithful vs initialised from `ft_fold0` (already measured 0.9254 fine-tuned / 0.9250 frozen MLP on fold 9 — this is the bar to beat); augmentations tuned on inner folds only.
- **C** (joint): masked dual-head training with the sampler/loss/normalisation/sharing grid in §8 (≤8 cells, 1 seed screen on inner folds, finalists 3 seeds); report negative transfer per head and per class (low-support classes).
- **D** (staged): PhysioNet→PTB-XL (new weights from A, plus the frozen `ft_fold0` as a reference — largely done already), PTB-XL→PhysioNet, vs single-dataset (B/A) and joint (C); freezing schedules (frozen / head-only / discriminative LR). Pretraining counted as helpful only if it beats scratch on **inner folds and fold 9 with non-overlapping paired CIs across seeds**.

## 10. Expected compute and storage (estimates; only a micro-benchmark was run) 
Measured micro-benchmark (random data, V2 architecture): fp32 batch 64 ≈ 318 records/s, 2.1 GB peak; fp16-autocast batch 128 ≈ 211 rec/s, 2.2 GB; batch 256 peaks ≈4.3 GB; bf16 55 rec/s. Data-loading/augmentation on 6 CPU cores will lower these. ⇒ PhysioNet epoch (35.6 k) ≈ 2–3 min, PTB-XL epoch (16.3 k) ≈ 1–1.5 min, joint epoch ≈ 3–4 min. Rough totals: one PhysioNet model 60–120 epochs ≈ 2–6 h (3-fold + fine-tune ≈ 8–20 h); one PTB-XL run 40–100 epochs ≈ 1–2 h; Exp B with CV + 3 seeds ≈ 15–30 h; C ≈ 30–60 h; D ≈ 20–40 h. **Total ≈ 80–160 GPU-hours** for the full programme on this machine, i.e. several weeks; a staged programme with early stopping on the screening results is necessary. Storage: float16 caches PhysioNet 44,476 × 12 × 5000 × 2 B ≈ 5.3 GB; PTB-XL eligible train+val ≈ 2.4 GB (+0.25 GB fold 10 later); checkpoints ~1.5–3 MB each (a few hundred ⇒ <2 GB); logs/OOF predictions <2 GB; total ≈ 10–15 GB of the 39 GB free. RAM 7.7 GB ⇒ memory-mapped caches, small worker count.

## 11. Recommended order [P]
0. (no GPU) Freeze the written protocol (this document → `docs/` version after your edits), register the primary metrics, implement tests: architecture/param/shape; bit-exact preprocessing equivalence to `preprocess`; masked-loss; split and leakage (patient overlap, duplicate groups, fold-10 persistent one-shot log); patient-clustered bootstrap; checkpoint immutability.
1. Build float16 caches + dataset-agnostic data loader; run a throughput benchmark with real data and augmentation (decides fp32 vs fp16).
2. **B pilot** (cheapest, validates the pipeline against the 0.925 bar).
3. **A single-fold reproduction** (fold 0 recipe) → decides how paper-faithful the recipe is before spending on 3 folds.
4. D (both directions) and C only after A/B baselines exist, so every gain is measured against like-for-like single-dataset models.
5. In parallel from step 2 on: the XAI battery on every finalist (below).
6. Fold-10 PTB-XL evaluation: one frozen candidate, after your explicit approval.

## 12. Explainability plan [P] and risks
Battery per model (reuses existing scripts, parameterised by model/thresholds): IG completeness/convergence, deletion/insertion faithfulness, attribution stability, lead attention/importance (incl. dual-head and DSBN variants), beat/region attribution, RR and atrial interventions as regression tests against the reference V2, and new broader controlled interventions on QRS morphology (width/amplitude scaling), ST-segment shift (J-point), T-wave amplitude/inversion, conduction-like widening, multi-lead interactions (lead-group ablations / Shapley over lead groups), and cross-head consistency for mapped classes. Expected directions of change are declared before running; findings are model-sensitivity statements, not physiological validation. Ensembles/TTA need an explicit explanation protocol (per-member attribution vs attribution of the averaged probability); faithfulness is first evaluated on single models.
**Main risks:**
1. Small headroom on PTB-XL superclasses (frozen 0.925 vs fine-tuned 0.9254); gains may be within seed noise. Published superclass AUROCs are around 0.92–0.93 (from memory, **not verified here**).
2. 4 GB GPU: no batch 256, no native bf16 ⇒ recipe deviations that could lower or change performance; compute budget forces few seeds.
3. PhysioNet estimates are optimistic (duplicates, no patient IDs, exposed folds); rare-class metrics on 3–9 positives are nearly meaningless.
4. Paper ambiguities (§1 items 2, 5, 7, 9) mean "paper-faithful" can only be "as described / as recovered"; the 0.68 M vs 0.376 M parameter mismatch is unresolved.
5. Shared backbones can suffer negative transfer or dataset shortcut features (device/hospital signatures); DSBN and dataset-balanced sampling may help but change the explanation pathway.
6. Selection pressure on fold 9 and the PhysioNet validation roles is already high; any further use must be counted and declared.
7. Better AUROC does not imply better attributions; mixup, lead-dropout, ensembling, TTA and dataset-specific BN can each degrade IG completeness/faithfulness or stability, so they are evaluated as factors, not assumed neutral.
8. Per-lead z-scoring removes absolute amplitude cues, limiting explanations of voltage-dependent classes.
9. Tooling: no PDF/pandas/sklearn libraries; metrics, bootstrap and PDF extraction are home-grown and need tests.

## 13. Decisions needed from you before any training
1. Confirm `ECG_paper.pdf` (new version) is the intended paper and check Fig. 2–5 against §1.
2. Protocol: approve PhysioNet tracks A1 (historical reproduction) and A2; and the PTB-XL inner patient-grouped CV.
3. Compute: accept staged programme / reduced seeds; fp32 or fp16-autocast with micro-batch 64×4.
4. Primary selection metric (proposal: inner-CV macro-AUROC with rare-class recall and ECE guards) and the hyperparameter budget.
5. Whether the 0.68 M parameter claim should be pursued (requires guessing block details) or documented as unresolved.
