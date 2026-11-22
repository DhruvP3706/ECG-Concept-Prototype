# Research protocol — Phase 1 (frozen before any training)

Project: *Explainable AI: Understanding and Validating Deep Learning Decisions in 12-Lead ECG Diagnosis*.
Objective: improve 12-lead ECG diagnosis using PhysioNet (45 classes) and PTB-XL (5 superclasses) **and** develop experimentally validated explanations. Explanation reliability is a co-primary outcome: no model is promoted on AUROC alone.
Status: written 2026-11-20, before Experiment B. Any change after approval must be listed in section 12 (this file's SHA-256 is part of the fold-10 release check, so edits invalidate an earlier approval).

## 1. Immutable references
`docs/REFERENCE_CHECKPOINTS.json` registers SHA-256 hashes of the reference checkpoints (`lightv2_ft_fold0.pt` = `7779719e…f79f72`, `lightv2_p1_fold0/1/2.pt`, thresholds), `lightecgnetv2.py`, `preprocessing.py`, `labels.py`, `inference.py`, the PTB-XL benchmark config/manifests and the Step 3.7 manifests. `provenance.verify_registry()` is called by every training run and by the tests; `assert_not_reference_output` forbids writing into `models/pretrained/`. New code lives in new files; previous experiment outputs are never overwritten (new run directories only).

## 2. Data roles and exposure (no role is called "untouched" unless it is)
| Data | Role in Phase 1 | Exposure |
|---|---|---|
| PhysioNet training role (35,568) | development (A2) | used to train the reference weights |
| PhysioNet original folds 0–8 (40,029) | development for A1 (paper protocol) | folds 0–2 historically used for selection |
| PhysioNet validation role (4,437), `folds[9]` test (4,447) | **report-only**, built only with `--allow-exposed` | exposed; never used to select models or tune hyperparameters |
| PTB-XL folds 1–8 (16,289 eligible) | development, patient-grouped inner CV | not exposed beyond earlier baselines |
| PTB-XL fold 9 (2,034 eligible) | exposed reference, **evaluated once per frozen finalist** | used for baseline/head selection earlier |
| PTB-XL fold 10 (2,050 eligible) | **locked**; no waveform access | labels in manifest only |
Patient IDs do not exist for PhysioNet: its estimates come from non-patient-aware, duplicate-aware splits and are optimistic by an unknown amount. No untouched PhysioNet test data exists.

## 3. PhysioNet protocols
**A1 — paper-described reproduction (fresh weights).** Original folds (`RandomState(42)` permutation, 10 index folds, as the notebook); CV models k = 0, 1, 2 train on folds {0..8}∖{k} and use fold k for early stopping and thresholds exactly as the paper's protocol does; fold 9 is the historically exposed test fold and is evaluated **only after the recipe is frozen**, once, labelled "historically exposed". A1 is a *reproduction*: the validation fold k selects epochs (this is the paper's own design and the one deviation from the "no historical split for selection" rule), but no hyperparameter is tuned against folds 0–2 or fold 9. Reported against the paper's 0.963 / 0.977 with the architecture caveats of `PAPER_VS_V2_ARCHITECTURE.md`.
**A2 — training-role-only development.** Only the 35,568 training-role records; inner 3-fold CV with duplicate groups kept together (`splits_v3.group_kfold`, seed 20261120, rarest-class-first balancing). All selection and hyperparameter decisions use the inner folds. The historical validation role and `folds[9]` are evaluated once, after freezing, as exposed references (cohorts B/C of Step 3.7 as sensitivity).
Historical validation/test results are never used to choose among models or hyperparameters in either track.

## 4. PTB-XL protocol (Experiment B)
Inner 4-fold CV over folds 1–8 with groups {1,2},{3,4},{5,6},{7,8} (patient-disjoint; verified by `splits_v3.ptbxl_inner_folds` and its tests). Labels: superclass vocabulary NORM, MI, STTC, CD, HYP, likelihood ≥ 50, unknown never positive; 1,129 records without a positive label excluded from supervised training (listed in the manifest). All development decisions use inner-CV macro-AUROC. Fold 9 is evaluated once per **frozen** finalist (never to choose among competitors). Fold 10: section 9. Patient-clustered bootstrap (B = 5000, seed 20261120) for every PTB-XL interval.

## 5. Experiment C (joint) and 6. Experiment D (staged)
Separate heads and taxonomies; no cross-dataset label merging; any cross-dataset concept analogy is used only for evaluation-time explanation consistency. C: masked per-dataset loss (`losses_v3`), balanced/alternating/√-size samplers, equal per-dataset loss weighting by default, gradient-conflict diagnostics, DSBN/partial sharing as declared grid cells — driver not yet implemented. D: staged fine-tuning with `init.type` = `run_checkpoint`/`v2_checkpoint`; pretraining is "helpful" only if it beats scratch on inner-CV with non-overlapping paired cluster-bootstrap/seed intervals.

## 7. Metrics (fixed definitions: `src/ecg_concept/eval_metrics.py`)
Primary selection metric: **inner-CV macro-AUROC** (mean over classes with ≥1 positive and ≥1 negative; undefined classes reported, never zero). Also reported: macro-AUPRC, per-class AUROC/AUPRC with support, macro-/micro-F1, macro sensitivity/specificity, **rare-class sensitivity** (training support < 100, ≥ 1 positive), ECE (15 equal-width bins, per class then macro), Brier, parameter count, runtime, GPU memory. Thresholds are never fitted on the evaluation set: they come from inner-CV out-of-fold predictions. A candidate is not promoted if it lowers rare-class sensitivity or calibration materially (pre-declared guard: macro-AUROC gain must not come with ECE worse by > 0.02 or rare-class sensitivity lower by > 0.05, both versus the comparator at the same threshold policy). Seeds: 3 for finalists (5 if time), mean ± SD, paired cluster bootstrap on identical predictions.

## 8. Hyperparameter budget
No hyperparameter search has been run. Planned budget: ≤ 20 configurations per dataset, evaluated on inner folds only, declared before running (list written to `results/phase1_runs/<exp>/search_plan.json` before the first job). Configs in `experiments/phase1/` use the paper's stated values and recovered V2 settings without tuning.

## 9. Fold-10 protection (persistent guard)
`src/ecg_concept/ptbxl_guard.py`: (i) the 2,198 fold-10 record keys are read from the frozen manifest; (ii) `install_hooks()` patches `open`, `io.open`, `os.open` and `numpy.fromfile` in the process, and every training/cache script installs it; (iii) every decision is appended to a **hash-chained persistent log** (`results/ptbxl_fold10_access_log.jsonl`, `verify_log()` detects edits); (iv) release requires a release file (approver, approval text, SHA-256 of this protocol, config hash, token hash), the matching environment token and a one-shot session event; a second session is refused even from a new process. No cache builder can build fold 10 (`build_cache.py` raises). **Limitations:** the guard binds only Python processes that install it; a deleted log, Explorer, PowerShell, or another script bypasses it; reading can also be done via `numpy.memmap`/C extensions not patched. Safer, **not applied** (needs approval): move the fold-10 `.dat/.hea` files to a separate directory with an NTFS deny-read ACL, or keep them in an encrypted archive/removable drive until release.

## 10. Preprocessing and deviations register
Reference preprocessing (`preprocessing.preprocess`: pad/crop 5000, zero-phase Butterworth 0.5–40 Hz with int16 truncation, per-lead z-score) is used unchanged for all datasets; caches store it as float16 (round-trip error < 5e-3, tested). Deviations from the paper/V2 recorded in every config: gradient accumulation 64×4 instead of batch 256 (BatchNorm statistics differ; measured logit difference vs true batch 256 on synthetic data: mean 0.0126, max 0.072 on logits of SD 0.745), FP32 instead of bf16, recovered V2 kernels/dropouts, details of augmentation not in the paper, Phase 1.5/ensemble/TTA as separate later stages, A1 selecting epochs on the paper's validation fold.

## 11. Explainability requirements for finalists
For every promoted model (single model first; ensemble/TTA protocol declared separately): IG completeness/convergence, deletion/insertion faithfulness, attribution stability, lead and beat/region attribution, RR/atrial regression tests against the V2 reference, and broader controlled interventions (QRS morphology, ST/T characteristics, conduction-like changes, multi-lead interactions, cross-dataset consistency). Expected response directions are declared before running. Findings are model-sensitivity statements, not physiological or clinical validation.

## 12. Change log
2026-11-20 initial Phase 1 version.
