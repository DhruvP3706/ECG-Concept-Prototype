# Combined Step 4.4–6.1 — first concept → anchored-prototype experiment (validation only)

**Read this first.** Concept targets are *automatically derived 10-second RR statistics* from the frozen Step 4.3 extractor, not expert-validated measurements; "concept prediction error" below is agreement with extractor targets. All results are on the **validation role only** (4,437 records); no test record was read, processed or scored. The validation set was also used for early stopping of every trained model, so validation numbers are mildly optimistic, equally so across models. Three seeds give a spread over initialisations, **not** a patient-level confidence interval, and no statistical-significance claim is made. This is an initial result for a minimal four-concept set, not evidence of publication readiness.

## 1. Headline answers

1. **Does the concept predictor beat a constant predictor? Yes, clearly (agreement with extractor targets).** Validation R² vs the train-only constant: rate 0.980 (MAE 2.74 bpm vs 24.20), RR CV 0.905, NRMSSD 0.870, median-abs-successive-difference 0.680. The irregularity concepts are predicted less well than rate, and all concepts generalise somewhat worse on validation than on training.
2. **Cost of the four-concept bottleneck: large.** Versus the same-embedding linear head (B), concept-only (C2) loses macro-AUROC -0.1663 (sd across seeds 0.0056), micro-AUROC -0.0502 (sd across seeds 0.0002), micro-AP -0.2705 (sd across seeds 0.0011); versus the original head (A) macro-AUROC -0.2028 (sd across seeds 0.0037). Rhythm/rate-defined classes keep most of their discrimination; morphology/conduction/axis classes fall to near chance (§5).
3. **Prototype layer vs concept-only head: a small, consistent gain, far short of the bottleneck gap.** D − C2: macro-AUROC +0.0093 (sd across seeds 0.0032), micro-AP +0.0651 (sd across seeds 0.0025), micro-F1 +0.0315 (sd across seeds 0.0023); positive for macro-AUROC in all three seeds. The largest single contribution is sinus rhythm (SR), a non-monotonic band of rate between bradycardia and tachycardia that a linear head cannot express. D is still far below the embedding head (D − B macro-AUROC -0.1570 (sd across seeds 0.0040)).
4. **Are masks predictive enough to confound? Not in aggregate.** The mask-only control reaches macro-AUROC 0.5190 ± 0.0013 (chance = 0.5; its micro-AUROC of 0.8861 ± 0.0003 reflects label prevalence) and every per-class AUROC is ≈ 0.5 (max 0.62 on 8 positives). Coverage does vary by label (e.g., ST and SA ≈ 92% accepted vs 96.6% overall), so coverage should keep being reported; but masks are not a usable shortcut here. Small subgroups (149 non-accepted validation records) are too few to conclude more.
5. **Prototypes distinct and explanations faithful? Computationally faithful, partly distinct, weights hard to read.** Logits equal bias + summed prototype contributions to ≤ 7.6e-6; all 16 anchors are distinct real training records in all seeds; no pairs are closer than 0.25 std in concept space and no similarity columns correlate > 0.95; every prototype is the nearest one for some validation records. But signed weights are very large and offsetting (the head had not converged at the 100-epoch cap), 14 of 48 prototype-seeds change macro-AUROC by < 0.001 when ablated, and each seed anchors a tiny outlier cluster of 3–6 training records (see §6). Computational faithfulness is not clinical validity, and four timing concepts cannot explain morphology diagnoses.
6. **Single best-supported next improvement:** add concepts that carry what timing cannot — start with **QRS duration and frontal axis** (the deferred delineation-based concepts, validated first against the public/expert references in the Step 4.2 plan). The bottleneck loses most on conduction/axis/morphology classes (§5), while head or prototype changes recovered only ~0.01 macro-AUROC. Do not add an embedding bypass to close the gap; that would remove the interpretability claim.

## 2. What was done

- Fingerprints verified before use: extractor sources and parameters equal the Step 4.3 frozen config (parameters `4f5e41ea05c53bd3…`), role and fold manifests unchanged, `lightv2_ft_fold0.pt` SHA-256 `7779719e9480a018…`. **No mismatch.**
- Records: only roles `train` (35,568) and `validation` (4,437) were admitted; `test_reserved` (4,447) and `excluded` (24) were rejected explicitly and never read. All artifacts are joined by record ID.
- Extraction: frozen Step 4.3 extractor over all 40,005 records (10 worker processes, ~8 min); 0 read failures. Embeddings: exact recovered backbone preprocessing, frozen `ft_fold0` in eval mode, pooled 512-d vector and original logits (~7 min on the GTX 1650); backbone state hash before/after identical (True); 0 embedding failures. Sanity check: the cached original-head validation macro-AUROC is 0.9594, equal to the Step 3.5 audit value for the same fold.
- Experiment configuration (seeds 42/43/44, four targets, hyperparameters, populations, thresholds) was written and hashed **before fitting** (`experiment_config.json`, SHA-256 `1e02d1ffcfa2fe4c…`); the training stage verified it, the merged cache hash and the code hashes before running.
- Common population: all 35,568 train and 4,437 validation records are embedding-eligible. Fully-accepted subsets (all four targets accepted): 34,341 train, 4,288 validation (validation records not fully accepted: 149).

**Correction of an earlier reporting claim.** The Step 4.3 report implied that a 0 ms *median* timing difference meant all matched detector beats had identical times. Actual differences between matched detector beats: 2,685 of 35,568 training records (7.5%) and 358 of 4,437 validation records have at least one matched pair with a nonzero difference; the maximum is 47 ms (train) / 42 ms (validation); the 99th percentile of the per-record maximum is 7 ms. Most matched times are identical, not all. No extractor bug was found; no detector parameter was changed.

### Extraction coverage (states are detector-consensus acceptance, not validity)

| Role | Group | ACCEPTED | UNCERTAIN | UNAVAILABLE |
|---|---|---|---|---|
| train | rate | 34,343 (96.6%) | 514 (1.4%) | 711 (2.0%) |
| train | irregularity | 34,341 (96.6%) | 511 (1.4%) | 716 (2.0%) |
| validation | rate | 4,288 (96.6%) | 66 (1.5%) | 83 (1.9%) |
| validation | irregularity | 4,288 (96.6%) | 66 (1.5%) | 83 (1.9%) |

Reasons (record counts, train / validation): DETECTOR_DISAGREEMENT 521/67; LOW_LEAD_SUPPORT 19/3; LOW_USABLE_LEADS 12/1; CLIPPING_SUSPECT 181/29; FLAT_LEAD 1,107/138; INSUFFICIENT_USABLE_LEADS 698/80; INSUFFICIENT_BEATS 13/3; INSUFFICIENT_BEATS_FOR_IRREGULARITY 5/0. Per-label coverage in `coverage_by_label.csv`; summaries in `coverage_summary.csv` and `failure_summary.json`. Within-training duplicate records remaining: 86 (43 exact pairs with 32 label-conflict flags; kept as-is, labels not merged or adjudicated; 1 validation record is in a duplicate group whose partner is outside validation/train roles; no train/validation duplicate-group overlap).

## 3. Declared models (one initial architecture each; no search)

| Model | Specification |
|---|---|
| Common | AdamW (weight decay 0.0001), batch 256, ≤ 100 epochs, early-stopping patience 10 on validation loss, CPU float32; threshold 0.5, no threshold optimisation or calibration |
| A_orig_head | frozen ft_fold0 45-way head, sigmoid, threshold 0.5 (historical validation exposure: fold 0 was its validation fold) |
| B_embedding_linear | {"arch": "Linear(512,45)", "loss": "BCE", "lr": 0.001, "selection": "validation BCE"} |
| C_concept_predictor | {"arch": "Linear(512,128)-ReLU-Linear(128,4)", "loss": "masked per-concept standardized MSE (equal concept weights)", "lr": 0.001, "selection": "validation masked MSE; frozen afterwards"} |
| C2_concept_only | {"arch": "Linear(4,45) on frozen PREDICTED standardized concepts", "loss": "BCE", "lr": 0.01, "selection": "validation BCE"} |
| D_prototype | {"K": 16, "init": "deterministic k-means (farthest-point init) in predicted standardized concept space on train records with all 4 targets accepted; prototype = nearest actual training record to each centre (anchored, NOT learned)", "similarity": "exp(-squared_distance / tau)", "tau_rule": "median over ALL train embedding-eligible records of the squared distance to the nearest prototype, floor 1e-6", "head": "Linear(16,45) on similarities only (signed weights allowed)", "loss": "BCE", "lr": 0.01, "selection": "validation BCE", "concept_predictor_updated_by_diagnosis_loss": false} |
| CTRL_mask_only | {"arch": "Linear(19,45)", "features": "acceptance states, quality-lead counts, reason flags (no values, no beat counts)", "lr": 0.01, "role": "control only; never feeds a proposed model"} |
| CTRL_extracted_target | {"arch": "Linear(4,45) on standardized EXTRACTED accepted targets", "population": "fully accepted subset", "lr": 0.01, "role": "auxiliary baseline (not an oracle)"} |
| Scaling | per-dimension standardization, mean/std fitted on TRAIN embedding-eligible records only (std floor 1e-6); per-concept standardization of ACCEPTED train targets only (std floor 1e-6); predictions inverse-transformed for error reporting |

## 4. Concept prediction (agreement with extractor targets; mean over 3 seeds)

| Concept | Split | n accepted | MAE | const MAE | RMSE | const RMSE | R² vs const |
|---|---|---|---|---|---|---|---|
| rate_mean_bpm | train | 34,343 | 2.3253 | 23.7682 | 3.3862 | 29.2709 | 0.987 |
| rate_mean_bpm | validation | 4,288 | 2.7406 | 24.2007 | 4.2342 | 29.5877 | 0.980 |
| rr_cv | train | 34,341 | 0.0131 | 0.0683 | 0.0195 | 0.0867 | 0.950 |
| rr_cv | validation | 4,288 | 0.0171 | 0.0677 | 0.0265 | 0.0857 | 0.905 |
| rr_nrmssd | train | 34,341 | 0.0207 | 0.0977 | 0.0310 | 0.1257 | 0.939 |
| rr_nrmssd | validation | 4,288 | 0.0283 | 0.0972 | 0.0450 | 0.1250 | 0.870 |
| rr_mad_succ | train | 34,341 | 0.0224 | 0.0668 | 0.0348 | 0.0968 | 0.871 |
| rr_mad_succ | validation | 4,288 | 0.0317 | 0.0672 | 0.0584 | 0.1032 | 0.680 |

Units: bpm for rate; dimensionless for the RR statistics (inverse-transformed from standardized predictions). The constant predictor is the train-only mean of accepted targets.

## 5. Diagnosis on validation (n = 4,437; mean ± SD over 3 seeds; threshold 0.5)

| Model | macro-AUROC | micro-AUROC | macro-AP | micro-AP | macro-F1 (all 45, undefined→0) | macro-F1 (evaluable) | micro-F1 |
|---|---|---|---|---|---|---|---|
| A: frozen original ft_fold0 head (threshold 0.5) | 0.9594 | 0.9835 | 0.4300 | 0.7417 | 0.3676 | 0.3759 | 0.6084 |
| B: linear head on frozen 512-d embedding | 0.9229 ± 0.0036 | 0.9825 ± 0.0002 | 0.4102 ± 0.0016 | 0.8468 ± 0.0005 | 0.3428 ± 0.0037 | 0.3506 ± 0.0038 | 0.7598 ± 0.0023 |
| C2: linear head on 4 predicted concepts | 0.7566 ± 0.0037 | 0.9322 ± 0.0001 | 0.1337 ± 0.0025 | 0.5763 ± 0.0009 | 0.0684 ± 0.0012 | 0.0700 ± 0.0013 | 0.5062 ± 0.0013 |
| D: anchored-prototype head (K=16) on the same predicted concepts | 0.7660 ± 0.0046 | 0.9392 ± 0.0004 | 0.1348 ± 0.0015 | 0.6414 ± 0.0019 | 0.0741 ± 0.0021 | 0.0757 ± 0.0022 | 0.5377 ± 0.0016 |
| Control: mask-only linear head | 0.5190 ± 0.0013 | 0.8861 ± 0.0003 | 0.0421 ± 0.0001 | 0.2484 ± 0.0009 | 0.0092 ± 0.0005 | 0.0094 ± 0.0005 | 0.0207 ± 0.0030 |

A is deterministic (no seed variance) and **its fold-0 validation exposure is historical**: fold 0 was the original validation fold for epoch, threshold and weak-class selection, so A is not a clean comparator; its head was trained with class weights and mixup, whereas B–D use plain BCE, which is why B has much higher micro-F1/micro-AP at a fixed 0.5 threshold but lower macro-AUROC. Undefined classes: AVRT has 0 validation positives and is excluded from macro-AUROC/AP; macro-F1 is shown under both conventions.

### Paired changes on identical validation IDs (mean over seeds; sd is the spread across seeds, not a CI)

| Comparison | macro-AUROC | micro-AUROC | micro-AP | micro-F1 |
|---|---|---|---|---|
| B_embedding_linear − A_orig_head | -0.036 (sd across seeds 0.004) | -0.001 (sd across seeds 0.000) | +0.105 (sd across seeds 0.000) | +0.151 (sd across seeds 0.002) |
| C2_concept_only − B_embedding_linear | -0.166 (sd across seeds 0.006) | -0.050 (sd across seeds 0.000) | -0.270 (sd across seeds 0.001) | -0.254 (sd across seeds 0.003) |
| D_prototype − C2_concept_only | +0.009 (sd across seeds 0.003) | +0.007 (sd across seeds 0.000) | +0.065 (sd across seeds 0.003) | +0.031 (sd across seeds 0.002) |
| D_prototype − B_embedding_linear | -0.157 (sd across seeds 0.004) | -0.043 (sd across seeds 0.000) | -0.205 (sd across seeds 0.001) | -0.222 (sd across seeds 0.004) |
| CTRL_mask_only − C2_concept_only | -0.238 (sd across seeds 0.003) | -0.046 (sd across seeds 0.000) | -0.328 (sd across seeds 0.002) | -0.486 (sd across seeds 0.003) |

### Fully-accepted subset and the extracted-target baseline (n = 4,288 validation records; same IDs for all models)

| Model | macro-AUROC | micro-AUROC | micro-AP | micro-F1 |
|---|---|---|---|---|
| A: frozen original ft_fold0 head (threshold 0.5) | 0.9592 | 0.9834 | 0.7418 | 0.6079 |
| B: linear head on frozen 512-d embedding | 0.9249 ± 0.0036 | 0.9826 ± 0.0001 | 0.8473 ± 0.0005 | 0.7598 ± 0.0023 |
| C2: linear head on 4 predicted concepts | 0.7559 ± 0.0037 | 0.9319 ± 0.0001 | 0.5798 ± 0.0014 | 0.5056 ± 0.0010 |
| D: anchored-prototype head (K=16) on the same predicted concepts | 0.7680 ± 0.0049 | 0.9400 ± 0.0005 | 0.6468 ± 0.0022 | 0.5438 ± 0.0012 |
| Control: mask-only linear head | 0.5043 ± 0.0004 | 0.8845 ± 0.0002 | 0.2421 ± 0.0007 | 0.0037 ± 0.0000 |
| Auxiliary: linear head on EXTRACTED accepted concepts (extracted-target baseline, not an oracle) | 0.7482 ± 0.0005 | 0.9293 ± 0.0003 | 0.5677 ± 0.0012 | 0.4991 ± 0.0038 |

The extracted-target baseline is not better than the head on *predicted* concepts (extracted − C2 macro-AUROC -0.0077). Predicted concepts come from the embedding, so they may carry diagnosis-related information beyond the four extracted statistics (concept leakage), which weakens the claim that C2/D rely only on rate and RR irregularity.

Records without all four accepted concepts (diagnostic only; n = 149): A macro-AUROC 0.9688; B macro-AUROC 0.9459 ± 0.0034; C2 macro-AUROC 0.7415 ± 0.0032; D macro-AUROC 0.6667 ± 0.0087; Control macro-AUROC 0.7382 ± 0.0178. Concepts are predicted from the embedding for these records too; the sample is small.

### Per-class AUROC (validation; classes with ≥ 30 positives; full table in `validation_per_class.csv`)

| Class | n+ | B embedding | C2 concept-only | D prototype | mask-only | B − C2 |
|---|---|---|---|---|---|---|
| ALS | 146 | 0.980 | 0.580 | 0.581 | 0.508 | +0.400 |
| AQW | 97 | 0.958 | 0.576 | 0.629 | 0.520 | +0.382 |
| RBBB | 62 | 0.983 | 0.610 | 0.626 | 0.517 | +0.373 |
| STE | 94 | 0.930 | 0.563 | 0.599 | 0.513 | +0.367 |
| TWO | 304 | 0.952 | 0.612 | 0.662 | 0.513 | +0.340 |
| ARS | 84 | 0.988 | 0.681 | 0.682 | 0.511 | +0.306 |
| LVH | 61 | 0.944 | 0.642 | 0.717 | 0.524 | +0.303 |
| TWC | 731 | 0.910 | 0.608 | 0.652 | 0.506 | +0.302 |
| SR | 733 | 0.985 | 0.698 | 0.954 | 0.504 | +0.287 |
| 1AVB | 112 | 0.982 | 0.696 | 0.639 | 0.506 | +0.286 |
| LFBBB | 30 | 0.962 | 0.683 | 0.618 | 0.522 | +0.279 |
| STDD | 161 | 0.948 | 0.677 | 0.724 | 0.513 | +0.272 |
| LVQRSAL | 116 | 0.885 | 0.638 | 0.611 | 0.501 | +0.247 |
| QTIE | 37 | 0.873 | 0.691 | 0.755 | 0.519 | +0.182 |
| STTC | 103 | 0.867 | 0.696 | 0.717 | 0.521 | +0.171 |
| SA | 273 | 0.964 | 0.839 | 0.905 | 0.537 | +0.125 |
| ERV | 35 | 0.909 | 0.807 | 0.746 | 0.503 | +0.102 |
| AT | 32 | 0.957 | 0.860 | 0.833 | 0.527 | +0.097 |
| APB | 142 | 0.958 | 0.863 | 0.881 | 0.509 | +0.094 |
| ST | 743 | 0.998 | 0.944 | 0.981 | 0.540 | +0.054 |
| AF | 809 | 0.975 | 0.922 | 0.928 | 0.526 | +0.053 |
| IVB | 74 | 0.941 | 0.891 | 0.898 | 0.509 | +0.050 |
| AFIB | 178 | 0.928 | 0.905 | 0.905 | 0.519 | +0.023 |
| SB | 1715 | 0.999 | 0.985 | 0.984 | 0.523 | +0.014 |
| SVT | 58 | 0.997 | 0.991 | 0.981 | 0.528 | +0.006 |

Classes that keep discrimination from four timing concepts are rate/rhythm classes (SB, SVT, AFIB, AF, ST, SA, APB, VPB); the largest losses are conduction, axis, hypertrophy and morphology classes (ALS, 1AVB, RBBB, AQW, STE, TWO, TWC, LFBBB; SR is also lost by the linear concept head and largely recovered by the prototype layer). Support for several rare classes is ≤ 10, so their per-class values are noisy.

## 6. Prototype analysis

| Seed | tau | logit identity max abs diff | prototypes never nearest in val | largest usage share | min pairwise distance (std) | pairs closer than 0.25 | pairs with similarity corr > 0.95 | anchors distinct training IDs |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.1224 | 7.63e-06 | 0 | 25.7% | 0.50 | 0 | 0 | 16 (True) |
| 43 | 0.1234 | 7.63e-06 | 0 | 29.8% | 0.58 | 0 | 0 | 16 (True) |
| 44 | 0.1185 | 7.63e-06 | 0 | 24.5% | 0.46 | 0 | 0 | 16 (True) |

- **Functional dependence (ablation: set one prototype's similarity to 0, validation):** macro-AUROC change per prototype ranges -0.0277 to +0.0033 (median -0.0030); 14 of 48 prototype-seeds change it by < 0.001. The three largest effects per seed come from large regular-rhythm prototypes (predicted rate / RR CV; macro-AUROC change): seed 42: ≈79 bpm / CV 0.03 (-0.021), ≈110 bpm / CV 0.01 (-0.019), ≈56 bpm / CV 0.02 (-0.011); seed 43: ≈78 bpm / CV 0.02 (-0.028), ≈56 bpm / CV 0.02 (-0.023), ≈111 bpm / CV 0.01 (-0.020); seed 44: ≈79 bpm / CV 0.02 (-0.022), ≈111 bpm / CV 0.01 (-0.020), ≈57 bpm / CV 0.02 (-0.014). This is a computational dependence on a rate profile; prototypes were defined from concepts only (no labels), and this is not clinical validity.
- **Signed weights** are reported as supporting (+) or opposing (−) logit contributions in `prototype_summary.csv`. They are very large and offsetting (max |weight| 85, max |bias| 7) because the similarity features are highly collinear and the head stopped at the 100-epoch cap (best epoch 100/100, 98/100, 100/100). Individual weights therefore should not be read as stand-alone evidence; the identity logit = bias + Σ contributions holds exactly, and the contribution decomposition is faithful to what the model computes.
- **Redundancy / outliers:** no near-duplicate prototype pairs by the thresholds above, but deterministic farthest-point initialisation anchored a small high-irregularity outlier cluster in every seed — seed 42: 3 records (7.5 std from nearest prototype, CV 0.48), 110 records (2.5 std from nearest prototype, CV 0.40); seed 43: 4 records (6.1 std from nearest prototype, CV 0.46); seed 44: 6 records (6.8 std from nearest prototype, CV 0.44), 141 records (2.3 std from nearest prototype, CV 0.39). 5 of 48 prototype-seeds have < 150 cluster members; these prototypes are distinct but rare and contribute almost nothing to validation performance. Prototypes are anchored to actual training records (IDs in the CSV); this is initial anchored selection, **not learned prototype optimisation**.
- **Scope of explanation:** the four timing concepts cannot explain morphology-related diagnoses; the prototype layer explains only how rate/regularity profiles map to labels in this model.

## 7. Engineering checks (all passed)

| Check | Result |
|---|---|
| backbone_state_unchanged_during_embedding | True |
| backbone_ckpt_sha256_matches_declared | True |
| no_train_val_id_overlap | True |
| no_train_val_dup_group_overlap | True |
| no_test_or_excluded_records_in_caches | True |
| roles_only_train_validation | True |
| embedding_scaler_fit_on_train_only | True |
| concept_scaler_fit_on_accepted_train_only | True |
| no_nan_in_predictions | True |
| cache_alignment_spot_check_300_ids | True |
| prototype_logits_identity_max_abs_diff | 7.62939453125e-06 |
| prototype_head_inputs_similarities_only | True |
| concept_predictor_frozen_during_head_training | True |
| reproducibility_seed42_rerun_max_abs_diff | {'B_embedding_linear': 0.0, 'C2_concept_only': 0.0, 'D_prototype': 0.0, 'CTRL_mask_only': 0.0, 'CTRL_extracted_target': 0.0, 'concept_Zhat_max_abs_diff': 0.0} |
| extraction_matches_frozen_config | True |

Unit tests: `tests/test_step4_4.py` (10 tests: masked-loss NaN safety, train-only scaling, deterministic k-means/anchors, tau safeguard, logit decomposition, similarity-only head, determinism) and the Step 4.3 tests (20) pass. The seed-42 chain was re-trained and reproduced bit-identically (max abs probability difference 0.0).

## 8. Limitations and what was not checked

- Validation is used for early stopping/selection of every trained model; validation numbers are mildly optimistic, equally for B–D and controls. The test fold was not touched.
- Concept heads (C2, D) are trained on predicted concepts of *training* records, which are in-sample for the concept predictor (lower error than on validation) — a train/validation shift in the head's inputs.
- The prototype head had not converged at the declared 100-epoch cap and the weights are ill-conditioned; no further tuning was done. Prototype selection is a single deterministic k-means with farthest-point initialisation (outlier-prone); K=16 and tau were not tuned.
- Concept targets are automatically derived and unvalidated against references; predicted concepts may encode information beyond them (see the extracted-target baseline).
- Original head A has historical validation exposure and a different training recipe (class weights, mixup), so A is a reference, not a fair competitor to B–D.
- Seed spread is not a confidence interval; no significance testing; within-training duplicates and conflicting labels (e.g., AF vs AFIB copies) are retained as-is.
- Not done: any test-set evaluation, threshold optimisation, calibration, hyperparameter search, bypass/residual variants, learned prototype optimisation, additional concepts, qualitative review of prototype records.

## 9. Files and commands

Outputs in `results/step4_4/`: `experiment_report.md`, `experiment_config.json`, `concept_prediction_metrics.csv`, `validation_metrics.csv`, `validation_per_class.csv`, `prototype_summary.csv`, `coverage_summary.csv`, `coverage_by_label.csv`, `failure_summary.json`, `engineering_checks.json`, `transforms.json`, `validation_ids_fingerprint.json`, `execution_log.txt`, `checkpoints/` (per-seed state dicts, anchors, histories), `cache/` (record index, chunked concept CSVs, chunked embeddings, `merged.npz`, fingerprints; no waveforms). Code: `src/ecg_concept/concept_proto.py`, `scripts/step4_4_cache.py`, `scripts/step4_4_train.py`, `scripts/step4_4_coverage.py`, `scripts/step4_4_render_report.py`, `tests/test_step4_4.py`.

```
.venv\Scripts\python.exe -I tests\test_step4_4.py
.venv\Scripts\python.exe -I scripts\step4_4_cache.py index
.venv\Scripts\python.exe -I scripts\step4_4_cache.py extract
.venv\Scripts\python.exe -I scripts\step4_4_cache.py embed
.venv\Scripts\python.exe -I scripts\step4_4_cache.py merge
.venv\Scripts\python.exe -I scripts\step4_4_train.py config
.venv\Scripts\python.exe -I scripts\step4_4_train.py train
.venv\Scripts\python.exe -I scripts\step4_4_coverage.py
.venv\Scripts\python.exe -I scripts\step4_4_render_report.py
```
