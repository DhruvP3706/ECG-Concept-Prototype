# Step 3.7 — Prospective evaluation protocol (frozen backbone `ft_fold0`)

Status: **proposed freeze, internally consistent** (checks in §9). Scope: the clinical-concept-grounded prototype study built on the existing pretrained LightECGNetV2. This protocol does not create independent evidence of generalisation; see §8. Historical folds are untouched; the new roles are a separate, prospective layer in `prospective_role_manifest.csv`. `results/` is git-ignored, so durable versioning of these artifacts is your later, manual decision.

## 1. Fixed backbone

| Item | Value |
|---|---|
| Checkpoint | `models/pretrained/lightv2_ft_fold0.pt`, SHA-256 `7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72` (verified again in this step) |
| Frozen? | Yes — weights, BatchNorm statistics and `eval()` mode fixed for all future experiments; no fine-tuning unless a separate, pre-declared ablation says otherwise |
| Initial representation | 512-d pooled vector `[max_t | mean_t]` of the `tcn3b` map (Step 3). This is the **initial design choice**, not a validated superiority claim; the per-step `(256×625)` map is kept as an optional localisation channel |
| Original training exposure | Weights trained on original folds **1–8** (initialised from `p1_fold0`, then rare-class fine-tuned); fold **0** used for epoch selection, threshold fitting and weak-class identification; fold **9** never in weights |
| Selection limitation | The choice of `ft_fold0` over `p1_fold0` was made after viewing test-fold comparisons (Step 3.5). The validation-only evidence is mixed (ft higher val AUROC 0.9594 vs 0.9564; p1 higher val macro-F1 0.4541 vs 0.4211). Treat the choice as partly test-informed |
| Original thresholds | `lightv2_ft_fold0_thresh.npy` (VERIFIED). They belong to the 45-way original classifier and are **not** inherited by any new head |

## 2. Roles (computed from original folds; see manifest for per-record values)

Policy, applied in this order (each exclusion has a recorded reason):

1. Original fold 9 → **test_reserved** (evaluation only; no member enters development).
2. Records outside the 44,476-record experiment set → excluded (never in any role; listed in `duplicate_group_manifest.csv`).
3. Non-test duplicate partners of test records (17) → **excluded** from training *and* validation (`nontest_partner_of_test_record`).
4. Fold-0 validation candidates whose duplicate lies in folds 1–8 (7) → **excluded** (`val_candidate_duplicate_in_backbone_training_folds`); they are *not* moved to training or used to rebalance counts.
5. Remaining fold-0 records → **validation**; remaining folds 1–8 → **train**.
6. Remaining duplicate groups never span active roles (verified: 64 groups with an active member, 0 spanning train+validation). 43 duplicate pairs lie wholly inside train (flag `within_train_dup_group`); 32 active train/validation records belong to label-conflicted pairs (flag `dup_label_conflict`). Original labels are kept; AF/AFIB and other conflicts are **not** merged or adjudicated automatically.

| Role | Records | Source |
|---|---|---|
| train | **35,568** | folds 1–8 minus 13 partners of test records |
| validation | **4,437** | fold 0 minus 4 partners of test records and 7 backbone-exposed duplicates |
| test_reserved | **4,447** | fold 9 |
| excluded (in experiment set) | 24 | 17 + 7 |
| out-of-experiment duplicate members | 20 | not in the 44,476 |

Class support (records carrying each label; multi-label so columns do not sum to record counts). Validation support is very thin for several classes — selection among rare-class settings on this fold will be noisy:

| Class | train | validation | test (A) | test cohort B | excluded |
|---|---|---|---|---|---|
| 1AVB | 900 | 112 | 128 | 128 | 0 |
| 2AVB | 55 | 6 | 5 | 5 | 0 |
| 2AVB1 | 25 | 4 | 2 | 2 | 0 |
| 3AVB | 65 | 8 | 3 | 3 | 0 |
| AF | 6453 | 809 | 795 | 793 | 3 |
| AFIB | 1440 | 178 | 158 | 155 | 4 |
| ALS | 1244 | 146 | 154 | 154 | 1 |
| APB | 1026 | 142 | 144 | 144 | 0 |
| AQW | 859 | 97 | 106 | 105 | 1 |
| ARS | 687 | 84 | 82 | 82 | 0 |
| AT | 236 | 32 | 29 | 29 | 0 |
| AVB | 185 | 29 | 30 | 30 | 0 |
| AVRT | 25 | 0 | 1 | 1 | 0 |
| CCR | 134 | 18 | 10 | 10 | 0 |
| CR | 64 | 3 | 9 | 9 | 0 |
| ERV | 286 | 35 | 45 | 45 | 0 |
| IVB | 624 | 74 | 73 | 73 | 0 |
| JEB | 65 | 3 | 7 | 7 | 0 |
| LFBBB | 194 | 30 | 16 | 16 | 0 |
| LVH | 536 | 61 | 50 | 50 | 0 |
| LVQRSAL | 826 | 116 | 101 | 101 | 0 |
| MISW | 89 | 16 | 18 | 18 | 0 |
| PRIE | 46 | 3 | 3 | 3 | 0 |
| PWC | 104 | 19 | 19 | 18 | 0 |
| QTIE | 308 | 37 | 49 | 49 | 0 |
| RAH | 29 | 5 | 2 | 2 | 0 |
| RBBB | 531 | 62 | 55 | 55 | 1 |
| RVH | 91 | 14 | 5 | 5 | 0 |
| SA | 2018 | 273 | 259 | 259 | 0 |
| SB | 13117 | 1715 | 1725 | 1724 | 2 |
| SR | 6553 | 733 | 831 | 825 | 8 |
| ST | 5818 | 743 | 691 | 689 | 3 |
| STDD | 1342 | 161 | 165 | 165 | 0 |
| STE | 612 | 94 | 95 | 95 | 0 |
| STTC | 946 | 103 | 109 | 107 | 0 |
| STTU | 143 | 15 | 18 | 18 | 0 |
| SVT | 586 | 58 | 76 | 73 | 4 |
| TWC | 5587 | 731 | 719 | 714 | 6 |
| TWO | 2309 | 304 | 264 | 264 | 0 |
| UW | 108 | 10 | 18 | 18 | 0 |
| VEB | 42 | 9 | 5 | 5 | 0 |
| VFW | 94 | 11 | 11 | 11 | 0 |
| VPB | 247 | 23 | 23 | 23 | 1 |
| VPE | 9 | 3 | 0 | 0 | 0 |
| WPW | 63 | 3 | 6 | 6 | 0 |

## 3. Evaluation cohorts (predeclared)

| Cohort | n | Use |
|---|---|---|
| **B** (exclude 17 test records with a duplicate in folds 0–8) | 4,430 | **Main leakage-screened evaluation cohort** for this existing-backbone study |
| A (all 4,447) | 4,447 | Historical comparability with Step 3.5 only |
| C (exclude all 23 duplicate-involved test records) | 4,424 | Stricter sensitivity analysis |

The choice of B rests on exposure status, not on scores (Step 3.7 shows B and C change headline metrics by < 0.001). **All compared future models — baselines, ablations, the concept-prototype model — must be evaluated on the same cohort IDs** (`cohort_B_member` etc. in the role manifest) with the same label definitions (45 classes, sorted-acronym order, original labels) and the same metric conventions (`src/ecg_concept/metrics.py`: AUROC/AUPRC on continuous scores; macro over evaluable classes plus the all-45 convention; strict `>` thresholds; record-level paired bootstrap B = 1000, seed 20261013, percentile 95%).

## 4. Three distinct kinds of exposure (removing future training partners does not erase the first two)

1. **Backbone weight training**: `ft_fold0` was trained on folds 1–8, so every record in the *train* role (and the duplicate partners of validation records) is a signal the backbone fitted. 7 fold-0 candidates were removed because their duplicate is such a signal. The 17 test records with duplicates in folds 0–8 were exposed to some checkpoint(s) (13 for `ft_fold0`) and are excluded from cohort B.
2. **Validation / epoch / threshold selection**: fold 0 (= the new validation role) was used for the backbone's epoch, threshold and weak-class selection. New layers selected on fold 0 are therefore selected on data that already influenced the backbone; the optimism is probably small but not zero. Historical regenerated thresholds also used validation folds containing duplicates.
3. **Future concept / prototype / head training**: governed by this protocol (§5) — only the *train* role may fit anything.

Open design risk, not decided here: backbone features of **train-role** records come from records the backbone was trained on (train-vs-held-out macro-AUROC gap ≈ 0.03), so a concept/prototype layer fitted on them may see better-separated features than on validation/test. Any design must report this gap (e.g., train vs validation behaviour) and must not hide it.

## 5. Rules for future work

- **Training role only** may be used to fit concept scaling/standardisation, imputation, clustering, prototype initialisation and projection, class weights, and any other learned or data-dependent preprocessing.
- **Validation role only** may be used for hyperparameters, epochs/early stopping, prototype count, loss weights, concept thresholds, **classification thresholds and calibration**.
- **Test role**: no use for redesigning concepts, choosing among models, selecting thresholds, or iterating. Test metrics are computed once per pre-declared final configuration on cohort B (with A and C reported as defined above). Any further look at the test fold must be logged and its effect on claims stated.
- **Concept targets** (e.g., measured intervals, axis, amplitudes) must be masked when missing or unreliable (flat/saturated leads, failed delineation, corrupt headers, unreliable demographics such as the 247 age-0 and 55 NaN-age records), their quality assessed on the training role, and the mask applied identically to all compared models. Extracted measurements are not automatically clinical ground truth; labels remain diagnostic codes from the source dataset.
- **No refitting of the folds**: any new split inside the pool must keep duplicate groups together (group IDs in the manifests) and must not touch fold 9.
- **Baselines/ablations** (backbone-only head, concept-only, prototype-only, frozen vs fine-tuned) use identical role IDs, cohort IDs, label definitions and metrics.

## 6. Claims that are and are not supported

Supported (within this single dataset): performance of models on the leakage-screened cohort B, relative comparisons under identical IDs, and faithfulness analyses of the concept-prototype mechanism. **Not supported**: patient-independent generalisation (no patient IDs; exact-duplicate removal does not remove same-patient repeats), external validity, clinical utility, or "unseen test set" language — the test fold has been inspected and used in baseline/backbone selection. Dataset-level label noise (e.g., AF vs AFIB conflicts on identical waveforms) bounds attainable performance on affected classes.

## 7. Recommendation for external evaluation

Before strong generalisation claims, evaluate the frozen pipeline once, with pre-registered settings, on an independent dataset or a prospectively untouched collection with patient-level identifiers and a compatible label mapping (the 45-class vocabulary is dataset-specific). Until then, describe all results as internal, single-dataset, duplicate-screened.

## 8. Limitations of the freeze

Exact byte-level duplicates only; no near-duplicate or patient-level screening. Fold membership is a hash-verified reconstruction, not stored split files. The validation role has ≤ 3 positives for six classes and 0 for AVRT; the test cohort has 0 for VPE and < 10 for 12 classes. The 24 excluded records and the out-of-experiment duplicates are bookkeeping, not an assessment of their clinical quality.

## 9. Consistency checks performed (all passed)

Roles are disjoint and cover all 44,476 records exactly once; test_reserved equals original fold 9 (4,447) and no test record is in an active role; no duplicate group has members in both train and validation; every non-test partner of a test record is excluded; cohort sizes 4,447 / 4,430 / 4,424; record order and test IDs hash-match Step 3.5; backbone hash re-verified; predeclared analysis config hash re-verified before use.

## 10. Artifacts

`original_fold_manifest.csv` (44,476 rows, immutable historical reconstruction + duplicate flags), `duplicate_group_manifest.csv` (174 rows incl. 20 out-of-experiment members), `prospective_role_manifest.csv` (44,476 rows, roles + reasons + flags + cohort membership), `analysis_config.json`, `sensitivity_analysis_report.md`. Generated by `scripts/step3_7_manifests.py` and `scripts/step3_7_sensitivity.py`.
