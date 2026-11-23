# Step 3.7 — Duplicate-leakage sensitivity analysis

Legend: **[V]** verified by computation in this step · **[H]** interpretation. Python in the project `.venv` is **3.14.6** (numpy 2.5.4, scipy 1.18.1, torch 2.12.0+cu126), checked directly. No inference, retraining, threshold/calibration fitting or hyperparameter search was done; only the saved Step 3.5 single-clean-pass predictions were used. Step 3.5 and Step 3.6 files are unchanged; **Step 3.5 remains the historical reproduction result.**

## 1. Evidence verified before analysis [V]

| Item | Result |
|---|---|
| Record order / test IDs | Rebuilt index (44,476 records) and `folds[9]` hash-match `evaluation_config.json` (`record_order_sha256`, `test_record_ids_sha256`) |
| Prediction alignment | `evaluation_predictions.npz` record IDs equal the reconstructed test IDs in order; class names equal the recovered 45-class list; stored targets equal labels reconstructed from the headers. Analysis matches rows by explicit ID |
| Duplicate totals (recomputed from the pair CSV + my own fold map, compared with `duplicate_summary.json`) | 87 pairs, 174 unique records, 17 cross-boundary pairs / 17 test records, 23 test records in any pair, 1 within-test pair, 50 pairs within folds 0–8, 19 pairs with a member outside the experiment set — **all equal** |
| The 19 "category 4" pairs | **18** have exactly one member outside the 44,476-record experiment set (4 of those have the in-set member in the test fold); **1** (JS45461 / JS45462) has **both** members outside. All 174 IDs exist in the dataset, so membership is *known* for every record; unresolved pairs: 0. "Outside" means no label among the 45 classes (never in any fold) |
| Ensemble | `ensemble3` probabilities equal the mean of `ft_fold0`, `p1_fold1`, `p1_fold2` (max diff < 1e-7) |
| Thresholds | `ft_fold0`, `p1_fold0`: arrays identical to the VERIFIED files in `models/pretrained`; `p1_fold1`, `p1_fold2`, ensemble: identical to the regenerated (validation-fold) arrays; the stale `p1_fold1_thresh.npy` is not used and differs from the array used |

## 2. Pre-declared analysis (written before any cohort metric was computed)

`analysis_config.json → declaration` (SHA-256 `85d98cc3…`, re-verified unaltered before the metrics run) defines three fixed cohorts, selected by duplicate exposure only (no labels, errors or scores), identical for every model:

| Cohort | Rule | n |
|---|---|---|
| A | all original test records | 4,447 |
| B | exclude the 17 test records with a duplicate in folds 0–8 | 4,430 |
| C | exclude all 23 test records in any exact-duplicate pair | 4,424 |

Models: `ft_fold0`, `p1_fold0` (original VERIFIED thresholds), `p1_fold1`, `p1_fold2` (regenerated validation-fold thresholds), original 3-model ensemble (regenerated CV thresholds). Paired bootstrap: B = 1,000 record-level resamples, seed 20261013, resamples shared across models within each cohort, percentile 95% CI. Exclusion ID lists are in the config.

**Deviation, disclosed:** the declared reproduction tolerance was 1e-8, but `aggregate_metrics.csv` stores 6 decimals, so the headline check could only be done at 5.1e-7. The paired bootstrap *was* checked at 1e-8.

## 3. Reproduction of Step 3.5 on cohort A [V]

- Headline metrics (all five models, ranking + thresholded) match the Step 3.5 CSV: max |Δ| = 4.96e-7 (CSV rounding).
- Paired bootstrap on cohort A (30 comparisons × mean, CI bounds, fraction positive) matches `paired_bootstrap.json`: max |Δ| = **0.0**.
- No discrepancy to investigate.

## 4. Results

Evaluable classes (≥1 positive and ≥1 negative): **44 in every cohort**; the only unevaluable class is **VPE (0 test positives)** in A, B and C. Because the evaluable set did not change, the "common-evaluable-class" macros equal the standard ones (columns `*_common` in the CSV). Positives removed (identical in all models): B — SR 6, TWC 5, AFIB 3, SVT 3, AF 2, ST 2, STTC 2, AQW 1, PWC 1, SB 1; C additionally removes more SVT (7 total), STTC (3), AF (3), AFIB (4), plus LFBBB 1.

Selected metrics, cohort **A / B / C** (full set, deltas and per-class values in `sensitivity_metrics.csv`, `sensitivity_per_class.csv`):

| Model | macro-AUROC | macro-AUPRC | macro-F1 (evaluable) | micro-F1 |
|---|---|---|---|---|
| ft_fold0 | .9628 / .9626 / .9627 | .3759 / .3762 / .3766 | .3803 / .3801 / .3801 | .6756 / .6756 / .6758 |
| p1_fold0 | .9600 / .9598 / .9598 | .3749 / .3753 / .3757 | .3858 / .3859 / .3858 | .6999 / .7000 / .7003 |
| p1_fold1 | .9576 / .9576 / .9577 | .3792 / .3792 / .3798 | .3861 / .3859 / .3859 | .6999 / .7000 / .7005 |
| p1_fold2 | .9552 / .9551 / .9551 | .3777 / .3779 / .3776 | .3849 / .3850 / .3848 | .6991 / .6990 / .6993 |
| ensemble3 | .9626 / .9625 / .9626 | .4015 / .4016 / .4017 | .3941 / .3939 / .3935 | .7199 / .7197 / .7198 |

Observed changes relative to A across the five models (descriptive):

| Metric | B − A | C − A |
|---|---|---|
| macro-AUROC | −0.00020 … −0.00004 | −0.00013 … +0.00003 |
| weighted-AUROC | +0.00003 … +0.00008 | +0.00006 … +0.00010 |
| micro-AUROC | −0.00002 … 0.00000 | 0.00000 … +0.00003 |
| macro-AUPRC | +0.00005 … +0.00037 | −0.00007 … +0.00080 |
| micro-AUPRC | +0.00005 … +0.00037 | +0.00020 … +0.00056 |
| macro-F1 (all 45 / evaluable) | −0.00020 … +0.00010 | −0.00062 … −0.00002 |
| micro-F1 | −0.00014 … +0.00008 | −0.00003 … +0.00060 |

Per-class changes are larger where support is small: for `ft_fold0`, SVT F1 −0.009 (B) and −0.021 (C) (76 → 73 / 69 positives); PWC AUROC −0.005; ensemble LFBBB F1 −0.022 (C, 16 → 15 positives). These are within the noise expected from removing a handful of positives of rare classes and should not be read as class-specific leakage effects.

Paired model comparisons (`paired_comparisons.csv`, 90 rows; all 1,000 replicates valid in every case; on average ≈43.2 of the 44 classes are evaluable per resample because rare classes are sometimes absent): **no comparison changed significance status or sign across cohorts** (e.g., ft_fold0 − p1_fold0 macro-AUROC +0.0029 [+0.0007, +0.0050] in A, [+0.0005, +0.0053] in B, [+0.0008, +0.0053] in C; ensemble micro-F1 advantage +0.019 to +0.044 in all cohorts; macro-F1 differences remain non-significant). The largest change of any point difference between A and B/C was 0.00016 (macro-AUROC), 0.0006 (macro-F1) and 0.0006 (micro-F1).

## 5. What this does and does not show

- [V] On the existing predictions, excluding the duplicate-exposed test records changes headline metrics by ≤ 0.001 and leaves every model comparison unchanged.
- [H] This is a **descriptive sensitivity difference, not a causal estimate of leakage inflation**: the cohort composition changes when records are removed, so a change mixes any memorisation effect with the changed class mix and sampling noise. Differences were not given confidence intervals; subset-vs-full CIs were not formed by subtracting independent CIs.
- [H] The small differences are *compatible with* little measurable benefit from duplicate exposure in this test fold, but they do not prove the absence of leakage: a model that memorised a duplicate can look correct for a different reason, 12–17 records carry little statistical power, and near-duplicates or same-patient recordings that are not byte-identical are undetected.
- Remaining dependence in the full cohort: one within-test duplicate pair (JS06370/JS18476) remains in A and B; unknown patient overlap and repeated recordings remain in every cohort. The record-level bootstrap treats records as independent, so these intervals are **not patient-level confidence intervals**.

## 6. Corrections to Step 3.6 wording (Step 3.6 files left unchanged)

1. Step 3.6 §5 said micro-level metrics "cannot move by more than a few tenths of a percentage point" and that a macro effect "would be on the order of 10⁻³". **Withdrawn as bounds**: a fraction of affected records does not bound the change in micro-F1, AUROC or AUPRC (a model could shift predictions on other records, rare-class denominators can be small, and the metrics are not linear in the number of records). The *measured* changes above are all < 0.001, which is an observation for this model set and fold, not a bound.
2. Step 3.6 §6 mentioned "consider dropping the 19 out-of-vocabulary partners": those records are not in the 44,476-record experiment set, so they are never in any training/validation pool; this is moot. Of the 19 pairs 18 have one member outside and 1 has both outside.
3. Step 3.6 §4.4 counted a partner in a checkpoint's own validation fold as "not trained on". That is right for the weights, but the partner was used for epoch/threshold selection (a different kind of exposure; see `evaluation_protocol.md` §4).

## 7. Preservation and recommended use

Step 3.5's full-test table (cohort A) is kept for historical comparability only; it is contaminated by 23 duplicate-involved test records and should not be designated the primary result for the paper. For the existing-backbone study the predeclared main cohort is **B** (exposure-based, not performance-based); C is the stricter sensitivity cohort; A is reported for comparability. The test fold has already been inspected, backbone/baseline choices were partly test-informed, historical validation contained duplicates, and patient independence is unknown — so B is *leakage-screened*, not pristine or unseen. Durable versioning of the manifests and protocol (the `results/` directory is git-ignored) is left to your decision; nothing was staged or committed.

## 8. Limitations / not checked

Exact (byte) duplicates only; no near-duplicate or patient-level search. Fold membership relies on the reconstructed order (hash-verified against Step 3.5, but the original split files do not exist). 12–17 affected records limit power; rare-class results are anecdotal. Thresholds are the Step 3.5 ones (including regenerated ones fitted on validation folds that themselves contain duplicates). No sensitivity analysis was done for TTA, float-filter or fixed-0.5 variants (not declared).

## 9. Reproduction

`.venv\Scripts\python.exe -I scripts\step3_7_manifests.py` then `…\step3_7_sensitivity.py` (both refuse to overwrite existing outputs; the second adds a `completed` section to `analysis_config.json`). Input fingerprints, exclusion IDs and software versions are in `analysis_config.json`.
