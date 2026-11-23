# Step 3.5 — Full held-out evaluation of the recovered LightECGNetV2 checkpoints

Legend: **[V]** verified by code/experiment · **[H]** inference/hypothesis. This is an engineering and model-performance reproduction on one dataset, **not a clinical validation**, and the numbers are **not comparable** to ProtoECGNet/PTB-XL (different dataset, label set, test split, and 45 mostly-rare rhythm/morphology labels). Nothing was trained, tuned on the test fold, committed or pushed.

## 1. Dataset and split provenance

- Dataset: `data/raw/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0` (all 45,152 file hashes matched `SHA256SUMS.txt` in Step 2).
- Record selection [V, notebook `ECGDataset`]: `os.walk(WFDBRecords)` with sorted filenames, keep records with ≥1 label among the 45 classes → **44,476** records (identical to the notebook's printed count). Class order = sorted acronyms (identical to the notebook's `class_names.npy`).
- Split [V, notebook cell 8]: `RandomState(42).permutation(44476)` → `np.array_split(…,10)`; **folds[9] = held-out test, 4,447 records** (notebook: 4,447). CV folds 0–8 have 4,447–4,448 records. Order fingerprint (SHA-256 of the ordered ID list): `2f9f2dba…efa`; test-ID list fingerprint `13ee2563…eb5` (full IDs in `evaluation_config.json`).
- Evidence the reconstructed fold membership is right (validation folds only, before test evaluation) [V]:
  1. The notebook's recorded validation macro-F1 is reproduced for all four checkpoints on the reconstructed validation folds: 0.4541 / 0.4211 / 0.4321 / 0.4432 vs 0.4541 / 0.4214 / 0.4321 / 0.4432 under bf16 emulation (|Δ| ≤ 3.4e-4; fp32 |Δ| ≤ 2.2e-3).
  2. Each checkpoint's macro-AUROC on its own validation fold is 0.955–0.965 but 0.988–0.991 on folds it trained on; on fold 0 the model that held it out scores 0.9564 while the two that trained on it score 0.9876/0.9876 (same pattern for folds 1 and 2). This only occurs if the fold indices match the training run.
  3. On the test fold every checkpoint scores 0.955–0.963 — the held-out level, not the 0.99 training level.
  4. The notebook's ensemble CV thresholds are reproduced 37/40 exactly (35/40 in fp32), remaining differences one grid step.
- Residual ambiguity [H]: membership assumes the same directory-walk order as the original run (NTFS order; both runs on Windows). Items 1–4 make a wrong order very unlikely, but a swap of a few records between folds cannot be excluded.

## 2. Checkpoint eligibility

Training/validation assignment is taken from the notebook training loops (cells 19–21); `folds[9]` appears in **no** training or validation set of any checkpoint. [V]

| Checkpoint (SHA-256) | Trained on folds | Validation fold (used for epoch + threshold selection) | Independent held-out on folds[9]? |
|---|---|---|---|
| `lightv2_ft_fold0` `7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72` | 1–8 (init from p1_fold0, then rare-class fine-tune) | 0 | **Yes** |
| `lightv2_p1_fold0` `042876f5cd9a8293267df91cc0d393e28296d5423e3d0f81f37e2d6d0b5251b6` | 1–8 | 0 | **Yes** |
| `lightv2_p1_fold1` `7e7559c095577c64997d3e1a8a67871642e0cda238f55bb28e1eb5007860da7b` | 0,2–8 | 1 | **Yes** |
| `lightv2_p1_fold2` `6e1ee03f3ef16578d0b39bb5355d8e06b25772e4c0793bd04843d2d07a54d434` | 0,1,3–8 | 2 | **Yes** |

Caveats on "independent": the *weights* never saw folds[9]. The notebook author did look at test results and tuned threshold *strategies* (offset, blend, oracle) on folds[9]; none of those are used here. The ft_fold0 and p1_fold0 models share fold 0 and training data (not independent of each other); each model's validation fold is training data for the others, so those folds must not be used to compare models. The 3-model ensemble covers folds 0–8 with every fold trained on by at least two members.

## 3. Reproduction details

fp32, `model.eval()`, `torch.inference_mode()`, GTX 1650, batch 64 (CUDA vs CPU Δ ≤ 1e-6 in Step 3). Preprocessing exactly as the recovered notebook pipeline including the int16 truncation after filtering. Every test record processed once per checkpoint (asserted: 4,447 unique IDs, all outputs finite and within [0,1], every record has ≥1 target label, class order equals the recovered list). Probabilities (pre-threshold) are saved in `evaluation_predictions.npz`. Primary rows = single clean pass; **secondary** = notebook-style 5-pass TTA (seeded, not identical to the original's unseeded noise) used only to compare with the notebook; **diagnostic** = 0.5 thresholds and float (non-truncated) filtering.

## 4. Results on folds[9] (n = 4,447; 44 of 45 classes evaluable — VPE has 0 test positives; 12 classes have <10 positives)

Primary rows, single clean pass, 95% CI from 1,000 record-level bootstrap resamples (shared across models).

| Model (thresholds) | macro-AUROC | weighted-AUROC | micro-AUROC | macro-AUPRC | micro-AUPRC |
|---|---|---|---|---|---|
| ft_fold0 | 0.9628 [0.9585, 0.9660] | 0.9701 [0.9682, 0.9723] | 0.9829 [0.9817, 0.9841] | 0.3759 [0.3689, 0.4237] | 0.7406 [0.7302, 0.7509] |
| p1_fold0 | 0.9600 [0.9549, 0.9636] | 0.9686 [0.9664, 0.9707] | 0.9837 [0.9824, 0.9850] | 0.3749 [0.3684, 0.4275] | 0.7767 [0.7661, 0.7873] |
| p1_fold1 | 0.9576 [0.9459, 0.9661] | 0.9697 [0.9677, 0.9718] | 0.9852 [0.9841, 0.9863] | 0.3792 [0.3707, 0.4267] | 0.7987 [0.7896, 0.8074] |
| p1_fold2 | 0.9552 [0.9425, 0.9649] | 0.9695 [0.9675, 0.9716] | 0.9849 [0.9837, 0.9861] | 0.3777 [0.3718, 0.4288] | 0.7977 [0.7891, 0.8071] |
| 3-model ensemble | 0.9626 [0.9524, 0.9701] | 0.9728 [0.9709, 0.9747] | 0.9866 [0.9855, 0.9876] | 0.4015 [0.3955, 0.4500] | 0.8035 [0.7952, 0.8127] |

Thresholded metrics (thresholds: ft_fold0 and p1_fold0 = original VERIFIED files; p1_fold1, p1_fold2 = **regenerated** on own validation fold; ensemble = **regenerated** on combined validation folds):

| Model | macro-F1 (nb def.¹) | macro-F1 (44 valid classes) | micro-F1 | weighted-F1 | macro P / R (nb) | exact match | Hamming loss |
|---|---|---|---|---|---|---|---|
| ft_fold0 | 0.3718 [0.350, 0.388] | 0.3803 [0.361, 0.408] | 0.6756 [0.667, 0.684] | 0.7528 [0.745, 0.761] | 0.310 / 0.569 | 0.457 [0.442, 0.471] | 0.0285 [0.0274, 0.0296] |
| p1_fold0 | 0.3773 [0.350, 0.397] | 0.3858 [0.362, 0.415] | 0.6999 [0.692, 0.708] | 0.7536 [0.746, 0.763] | 0.337 / 0.529 | 0.495 [0.480, 0.509] | 0.0250 [0.0240, 0.0260] |
| p1_fold1 | 0.3775 [0.352, 0.397] | 0.3861 [0.364, 0.415] | 0.6999 [0.691, 0.709] | 0.7490 [0.741, 0.758] | 0.337 / 0.506 | 0.497 [0.482, 0.511] | 0.0248 [0.0238, 0.0258] |
| p1_fold2 | 0.3763 [0.351, 0.393] | 0.3849 [0.362, 0.413] | 0.6991 [0.691, 0.708] | 0.7478 [0.740, 0.757] | 0.336 / 0.503 | 0.488 [0.473, 0.502] | 0.0249 [0.0239, 0.0259] |
| ensemble | 0.3854 [0.359, 0.406] | 0.3941 [0.369, 0.425] | 0.7199 [0.712, 0.728] | 0.7622 [0.754, 0.771] | 0.369 / 0.471 | 0.504 [0.489, 0.518] | 0.0226 [0.0217, 0.0235] |

¹ "nb def." = the notebook/scikit-learn convention: average over all 45 classes with undefined → 0 (VPE contributes 0, lowering the mean by ~1/45).
Fixed-0.5 diagnostic rows (not original thresholds) and everything else are in `aggregate_metrics.csv`; per-class AUROC/AUPRC/P/R/F1/thresholds/counts with CIs in `per_class_metrics.csv`; `per_class_auroc.svg` plots per-class AUROC.

Paired bootstrap (`paired_bootstrap.json`, same resamples): ft_fold0 − p1_fold0 macro-AUROC +0.0029 [+0.0007, +0.0050], micro-F1 −0.0242 [−0.029, −0.019], macro-F1 −0.005 [−0.019, +0.011] (n.s.). Ensemble − single models: micro-F1 +0.020 to +0.044 (all CIs exclude 0); macro-AUROC +0.005/+0.007 over p1_fold1/p1_fold2 (excluded 0) but ≈0 vs ft_fold0 (−0.0003 [−0.007, +0.011]); macro-F1 +0.008–0.013 with CIs spanning 0. p1_fold0/1/2 are statistically indistinguishable from each other.

Interpretation notes: (i) macro metrics are dominated by many rare classes — per-class results for classes with <10 test positives (e.g. AVRT 1, 2AVB1 2, RAH 2, PRIE 3, 3AVB 3, 2AVB/RVH/VEB 5, WPW 6) are essentially anecdotal. (ii) The macro-AUPRC percentile CIs are asymmetric (point estimate near the lower end) because resampling inflates average precision of rare classes; treat them as indicative. (iii) AUROC is high (0.96–0.99) while macro-F1 is only ~0.37–0.39: ranking is good, but per-class decisions on rare/overlapping classes (TWC, STTC, LVQRSAL, QTIE, AFIB, AVB; per-class AUROC 0.86–0.93) are weak. Frequent rhythm classes are strong (SB F1 0.977, ST 0.951, SR 0.889, AF 0.881).

## 5. Comparison with the original notebook (Mission D)

| Config | Metric | Notebook | Reproduced | Δ |
|---|---|---|---|---|
| ft_fold0, TTA5, original thresholds (cell 24) | macro P / R / F1 | 0.3095 / 0.5726 / 0.3725 | 0.3095 / 0.5681 / 0.3712 | 0.0000 / −0.0045 / −0.0013 |
| | micro-F1 · macro-AUROC · mean acc | 0.6763 · 0.9628 · 0.9715 | 0.6758 · 0.9627 · 0.9715 | −0.0005 · −0.0001 · 0 |
| ft_fold0, TTA5, fixed 0.5 (cell 24) | macro P / R / F1 | 0.2721 / 0.6585 / 0.3478 | 0.2689 / 0.6591 / 0.3467 | −0.003 / +0.001 / −0.001 |
| 3-model ensemble, TTA5, CV thresholds (cell 22) | macro P / R / F1 | 0.3673 / 0.4786 / 0.3856 | 0.3691 / 0.4701 / 0.3852 | +0.002 / −0.009 / −0.0004 |
| | micro-F1 · macro-AUROC · mean acc | 0.7172 · 0.9626 · 0.9771 | 0.7196 · 0.9627 · 0.9774 | +0.002 · +0.0001 · +0.0003 |
| ensemble, TTA5, fixed 0.5 (cell 22) | macro F1 · micro-F1 | 0.3670 · 0.6415 | 0.3675 · 0.6418 | +0.0005 · +0.0003 |

**The notebook's headline test results are reproduced** (macro-F1 ≈ 0.39 for the CV-threshold ensemble: 0.3852 vs 0.3856; AUROC to 4 decimals). Remaining differences are ≤ 0.009 and are explained as follows:
- *Precision (bf16 vs fp32)* [V]: the original evaluated under bf16 autocast. Emulating it changes ft_fold0 test probabilities by at most 0.045 (mean 4e-4), macro-recall 0.5694→0.5710, macro-F1 unchanged (0.3718); validation F1 recorded by the notebook is reproduced ~10× more closely under bf16 (≤3.4e-4 vs ≤2.2e-3).
- *TTA randomness* [V]: the original TTA noise came from the global NumPy RNG (seeded at kernel start but consumed by training since), so it cannot be replayed; ours is explicitly seeded (+4 passes). TTA moves metrics by ≤0.001 relative to a clean pass (ft_fold0 macro-F1 0.3718 clean vs 0.3712 TTA), so it cannot explain 0.0045 of recall either; the rest is attributed to bf16 numerics near thresholds and to sampling noise in the TTA draw [H].
- *Regenerated ensemble thresholds* [V]: 3/40 printed ensemble thresholds differ by one grid step (bf16) and STTU is 0.85 here but within 0.4–0.6 in the original (it was not printed); this moves ensemble recall by up to ~0.009 [H].
- *Integer truncation before z-scoring* [V]: removing it changes macro-AUROC by ≤0.0001 and macro-F1 by ≤0.0002 for every checkpoint (`floatfilt` rows) — negligible but kept for fidelity.
- *Metric implementation* [V]: our NumPy/SciPy metrics were checked against brute-force definitions and the scikit-learn documented examples; scikit-learn itself was not available to cross-check.
- *Dataset order / class order / checkpoint version*: ruled out by the evidence in §1 and by exact reproduction of the recorded validation F1. Unresolved: the original's exact RNG for TTA and its bf16 kernels (GPU) cannot be replicated on a GTX 1650.

Not claimed: the notebook's "oracle", offset and blend strategies were tuned on the test fold and were deliberately not reproduced.

## 6. Threshold provenance (details in `threshold_audit.md`)

ft_fold0 and p1_fold0 files: VERIFIED. **p1_fold1 file: UNVERIFIED/stale** (from an interrupted earlier run; costs ~0.042 validation macro-F1; on the test fold it shows macro-F1 0.3875 vs 0.3775 for regenerated thresholds with micro-F1 0.692 vs 0.700 — a higher macro but not a trustworthy configuration, shown only as a diagnostic and not used). **p1_fold2: no original file exists**; regenerated thresholds are labelled as such. No threshold was selected on the test fold.

## 7. Best defensible baseline

- **Single backbone for Step 4 (recommended): `lightv2_ft_fold0`.** Reasons: (a) independent of folds[9] and verified provenance (weights, original thresholds, validation fold all consistent); (b) highest macro-AUROC of the single models on the test fold (0.9628, +0.003 over p1_fold0 with CI excluding 0) and also higher than p1_fold0 on its shared validation fold (0.9594 vs 0.9564), which is the criterion usable without the test set; (c) ranking quality, not thresholded F1, is what a frozen representation needs. Counterpoints: p1_fold0 has better micro-F1 (+0.024) and calibration on frequent classes (fine-tuning with up-weighted rare classes shifts probabilities), and its validation macro-F1 is higher (0.4541 vs 0.4211); macro-F1 differences are not significant. The choice between ft_fold0 and p1_fold0 is a judgement, flagged as such; selecting it after seeing the test comparison is a mild form of test-informed selection.
- **Best prediction baseline overall: the 3-model ensemble** (best weighted-AUROC 0.9728, micro-AUROC 0.9866, both AUPRCs 0.40/0.80, micro-F1 0.720, lowest Hamming loss 0.0226), but its thresholds are regenerated, and it is three networks, which does not fit a single-backbone prototype architecture.
- p1_fold1/p1_fold2 are eligible but offer no advantage; p1_fold1's threshold file must not be used.

## 8. Limitations and open issues

1. Single dataset, single held-out fold, no external validation; many classes with <10 test positives; VPE not evaluable on this fold.
2. Fold membership relies on a reconstructed ordering (strong, not absolute, evidence). The same records were used in training and validation by the original author's other folds, so the trained-on versus held-out distinction rests on the notebook, not on stored split files.
3. Record-level bootstrap treats records as independent, but the dataset has no patient IDs; repeated recordings per patient (and 87 exact-duplicate signal pairs found in Step 2) could make CIs optimistic and even leak across folds. Not checked here: whether duplicate pairs straddle folds[9] and training folds.
4. Regenerated thresholds come from one validation fold of ~4.4k records; rare-class thresholds are noisy.
5. bf16 emulation was CPU autocast, not the original GPU kernels.
6. SHAP script (`shap-project/…v2.py`) still uses a model definition that does not match these checkpoints (Step 3); no SHAP results are used or evaluated here.
7. scikit-learn/matplotlib are not installed; metrics are NumPy/SciPy and the plot is a hand-written SVG.

## 9. Recommended next step

Before Step 4: (1) check whether the 87 duplicate-signal pairs cross the folds[9]/training boundary and, if so, report metrics with duplicates removed; (2) decide the backbone (recommended `ft_fold0`) and freeze its split (folds[9] as the only test set; folds 3–8 are training data for every checkpoint and must not be used for validation of new heads — use fold 0 only for ft_fold0/p1_fold0-based work, or define a new validation split inside the CV pool for any newly trained layer); (3) begin Step 4 (concept definitions) with the 512-d pooled representation as planned in Step 3. No Step 4 work was started.

## 10. Files

`results/step3_5/`: `evaluation_report.md`, `aggregate_metrics.csv`, `per_class_metrics.csv`, `threshold_audit.md`, `threshold_audit.json`, `evaluation_config.json`, `evaluation_predictions.npz`, `regenerated_thresholds.npz`, `notebook_comparison.json`, `paired_bootstrap.json`, `bf16_diagnostic.json`, `bf16_test_diagnostic.json`, `per_class_auroc.svg`, `cache/` (per-fold raw outputs, no waveforms), run logs. Code: `src/ecg_concept/metrics.py`, `scripts/step3_5_{predict,threshold_audit,evaluate,paired,finalize,bf16_diag,bf16_test_diag}.py`. `results/` is git-ignored; nothing was staged, committed or pushed.
