# Step 3.6 — Duplicate-signal leakage audit (read-only)

Legend: **[V]** verified by computation in this step · **[H]** inference. No model was run, no metric was recomputed, no threshold or split was changed, nothing was staged or committed. Waveforms are not shown; only record IDs, folds, labels and header fields.

## 1. Bottom line

| Quantity | Value |
|---|---|
| Duplicate pairs (byte-identical `.mat`) | **87** (174 unique records; every duplicate group has exactly 2 members) |
| Pairs with both records in the test fold (`folds[9]`) | 1 |
| **Pairs crossing the test boundary** (one test, one in folds 0–8) | **17** |
| Pairs entirely outside the test fold (folds 0–8) | 50 (44 of them put the two copies in different folds) |
| Pairs where one record is not in the 44,476-record experiment set | 19 (4 of them have the in-set copy in the test fold) |
| **Test records in cross-boundary pairs** | **17 of 4,447 (0.38%)** |
| …of which the duplicate has identical labels | 12 (0.27%) |
| …of which the duplicate shares ≥1 label | 16 |
| Test records involved in any duplicate pair | 23 |

The held-out evaluation is **very slightly optimistic in principle, but the possible effect is small** (§5). A split-aware sensitivity analysis is warranted before reporting final numbers, and the duplicate handling must be fixed before any new training in Step 4.

## 2. How the 87 pairs were originally detected

[V — from the Step 2 audit script, not stored in the repo] Step 2 computed the MD5 of the **entire `.mat` file** for all 45,152 records and grouped identical hashes, then printed "87 groups / 174 records" (and only the first five groups). The comparison was therefore on raw file bytes: not on processed waveforms, not on `.hea` files, and not a similarity threshold. The pair list itself and its fold membership were never saved or checked — hence this step. The Step 2 script is not in the repository (provenance gap; the method above is what I ran then and re-ran now).

## 3. Methodology (this step, `scripts/step3_6_duplicate_audit.py`, `scripts/step3_6_exposure.py`)

1. Re-hash all 45,152 `.mat` files: MD5 of the full file (**exact duplicates**), and MD5 of each lead's 5,000 int16 samples (**partial duplicates**: records sharing an identical non-flat lead without being byte-identical; flat/constant leads excluded because they match trivially).
2. Fold membership: `labels.build_index` + `labels.reproduce_folds` (the same code as Step 3.5: `os.walk` order, 44,476 labelled records, `RandomState(42)` permutation, 10 folds, folds[9] = test with 4,447 records). Each member of each pair is mapped to its fold; records without any label among the 45 classes are not in the experiment set and have no fold.
3. Classification of each pair into 1) both in test, 2) one in test / one in folds 0–8, 3) both in folds 0–8, 4) a member outside the experiment set (membership "unresolved" only in the sense that the record never enters any fold).
4. For each pair: labels (45-class acronyms), raw `#Dx` code sets, age, sex, ID distance. For each checkpoint, which test records have a duplicate in a fold that checkpoint trained on (training folds = 0–8 minus its validation fold, per the notebook).

## 4. Findings

**4.1 Reproduction of Step 2** [V]: 87 groups, all of size 2, 174 records, 45,065 distinct file hashes among 45,152 files. Identical MD5 over the full 120,024-byte file means identical header and identical samples in all 12 leads. **No partial duplicates**: 0 pairs share a non-flat lead without being byte-identical. So all 87 are exact duplicates and there are no near-duplicates *of the kinds tested* (identical lead content). Shifted, cropped, resampled, re-filtered or noisy copies were **not** searched for (§7).

**4.2 Fold membership of the 87 pairs** [V given the reconstructed folds]

| Category | Pairs |
|---|---|
| 1. both in test | 1 (JS06370 / JS18476) |
| 2. one in test, one in folds 0–8 | **17** |
| 3. both in folds 0–8 | 50 (6 in the same fold, 44 in different folds) |
| 4. a member is not in the experiment set | 19 (18 are JS10607…JS10634 ↔ JS45510…JS45529; the JS455xx copy has no label in the 45-class vocabulary) |

The 17 cross-boundary partners are spread over folds 0 (4), 2 (1), 3 (2), 4 (2), 5 (1), 6 (2), 7 (3), 8 (2); none lies in fold 1 (validation fold of `p1_fold1`).

**4.3 Characteristics of the cross-boundary pairs** (full list in `duplicate_pair_summary.csv`)
- 12 have identical 45-class labels (SR ×5, SR+TWC ×1, SVT ×3, AFIB+TWC ×1, ST ×1, SB ×1); 5 differ: **3 differ only by AF ↔ AFIB** (one copy coded atrial flutter 164890007, the other fibrillation 164889003), 1 by AF ↔ AFIB plus an extra STTC, and 1 by extra labels (PWC+ST+STTC vs ST).
- 10 of the 17 have age 0 on one copy and a real age on the other. Across all 87 pairs 55 have age 0 on one side, versus 247 age-0 records in the whole dataset. [H] Many duplicates are therefore **re-exports of the same recording with missing demographics**, i.e. very probably the same patient and not merely coincidental files; copies often sit far apart in ID (median gap ≈ 31,000), a few close (gap 178, 325).
- Label disagreement between copies (24 of 68 in-set pairs) shows these records also carry **label noise**: the same waveform with different diagnoses.

**4.4 Exposure by checkpoint** (test records whose duplicate lies in a fold the checkpoint trained on)

| Checkpoint | any cross pair (of 17) | identical-label pairs (of 12) | shares ≥1 label (of 16) |
|---|---|---|---|
| `ft_fold0`, `p1_fold0` (train folds 1–8) | 13 | 8 | 12 |
| `p1_fold1` (train 0,2–8) | 17 | 12 | 16 |
| `p1_fold2` (train 0,1,3–8) | 16 | 11 | 15 |
| 3-model ensemble (ft_fold0, p1_fold1, p1_fold2) | 17 | 12 | 16 |

For the remaining test records the duplicate was in the checkpoint's own validation fold (not trained on, but used for epoch/threshold selection).

**4.5 Other contamination** [V]: 44 duplicate pairs lie in different CV folds. Pairs straddling a checkpoint's validation fold (training copy vs validation copy): 7 for `ft_fold0`/`p1_fold0` (fold 0), 12 for `p1_fold1` (fold 1), 7 for `p1_fold2` (fold 2). This makes validation scores, early stopping and the regenerated thresholds marginally optimistic, and slightly weakens the "held-out vs trained-on" AUROC gap used as split evidence in Step 3.5 (it does not invalidate it: ≤12 of ~4,448 records). The 1 both-in-test pair is not leakage but double-counts a (label-conflicted) recording.

## 5. Implications for the Step 3.5 test evaluation

- Scope: 17 records = 0.38% of the test fold; 12 (0.27%) with matching labels. By construction, record-level bootstrap CIs assume independence, so the CIs already ignore all such dependence.
- Upper bound reasoning (not a recomputation): at most ~0.27–0.38% of records can be affected, so micro-level metrics (micro-F1/AUROC/AUPRC, Hamming loss, exact match) cannot move by more than a few tenths of a percentage point. Macro metrics are the more sensitive ones because rare classes have few test positives: exposed test records concentrate in SR (6), TWC (5), SVT (3 of 76 test positives = 3.9%), AFIB (3 of 158), ST/STTC/AF (2 each), and singles PWC/AQW/SB. A fully "memorised" duplicate could raise one class's F1 by a few hundredths; spread over 44 classes that is on the order of 10⁻³ in macro-F1 and less in macro-AUROC. This is a bound on possible bias, **not** a measured effect.
- Whether the models actually memorised these copies is unknown (the train-vs-held-out AUROC gap of ~0.03 shows memorisation exists in general). The 5 label-conflicted pairs would, if anything, work against the model.
- Conclusion: **the Step 3.5 numbers are probably optimistic by a negligible-to-small amount, but this has not been measured.** The conclusions drawn (the ranking of checkpoints, reproduction of the notebook) are unlikely to change.

## 6. Recommended Step 3.7 (not performed here)

1. **Sensitivity analysis, no retraining:** from the already saved `evaluation_predictions.npz`, recompute the headline metrics (and paired differences) excluding the 17 cross-boundary test records, and separately excluding all 23 duplicate-involved test records (+ a variant keeping only the 12 identical-label ones if desired), reporting changes with the same bootstrap. Pre-declare that the original Step 3.5 table stays the primary result and the exclusion is a sensitivity check.
2. **Duplicate-aware splitting for Step 4:** group records by file hash (and by a shared-lead hash) and keep each group in one fold; fix `folds[9]` as the test fold but remove, from any *new* training pool, the 17 training-side partners of test records (and consider dropping the 19 out-of-vocabulary partners), so no new layer sees a copy of a test recording. Store the explicit fold assignment (record ID → fold) in the repo instead of relying on regeneration.
3. Optional: a broader near-duplicate search (shift/crop-tolerant, correlation-based) and an AF/AFIB label-consistency review.

## 7. Limitations and what is missing

- Fold membership relies on the reconstructed record order (strong but not absolute evidence in Step 3.5); a few swapped records would change individual pair classifications, not the overall picture.
- Only exact (byte) and identical-lead duplicates were searched. Near-duplicates (resampled, time-shifted, filtered, noise-added, truncated copies) and same-patient recordings with different waveforms cannot be detected without patient IDs, which the dataset lacks; true patient-level leakage is therefore a lower bound here.
- Which training records each checkpoint actually saw is taken from the notebook training code, not from stored split files.
- The Step 2 script and its full pair list were not preserved; pairs were recomputed (matching totals 87 / 174).
- The 19 pairs with a member outside the experiment set cannot leak into this experiment, but they show the dataset contains labelled/unlabelled copies of the same recording.

## 8. Files

`results/step3_6/duplicate_leakage_report.md` (this file), `duplicate_pair_summary.csv` (87 exact pairs; partial-duplicate rows would appear here but there are none), `duplicate_summary.json`, `exposure_counts.json`, `audit.log`. Code: `scripts/step3_6_duplicate_audit.py`, `scripts/step3_6_exposure.py` (new; no existing file modified). `results/` is git-ignored.
