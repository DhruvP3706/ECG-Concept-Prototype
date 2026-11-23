# Step 3.5 — Threshold-file audit

Legend: **[V]** verified by code/experiment · **[H]** inference · everything here uses **validation folds only** (folds 0,1,2,5); the test fold was not read by the audit (`scripts/step3_5_threshold_audit.py`, `scripts/step3_5_bf16_diag.py`).

## Method

The notebook saved a threshold file together with a checkpoint at the epoch of best validation macro-F1. Therefore, if a file belongs to a checkpoint, applying it to that checkpoint's probabilities on its own validation fold must give back the macro-F1 the notebook printed when it saved them, and re-running the notebook's search (`find_optimal_thresholds`: per class, grid `np.arange(0.1,0.9,0.05)`, strict `>`, first maximum wins, fall back to all-0.5 if that scores higher) should land on (almost) the same values. Notebook-recorded validation macro-F1 values: p1_fold0 0.4541 (cell 19), ft_fold0 0.4211 (cell 20), p1_fold1 0.4321 and p1_fold2 0.4432 (cell 21). Rule fixed before looking at results: **VERIFIED** iff |F1(file thresholds, own val fold) − recorded| ≤ 0.003; otherwise UNVERIFIED.

Two precisions were run: fp32 (primary) and CPU bf16 autocast (an approximation of the original RTX-4060 bf16 evaluation, which the GTX 1650 cannot run natively).

## Per-file results

| File | Where written [V] | Val fold | F1 recorded | F1 with file (fp32 / bf16) | F1 regenerated (fp32 / bf16) | File = regenerated (classes, fp32 / bf16) | Status |
|---|---|---|---|---|---|---|---|
| `lightv2_ft_fold0_thresh.npy` | cell 20: `np.save(...)` beside `torch.save` at best FT epoch (ep 25, F1 0.4211) | 0 | 0.4211 | 0.4215 / 0.4214 | 0.4221 / 0.4214 | 39 / 42 of 45 | **VERIFIED** |
| `lightv2_p1_fold0_thresh.npy` | cell 19 (fold index 0): saved beside the checkpoint at ep 70 (F1 0.4541) | 0 | 0.4541 | 0.4519 / 0.4542 | 0.4521 / 0.4542 | 41 / 44 of 45 | **VERIFIED** |
| `lightv2_p1_fold1_thresh.npy` | cell 19 (fold index 1) — **interrupted**; see below | 1 | 0.4321 | **0.3900 / 0.3900** | 0.4326 / 0.4321 | **9 / 9 of 45** | **UNVERIFIED — stale, do not use** |
| `lightv2_p1_fold2_thresh.npy` | never written (cell 21's p1 loop saves only the checkpoint; the FT block saves thresholds only if FT improved, which it did not) | 2 | 0.4432 | n/a | 0.4439 / 0.4432 | n/a | **NO ORIGINAL FILE** |

- ft_fold0 and p1_fold0: with bf16 emulation the file thresholds reproduce the recorded F1 to ≤ 3e-4 and the re-search reproduces it to ≤ 1e-4 — the remaining 1–6 differing classes are single grid-step flips in near-tie classes (fp32 vs bf16 probabilities). The thresholds, checkpoint, validation fold and search code are mutually consistent. They are the thresholds of these checkpoints.
- **p1_fold1 — why stale.** Notebook cell 19's loop started fold index 1 ("PHASE 1 — FOLD 2/9") and its saved output stops after epoch 25 (val F1 0.4100) with no "best val F1" line — the run was interrupted, but the loop writes `*_thresh.npy` at every improvement, so a threshold file from epoch ≤ 25 of that abandoned run remained. Cell 21 (executed later, exec count 18 vs 12) retrained fold index 1 for 80 epochs (best val F1 0.4321, early-stopped by patience) and wrote only the checkpoint (`torch.save`), never the thresholds. The repo's `p1_fold1.pt` is that later checkpoint [V: regenerated F1 0.4326 / 0.4321 matches 0.4321; the checkpoint also took part in cell 22's ensemble]. The threshold file therefore belongs to a different (earlier, weaker) model: only 9 of 45 values coincide with the correct ones and using it costs ~0.042 macro-F1 on fold 1 (0.390 vs 0.432). Both statements hold in fp32 and bf16.
- **Do not interchange files:** thresholds are fit to one model's probabilities; no file was applied to another checkpoint.

## Regenerated thresholds (clearly separate from original artifacts)

`results/step3_5/regenerated_thresholds.npz` — fp32 probabilities, notebook search, **own validation fold only** (`p1_fold1` → folds[1]; `p1_fold2` → folds[2]; `ft_fold0`/`p1_fold0` → folds[0], used only as a secondary row); key `ensemble_cv` = search on the 13,344 combined validation records (ft_fold0@fold0, p1_fold1@fold1, p1_fold2@fold2), as in notebook cell 22. These are **regenerated, not original**. No fold-2 file was written to `models/pretrained/`, and nothing was fitted on the test fold.

Ensemble check against the 40 values the notebook printed in cell 22 (classes with |t − 0.5| > 0.1): 35/40 exact in fp32, 37/40 exact under bf16 emulation; every mismatch is one 0.05 grid step. Of the 5 classes the notebook did not print, 4 are within ±0.1 of 0.5 here (STTU is the exception). This independently confirms that the three ensemble checkpoints in `models/pretrained/` and folds[0..2] are those the notebook used.

## Rule used in the evaluation

| Checkpoint | Primary thresholds | Other rows (labelled) |
|---|---|---|
| ft_fold0, p1_fold0 | original file (VERIFIED) | regenerated (secondary), fixed 0.5 (diagnostic) |
| p1_fold1 | regenerated on folds[1] | original file shown as UNVERIFIED diagnostic only; fixed 0.5 |
| p1_fold2 | regenerated on folds[2] (no original exists) | fixed 0.5 |
| 3-model ensemble | regenerated on combined val folds | fixed 0.5 |

## Residual uncertainty

- [H] Regenerated thresholds come from a single validation fold (≈4.4k records); classes with few validation positives have noisy thresholds (e.g. 5 positives on a fold). Thresholds for such classes should be read as unstable.
- [H] bf16 vs fp32 changes a few thresholds by one grid step; the effect on test metrics is small (original vs regenerated thresholds differ by ≤ 0.001 macro-F1 for ft_fold0 and p1_fold0).
