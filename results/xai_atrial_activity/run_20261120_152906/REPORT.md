# Experiment 8 — Atrial activity and AFIB discrimination (frozen LightECGNetV2 ft_fold0)

Question: with RR intervals and ventricular morphology held fixed, how does changing the atrial region of the raw ECG move the model's AFIB and SR logits?
This is a model-sensitivity study of the frozen network. It makes no claim of clinical validity, physiological mechanism or cardiologist validation. No training, no weight change, no test_reserved / PTB-XL access.

## 0. Integrity
- Checkpoint `models/pretrained/lightv2_ft_fold0.pt` SHA-256 `7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72`: verified before the run, after the run (in `run_log.txt`; weights also compared), and again after analysis/figures. Thresholds file `lightv2_ft_fold0_thresh.npy` SHA-256 `e722ca94…12e8` (frozen, used for flip rates).
- The 'orig' items reproduce the stored Exp 7 logits exactly (max abs diff 0.0).
- Population: original PhysioNet validation role only (Exp 7 population copied to `population_exp7_copy.json`). SR 677 usable, AFIB 169 usable. Exp 7 artifacts untouched (read only).
- Artifact hashes: items.csv `e57c2fec…9a59`, logits.npy `1c1b54f6…212`, analysis_results.json `f1e18720…c31`, stage1_audit.json `a0a0d8d5…dec`, config.json `d8046cc5…fe`.

## 1. Feasibility audit (Stage 1; details in stage1_audit.json)
- Coordinates: raw 500 Hz, 5000 samples, interventions applied to the raw int16 record and then passed through the original preprocessing. Beat positions come from the Step 4.3 extractor (automatic, not expert-validated).
- The 50-sample pre-QRS region used earlier is **not** a complete P wave. I used r-160…r-51 (≈320–102 ms before R), which covers the P-wave extent in SR records. Only the region location is defensible from SR data; the P onset/offset is not delineated per beat.
- In AFIB there is no discrete P wave to localise; the same window is used as "the atrial-activity region" by position only. This is the main assumption-sensitive element of the AFIB-origin conditions.
- Eligibility: the region must start ≥220 samples after the previous R (previous T wave excluded) and a record needs ≥4 eligible beats. This excludes short-RR (fast) records; see §3.

## 2. Interventions (exact definitions; src/ecg_concept/atrial_ops.py)
Per eligible beat region [r-a1, r-51], per lead, with L = straight line between the region's first and last sample (region ends unchanged, so no step):
- A original.
- B/C gain: y = L + s(x−L); s=0 flatten, 0.5 attenuate, 1.5 and 2.0 amplify. Region-length variants flat_short (a1=120), flat_long (a1=200). In AFIB, s=0.5/1.5/2 are the controlled modification (not arbitrary erasure).
- D matched sham: smooth zero-edge random sine series (4 terms) per lead and region, RMS equal to the flatten modification (median sham/real RMS ratio 0.99997; 5–95% 0.9998–1.0002). Two independent seeds (sham_a, sham_b). Same location and energy, not a removal.
- E synthetic f-wave (SR only): flattened region + 4–9 Hz band-limited noise, per-lead RMS calibrated to the AFIB median region deviation, independent across leads (unrealistic; four seeds). Factorial (SR only): RR-CV moved to 0.20 with a rigid atrial region (`rr_only`), then flatten (`rr_flat`) or flatten+f-wave (`rr_flat_fwave`).

## 3. QC and acceptance (acceptance.svg; analysis_results.json → acceptance)
QC enforced per item: QRS windows (r±40) bit-identical; outside-region samples identical; shape/dtype; max first-difference ratio ≤1.0 of the record's own p99.9; amplitude within record max, no new clipping; modification RMS ≥2 counts; sham/real ratio in [0.8,1.25]; extractor re-run on the modified signal must return the same beats (±2 samples). Accepted and rejected items with reasons are in items.csv.

| Condition | SR accepted/677 | AFIB accepted/169 |
|---|---|---|
| flat | 461 (68%) | 75 (44%) |
| flat_short / flat_long | 592 / 296 | 108 / 49 |
| g05 | 462 | 75 |
| g15 / g20 | 289 / 188 | 45 / 28 |
| sham_a / sham_b | 403 / 405 | 71 / 66 |
| f-wave a–d | 455–456 | n/a |
| rr_only / rr_flat / rr_flat_fwave | 286 / 284 / 282 | n/a |

- Dominant rejection is **ineligibility (short RR)**, not QC: SR flat 211 ineligible, 5 QC-rejected (4 timing, 1 discontinuity); AFIB 92 ineligible, 2 QC-rejected. Among eligible records flat is accepted 99%.
- Amplification (g15/g20) is rejected mostly for discontinuity (171/269 of 466 eligible SR) and amplitude, so the amplified sets are a biased subset (records with small, smooth atrial deviation).
- Selection: accepted records have longer RR (SR 864 vs 684 ms ineligible; AFIB 842 vs 543 ms). RR-CV and baseline logits are similar (AFIB baseline AFIB logit 2.44 accepted vs 2.52 ineligible; SR 0.024 vs 0.024 RR-CV). **Results do not apply to fast-rate records.**

## 4. Results (within-record paired change = modified − original logit; record-level bootstrap 95% CI, B=5000, seed 20261120; patterns averaged within record)
### SR-origin (n=461 for flat)
| Condition | ΔAFIB logit | ΔSR logit |
|---|---|---|
| flatten | **+8.25 [+7.75, +8.77]** | **−2.32 [−2.47, −2.19]** |
| g05 | +1.17 [+1.01, +1.33] | −0.55 [−0.60, −0.51] |
| g15 (n=289) | −0.02 [−0.13, +0.10] | +0.14 [+0.10, +0.18] |
| g20 (n=188) | +0.28 [+0.04, +0.52] | +0.10 [+0.02, +0.18] |
| flat_short / flat_long | +8.72 / +6.89 | −2.23 / −1.97 |
| matched sham (2 seeds, mean) | +1.66 [+1.51, +1.81] | −0.07 [−0.11, −0.03] |
| **flatten − sham, same records (n=413)** | **+6.51 [+6.01, +7.00]** | **−2.22 [−2.36, −2.08]** |
| synthetic f-wave (4 patterns) | +7.55 [+7.16, +7.95] | −0.96 [−1.07, −0.86] |
| f-wave − flatten (n=457) | −0.78 [−1.01, −0.54] | +1.38 [+1.31, +1.45] |

Threshold flips (frozen thresholds): flatten flips AFIB to positive in 1.3% [0.4, 2.4] of SR records and SR to negative in 19.7% [16.3, 23.4]; sham 0% / 1.6%; f-wave 0.3% / 4.8%. The pattern is consistent across the four f-wave and two sham patterns (CIs overlap tightly).
Secondary (flatten − sham, SR-origin): AF (flutter) +4.88 [+4.52, +5.22], SA +1.80, SB +0.73, TWO +0.73, TWC +0.42, ST −0.43, STTC −1.02 [−1.17, −0.88]. Flatten shifts many classes, not only AFIB.
Gain is non-monotonic: the response is concentrated at s=0 (very large) and s=0.5 (≈15% of the s=0 AFIB effect); amplification does almost nothing for AFIB.
Correctness: 437 correct SR records ΔAFIB +8.36; 24 originally misclassified records ΔSR −0.95 (small n).

### AFIB-origin (n=75 for flat)
| Condition | ΔAFIB | ΔSR |
|---|---|---|
| flatten | −0.01 [−0.10, +0.08] | −0.03 [−0.30, +0.22] |
| g05 | +0.17 [+0.11, +0.24] | −0.73 [−0.92, −0.56] |
| g15 (n=45) | −0.21 [−0.28, −0.14] | +1.14 [+0.96, +1.32] |
| g20 (n=28) | −0.34 [−0.52, −0.17] | +2.05 [+1.73, +2.36] |
| flat_short / flat_long | 0.00 / −0.29 | −0.72 / +1.29 |
| matched sham | −0.15 [−0.21, −0.09] | +0.76 [+0.63, +0.89] |
| flatten − sham (paired) | +0.15 [+0.03, +0.27] | −0.83 [−1.13, −0.53] |

AFIB-origin logits barely react: flatten leaves AFIB at its original value (AFIB-positive flips 6.7% [1.3, 13.3] vs sham 3.5%). The SR logit rises with amplification and a matched sham (generic perturbation effect) and the flatten−sham SR difference is negative. Supplementary same-record comparison (supplementary_region_extent.json, n=49 with all three extents) shows the AFIB-origin flatten effect on SR changes sign with region length (short −0.46, 160 +0.08, long +1.29); the SR-origin ΔAFIB shrinks with length (8.2, 8.0, 6.9). The AFIB-origin sign pattern is therefore not a stable dependence on atrial content and is confounded by region length and the slower-rate accepted subset (corr(ΔAFIB, coverage) −0.42, corr with mean RR −0.45).

### Factorial (SR records, n=282; RR-CV 0.20 with rigid atrial region)
| Condition | AFIB logit | SR logit |
|---|---|---|
| original | −16.60 | +4.28 |
| RR only | −7.98 [−8.26, −7.68] | −0.10 |
| RR + flat atrial | −3.32 [−3.76, −2.87] | −1.03 |
| RR + flat + f-wave | −3.07 [−3.46, −2.67] | −0.39 |
| natural AFIB, CV 0.17–0.23 (n=70) | +2.46 [+2.26, +2.65] | −7.74 |
RR variability alone moves the AFIB logit by +8.6 but flips no record to AFIB. Adding flatten closes 44.6% of the remaining 10.4-logit gap to natural AFIB (+4.66 [4.27, 5.06] over RR only); flatten+f-wave 47.0%. AFIB-positive rate: 0% / 8.9% / 5.7%. The remaining >50% gap is unexplained by anything varied here (RR pattern details, ventricular morphology, noise, other features).

## 5. Interpretation (model sensitivity only)
1. Specific vs generic: in SR, removal of the atrial-region deviation has an AFIB effect ~5× that of an equal-energy random modification (+6.5 net), so most of the response is specific to the removal, not to local perturbation magnitude. Part (~20%) of the flatten effect is generic (sham alone +1.7).
2. SR vs AFIB: strongly asymmetric. The model's AFIB logit in SR records is highly sensitive to the absence of P-wave-region structure; AFIB records are essentially insensitive to removing or attenuating their atrial region, and their SR logit responds mainly to generic perturbation.
3. Atrial activity and the RR-matched gap: flattening explains roughly 45% of the remaining gap in SR records at matched RR-CV; the rest does not close.
4. Consistency: two sham seeds, four f-wave seeds and three region extents give consistent SR-origin results (the effect decreases with region length, 8.7→6.9). Amplification and AFIB-origin conclusions are not robust (small/biased subsets, sign changes with region extent).
5. Not shown: the flat region is noise-free and unphysiological. Replacing it with independent-across-leads f-wave noise leaves ~92% of the AFIB effect (+7.55 vs +8.25), so the effect does not depend on smoothness alone but also does not demonstrate that the model reads physiological f-waves; it may respond to "absence of P-wave morphology". A logit change is not evidence that the model has identified a physiological mechanism.

## 6. Limitations
Automatic beat detection; atrial region defined by position (320–102 ms pre-R), not delineated P waves; eligibility excludes short-RR records (selected subset, SR 68% / AFIB 44% coverage); AFIB = atrial fibrillation and AF = atrial flutter in this dataset (flatten also raises the AF/flutter logit by ~5, so the AFIB response is not AFIB-specific); AFIB-origin n is small (75, 28 for g20); amplification subsets biased; synthetic f-waves unrealistic; one checkpoint, one split; no clinical validation.

## 7. Artifacts and tests
Run dir `results/xai_atrial_activity/run_20261120_152906/`: config.json, stage1_audit.json, items.csv (12,353 items incl. rejected+reasons), logits.npy (7,132×45), analysis_results.json, supplementary_region_extent.json, records_beats.json, population_exp7_copy.json, example_signals.npz, run_log.txt/log.txt, `figures_v2/` (final: acceptance, atrial_gain_and_sham, factorial_rr_by_atrial, waveform_examples; `figures/` are the first draft with two layout defects, kept for provenance). All SVGs were rendered with headless Edge and inspected.
Code: src/ecg_concept/atrial_ops.py, scripts/atrial_core.py (torch-free worker), xai_atrial_activity.py, xai_atrial_analysis.py; tests/test_atrial_ops.py.
Tests (all OK): atrial_ops 8, graded_rr 7, rr_warp 11, beat_regions 8, competitor_rules 6, contrastive 6, lead_attention 7, attribution 11.
Reproduce: `.venv\Scripts\python.exe -I scripts\xai_atrial_activity.py population | run <dir> | analyze <dir>`.
