# Experiment 10 — P-wave-like atrial patch transplantation into AFIB recipients (frozen LightECGNetV2 ft_fold0)

Question: does introducing naturally occurring, **P-wave-like** SR atrial structure into AFIB ECGs lower the AFIB logit, and does the effect depend on the temporal position of the transplanted patch relative to the QRS?
Model-sensitivity study only. "P-wave-like" means a template-localised phase-locked pre-QRS deflection; it is **not** an expert-validated P-wave delineation. No physiological or clinical claim.

## 0. Integrity
- Checkpoint SHA-256 `7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72` verified before and after the run (weights also compared in memory) and again after analysis. No training. Validation role only; no test_reserved, no PTB-XL. No installs, no Git operations. Experiments 1–9 untouched (the Experiment 9 region bank was read only).
- Re-generated flatten and original logits are bitwise identical to Experiment 9 (orig 169 records, flatten 75 records, max |Δlogit| = 0.0), so Exp 8/9 operations are reproduced exactly.

## 1. P-wave localisation: feasibility (stage1_audit.json, pwave_localization_sr.csv, figures/localization.svg)
- The repository has R-peak detectors only; no P-wave delineator exists and none was installed. I used a transparent template method on the Exp 8/9 region (r-160..r-51): per SR record, mean deviation over eligible beats, lead-pooled energy peak = lag; record accepted if template SNR ≥ 5 and ≥ 4 beats correlate ≥ 0.6 (±20 samples) with the template; beat accepted individually by the same correlation rule.
- SR records (466 with ≥4 eligible beats): median SNR 8.8; median lag 76 samples = 152 ms before R (5–95%: 64–95 samples). **AFIB recipients under the same localiser: SNR median 1.3, 0% reach 5** — no phase-locked deflection, as expected.
- A defensible localisation is therefore possible for most SR records, but only as a record-level phase-locked deflection verified per beat. Not handled: PR-interval variability beyond what the per-beat correlation tolerates, AV block, expert verification, per-beat onset/offset. Lead-specific quality is not selected per lead (leads are pooled).
- Overlapping T waves: eligibility rule from Exp 8 (region starts ≥220 samples after the previous R). Short RR and ectopy: short-RR records are excluded by eligibility; ectopic/mixed rhythms are excluded only via the Exp 7 annotation-based states (e.g. AFIB 166 accepted, 3 detector-disagreement; SR 660 accepted plus flat-lead/clipping cases) and the beat-correlation rule.

## 2. Populations
- AFIB recipients: 169 usable annotated records → **77 eligible** (≥4 eligible beats; 92 excluded for short RR). Selection bias: eligible median mean RR 805 ms vs ineligible 554 ms, so fast AFIB is under-represented; results do not apply to fast AFIB. Baseline: 72/77 eligible records are AFIB-positive at the frozen threshold; mean AFIB logit +2.42, SR logit −7.14.
- SR donors: 466 candidate records → **222 accepted donors** (2,147 validated beats). Exclusions (non-exclusive): lag outside the feasible range −88…−72 samples 207, SNR < 5 76, too few beats matching the template 14. The lag range is chosen so every offset in the design is placeable inside the protected region (donor choice by P-lag feasibility only).

## 3. Donor selection (donors.json, donor_selection.csv)
Deterministic; no model output used. Distance = |ln(mean RR ratio)| + median over leads |ln(donor patch RMS / recipient region RMS)|; greedy in a CRC32-seeded order with usage penalty 0.05 and cap 3. Per recipient 5 matched donors plus 3 random donors (seeded). Recipient (AFIB) and donor (SR) classes are disjoint; validation role only (training records not used: the checkpoint was fine-tuned on them). 152 of 222 donors used (median 2, max 3 recipients each); matched distance median 0.43 (p5 0.17, p95 0.86). The pool is small relative to the matched-distance spread, so matching quality is limited.

## 4. Interventions (src/ecg_concept/pwave_ops.py, scripts/pwave_core.py)
Region = r-160..r-51 as in Experiments 8/9. All interventions act on the deviation from the region's straight baseline (region ends and QRS/non-target samples bit-identical).
- **flat** (reference, Exp 8). **Pal (aligned P-like patch)**: flatten the recipient region, then add a donor patch (40 samples = 80 ms: 25 before and 15 after the donor's lag, cosine-tapered 6 samples) placed with its peak at the donor's own lag (natural PR relationship to the recipient's R). Each recipient beat receives a different donor beat (cycled).
- **Pd<offset> (shifted)**: same patch, peak moved by −45, −30, −15, +10 samples (−90…+20 ms; negative = earlier). Later shifts are limited to +10 (and only for donors with lag ≤ −78) by the 2-sample margin before the QRS protection window; the +10 condition is accepted for 25/77 records.
- **shP (matched sham)**: flatten + random-phase surrogate of the donor patch (same per-lead amplitude spectrum, RMS, taper and position; no P-like waveform). **shM**: spectrum-matched surrogate of the total modification (Exp 9 sham).
- **Ctl (non-P control)**: donor deviation segment from region samples 5–45 (offsets −155…−115, before every accepted P window), amplitude-scaled per lead to the P patch RMS, placed at the aligned position. **Full**: whole donor region deviation (validated beats). **Prnd**: aligned patch from random donors. **PalW/PalN**: 50- and 30-sample windows. Lead subsets: II, V1, limb, chest.
- Combined: RR-CV warped to 0.03 (SR-like; rigid atrial region as in Exp 8), then flatten / patch / sham.

## 5. Acceptance and QC (items.csv: 11,323 items with reasons; figures/acceptance.svg)
QC (all items): QRS windows, non-target samples, shape and dtype bit-identical; first-difference ratio ≤ 1 of the record's own p99.9; amplitude/clipping; modification RMS ≥ 2 counts; beat detector re-run returns the same beats (±2 samples); patch/recipient RMS ratio in [0.1, 5] (median 1.44); sham/real RMS ratio in [0.8, 1.25].
Accepted of 77 eligible recipients: flat 75, Pal0 68, shifted −45/−30/−15: 71/71/70, +10: 25 (47 infeasible placements), shP0 68, shM0 50, Ctl0 34 (43 discontinuity rejections), Full0 63, Prnd0 69, rr_only 26 (50 RR-warp quality rejections, all in the faster half), rr_Pal0 23. Rejections by group (slow/fast, split at mean RR 805 ms): Pal0 33/39 vs 35/38; Ctl0 14/39 vs 20/38; RR-warp factorial 26/39 vs 0/38. Rejected vs accepted records differ little in RR-CV and baseline logit. Lead-subset acceptance is 90–95% (II, V1, limb, chest).
Sham matching: median RMS ratio shP/real 1.00 (5–95% 0.93–1.08), band-energy L1 0.17; shM ratio 1.00, L1 0.22. Real modification RMS (counts, median): flat 27, Pal 36, Ctl 34, Full 40; sham of patch 36. Shams are locally rougher than the real patches (max first-difference ratio 0.60 vs 0.49).

## 6. Results (AFIB recipients; mean [95% record-level bootstrap CI]; paired within record)
| Condition | n | ΔAFIB | ΔSR | AFIB label lost | SR label gained |
|---|---|---|---|---|---|
| flatten | 75 | −0.01 [−0.10, +0.08] | −0.03 [−0.30, +0.22] | 6.7% | 0% |
| **aligned P-like patch (5 donors)** | 76 | **−1.17 [−1.48, −0.89]** (median −0.84) | **+3.54 [+3.23, +3.84]** | **34.5%** | 0% |
| matched sham of patch (shP) | 76 | −0.67 [−0.83, −0.53] | +2.66 [+2.38, +2.95] | 21.7% | 0% |
| total-modification sham (shM) | 55 | −0.30 [−0.41, −0.20] | +1.71 | 11.5% | 0% |
| non-P donor segment (Ctl) | 64 | −0.33 [−0.47, −0.21] | +1.52 | 12.5% | 0% |
| full-region donor segment | 76 | −1.20 [−1.52, −0.89] | +3.37 | 34.6% | 0% |
| aligned patch, random donors | 75 | −1.27 [−1.63, −0.96] | +3.84 | 36.0% | 0.9% |
| aligned patch, 50 / 30-sample window | 75 | −1.19 / −1.13 | +3.61 / +3.48 | 36% / 35% | 0% |

Net of paired matched sham: aligned patch **−0.50 [−0.77, −0.30]** AFIB (n 75; SR +0.86 [+0.69, +1.05]); vs total-modification sham −0.82 [−1.26, −0.47] (n 54). Aligned patch − flatten −1.13 [−1.45, −0.86]; − non-P control −0.75 [−0.99, −0.54]; − full-region segment +0.03 [−0.03, +0.08]; − random-donor patch +0.10 [−0.23, +0.35]. Individual responses (figures/individual_responses.svg): 92% of records move AFIB down and SR up; the response is skewed (median −0.84, p10 −2.6, p90 −0.09).
**Other logits (aligned patch, net of sham):** SR +0.86, SB +0.83, SA +0.87, ST +0.43, AF (flutter) −0.42 [−0.63, −0.26], STTC −0.23, TWC −0.10, TWO −0.09. Raw changes: SB +3.29, AF −1.06.

### Temporal offsets (same records and donors; 73 records, 206 record-donor units for −45…0)
| shift | ΔAFIB | ΔSR | ΔAFIB vs aligned |
|---|---|---|---|
| −45 | −0.75 [−0.88, −0.63] | +1.95 [+1.62, +2.30] | +0.38 [+0.10, +0.70] |
| −30 | −0.85 [−1.03, −0.70] | +2.59 | +0.28 [+0.02, +0.59] |
| −15 | −0.92 [−1.12, −0.73] | +3.17 | +0.21 [+0.05, +0.43] |
| 0 (aligned) | −1.13 [−1.48, −0.83] | +3.54 [+3.22, +3.88] | 0 |
| +10 (56 records) | −1.15 [−1.57, −0.79] | +3.56 | −0.01 [−0.11, +0.10] |
Reference lines on the same records: flatten −0.02, matched sham −0.67, non-P control −0.28 (AFIB); SR +0.01, +2.70, +1.47. The effect is graded and approximately monotonic with proximity to the donor-natural position, not a sharp optimum, and the −45 point is at the matched-sham level.

### Heart-rate group, donors, leads
- Slow AFIB (mean RR ≥ 805 ms, n 39): aligned patch −1.55 [−2.13, −1.05], net of sham −0.74 [−1.24, −0.37], 44% lose the AFIB label; fast AFIB (n 38): −0.79 [−1.03, −0.57], net −0.27 [−0.41, −0.14], 25% lose it. Flatten: −0.15 / +0.12.
- Donors: by rank −0.99, −1.17, −1.04, −1.07, −1.35 (overlapping CIs). Of 85 donors with ≥3 recipients 96% have a negative mean (median −0.97, p5–p95 −2.8…−0.16). Not related to donor template SNR (r = −0.01); donors with earlier lag (≤ −80) show a smaller effect (−0.84 vs −1.24).
- Leads (aligned patch donor 0 vs flatten): all 12 −0.99 [−1.24, −0.75] (net of sham −0.30 [−0.50, −0.11]); limb leads −0.61 (net −0.36); chest −0.32 (net −0.09); lead II alone −0.11; V1 alone −0.03. The effect needs several leads and is carried mainly by limb leads.

### Combined with earlier experiments (slow-AFIB subset, n = 21: RR warp succeeded and every condition was accepted)
| Condition | AFIB logit | SR logit | AFIB-positive | SR-positive |
|---|---|---|---|---|
| original AFIB (new subset) | +2.32 | −7.31 | 95% | 0% |
| RR-CV 0.03 only (new) | +0.99 [0.52, 1.43] | −5.68 | 38% | 0% |
| + flatten (new) | +0.82 [0.26, 1.34] | −5.41 | 43% | 5% |
| + aligned patch (3 donors, new) | −2.68 [−3.65, −1.74] | −2.23 | 0% | 13% |
| + matched sham (new) | −1.14 [−1.98, −0.41] | −2.80 | 0% | 10% |
Patch − sham −1.54 [−2.08, −1.03]; patch − flatten −3.50 [−4.42, −2.58]. Reused values: Experiment 7 (AFIB recipients, RR-CV to SR median 0.02, n 115, different subset): ΔAFIB −1.12 [−1.34, −0.92], ΔSR +2.06; Experiment 9 (full-region SR-class donors, unvalidated, same recipients, n 75): ΔAFIB −0.95 [−1.23, −0.71], ΔSR +2.80; new aligned patch minus that Exp 9 effect on the same records: ΔAFIB −0.22 [−0.36, −0.10], ΔSR +0.73 [+0.49, +0.97]. Exp 8/9 flatten on AFIB records reproduced (−0.01). The factorial result is on the 21 slowest-rate records only (fast records fail RR-warp QC), so it is not representative of all AFIB.

## 7. Answers to the research questions
1. **Lowers AFIB logits?** Yes, modestly: −1.17 [−1.48, −0.89] (34% of records lose the AFIB label; baseline mean +2.42). Flattening alone does nothing (−0.01), so the effect comes from added structure, not from removing the recipient's atrial waveform.
2. **Raises SR logits?** Yes: +3.54 [+3.23, +3.84], but no record crosses the SR threshold in the unmodified-RR setting (0%); SB (+3.3) and SA also rise.
3. **Greater than matched sham?** Yes but by less than the raw effect: net −0.50 [−0.77, −0.30] (SR +0.86); about 57% of the AFIB effect and 75% of the SR effect is reproduced by an equally energetic spectrum-matched random bump at the same position.
4. **Does alignment matter?** Partly: the effect declines gradually as the patch is moved earlier (−0.75 at −45 vs −1.13 aligned; +0.38 [0.10, 0.70]); the −45 shift is at sham level; the later side could not be tested beyond +10 (no further change).
5. **Shifted patches smaller?** Yes (above), by about one third at −45.
6. **Consistent across donors?** Yes in sign (96% of donors), magnitude varies 0.2–2.8; consistent across donor ranks; random unmatched donors are as effective as matched.
7. **Lead dependence?** Yes: limb leads carry the effect; II or V1 alone do almost nothing.
8. **After accounting for energy/boundary?** Partly: net of sham −0.50; boundary steps are absent by construction and shams are rougher than real patches. A non-P donor segment of equal amplitude produces only −0.33, so waveform smoothness/shape matters in addition to energy.
9. **Other rhythm logits?** Not AFIB-specific: SB, SA, ST rise; flutter falls (−0.42 net) similar in size to AFIB.
10. **Robust to localisation choices?** Window width (30/40/50 samples), matched vs random donors, donor SNR and the full-region donor segment give the same AFIB change (−1.1 to −1.3); localisation errors of ±45 samples reduce the effect but do not remove it.

**Interpretation (conservative):** the data support sensitivity of the model's AFIB/SR output to the presence of smooth, localised signal structure in the pre-QRS window, increasing when it sits near the donor-natural P position and in regular rhythm (factorial), and mostly driven by limb leads. They do not show that the model recognises P waves: a spectrum-matched random bump gives >half of the effect, the effect is broad rather than sharply tuned in time, and sinus-bradycardia/arrhythmia logits move with it. Signal-region sensitivity: demonstrated. Content sensitivity: weak (patch beats sham by 0.5 logits; beats a noise-like control by 0.75). Temporal alignment sensitivity: graded, modest. Physiological interpretation: not supported by these data.

## 8. Limitations and selection bias
P-like localisation is template based and unvalidated; only 77/169 AFIB records (slower rates) and 222/466 SR donors qualify; fast AFIB is absent from the factorial; the donor pool is small, so many donors serve several recipients; patch tapering alters P-wave tails; later-than-natural shifts could not be tested; sham is rougher than the real patch while the control is limited by 43 discontinuity rejections; paired subsets differ slightly in n; 12 leads are modified jointly; one checkpoint, one split; no cardiologist validation. Effects are small relative to the Experiment 8 SR-origin effect (+8).

## 9. Artifacts and hashes (`results/xai_pwave_transplant/run_20261120_160659/`; first 20 hex digits)
items.csv `5efd5826ba269245a0aa`; logits.npy `880369e64bdb631ba6be`; analysis_results.json `9f2efc9166beed3988fd` (all tables, bootstrap summaries, donor stats); stage1_audit.json `69eadd786dd41648e881`; config.json `c624899361da4d3f948e`; donors.json `992211513f503e050558`; donor_selection.csv `ed2dc726f00485e3244b` (616 rows); pwave_localization_sr.csv `a84d2794b654e02cd998`; bank/ (donor patches), sr_templates.npz, example_signals.npz, run_log.txt; `figures/` (pwave_effects, temporal_offsets, individual_responses, robustness, localization, factorial_rr_pwave, acceptance, waveform_examples; all rendered with Edge and inspected); `_old/` = first analysis draft (identical numbers, figures with overlapping captions in the localisation panel).
Code: src/ecg_concept/pwave_ops.py, scripts/pwave_core.py, xai_pwave_transplant.py, xai_pwave_analysis.py; tests/test_pwave_ops.py (7 tests). Reproduce: `.venv\Scripts\python.exe -I scripts\xai_pwave_transplant.py population | run <dir> | analyze <dir>`.
Tests (all OK): pwave_ops 7, atrial_content 7, atrial_ops 8, graded_rr 7, rr_warp 11, beat_regions 8, competitor_rules 6, contrastive 6, lead_attention 7, attribution 11.
