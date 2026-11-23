# Experiment 9 — Atrial content substitution and counterfactual analysis (frozen LightECGNetV2 ft_fold0)

Question: does the **content and temporal structure** of the atrial-region signal influence the model's rhythm output beyond the effect of removing/perturbing signal information?
Model-sensitivity study only. No claim of physiological mechanism, clinical validity or cardiologist validation. Donor segments are positional windows (r-160..r-51), **not** verified fibrillatory-wave segments.

## 0. Integrity and reuse
- Checkpoint `lightv2_ft_fold0.pt` SHA-256 `7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72`: verified before and after the run (weights also compared in memory) and again after analysis. No training, no weight change.
- Validation role only; no test_reserved, no PTB-XL. Experiments 1–8 untouched (read only). No packages installed, no Git operations.
- Population, beats, region convention (r-160..r-51, region start ≥220 samples after previous R, ≥4 eligible beats), preprocessing, thresholds, bootstrap (record-level, B=5000, seed 20261120) are those of Experiment 8 (`records_beats.json` SHA verified equal to Exp 8's recorded hash).
- Reproduction check against saved Experiment 8 logits (bitwise): orig 846 records, flatten 536, RR-only 284, RR+flatten 284 — **max |Δlogit| = 0.0** in all. The Exp 8 flatten and RR-factorial results are therefore re-generated, not re-estimated differently.

## 1. Feasibility findings (stage1_audit.json)
- Recipient pools (≥4 eligible beats): SR 466, AFIB 77 (Exp 8 flatten accepted 461 / 75; the other usable records are ineligible because of short RR).
- Donor transplant is defensible when applied to the **deviation** (x − line through the region end samples) and added to the recipient's own baseline line: identical length (110 samples), region end values unchanged (zero boundary step), QRS/RR/non-target samples untouched. Raw-sample substitution changes the baseline and creates steps (median end step 75 counts per lead, p90 135); it is kept only as a documented variant and is mostly rejected (21/677 SR accepted).
- Window adequacy: unchanged from Exp 8 — good for SR (coherent pre-QRS deflection, SNR median 21 in lead II), only positional for AFIB. Region-length variants 120/200 run as sensitivity analyses.
- Spectra of deviations (fractions of energy in 0–4/4–9/9–20/20–250 Hz): SR recipients 0.50/0.26/0.13/0.12; AFIB donors 0.53/0.21/0.11/0.16 — similar, so donor and recipient segments differ in waveform/timing more than in broad spectrum.
- Scale compatibility: donor/recipient per-lead region RMS ratio median 0.85 (p5–p95 0.66–1.06) for the first matched donor; QC accepts a median ratio within [0.25, 4].

## 2. Donor selection (donors.json, donor_selection.csv)
- Source: **validation-role records only** (same population as Exp 8). Training-role records were not used: the checkpoint was fine-tuned on them (possible memorisation confound) and the task restricts access to the validation role. Recipient and donor classes are disjoint for the primary analysis (SR recipients ← AFIB donors); SR-class donors and AFIB recipients exclude the recipient itself.
- No model output used. Distance = |ln(mean RR ratio)| + median over leads |ln(region-deviation RMS ratio)|; greedy in a deterministic order (CRC32 of record id + seed) with a usage penalty (0.02 per prior use) and hard cap (AFIB donors 53, SR donors 6). Per recipient: 5 matched AFIB donors, 3 matched SR donors, 3 random AFIB donors (seeded, unmatched).
- Usage: all 77 AFIB donors used (17–51 times, median 35); 464/466 SR donors used (max 6). Matched distance median 0.28 (p5 0.12, p95 0.52). Same lead = same row index; each recipient region receives a different donor region (cycled starting at a deterministic offset).

## 3. Transformations (src/ecg_concept/atrial_content.py, scripts/content_core.py)
All act on per-region deviation dev; y = L_recipient + new_dev (Exp 8's flatten is new_dev = 0, verified identical by test and by logits).
- **flat**: Exp 8 operation. **rev**: dev reversed in time (same energy; ends stay zero). **shuf**: dev cut into 5 equal blocks (22 samples ≈ 44 ms, about half a P-wave duration), one permutation per region shared by all leads (not identity, not reversal), end-to-end ramp removed afterwards so region ends are unchanged (this changes energy slightly).
- **dAF/dSR/dRND**: donor deviation (matched AFIB / matched SR / random AFIB). **dAFsc**: donor deviation scaled per lead to the recipient's region RMS. **rawAF**: donor raw samples.
- **Matched sham** (new, replaces the Exp 8 sine sham): surrogate of the real modification m = new_dev − old_dev with the same per-lead per-region amplitude spectrum and RMS, random phase common to all leads, end ramp removed. Two realisations for flat/rev/shuf; one per donor (5 AFIB, 3 SR) for donors.
- Sensitivity: 120/200-sample windows, lead subsets (II, V1, limb, chest), RR-CV 0.20 factorial (same RNG as Exp 8).

## 4. Population, acceptance, QC (items.csv: 49,746 items incl. rejected, with reasons)
SR recipients accepted / 677 (466 eligible): flat 461, rev 413, shuf 266, dAF0 386 (83% of eligible), dAFsc0 332, dRND0 406, dSR0 376, rawAF0 21, sh_flat 323, sh_rev 311, sh_shuf 258, shAF0 229 (49% of eligible), dAF0_120 384, dAF0_200 242. AFIB recipients (77 eligible): flat 75, rev 71, shuf 42, dAF0 69, dSR0 71, shAF0 39, rawAF0 2.
- Main rejection reasons: ineligible short RR (211 SR, 92 AFIB) then **discontinuity** (shuf 199, shAF0 226, rawAF0 445, dAF0 75) and amplitude; beat-timing changes are rare (4–13).
- Dependence: accepted vs rejected-among-eligible records do not differ materially in mean RR or RR-CV; rejected shuffle/donor records had somewhat higher (less negative) baseline AFIB logits (shuf −16.2 vs −17.6; dAF0 −16.2 vs −17.3). Eligibility itself selects slower rates. Recipient class: dAF0 83% (SR) vs 90% (AFIB). Lead dependence: lead-subset acceptance 56–59% (SR) is similar to all-lead donors; individual leads were not rejected separately.
- Sham matching: RMS ratio sham/real 5–95% 0.9998–1.0002 for all families. Band-energy profile L1 distance (0 identical, 2 disjoint): 0.17–0.22 for flat/shuffle/donor shams but **1.08 for the reversal sham (poor spectral match)**. Boundary roughness: shams are rougher than the real edits (max first-difference ratio 0.59–0.70 vs 0.26–0.54), i.e. they are the more disruptive modification locally. Real median modification RMS (counts): flat 34, rev 42, shuf 42, dAF 44, dSR 37.
- Paired analyses use records where both real and sham are accepted, so n differs from unpaired rows (stated per table; sham-acceptance is lower than real-acceptance).

## 5. Results (SR recipients, primary; within-record change, mean [95% record-level bootstrap CI])
| Intervention | n | ΔAFIB | ΔSR | AFIB→pos flips | SR→neg flips |
|---|---|---|---|---|---|
| flatten (Exp 8) | 461 | +8.25 [7.75, 8.77] | −2.32 [−2.47, −2.19] | 1.3% | 19.7% |
| time reversal | 413 | +9.09 [8.68, 9.48] | −1.51 [−1.62, −1.40] | 0% | 5.6% |
| block shuffle | 266 | +6.57 [6.02, 7.12] | −0.77 [−0.89, −0.65] | 0% | 2.3% |
| AFIB donors (5, matched) | 457 | +8.29 [7.88, 8.71] | −1.27 [−1.39, −1.15] | 0.5% | 6.3% |
| AFIB donors, amplitude-scaled | 425 | +7.89 [7.48, 8.30] | −1.12 [−1.23, −1.02] | 0.2% | 4.8% |
| AFIB donors, random (unmatched) | 457 | +8.21 [7.80, 8.63] | −1.26 [−1.38, −1.14] | 0.6% | 6.6% |
| AFIB donors, raw samples | 39 | +9.85 [8.44, 11.26] | −1.79 [−2.10, −1.51] | 0% | 10.3% |
| **SR donors (3)** | 453 | **+1.03 [0.82, 1.24]** | −0.10 [−0.16, −0.03] | 0.3% | 1.4% |
| sham of flatten | 347 | +1.18 [1.03, 1.34] | −0.07 | | |
| sham of reversal | 334 | +1.62 [1.44, 1.79] | −0.17 | | |
| sham of shuffle | 289 | +1.98 [1.76, 2.21] | −0.28 | | |
| sham of AFIB donors | 320 | +1.90 [1.71, 2.09] | −0.16 | | |
| sham of SR donors | 326 | +1.37 [1.20, 1.55] | −0.16 | | |

Net of paired matched sham (real − sham, same records): flatten +6.70 [6.15, 7.28] (n 346); reversal +7.25 [6.85, 7.64]; shuffle +4.27 [3.76, 4.78]; **AFIB donors +6.27 [5.82, 6.74]** (n 316; ΔSR −1.02 [−1.15, −0.90]); SR donors −0.46 [−0.70, −0.21]. Distributions of individual records are in `figures/individual_responses.svg` (AFIB-donor and flatten responses are broad, 0 to +18, with 98% of records increasing).
Secondary (net of sham, AFIB donors): AF(flutter) +4.49 [4.16, 4.81], SB +0.53, STTC −0.52 [−0.65, −0.39]; the flutter logit moves almost as much as AFIB logit.

AFIB recipients (secondary, n=77 eligible; interpret with caution): AFIB donors ΔAFIB −0.10 [−0.17, −0.04], ΔSR +0.38 [0.18, 0.58]; flatten −0.01 / −0.03; reversal −0.16 / +0.32; shuffle −0.10 / +0.82. **SR-class donors inserted into AFIB records**: ΔAFIB −0.95 [−1.23, −0.71], ΔSR +2.80 [2.50, 3.10]; 30.9% of AFIB records lose the AFIB label (net of sham: ΔAFIB −0.46, ΔSR +1.22; n 51).

## 6. Critical questions
1. **AFIB-donor substitution vs matched sham:** yes. Net +6.27 [5.82, 6.74] over the same-record sham (sham itself +1.90).
2. **Different from flattening?** On the AFIB logit no: donor − flatten = −0.02 [−0.24, +0.20] (n 457); net-of-sham difference −0.26 [−0.56, +0.04]. The SR logit falls less with donors (difference +1.07 [1.01, 1.14]); the donor − flatten AFIB difference depends on window length (below).
3. **Beyond generic signal disruption?** The AFIB-logit effect exceeds matched generic disruption by ~6.3 logits, but it is **not donor-specific**: random unmatched AFIB donors (+8.21; difference to matched +0.08 [−0.06, +0.22]), flattening, reversed own content (+9.09) and shuffled own content (+6.57) give comparable or larger effects, and an equally energetic, equally positioned **SR-class donor** changes the AFIB logit by only +1.03 (AFIB donor − SR donor +7.25 [6.90, 7.59]; SR-donor effect is at the sham level).
4. **Time reversal:** similar in sign, larger: +9.09, i.e. reversal exceeds AFIB donors by 0.83 [0.54, 1.12]. (Reversal sham is spectrally poorly matched, so its net value is least reliable.)
5. **Shuffling:** same direction, smaller (+6.57; donors exceed shuffle by 1.48 [1.18, 1.80], n 264, a non-random subset because 43% of shuffles are rejected for discontinuities).
6. **Consistency:** net AFIB-donor effect by donor rank 0–4: +5.99, +5.76, +6.16, +5.93, +5.94 (CIs overlap); 67 donors with ≥5 recipients: all positive on average, donor means p5–p95 2.4–8.2; within-recipient SD across 5 donors median 1.6 logits. **Leads:** the effect needs many leads: flatten limb-only +3.14, chest-only +0.47, II only +0.27, V1 only +0.18 vs all 12 leads +8.25 (donor 0: limb +4.20, chest +1.87, II +0.36, V1 +0.62 vs +8.42); net of sham limb +3.23, chest −0.34.
7. **Content or artefact?** Not boundary/energy artefact: region ends and QRS are fixed, shams that are rougher and equally energetic give +1–2 logits, and the SR-class donor with identical handling gives +1.0. But it is also not evidence for AFIB-specific content: the output is consistent with the model reacting to the **absence of a correctly placed, recipient-consistent phase-locked P-wave-like deflection** in the window (flatten, AFIB donor, reversed or shuffled P all remove it; another SR record's window restores one). This is a hypothesis; P-wave presence was not directly measured per beat.
8. **Opposite directions:** in SR recipients yes: AFIB up and SR down in 87% of records (r = −0.71 between the two changes for AFIB donors). In AFIB recipients the changes are small (AFIB −0.10, SR +0.38).
9. **Specificity:** not AFIB-specific; atrial flutter logit rises almost as much (+5.8 for donors, +6.0 flatten), SB/SA rise, STTC falls.
10. **Robustness:** AFIB-donor effect stays large for 120/160/200-sample windows (+7.46/+8.29/+8.66; net of sham +5.04/+5.99/+5.87), flatten decreases with length (+8.70/+8.25/+6.89), so donor − flatten changes from about −1.2 to +1.8; same-record check (n 92) agrees (flatten 7.54/7.20/6.10; donor 6.58/8.15/8.61). Matched vs random donors and amplitude scaling change the result by ≤0.3. Conclusions about donor vs flatten are therefore window-dependent; the donor ≫ SR-donor and donor ≫ sham results are not.

## 7. Combined with Exp 7/8 (RR-CV 0.20, SR records with every condition accepted, n=117; subset of the 282 in Exp 8)
Mean AFIB logit: original −17.35; RR-only −8.26; +flatten −4.22; +AFIB donors (3) −3.44; +matched sham −7.54; natural AFIB (CV 0.17–0.23, n 70) +2.46. Remaining gap after RR-only 10.73 logits; closed by flatten 38%, AFIB donors 45%, sham 7%. Change vs RR-only: flatten +4.04 [3.49, 4.60], AFIB donors +4.83 [4.37, 5.29], sham +0.72 [0.50, 0.95]; donor − flatten +0.79 [0.54, 1.04]. AFIB-positive records: 0% (RR-only), 6.8% (flatten), 6.8% (donors), 0% (sham). Atrial content substitution explains additional prediction change beyond RR variability, but not more than ≈ the flatten effect, and over half of the gap remains unexplained. Earlier values from Exp 8 (previously observed): flatten closes 44.6% of the gap (n 282); the present subset differs, bitwise-equal logits confirm the same transformations.

## 8. Limitations and confounders
Positional atrial window (no P-wave delineation; AFIB window not a verified f-wave segment); beat detection automatic; eligibility excludes fast-rate records (SR 211, AFIB 92 ineligible), so results do not apply to them; rejection of shuffles/shams (up to 51%) means paired subsets are not random samples; reversal sham spectrally mismatched; shuffle changes energy slightly; donors drawn from the same (validation) role, a small AFIB pool (77); the AFIB-recipient analysis is small and rests on 31–77 records; interventions apply to 12 leads jointly (lead-wise effects non-additive); all 3-donor/5-donor averages weight donors equally; one checkpoint, one split; no cardiologist validation. A larger AFIB-logit change is not evidence that the model recognises AFIB physiology.

## 9. Artifacts and hashes (this run dir `results/xai_atrial_content/run_20261120_154832/`)
items.csv `20de4f34…a8964`; logits.npy `443beed3…3150d`; analysis_results.json `40f9fe3b…f8c46` (full tables, bootstrap summaries, per-donor stats, lead/length/donor consistency); stage1_audit.json `27b079b5…45c3`; config.json `4f7a2827…b04bb8`; donors.json `c366ba42…2b4e`; donor_selection.csv `d23cb28e…e673` (5,973 rows); records_beats.json (= Exp 8 copy); bank/ (donor deviation bank); example_signals.npz; run_log.txt; `figures/` (content_effects, individual_responses, robustness, factorial_rr_content, acceptance, waveform_examples; all SVGs rendered with Edge and inspected); `_old/` = first analysis draft (identical numbers; figures with two layout defects, superseded).
Code: src/ecg_concept/atrial_content.py, scripts/content_core.py, xai_atrial_content.py, xai_atrial_content_analysis.py; tests/test_atrial_content.py (7 tests). Reproduce: `.venv\Scripts\python.exe -I scripts\xai_atrial_content.py population | run <dir> | analyze <dir>`.
Tests (all OK): atrial_content 7, atrial_ops 8, graded_rr 7, rr_warp 11, beat_regions 8, competitor_rules 6, contrastive 6, lead_attention 7, attribution 11.
