# Step 4.3 — Minimal RR-concept extraction pipeline: engineering pilot report

**Verdict: ENGINEERING READY** (all automated engineering checks passed: unit_tests_pass, no_read_failures, determinism_identical, duplicate_identity_all, no_value_with_unavailable_state).

This is an **engineering** pilot on **training-role** records. The extractor works and its limits are documented. There are **no reference annotations**, so no sensitivity, PPV, physiological error or accuracy is computed, and a high detector-agreement rate is **not** evidence of clinical or physiological validity. Outputs are *automatically derived* ventricular-rate values and **10-second RR statistics** (not conventional HRV; not expert-validated). The locked sets are engineering assessment subsets inside the training role; they are not the project's validation or test fold.

## 1. What now runs

| Component | File |
|---|---|
| Detectors (ADAPT: Pan–Tompkins-*style*; TMA: Elgendi-*style*), per-lead → cross-lead fusion, one-to-one matching | `src/ecg_concept/beat_detection.py` |
| Concept formulas, quality flags/states/reason codes, record interface `extract_record` / `extract_path` / `extract_many` | `src/ecg_concept/rr_concepts.py` |
| 20 engineering tests (synthetic) | `tests/test_rr_concepts.py` |
| Pilot runner (sample → develop → freeze → lock → report) | `scripts/step4_3_pilot.py` |
| DEV-S perturbation diagnostic; report renderer | `scripts/step4_3_stability.py`, `scripts/step4_3_render_report.py` |

Both detectors are our own SciPy implementations (no ECG package is installed); they are named after the published methods they take inspiration from, **not verified reproductions**. Parameters are in `extraction_config.json`. Concepts: primary rate = 60/mean(RR); secondary = 60/median(RR); RR coefficient of variation (sample SD); normalised RMS of successive differences; median absolute successive RR difference / median(RR); beat/interval counts; states `ACCEPTED/UNCERTAIN/UNAVAILABLE` with reason codes; missing values are empty/NaN (never 0). Nothing but the signal enters a value or mask (interface test).

Detection preprocessing is independent of the backbone: raw integer signal, per-lead detector band-pass for timing only (zero phase), polarity-agnostic energy detection, |x| localisation. Leads I, II, V1–V6 (8 non-redundant channels, not independent votes). Beats within 100 ms of a record edge are excluded; ≥3 supporting leads (within 50 ms of the cluster's first member, no chaining); detector outputs matched one-to-one (optimal assignment, 50 ms).

## 2. Exact commands

```
.venv\Scripts\python.exe -I tests\test_rr_concepts.py
.venv\Scripts\python.exe -I scripts\step4_3_pilot.py sample
.venv\Scripts\python.exe -I scripts\step4_3_pilot.py develop
.venv\Scripts\python.exe -I scripts\step4_3_pilot.py freeze
.venv\Scripts\python.exe -I scripts\step4_3_pilot.py lock
.venv\Scripts\python.exe -I scripts\step4_3_pilot.py report
.venv\Scripts\python.exe -I scripts\step4_3_stability.py
.venv\Scripts\python.exe -I scripts\step4_3_render_report.py
```

Tests: 20 run, 0 failures, 0 errors (first run had 2 failures that exposed a clipping-plateau false positive on smooth baseline minima, fixed before any pilot record was opened; the other was a test-construction bug).

## 3. Sample counts and processing success

| Set | Type | Units | Records read | Selection |
|---|---|---|---|---|
| DEV-S | stratified stress | 120 | 121 (incl. 1 duplicate partner) | {'stratum': 115, 'fallback': 5} |
| LOCK-S | stratified stress (locked) | 120 | 120 | {'stratum': 110, 'fallback': 10} |
| LOCK-R | representative simple random sample | 100 | 100 | {'srs': 100} |
| DUP-CHECK | remaining within-train duplicate pairs, identical-output check only | 42 pairs | 84 | not sampled units; never in metrics |

Seed 20261111; units = duplicate groups or records; key = SHA-256(seed|unit); draw order LOCK-R → DEV-S → LOCK-S; sets disjoint (asserted). Lists were written and hashed (`pilot_manifest.csv` SHA-256 `4347a4aa1c7d4860…`, list hash `836b91deed5aa7c4…`) **before any waveform was read**. **Shortfall:** stratum S12_LVQRSAL has only 5 eligible training units under the declared priority order (LVQRSAL records overwhelmingly carry an earlier-priority rhythm label); DEV-S received 5 and LOCK-S 0; the remaining quota (DEV-S:S12_LVQRSAL: 5, LOCK-S:S12_LVQRSAL: 10) was filled by the deterministic fallback (next units by hash order from any stratum, marked `fallback`). The stress sets are **not representative**; only LOCK-R supports population-level statements about coverage.

Processing: 425 records read from training `.mat` files (frozen code), **0 read failures, 0 structural failures**; runtime per record (5/25/50/75/95th percentile, s): [0.054, 0.063, 0.069, 0.072, 0.076] (LOCK-R). The lock stage took 33 s for 425 records, so a full extraction over the 35,568 training records is roughly 45 minutes single-threaded (arithmetic estimate; not run).

## 4. Measurement coverage (states are detector-consensus acceptance, not validity)

| Set | Units | ACCEPTED | UNCERTAIN | UNAVAILABLE | Reasons present (record counts) |
|---|---|---|---|---|---|
| DEV-S | 120 | 116 | 2 | 2 | {'DETECTOR_DISAGREEMENT': 2, 'FLAT_LEAD': 5, 'INSUFFICIENT_USABLE_LEADS': 2, 'CLIPPING_SUSPECT': 1} |
| LOCK-S | 120 | 119 | 1 | 0 | {'DETECTOR_DISAGREEMENT': 1, 'FLAT_LEAD': 1} |
| LOCK-R | 100 | 98 | 2 | 0 | {'FLAT_LEAD': 2, 'DETECTOR_DISAGREEMENT': 2} |

Rate and irregularity states coincide in every record here (irregularity needs ≥3 RR; every record with a rate had ≥6 consensus beats, i.e. ≥5 RR). Coverage by stratum (ACCEPTED/units per set; full table in `work/pilot_summary.json`):

| Stratum | DEV-S | LOCK-S | LOCK-R |
|---|---|---|---|
| S01_timestamp_block | 9/10 | 10/10 | 3/3 |
| S02_age_missing | 10/10 | 10/10 | - |
| S03_AFIB | 10/10 | 10/10 | 3/3 |
| S04_AF_flutter | 9/10 | 10/10 | 20/20 |
| S05_SVT_AT | 10/10 | 10/10 | 3/3 |
| S06_ectopy | 10/10 | 10/10 | 5/5 |
| S07_conduction_wide_QRS | 10/10 | 10/10 | 1/1 |
| S08_SA | 9/10 | 9/10 | 7/7 |
| S09_SB | 15/15 | 18/18 | 32/34 |
| S10_ST | 9/10 | 10/10 | 9/9 |
| S11_SR | 10/10 | 12/12 | 15/15 |
| S12_LVQRSAL | 5/5 | - | - |

Batch metadata (aggregate only; all three sets pooled): age_missing=False: ACCEPTED 313/320; age_missing=True: ACCEPTED 20/20; JS08344-09373: ACCEPTED 22/23; other: ACCEPTED 311/317. No batch or stratum shows a coverage deficit beyond the isolated failures listed below, but the sample is small (descriptive only).

## 5. Detector agreement (matching tolerance 50 ms; one-to-one assignment)

| Set | Records with detections | Perfect beat-set agreement | Dice agreement (5/25/50/75/95th pct) | |ΔRate A−B| bpm (pct) | median |Δt| of matched beats, ms |
|---|---|---|---|---|---|
| DEV-S | 118 | 0.983 | [1.0, 1.0, 1.0, 1.0, 1.0] | [0.0, 0.0, 0.0, 0.0, 0.022] | [0.0, 0.0, 0.0, 0.0, 0.0] |
| LOCK-S | 120 | 0.992 | [1.0, 1.0, 1.0, 1.0, 1.0] | [0.0, 0.0, 0.0, 0.0, 0.0] | [0.0, 0.0, 0.0, 0.0, 0.0] |
| LOCK-R | 100 | 0.980 | [1.0, 1.0, 1.0, 1.0, 1.0] | [0.0, 0.0, 0.0, 0.0, 0.0] | [0.0, 0.0, 0.0, 0.0, 0.0] |

**Interpretation limits.** Agreement is between two detectors that share the same band-pass family, the same refractory logic and — crucially — the **same peak-localisation step**, so matched beats have identical timing (|Δt| = 0 ms) and only *detection decisions* can differ. Agreement therefore cannot reveal errors common to both (e.g., a T wave or pacemaker spike detected by both, or a low-amplitude beat missed by both), and it says nothing about accuracy against annotations.

## 6. Concept-value distributions (ACCEPTED values; percentiles 5/25/50/75/95)

| Set | Metric | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| DEV-S | beats (consensus) | 8.000 | 10.000 | 13.000 | 18.000 | 27.100 |
| DEV-S | rate mean, bpm | 52.672 | 59.818 | 79.404 | 112.455 | 164.590 |
| DEV-S | RR CV | 0.005 | 0.011 | 0.030 | 0.122 | 0.219 |
| DEV-S | RR NRMSSD | 0.005 | 0.013 | 0.040 | 0.186 | 0.314 |
| DEV-S | RR median-abs-succ-diff / median RR | 0.003 | 0.008 | 0.025 | 0.083 | 0.215 |
| LOCK-S | beats (consensus) | 8.000 | 9.000 | 13.000 | 17.000 | 27.000 |
| LOCK-S | rate mean, bpm | 51.261 | 57.977 | 77.775 | 106.218 | 165.138 |
| LOCK-S | RR CV | 0.004 | 0.014 | 0.031 | 0.148 | 0.244 |
| LOCK-S | RR NRMSSD | 0.005 | 0.014 | 0.035 | 0.186 | 0.392 |
| LOCK-S | RR median-abs-succ-diff / median RR | 0.003 | 0.010 | 0.024 | 0.087 | 0.304 |
| LOCK-R | beats (consensus) | 8.000 | 9.000 | 11.000 | 15.250 | 22.000 |
| LOCK-R | rate mean, bpm | 51.821 | 56.558 | 65.996 | 97.994 | 135.410 |
| LOCK-R | RR CV | 0.007 | 0.016 | 0.033 | 0.125 | 0.263 |
| LOCK-R | RR NRMSSD | 0.008 | 0.021 | 0.038 | 0.174 | 0.363 |
| LOCK-R | RR median-abs-succ-diff / median RR | 0.004 | 0.014 | 0.028 | 0.077 | 0.269 |

Median per stress stratum (all sampled units, consensus values; shown only to illustrate that outputs vary directionally with the strata used for sampling — **diagnostic labels are not reference values**, no cutoffs are applied):

| Stratum | n | median rate (bpm) | median RR CV | median beats |
|---|---|---|---|---|
| S01_timestamp_block | 23 | 58.9 | 0.023 | 10 |
| S02_age_missing | 20 | 83.4 | 0.036 | 14 |
| S03_AFIB | 23 | 102.6 | 0.184 | 17 |
| S04_AF_flutter | 40 | 83.3 | 0.194 | 13 |
| S05_SVT_AT | 23 | 165.0 | 0.007 | 27 |
| S06_ectopy | 25 | 74.4 | 0.135 | 12 |
| S07_conduction_wide_QRS | 21 | 58.0 | 0.017 | 10 |
| S08_SA | 27 | 67.2 | 0.088 | 11 |
| S09_SB | 67 | 54.2 | 0.021 | 9 |
| S10_ST | 29 | 112.1 | 0.010 | 18 |
| S11_SR | 37 | 73.5 | 0.020 | 12 |
| S12_LVQRSAL | 5 | 75.8 | 0.015 | 12 |

## 7. Internal stability (DEV-S only; perturbed copies of the parameters; not accuracy)

120 DEV-S records re-extracted under 16 single-parameter perturbations (lead support, cluster span, edge zone, detector thresholds/windows, localisation window, matching tolerance). Records whose primary rate moved by > 1 bpm, per variant: fuse_k_min=2: 1; fuse_k_min=4: 1; edge_s=0.05: 3; edge_s=0.20: 1. The largest single change was 13.9 bpm (relaxing lead support to 2 leads). Accepted-status changed in 0–4 of 120 records per variant. Values are stable to reasonable parameter changes on this sample; this does not show they are correct.

## 8. Duplicate-pair output identity

All **43** within-train duplicate pairs (1 sampled pair + 42 read only for this check, counted separately from sampled units) have byte-identical raw arrays and identical extracted outputs (43/43). Determinism re-run on 20 DEV-S records: identical = True.

## 9. Actual failures and non-accepted records (IDs and reasons only)

| Record | Set | Stratum | Rate state | Reason codes | Beats A/B/consensus | Usable/flat leads |
|---|---|---|---|---|---|---|
| JS08409 | DEV-S | S01_timestamp_block | UNCERTAIN | DETECTOR_DISAGREEMENT | 12/10/10 | 8/0 |
| JS11245 | DEV-S | S10_ST | UNAVAILABLE | FLAT_LEAD;INSUFFICIENT_USABLE_LEADS | 0/0/0 | 2/6 |
| JS17843 | DEV-S | S04_AF_flutter | UNCERTAIN | DETECTOR_DISAGREEMENT | 14/13/13 | 8/0 |
| JS35773 | DEV-S | S08_SA | UNAVAILABLE | FLAT_LEAD;INSUFFICIENT_USABLE_LEADS | 0/0/0 | 2/6 |
| JS29612 | LOCK-R | S09_SB | UNCERTAIN | DETECTOR_DISAGREEMENT | 9/10/9 | 8/0 |
| JS30940 | LOCK-R | S09_SB | UNCERTAIN | DETECTOR_DISAGREEMENT | 9/10/9 | 8/0 |
| JS34615 | LOCK-S | S08_SA | UNCERTAIN | DETECTOR_DISAGREEMENT | 16/11/11 | 8/0 |

Failure counts across the 340 sampled units: **UNAVAILABLE 2** (both with only 2 usable non-redundant leads after 6 flat leads — the rule requires ≥3 usable leads and ≥3 supporting leads, so a record with only leads I and II usable is reported unavailable although a human might still read a rate), **UNCERTAIN 5** (detector disagreement: one detector reports extra or missing beats; values reported separately in `*_A`, `*_B`, `*_cons` columns but not accepted). Detector disagreement is not automatically an error and mean-vs-median rate differences are not treated as errors. Reason-code examples: {'DETECTOR_DISAGREEMENT': ['JS08409', 'JS17843', 'JS34615', 'JS30940', 'JS29612'], 'FLAT_LEAD': ['JS12431', 'JS10847', 'JS35773', 'JS35959', 'JS11245'], 'INSUFFICIENT_USABLE_LEADS': ['JS35773', 'JS11245'], 'CLIPPING_SUSPECT': ['JS40452']}.

## 10. Earlier plan gates — status

| Gate | Status |
|---|---|
| G1 trusted reference available | **not assessed — reference unavailable** |
| G2 detection sensitivity/PPV vs reference | **not assessed — reference unavailable** |
| G3 rate agreement vs reference | **not assessed — reference unavailable** |
| G4 coverage | descriptive only: ACCEPTED DEV-S 116/120, LOCK-S 119/120, LOCK-R 98/100 (acceptance = detector consensus; the gate wording referred to validated states) — **not claimed as a gate pass** |
| G5 mask-leakage control | **not run** (mask-only diagnostic classifier deferred) |
| G6 irregularity vs reference-derived values | **not assessed — reference unavailable** |
| G7 stratum-based morphology marking | **not adopted** (no morphology is marked unreliable from stratum performance) |

## 11. Limitations (stated, not resolved)

- No reference annotations: nothing here measures accuracy; agreement and stability are internal consistency only.
- Both detectors share filtering family, refractory logic and peak localisation; common-mode errors (T-wave double counting, pacemaker spikes, missed low-amplitude beats, very high rates) are invisible to the agreement check. No noise/SQI layer beyond flat, non-finite, clipping indicators and insufficient detections was implemented.
- 10-second RR statistics are descriptive (roughly 5–30 RR intervals per record in this sample); not conventional HRV, not NN-based, no sinus-rhythm assumption, and not validated for arrhythmia.
- Physical calibration (1 µV vs 4.88 µV per count) and raw-vs-denoised provenance remain unresolved and untested; the pilot uses timing only, which is unaffected by a constant gain.
- Clipping and flat-lead rules are engineering conventions (mode fraction ≥ 0.8; plateau of ≥ 4 samples at the lead extreme with steep flanks; int16 limits) with no external validation; clipped leads remain in detection and are only flagged.
- Sampling: strata assigned by a fixed priority order starve S12_LVQRSAL; fallbacks fill the quota; stress sets are not representative; LOCK-R is 100 units, so rare failure modes may be unobserved (a 3% record-level failure prevalence is detected with ≈95% probability).
- Locked sets lie inside the training role, so the extractor was developed on data the backbone was trained on; this is irrelevant for timing extraction but must be remembered when concepts are later paired with backbone embeddings.
- A single configuration was used; the stability diagnostic perturbed parameters on DEV-S only.

## 12. Reuse and next step

The extractor accepts a record array or `.mat` path and returns concept values, states, reason codes and provenance; `extract_many` is the hook for processing a role manifest. **Use only `accepted_*` values (state ACCEPTED) for later supervision; keep UNCERTAIN values and masks separate; never impute 0.** Suggested next combined step (not started): frozen extraction over the training role with these exact parameters, backbone embeddings, and an initial validation-only concept/prototype experiment, reporting masks and coverage by label and batch and treating the concepts as automatically derived.

## 13. Fingerprints (replace a git commit; no Git operations were used)

| Item | SHA-256 |
|---|---|
| parameters (`extraction_config.json`) | `4f5e41ea05c53bd39bf0267039047a30a668652c2229b83e9ba0b11dfb00403c` |
| `src/ecg_concept/beat_detection.py` | `eee2c017a57d42b18a3b2dbee4c51880fad4cd3f7960e18f7389b208fa0158b5` |
| `src/ecg_concept/rr_concepts.py` | `acbcfc67ad67108e7495684f82e99c20899f40329754b3ccefc2c7285fc807dd` |
| `src/ecg_concept/preprocessing.py` | `4b6a6a54b573b2ef95ddb418e14bb80d1aca287ad25e62b0e9e52acb12070d66` |
| `scripts/step4_3_pilot.py` | `da9fbe583238995e3220650394ac5c6542b6cf11cb7f32f8225f0acf6cd203bd` |
| `tests/test_rr_concepts.py` | `25a9939421c79f84a776ede7f4b03ccc0029352bb7843d8e1f184a2f6da2db55` |
| input: role_manifest | `8595e4d68f178f6e7005e78959c20a57a000fb0fc13ceab5d7ab29ba61e5b8d7` |
| input: original_fold_manifest | `5fc86602f95a1668ad66d6a27747e42430e9603327382a9673a7c1b4cc9595f1` |
| input: pilot_manifest | `4347a4aa1c7d4860c81b66d9ae8e45212e94c7e23b46a9b9a32202df3185d790` |
| input: dup_check_manifest | `790272bcebfdcea6d7e077b415ea9065712ae52bbcd3475272e4e3c3d9fec109` |
| input: label_dictionary | `05d6fa23acdcbb466b378310ccfad22ea6cef4d3abfa83e8f660c77016be028d` |
| sampled lists | `836b91deed5aa7c424ef51d5a3242ae69c871361809a9eda779f19304640e34f` |
| `pilot_manifest.csv` | `4347a4aa1c7d4860c81b66d9ae8e45212e94c7e23b46a9b9a32202df3185d790` |

Frozen at 2026-11-11 12:09:56 (before the lock sets were processed); software: {'python': '3.14.6', 'numpy': '2.5.4', 'scipy': '1.18.1'}. Execution log: `execution_log.txt`. Beat times (no waveforms) in `pilot_beat_times.csv`. Intermediate files (header flags, DEV run, hashes) in `work/`.

## 14. Skipped / not checked

- No accuracy, sensitivity, PPV or physiological-error computation (no references).
- No axis, amplitude, low voltage, delineation, PR, QT, ST, T-wave or ectopy classification.
- No noise-level SQI, no pacemaker-spike handling, no mask-only leakage control, no full-training-set run, no validation/test records read.
- Detector implementations were not compared with reference implementations (none installed).
- Calibration and raw/denoised provenance not examined.
