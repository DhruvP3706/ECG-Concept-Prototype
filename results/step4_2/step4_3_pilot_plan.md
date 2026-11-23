# Step 4.3 — Measurement feasibility pilot: PLAN ONLY (not run, not approved)

Everything below is **provisional** and needs the owner's decisions in §9 before execution. Scope: training-role records only (35,568; `results/step3_7/prospective_role_manifest.csv`, `role == train`); validation, test and excluded records are not touched; no model inference, no embeddings, no concept use in any classifier; waveforms are read only for the measurement pilot and never exported. Labels and header fields are used **only for stratification**, never as ground truth. Parameter labels as in `measurement_protocol_draft.md`: **[ENG]** engineering convention, **[DEV]** set on development data, **[CITED]** from a source.

## 1. Objectives and what the pilot may conclude

1. Can R-peaks be detected reliably enough, on 12-lead 500 Hz 10 s training ECGs, to define C01 (ventricular rate) and C02 (10 s RR irregularity, descriptive)?
2. Can quality/masking (C05) be defined without hiding diagnoses or batches?
3. Which detector principle(s) and fusion rule perform best on a **development** set, then how do they perform on a **locked** set?
It may **not** conclude that concept values are physiologically accurate unless an expert/trusted reference exists, and detector agreement or label plausibility is never reported as accuracy.

## 2. Sampling frame and deterministic rule

- Unit = duplicate group if the record is in one (43 within-train pairs flagged in the role manifest), else the record. Both members of a group go to the **same set**; only the lowest-ID member enters metrics (its partner is used only for an identical-output consistency check).
- Seed `20261111` (declared here). Key for each unit: `SHA-256("20261111|" + unit_id)`; within a stratum, units are ordered by ascending key and taken in order. Draw order: (1) LOCK-R, (2) DEV-S, (3) LOCK-S, so sets are disjoint by construction; units already drawn are skipped.
- Selection never uses detector output, model outputs, errors, or any test/validation information. The three ID lists and their SHA-256 are written to disk **before** any pilot code runs and are not edited afterwards.

| Set | Type | n | Purpose |
|---|---|---|---|
| DEV-S | stratified **stress test** (not representative) | 120 | method development, parameter (`[DEV]`) setting, failure analysis |
| LOCK-S | stratified **stress test**, locked | 120 | post-freeze performance by stratum (descriptive) |
| LOCK-R | **representative** simple random sample of training units | 100 | post-freeze estimate of failure prevalence and overall performance |

Stress-test results describe behaviour on challenging cases and **must not be pooled or presented as population performance**; only LOCK-R supports representative statements.

**Stress strata** (10 units per stratum per stress set; assigned by this fixed priority order so each unit has one stratum): (1) timestamped ID block JS08344–JS09373; (2) age 0 or `NaN` (late-ID batch); (3) AFIB; (4) AF (atrial flutter); (5) SVT/AT; (6) ectopy (APB, VPB, JEB, VEB); (7) conduction/wide-QRS (RBBB, LFBBB, IVB, WPW, VPE, AV blocks); (8) SA; (9) SB; (10) ST; (11) SR; (12) LVQRSAL. Optional stratum 13: records with a flat (zero-variance) or clipped lead by a pre-declared screen — requires a waveform screening pass over the training set (see §9). Quotas are halved for any stratum with fewer than 40 eligible units.

## 3. Sample-size and workload justification (planning arithmetic, assumptions stated)

- LOCK-R: with independent records and a per-record failure prevalence of 3%, 100 records give a probability of seeing ≥ 1 failure of 1 − 0.97¹⁰⁰ ≈ 0.95; zero failures would bound the prevalence below ≈ 3% (rule of three: 3/100). With ~12 reference beats per record (≈ 1,200 beats) and an **assumed** design effect of 3 for within-record correlation, beat-level sensitivity near 0.98 has a 95% half-width of about ±1.4 percentage points; record-level bootstrap will be used instead of binomial formulas.
- Each stress stratum (n = 10): zero failures gives a 95% upper bound of about 26% (exact binomial) — strata are descriptive, not for confirmatory claims.
- Annotation: ≈ 220 assessment records (LOCK-R + LOCK-S) × 3–5 min ≈ 11–18 h for annotator A; overlap ≥ 55 records for annotator B ≈ 3–5 h; a calibration/development subset (~40 records of DEV-S) ≈ 2–3 h. Estimates to be confirmed with annotators.

## 4. Method development vs locked assessment

1. **Freeze the lists** (hash, git-ignored but copied to a permanent location chosen by the owner).
2. **Calibration diagnostics on DEV-S only** (non-clinical data checks): histogram of sample-to-sample value steps and value ranges to resolve the 4.88 µV/bit vs 1 µV/count question and check clipping limits; band power above 60 Hz and below 0.5 Hz to test whether the WFDB copy is raw or denoised (S02: denoising = low-pass 50/60 Hz, LOESS baseline removal, NLM). Results inform deferred concepts and detector filtering; they are **not** go/no-go criteria for C01/C02.
3. **Develop** detectors, fusion and quality parameters on DEV-S (and public references if approved, `reference_validation_plan.md`); iterate freely; record every parameter value and rationale. A parameter file is hashed when development ends.
4. **Lock**: commit the parameter file, code version and detector set; only then run on LOCK-S and LOCK-R **once**; any change afterwards creates a new, documented iteration and the assessed sets are then declared used.
5. **Assess** against the reference (expert/trusted) if available; otherwise report internal consistency only.

## 5. Candidate detector comparison (to be pre-declared before development ends)

| Arm | Detector principle | Implementation | Note |
|---|---|---|---|
| A | energy/adaptive-threshold (Pan–Tompkins-type) | scipy-only implementation written for the pilot | original paper unreadable (S18); implementation fidelity to the original cannot be claimed |
| B | two moving averages (Elgendi-type) | scipy-only; parameters re-expressed in ms for 500 Hz; offset unresolved (S15) | different principle from A |
| C (optional) | wavelet/matched-filter/other (e.g., EngZee Mod, Kalidas SWT, Hamilton) | **needs a package with reference implementations (NeuroKit2/wfdb) — not installed; installation requires approval** | provides an independently implemented third opinion |
| Fusion | per-lead detections → cross-lead clustering over the 8 independent channels → consensus | scipy/numpy | parameters `[DEV]` |

No arm is declared best in advance; selection uses DEV-S/public-reference results and is then frozen.

## 6. Predetermined outputs

Per detector arm and fusion: detections per lead and fused; beat-level sensitivity, PPV, F1 at tolerance(s) declared in advance (50 ms as in S13 and sample-accurate as in S14), timing error summaries; rate error (Bland–Altman bias and limits of agreement); irregularity-statistic error; cross-detector agreement; coverage by concept state (`VALID/UNRELIABLE/UNDEFINED`) and reason code, by diagnostic stratum, by batch/ID block/timestamp presence, by age-missingness and by duplicate-group (identical signals ⇒ identical outputs); failure case list by *record ID only* (no waveform export); runtime per record. Plus the later mask-only control design (not run unless approved).

## 7. Go / no-go criteria — PROVISIONAL, to be signed off by the owner before any run

These are engineering targets, **not clinical cutoffs**; they are informed by detector performance reported on clean/arrhythmia single-lead sets (S13) and are meant to avoid post-hoc rationalisation.

| Gate | Criterion (proposal) | If failed |
|---|---|---|
| G1 reference | trusted R-peak reference exists for LOCK-R and LOCK-S (expert double-annotated overlap ≥ 25%) | pilot becomes **method-development/internal-consistency only**; no accuracy claim; concepts labelled "automatically derived" |
| G2 detection (LOCK-R) | beat sensitivity and PPV each ≥ 0.98 with lower 95% record-bootstrap bound ≥ 0.95 at the declared tolerance **[ENG]** | NO-GO for C01/C02 as model concepts; revise methods or stop |
| G3 rate agreement (LOCK-R) | Bland–Altman 95% limits for `VR_mean` vs reference-derived rate within ±5 bpm **[ENG, value to be confirmed]** | C01 not adopted until resolved |
| G4 coverage | `VALID` C01 in ≥ 95% of LOCK-R records and no diagnostic or batch stratum below 80% without an identified, documented cause **[ENG]** | masks redesigned or concept limited with explicit coverage statement |
| G5 mask leakage | mask-only discrimination for diagnostic labels/batch reported; **review trigger** if the lower CI bound ≥ 0.70 for any class or batch indicator **[ENG]** | decide, in a pre-declaration, whether masks may enter a model, and how |
| G6 irregularity | C02 advances only if G2–G4 pass and reference-derived vs detected-peak-derived irregularity agree within a pre-declared Bland–Altman band; **never** described as validated HRV | C02 dropped or kept as an undefined-validity descriptor excluded from claims |
| G7 stress strata | strata with sensitivity < 0.90 are listed; the corresponding morphology gets `UNRELIABLE` by rule (not excluded) | documented failure mode |

Outcomes: **GO** = C01 + C02 (descriptive) + C05 · **PARTIAL** = C01 + C05 only · **NO-GO** = no extracted concepts from this pilot; reassess methods. The pilot is valuable even for PARTIAL.

## 8. Explicit exclusions

No axis, amplitude, low-voltage, QRS duration, PR, QT, ST, T or P-wave extraction; no use of diagnostic labels as concept ground truth; no AF/AFIB merging or label adjudication; no masking of AF, flutter, ectopy, wide QRS or other abnormal rhythms because they are abnormal; no touching of validation/test records; no model training; no use of embeddings.

## 9. Readiness and blocking decisions

**Status: not ready to execute; ready to start immediately after the decisions below.**

1. **Approve reading training waveforms** for the pilot (read-only; measurement scripts only; no waveform export).
2. **Dependencies:** approve (or decline) installing a reference-implementation package (NeuroKit2 and/or wfdb) for arm C; if declined, arm C is dropped and the cross-detector agreement relies on arms A/B only. (Current environment: numpy 2.5.4, scipy 1.18.1, torch 2.12.0 — no signal-processing packages.)
3. **Public reference data:** approve downloading LUDB (and optionally MIT-BIH and INCART) to a location outside `data/raw` (suggest `data/external/`, which is not currently git-ignored — decide on versioning); approve or decline the figshare attribute files; note a join to our IDs is undocumented.
4. **Expert availability, tool and budget** for the annotation plan, or acceptance of the fallback (method-development only).
5. **Sign-off on §7 criteria and tolerances**, including the rate-agreement band.
6. **Waveform screening pass** over all 35,568 training records for the optional flat/clipped stratum, or drop that stratum.

If decisions 1–3 are given and 4 is deferred, a reduced **Step 4.3a** (public-reference detector validation + DEV-S internal consistency, no accuracy claim on our data) can start without an expert.
