# Step 4.2 — Measurement protocol DRAFT (provisional; nothing here has been implemented or run)

Status legend: **PROVISIONAL** = proposed design to be tested in the Step 4.3 pilot; **DEFERRED** = not supported well enough to include; every parameter carries a label: **[CITED]** supported by a cited source (see `evidence_review.md`), **[ENG]** engineering convention requiring pilot assessment, **[DEV]** to be determined on training-only development data. **No clinically validated cutoff is proposed.** Source IDs (S01…S33) refer to `evidence_review.md`.

## 0. Scope and first concept set

| Concept | Decision | Why (evidence) |
|---|---|---|
| C01 Ventricular rate / mean RR | **PROCEED to pilot** | R-peak timing is the most mature measurement (S13); definition is simple and independent of the unresolved amplitude calibration |
| C02 RR irregularity (descriptive, 10 s) | **PROCEED to pilot, with restricted wording** | Same detections; but conventional-HRV validity at 10 s is not established for arrhythmia (S25, S26) |
| C05 Quality and measurement-specific masks | **PROCEED to pilot** | Prerequisite for any concept; SQI evidence is single-lead/ICU and warns about arrhythmia (S21) |
| C03 Frontal QRS axis | **DEFERRED** | no accessible validated method; standard unreadable (S28); depends on delineation and calibration (§6) |
| C04 QRS amplitude / low voltage | **DEFERRED** | calibration conflict (S01/S02 vs headers), raw-vs-denoised unknown, low-voltage definitions inconsistent and not official (S29, S30) (§6) |
| C06–C12 (QRS duration, PR, QT/QTc, ST, T, P, ectopy burden) | **DEFERRED** (as in Step 4.1) | need delineation and reference measurements |

The pilot is designed to remain useful if only C01, C02 (descriptive) and C05 prove defensible.

## 1. Inputs and preprocessing (kept separate from the backbone)

- Input: the raw integer signal `(12, 5000)`, 500 Hz, header-specified (`1000/mV`, ADC zero 0; S04). **Never** the backbone's bandpassed, int16-truncated, per-lead z-scored tensor, which discards physical amplitude and whose filtering alters waveform shape. `[CITED]` for the header semantics.
- Rate and RR depend only on beat *times*; they are invariant to a constant gain error, so the unresolved 4.88 µV vs 1 µV calibration question and the raw-vs-denoised question do **not** change C01/C02 definitions, although denoising could change detector behaviour (to be checked in the pilot).
- Detection-only filtering: detector-specific band-pass (for example 5–15 Hz in Pan–Tompkins-style detectors, 5–20 Hz in XQRS; S13, S17) `[ENG]`. This attenuates P/T waves and baseline and distorts QRS shape and amplitude, so **filtered signals are used for timing only, never for amplitude, axis or ST measurements**. Zero-phase (`filtfilt`) or a fixed delay correction must be used so detection delay is constant `[ENG]`; peak time is refined to the local extremum of |x| on a lightly filtered copy within a small window `[DEV]`.
- Polarity: detection on squared derivative/energy (polarity-agnostic), localisation on |x|, so negative-dominant QRS complexes are handled `[ENG]`.

## 2. Beat detection and fusion (C01/C02)

1. Per-lead detection with **two detectors of different principle** (candidates: energy/threshold Pan–Tompkins-type; two-moving-average Elgendi-type; third principle only if reference implementations are approved — see `method_decision_table.csv`). `[ENG]`; selection by pilot, not by popularity.
2. Cross-lead clustering of per-lead beat times within tolerance τ `[DEV]` (scoring tolerance in S13 was 50 ms; this is a scoring convention, not a clustering parameter).
3. Independence: lead II = lead I + lead III and aVR + aVL + aVF = 0 (S02, Technical Validation), so only **8 channels are independent (I, II, V1–V6)**; consensus counts should be over independent channels, not 12 leads `[H]`.
4. A beat is accepted when supported by ≥ *k* usable independent channels `[DEV]`; accepted beat time = median of supporting lead times.
5. Beats are never classified (no NN/ectopic labels are available), so all intervals are **RR intervals**.

## 3. C01 — Ventricular rate

Let accepted beat times be `t_1 < … < t_n` (seconds), `RR_i = t_{i+1} − t_i`, `m = n − 1`.

- **Primary (provisional): `VR_mean = 60 / mean(RR) = 60·m / (t_n − t_1)` [beats/min].** Interpretation: average ventricular activation rate over the observed span; includes ectopic beats and AF irregularity.
- **Secondary: `VR_median = 60 / median(RR)`.** Robust to a single detection error; differs from `VR_mean` when rhythm is irregular. Both are reported with `m`.
- The choice of primary is a *definition*, not something detector validation can confirm; it is declared now and its sensitivity to the choice is reported. The definition used by the device that generated figshare `VentricularRate` is undocumented in accessible sources (S02), so no equivalence is claimed.
- Minimum usable information: `m ≥ m_min` `[DEV]` (arithmetic: with ~6–30 beats in 10 s, one missed beat changes the mean rate by roughly 1/m; the pilot quantifies it). Edge handling: the first/last beats within `E` ms of the record boundary may be partial QRS complexes `[DEV]`; only RR between *two accepted beats* are used; no extrapolation to a 60 s rate.
- Failure/uncertainty: missed beats (RR doubled, rate underestimated), extra detections (RR halved), noise bursts, flat leads; uncertainty = disagreement between detectors/leads and between `VR_mean` and `VR_median`.
- Reference needed: expert R-peak times (or an equivalent verified reference) on a sample (see `reference_validation_plan.md`); rate error computed from reference-derived rate with Bland–Altman (S26 recommends limits of agreement rather than correlation alone).

## 4. C02 — RR irregularity (descriptive; not validated HRV)

For `m ≥ 3`: `CV_RR = SD(RR)/mean(RR)` (sample SD) and `NRMSSD = sqrt(mean((RR_{i+1} − RR_i)²)) / mean(RR)`; robust companion `MADSD = median(|RR_{i+1} − RR_i|)/median(RR)`. All are dimensionless and computed on **all detected RR** in the 10 s record.

- **Wording rule:** call it "10-second RR irregularity statistic", never "HRV", "SDNN" or "RMSSD" in the conventional sense. Reasons: (a) the conventional short-term standard is 5 min and ultra-short measures are called "controversial proxies" (S26); (b) the only 10 s validation read (S25) used sinus-edited intervals from finger-pressure recordings, found SDNN a poor proxy, and states its agreement statistics are inflated; (c) NN intervals require beat classification we do not have; (d) in AF, flutter and ectopy irregularity is the *signal*, so sinus-rhythm validity evidence does not transfer.
- **Detection error vs true irregularity:** a missed beat creates RR ≈ 2×local RR, an extra detection ≈ ½×; true ectopy creates a short–long pair. These are indistinguishable from one detector's output. Ambiguous intervals (relative to the local median, band `[DEV]`) are **flagged, not removed**; they are resolved only by cross-detector/cross-lead support (supported ⇒ treated as physiological) or reference review. Records with unresolved ambiguous intervals above a `[DEV]` fraction get `UNRELIABLE` state for C02 even though the diagnosis may be AF.
- Minimum usable information `m_min2 ≥ m_min` `[DEV]`; at 40 bpm a record contains only ~6 beats, at 180 bpm ~30 — the statistic's variance depends strongly on `m` and must be reported with it.
- Reference needed: irregularity derived from expert R-peaks (tests detection errors only); true physiological validity of the statistic as a concept is **not** established by that test. Diagnostic labels (SA, AFIB, AF, APB, VPB) overlap by definition and cannot validate it.

## 5. Quality and masking (C05): five separate layers

| Layer | Checks | Parameters | Effect |
|---|---|---|---|
| L0 structural validity | shape `(12,5000)`, header/record consistency, non-finite samples, missing leads, malformed header | none (pass/fail) | record has no valid input → all concepts `UNDEFINED(structural)` |
| L1 signal limitations (per lead) | flat/constant run fraction; clipping at ±32,767/−32,768 (limits stated in S01 **[CITED]**); baseline wander (cubic-spline estimate, S21 **[CITED definition]**); high-frequency noise/power-line; kurtosis and power-ratio SQIs (definitions S21 **[CITED]**) | all thresholds `[DEV]`/`[ENG]` (S13 used a flat-line portion >80% in its own pipeline as an example convention) | marks leads unusable *for that measurement* |
| L2 detector uncertainty | cross-detector agreement (F1 at τ), supported-beat fraction, ambiguous-interval fraction (§4) | `[DEV]` | `UNRELIABLE` state |
| L3 concept-specific validity | `m ≥ m_min` for C01, `m ≥ m_min2` and ambiguity limit for C02 | `[DEV]` | concept state per record |
| L4 physiological absence/undefinedness | e.g., <2 accepted beats (no RR defined), an interval concept without the required wave | none | `UNDEFINED(physiological)` |

Rules: (1) **Abnormal rhythm or morphology is never a mask by itself** — AF, flutter, ectopy, wide QRS, paced rhythm, bradycardia, tachycardia are retained; masks fire only on *measured limitations* (L0–L3). (2) Three states per concept — `VALID`, `UNRELIABLE` (value kept, uncertainty flagged), `UNDEFINED` — with reason codes, so physiologically undefined concepts are not confused with unreliable estimates. (3) Detection failure caused by abnormal morphology is `UNRELIABLE` pending reference review, not excluded. (4) No SQI is used as a record-level exclusion until its behaviour on arrhythmia-containing training records is evaluated (S21 warns SQIs can classify arrhythmia as noise). (5) Lead-integrity check from the dataset paper (II = I + III; aVR + aVL + aVF = 0; S02) may serve as an L0/L1 flag with tolerance `[DEV]`; it was documented for the figshare release and not verified for our records.

**Leakage risk and controls (to be designed, not implemented):** masks and missingness may encode diagnoses (e.g., noisy AF, pacemaker records) or acquisition batch (ID block with 2021 timestamps; age-missing late-ID batch; Step 4.1). Planned later controls: (a) **coverage reporting** by diagnostic label, by batch/ID block and by timestamp presence, and by duplicate-group; (b) a **mask-only diagnostic control** — predict diagnostic labels and batch indicators from the mask/quality/coverage features alone on training data with cross-validation, and report discrimination; a model that gains from the masks must be declared as using them. Neither is run in this step.

## 6. Deferred concepts — design notes only (not a method decision)

**C03 axis.** Needed before adoption: (i) an accessible authoritative definition and angle convention (AHA/ACCF/HRS Part I was not readable); (ii) QRS onset/offset or a justified window rule — fixed R-centred windows truncate wide QRS and mix in ST/T; (iii) a stated lead pair or least-squares scheme with the lead-scaling relations (augmented leads) taken from a primary source rather than from this draft; (iv) a per-beat isoelectric baseline definition (PR/TP) and its effect on the net deflection; (v) comparison by **circular statistics** (wrap differences to (−180°, 180°], report circular mean error/dispersion); categorical left/normal/right agreement needs boundaries from an official source; (vi) physical amplitude (calibration question). The available open evidence (S31, S32) includes no reference comparison. LUDB (categorical axis labels) and PTB-XL (axis from reports) offer only coarse checks.

**C04 amplitude / low voltage.** Per-lead amplitude (R, S, peak-to-peak in a QRS window relative to an isoelectric reference) is a *measurement*; "low voltage" is a *classification* needing an authoritative definition (S29 and S30 differ in units and lead groups, neither official). Needed before adoption: calibration resolved from the data (sample-value quantisation vs 4.88 µV/bit and 1 µV/count), raw-vs-denoised status (denoising includes baseline removal and NLM, S02, which alter amplitudes), bandwidth effects, and manual caliper references.

## 7. Outputs per record (for the pilot)

Concept values (`VR_mean`, `VR_median`, `m`, `CV_RR`, `NRMSSD`, `MADSD`), a state and reason codes per concept, quality indicators per lead and record, detector-agreement indicators, and the exact parameter set/seed — all stored with record ID and duplicate-group ID; no waveforms are exported.
