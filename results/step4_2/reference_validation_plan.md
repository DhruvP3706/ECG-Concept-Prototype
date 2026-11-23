# Step 4.2 — Independent references and annotation plan (draft)

Access date for web claims: 2026-11-11; source IDs refer to `evidence_review.md`. Machine-generated values are never called expert ground truth. Nothing was downloaded; no annotation work was started.

## 1. Candidate external reference datasets (verified facts only)

| Dataset | Annotation actually present | Annotated by | Leads / fs / duration / population | License / access | Supports validation of | Domain mismatch and limits |
|---|---|---|---|---|---|---|
| **LUDB** (S06) | Boundaries **and peaks** of P, QRS, T per lead; categorical rhythm and "electric axis of the heart" per record | Certified cardiologists, by eye, independently per lead ("number of annotators/consensus not stated") | 12 leads, 500 Hz, 10 s; 200 records, 200 subjects, healthy volunteers + cardiovascular patients (some pacemakers), ages 11–>89; one hospital; AF 15, typical flutter 3 | Open Data Commons Attribution v1.0, open | **R-peak/QRS timing** (QRS peak and onset/offset), QRS duration in later steps, amplitude only indirectly (manual windows), **coarse categorical** axis | Closest signal format; small, one centre, few arrhythmias; no quality annotation; PhysioNet page and README differ on a category label (Undetermined vs None (pacemaker)) |
| **QT Database** (S07) | Onset/peak/end of P, QRS, T(, U) for 30–50 beats/record (manual, annotator 1; annotator 2 for 11 records) plus automatic marks | Annotators not named | 105 two-lead 15 min records; sampling rate not stated on the page; population not described | ODC-By, open | QRS delineation, T-end (later) | Two leads, ambulatory, small beat sample; not 12-lead |
| **MIT-BIH Arrhythmia** (S08) | ~110,000 beat annotations (+rhythm, signal quality, comments) | Automatic initial labels, two cardiologists independently, consensus | 2 channels, 360 Hz, 48 × 30 min; 47 subjects; deliberately arrhythmia-enriched | ODC-By, open | **R-peak/beat timing, arrhythmia/ectopy handling** | 2-channel Holter, 360 Hz, 1975–79; not representative of resting 12-lead; detectors often tuned on it (S13) |
| **INCART** (S09) | >175,000 beat annotations, automatic then "corrected manually" (page also says locations not manually corrected) | Not named; verification by named individuals | 12 leads, 257 Hz, 75 × 30 min; coronary-disease Holter cohort | ODC-By, open | multi-lead R-peak detection (approximate timing) | 257 Hz, Holter; annotation positions may be misaligned |
| **PTB-XL** (S10) | Diagnostic statements (cardiologist or device); `heart_axis` from the report; quality fields without stated producer. **No R-peak/amplitude annotation** | Cardiologists and/or ECG device (not separable on the page) | 12 leads, 500/100 Hz, 10 s; 21,799 records; Schiller 1989–96 | CC BY 4.0, open | only coarse axis-category plausibility; quality-flag comparison (producer unknown) | not an expert measurement reference |
| **PTB-XL+** (S11) | Algorithmic features/fiducials from 12SL, Uni-G (closed), ECGDeli (open) | **Algorithms**, not experts | derived from PTB-XL | CC BY 4.0, open | inter-algorithm agreement only | "cannot guarantee generalizability"; not truth |
| **CPSC2018** (S12) | labels only; no beat/measurement annotation | annotators not named | 12 leads, 500 Hz, 6–60 s | no licence text on the page; test data sequestered | none for measurements | — |
| **Chapman figshare `Diagnostics.xlsx`** (S02, S03) | Device-generated `VentricularRate`, `RAxis`, `QRSDuration`, … for 10,646 patients | GE MUSE software | same acquisition source as ours | CC0 (figshare API), not downloaded | **machine-generated comparison values** for rate (if joinable) | join to PhysioNet IDs undocumented; definitions undocumented; device output, not truth |
| CSE reference database | not reviewed | — | — | access not verified | possibly multi-lead wave boundaries | **unreviewed** |

**Verified conclusion:** none of the open datasets reviewed provides expert-annotated R-peaks on 12-lead 500 Hz hospital ECGs *of our population*; LUDB and INCART are the closest in lead count/rate or format and MIT-BIH/INCART carry the largest beat-level references. There is no open amplitude or quality reference from experts (PTB-XL quality fields have no stated producer).

## 2. What can be validated without an expert on our data

| Claim | Feasible without our own expert? | Method | Residual limitation |
|---|---|---|---|
| Detector class works on expert-annotated R-peaks | **Yes (generic)** | run the frozen detectors on LUDB (12-lead 500 Hz) and MIT-BIH/INCART (resampled; effect of resampling must be reported); sensitivity/PPV with a stated tolerance, timing error, record-level bootstrap | domain shift; does not establish accuracy on our records |
| Detector internal consistency on our training data | Yes | cross-detector/cross-lead agreement, duplicate-group identical output (identical signals ⇒ identical values), stability to small, known perturbations | agreement ≠ accuracy |
| Coarse plausibility of rate | Yes (weak) | direction only, e.g., rate distributions in records labelled SB vs ST; labelled-diagnosis counts from the manifest | diagnoses define rate classes, so this is a sanity check, not validation; no cutoffs are invented |
| Comparison with device-generated rates | Conditional | only if figshare `Diagnostics.xlsx` can be downloaded (needs approval) and joined to our IDs (no documented cross-walk) | machine-generated; definition undocumented |
| Concept-bottleneck *mechanistic* faithfulness | Yes | later modelling steps | says nothing about clinical validity of concept values |

**Claims that remain unsupported without expert or suitable reference validation:** that extracted rate/RR/irregularity values are physiologically accurate on this dataset; that irregularity corresponds to a clinical rhythm concept beyond the detected RR; that any quality mask corresponds to clinically unusable signals; any axis, amplitude or low-voltage claim; that concept-based explanations are clinically meaningful (as opposed to faithful to the model).

## 3. Feasible expert-annotation plan (to be confirmed; no expert is assumed)

**Annotators.** Preferably two independent annotators with ECG interpretation expertise (a cardiologist or an experienced ECG technician for R-peak marking; a cardiologist for any later delineation). Availability, time and compensation are open questions.

**Material.** Training-role records only, drawn by the deterministic rule in `step4_3_pilot_plan.md`; presented as 10 s, 12-lead plots at fixed gain/speed from raw samples in a neutral viewer; **no header text, no diagnostic labels, no ID, no model or detector output**; duplicate groups assigned to the same set and shown once. Annotators receive a written protocol and a calibration set not used in assessment.

**Task and definitions (draft).** Mark every QRS (R-peak = the location of the dominant QRS deflection extremum in a pre-specified lead rule, e.g., the lead where the deflection is clearest, recorded per beat); mark questionable beats and "unreadable" intervals separately; record the lead(s) used and a 3-level confidence; mark records/leads with unusable signal for rate determination. R-peak reference values are timing points; **amplitude/axis are not annotated in this pilot.**

**Independent annotation and disagreement.** Annotator A annotates all assessment records, annotator B a pre-selected overlap subset (≥ 25% of records, stratified); disagreements (beat sets matched at the tolerance) are adjudicated by a third reader or by consensus meeting; unresolved beats are excluded and counted. The adjudicated set is the reference; unadjudicated annotator-A marks are never mixed in silently.

**Blinding.** Blinded to detector outputs, diagnostic labels and header fields; blinding to rhythm is imperfect (the waveform shows it) and is stated as a limitation. Pre-annotation by an algorithm is **not** used (anchoring) unless it is a documented, separate condition.

**Separate samples.** Method-development annotations (used to tune `[DEV]` parameters) and a **locked assessment set** annotated afterwards; the assessment set is fixed, hashed and unseen by parameter tuning. Duplicate groups never straddle the two.

**Agreement and error measures.** Beat-level sensitivity, positive predictive value and F1 at a pre-declared tolerance (S13 used 50 ms; sample-accurate scoring is also reported because tolerance choice alters conclusions, S14); timing-error distribution (median and 95th percentile of |Δt|); rate error and irregularity-statistic error with Bland–Altman limits of agreement and bias (S26); inter-annotator F1 and timing error; record-level bootstrap CIs; per-stratum results.

**Workload estimate (assumption, to be checked with annotators).** ~3–5 min per 10 s record for from-scratch marking ⇒ 200 assessment records ≈ 10–17 h for annotator A, a 50-record overlap ≈ 3–4 h for annotator B, plus a ~40-record calibration/development set.

## 4. Fallback if no expert is available

1. Treat the pilot as **method development and internal-consistency only**; report no accuracy claim for our dataset.
2. Validate detector classes on LUDB (closest format) and MIT-BIH/INCART (beat references), with resampling effects reported; state the domain gap.
3. Optionally compare with device-generated rates (if obtainable and joinable), labelled "machine-generated agreement".
4. Use cross-detector/cross-lead agreement, duplicate-pair identity, and diagnostics-direction checks as consistency evidence only.
5. In the paper, describe concepts as "automatically derived descriptors" and limit claims to model faithfulness; leave physiological accuracy as an explicit limitation.

## 5. Open decisions requiring the project owner

Whether an expert can be recruited; whether to approve downloading public reference sets (LUDB, MIT-BIH, INCART) and the figshare attribute files; whether to review CSE access; time budget for annotation.
