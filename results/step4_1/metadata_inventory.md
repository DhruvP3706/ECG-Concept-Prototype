# Step 4.1 — Metadata inventory and clinical-concept feasibility

Legend: **[V]** verified in this step · **[H]** inference. Scope: header/metadata inspection only. No waveform samples were opened (no `.mat` access), no features, no model inference. Record-level inspection was limited to the **35,568 `train`-role records** selected explicitly from `results/step3_7/prospective_role_manifest.csv` (count asserted; no new split). Validation/test/excluded IDs were never read. Python 3.14.6, numpy 2.5.4; script `scripts/step4_1_metadata_inventory.py`; details in `inspection_config.json`.

Two disclosures: (1) an early `head -c` of the role manifest displayed one truncated validation-role row (ID, fold, partial role; no labels/header content). (2) My first parser pass was too strict and failed on 811 headers (optional timestamp field, see §4); that run's output is kept as `inspection_config_run1_strict_parser_superseded.json` and was superseded by the corrected run — the script refuses to overwrite files.

## 1. Bottom line

**The dataset provides no clinical measurements at all.** No heart rate/RR, PR, QRS duration, QT/QTc, electrical axis, ST or T measurements, no beat locations, fiducials or delineation, and no signal-quality annotations. Headers contain only demographics (Age, Sex), diagnostic SNOMED codes (Dx), three always-"Unknown" fields (Rx, Hx, Sx), the signal specification, and — for one ID block — a timestamp. Everything concept-like must be **extracted from the waveform** and, absent trusted references, its physiological accuracy **cannot yet be established**.

## 2. Sources inspected

| Source (relative to `data/raw/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/`) | Format | Join rule / coverage | Fields | Type | Provenance |
|---|---|---|---|---|---|
| `WFDBRecords/<xx>/<xxx>/<id>.hea` | WFDB header text | `record_id` = file stem; path from `original_fold_manifest.csv`; **35,568/35,568** train IDs resolved, 35,567 parsed | record line (id, 12, 500, 5000 [, time date]); 12 signal lines; comments `#Age #Sex #Dx #Rx #Hx #Sx` | demographics, diagnostic labels, signal spec | Not documented locally |
| `WFDBRecords/…/<id>.mat` | MATLAB Level-4 int16 | **not opened** | — | waveform | — |
| `ConditionNames_SNOMED-CT.csv` | CSV (Acronym Name, Full Name, Snomed_CT), 55 rows | SNOMED code in `#Dx` → acronym (last row wins for shared codes, as in Step 3) | names only | label dictionary | Not documented; 43 of 94 data codes have no entry (Step 2) |
| `RECORDS`, `LICENSE.txt`, `SHA256SUMS.txt` | lists / CC BY 4.0 text / hashes | n/a | none relevant | documentation | — |

File types in the whole tree (names only): `.hea` 45,152, `.mat` 45,152, `RECORDS` 453, `.txt` 2, `.csv` 1. **There is no README, attribute dictionary, measurement table, annotation file or quality table** anywhere in the project data tree, and the HonsExAI folder has no dataset copy with extras. Unit and missing-value conventions are therefore **undocumented locally** for all header fields. (Published descriptions of this dataset may document more; that belongs to the literature step and is not assumed.)

## 3. Measurement availability

| Candidate measurement | Present in dataset? | Evidence |
|---|---|---|
| Heart rate / RR | **No** | no field; only diagnoses SB/ST/SR/SA/AF… (labels, not measurements) |
| PR interval | **No** | no field; labels 1AVB, PRIE |
| QRS duration | **No** | no field; labels RBBB, LFBBB, IVB, WPW |
| QT / QTc | **No** | no field; label QTIE |
| Electrical axes | **No** | no field; labels ALS, ARS |
| ST measurements | **No** | no field; labels STDD, STE, STTC, STTU |
| T-wave measurements / morphology annotation | **No** | labels TWC, TWO only |
| Beat locations / fiducials / delineation | **No** | no files |
| Signal-quality annotations | **No** | none (flat/saturated leads known from Step 2 waveform checks) |
| Other measurements | **No** | Rx/Hx/Sx are "Unknown" in every parsed training header |

Diagnostic labels are **not** measurements and must not be repackaged as independent concept ground truth. No diagnoses were merged or adjudicated (e.g., AF = atrial flutter and AFIB = atrial fibrillation stay separate).

## 4. Training-header summary (n = 35,568 train; 35,567 parsed)

- **Parse status [V]:** 1 failure — JS01052 (the known malformed header from Step 2; its record line is merged with the first signal line). No duplicate IDs.
- **Signal specification [V]:** all 35,567 parsed headers: 12 signals, lead order I, II, III, aVR, aVL, aVF, V1–V6, 500 Hz, 5000 samples (10 s). Every one of the 426,804 signal lines has format `16+24`, gain `1000/mV`, ADC resolution 16, ADC zero 0, block size 0; file name = `<id>.mat`.
- **Comment fields [V]:** `Age`, `Sex`, `Dx`, `Rx`, `Hx`, `Sx` in all parsed headers; no other comment keys. Rx = Hx = Sx = "Unknown" in 35,567/35,567.
- **Age [V]:** numeric 35,526 (99.88%); literal `NaN` 41; no absent key; integer-valued; min 0, max 89, mean 58.2, median 61. **0 in 171** records (convention undocumented; genuine infants vs missing marker unresolved — [H] partly a marker: age 0 occurs only at IDs ≥ ~JS10696, and in the 43 within-train duplicate pairs 26 pairs disagree on age); **89 in 726 records, none ≥ 90**, suggesting top-coding (observed only). `NaN` also occurs only at IDs ≥ ~JS10867.
- **Sex [V]:** Male 19,958 (56.1%), Female 15,591 (43.8%), Unknown 18.
- **Dx [V]:** 1–12 codes per record (16,943 with one code). Labels = diagnostic codes only.
- **Conflicts [V]:** within the 43 training-only duplicate pairs (identical signals): age differs in 26, sex in 1, mapped labels in 14.
- **Timestamp [V]:** 811 headers carry a WFDB base time/date on the record line (e.g., `24-Mar-2021 05:29:25`), **all dated 2021**, and they are exactly the 811 training records with IDs JS08344–JS09373 (every training record in that range, none outside). Whether it is acquisition or export time is undocumented. The label mix of that block differs (e.g., AFIB 19.7% vs lower elsewhere; AF 5.4% vs 18.4%) — [H] a batch/period structure by ID range (the 1,030-ID range also contains non-training records that were not inspected). Batch/ID range is a **potential confound** for any extracted concept and should be tracked as a covariate, never used as a feature.

## 5. Can headers support conversion to physical amplitudes?

[V/H] Yes, with caveats. The signal lines specify 16-bit samples at **1000 ADC units per mV** with ADC zero 0 and no separate baseline, so `mV = ADC / 1000` is what the header supports (WFDB convention; the storage `16+24` matches the Level-4 `.mat` layout verified in Step 2). Caveats: the calibration cannot be checked from headers; hardware, filtering and lead-wise gain accuracy are undocumented; the checksum/initial-value fields were not used. **Future measurement preprocessing must start from the raw integer signal in mV and stay separate from the recovered backbone preprocessing** (0.5–40 Hz bandpass, int16 truncation, per-lead z-score), whose output no longer carries physical amplitude. Lead order is consistent across all parsed headers, so lead identity is reliable at the header level; lead misplacement/reversal cannot be assessed from headers.

## 6. Label dictionary notes (diagnoses that mention measurements)

The dictionary describes diagnoses, e.g., "PR interval extension" (PRIE), "QT interval extension" (QTIE), "Axis left/right shift" (ALS/ARS), "lower voltage QRS in all lead" (LVQRSAL), "ST drop down" (STDD), "ST extension" (STE), "ST tilt up" (STTU), "T wave opposite" (TWO). None gives a numeric definition; "ST extension" and "T wave opposite" are ambiguous and unresolved. LVQRSCL and LVQRSLL are listed but have no records. Their training counts (§ `concept_feasibility.csv`) show where weak, **non-independent** plausibility checks exist.

## 7. Feasibility outcome (see `concept_feasibility.csv`)

Initial set recommended for a pilot, chosen because it depends on the most robust waveform step (R-peak detection plus fixed QRS windows) rather than on full delineation: **C01 ventricular rate/RR, C02 RR irregularity, C03 frontal QRS axis, C04 QRS amplitude/low-voltage**, with **C05 signal-quality masking** as a prerequisite. QRS duration, PR, QT/QTc, ST, T, P-wave and ectopy concepts are deferred (need delineation; PR/P undefined in AF/AFIB; QT method-dependent; ST/STE naming unresolved). Age/sex are contextual covariates, not ECG concepts. All extraction methods are candidates, not validated choices.

**Reference data needed:** trusted annotations do not exist locally. To establish physiological accuracy one needs (a) an expert-annotated sample (R-peaks, axis, QRS amplitude, quality) drawn from training-role data with rules fixed before assessment on held-out data, and/or (b) an external annotated ECG database — candidates to be identified and verified in the literature step. Diagnoses can only serve as coarse plausibility checks (e.g., direction of rate in SB vs ST), never as accuracy evidence, and cannot validate a measurement that the diagnosis defines.

## 8. Limitations and not checked

Only training-role headers were examined; counts for validation/test were not assessed (by design). Headers beyond the Age/Sex/Dx/Rx/Hx/Sx and record-line fields might differ in non-training records. No waveform-level quality, noise or lead-reversal checks were made. Amplitude calibration, filtering history and acquisition device are unknown. Units/missing conventions are inferred from observed values, not documentation. The meaning of the timestamp block is unknown. No expert review was available.
