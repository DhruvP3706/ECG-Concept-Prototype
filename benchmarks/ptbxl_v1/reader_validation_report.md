# PTB-XL v1.0.3 WFDB-free reader: validation report (benchmark ptbxl_v1)

Date: 2026-11-20. Status legend: **verified** = measured in this work; **not verified** = stated as open.

## 1. Reader design (`src/ecg_concept/ptbxl_reader.py`, `ptbxl.load_signal_native`)
Supports exactly one layout, taken from the real headers: one `<name>.dat` per record, 12 signals, WFDB **format 16** (signed little-endian int16), signals interleaved sample by sample (sample-major), no skew / samples-per-frame / byte offset, `1000.0(0)/mV`, ADC resolution 16, ADC zero 0, 500 Hz, 5000 samples, no timestamp on the record line.
- Anything else (other format such as `212`, `16x2`, `16+24`, other rate/length/signal count, other gain/baseline/units, other file name, malformed or non-printable or non-ASCII header, wrong record name) raises `RecordError` with the record name.
- Byte count must equal 12 x 5000 x 2 exactly.
- Per channel: first sample must equal the header initial value; the sum of samples modulo 65536, viewed as signed int16, must equal the header checksum, also viewed as signed int16 (the official headers print some checksums unsigned, e.g. 64050, others signed).
- Channels are reordered by **name** (case-insensitive: AVR -> aVR) to the model order I, II, III, aVR, aVL, aVF, V1-V6 using the existing `ptbxl.reorder_leads`; missing, duplicate or unexpected names raise.
- Output: digital `int16 (12, 5000)` unchanged: no calibration, filtering, normalisation or mV conversion. `preprocessing.py` is untouched.
- Fold-10 guard: `load_signal_native` raises `PermissionError` for fold-10 records before any file is opened unless `allow_test=True` (tested with `open` patched to fail).
- The older wfdb-based `load_signal` is unchanged and still unverified (wfdb not installed).

## 2. Tests (verified)
| File | Tests | Result |
|---|---|---|
| `tests/test_ptbxl_reader.py` (new) | 16 | OK |
| `tests/test_ptbxl.py` | 22 | OK |
| `tests/test_rr_concepts.py` | 20 | OK |
| `tests/test_step4_4.py` | 10 | OK |

Total 68, 0 failures. New tests cover little-endian decoding (byte-level), sample-major multiplexing, name-based reorder from a shuffled header, signed and unsigned checksums (full int16 range, so wraparound is exercised), a single flipped bit detected by checksum, bad initial value and checksum (naming the channel), too short/long byte counts, wrong format/rate/length/signal count/gain/baseline/units, duplicate/unexpected/missing leads, empty/garbage/short/malformed/wrong-name/other-file headers, non-ASCII header bytes, missing files, and the fold-10 guard. During development two of my own tests failed (an error-message regex and a cp1252 encoding issue in the test fixture); the reader logic was not changed for them except the error message text.

## 3. Real-data validation (verified; folds 1-9, 500 Hz only)
- 19,601 of 19,601 records (train 17,418 + validation 2,183) decoded: **0 failures**, 0 checksum failures, 0 initial-value failures (each of the 12 channels per record).
- Per-fold counts equal the metadata (2,173-2,192 per fold, fold 9 = 2,183). All headers list leads in the order I, II, III, AVR, AVL, AVF, V1-V6 (19,601/19,601).
- All arrays `(12, 5000)` int16. Global sample range -32,692 to 32,716. Runtime 317 s.
- Source-file checksums: earlier audit hashed all fold 1-9 `.dat` files (both rates) against `SHA256SUMS.txt`, 0 mismatches. That check was not repeated here; the manifest stores the `SHA256SUMS.txt` values.
- Fold-10 `.hea`/`.dat` files opened: **0** (script counter, plus the fold-10 test records are never passed to the reader).

## 4. Constant-lead finding (verified; record not modified or excluded)
One record has a constant lead: **ecg_id 12722** (`records500/12000/12722_hr`, fold 2, train, patient 2818), lead **V5**: all 5,000 decoded samples equal **0** (min = max = 0), so the signal is genuinely constant in the file, not a decoding artefact (the checksum and initial value passed). Labels: ISC_ 100, ASMI 100 (likelihood 0: LPR, STD_, SBRAD). In the same record V4 has max 3,611 against min -532 (an outlier relative to its neighbours), and other leads look normal. The cause (lead-off / missing electrode) is not established; PTB-XL's `electrodes_problems` column was not checked. The record stays in the manifest unflagged; whether to exclude or mask it is an open decision. It is the only record with a zero-variance lead among the 19,601 checked.

## 5. Manifests (verified)
Files in `benchmarks/ptbxl_v1/` (config `config.json`, sha256 `3c59a5c4604beb070167af70b792ad287935c545cff4d7228707e203860f98c1`):
- `manifest_train_val.csv`: 19,601 rows (train 17,418, role `val` 2,183; the config calls this role "validation"), columns: ecg_id, patient_id, strat_fold, role, filename_hr, scp_codes_raw, hea_sha256, dat_sha256 (copied from `SHA256SUMS.txt`), reader_status (all `ok`), the five superclass targets NORM, MI, STTC, CD, HYP, n_positive, excluded, exclusion_reason. sha256 `2abf08ca...9d24`.
- `manifest_test_reserved.csv`: 2,198 rows, role `test_reserved`, fold 10 only; same columns **without** hashes, reader_status or any signal-derived field (metadata only; it does contain the fold-10 labels derived from the CSV). sha256 `cd5b4d7f...70ac`.
- Checks: 21,799 unique ecg_ids across both files; patients train 15,023 / validation 1,942 / test 1,904 with 0 overlap; every row matches `ptbxl_database.csv` (patient, fold, scp_codes, path); targets recomputed independently from `scp_statements.csv` (diagnostic, likelihood >= 50, superclass) match all 21,799 rows; exclusion flag and reason are consistent with zero positives.
- Excluded (no positive diagnostic superclass): train 1,129, validation 149, test 148. They are kept in the files with `exclusion_reason=no_positive_diagnostic_superclass`.
- Support (train / validation): NORM 7,542 / 942, MI 3,307 / 412, STTC 4,062 / 510, CD 3,900 / 495, HYP 1,820 / 216.
- `reader_validation_results.json` also records the sha256 of `ptbxl.py` (`29e817...ef539`) and `ptbxl_reader.py` (`9b595c...c8c8`) used for the run; changing either file changes these hashes.

## 6. Fold-10 waveform data untouched (verified for this work)
No fold-10 `.hea` or `.dat` file was opened, read, hashed or stat-listed by the reader, the validation script or the manifest script (they read the CSV metadata only). The earlier audit read fold-10 header text and file sizes only. The in-process one-time test release (`open_test`) was used to write the metadata manifest, so that object cannot release test records again; this is not a persistent log.

## 7. Open items
- Backbone preprocessing for PTB-XL (int16 truncation quirk, scale 1000 counts/mV vs the PhysioNet data's 4.88 uV/bit assumption), not decided.
- Constant-lead record 12722 policy; V4 outlier of that record unexplored.
- Role label `val` (manifest) vs `validation` (config) naming; manifests not regenerated.
- `docs/PTBXL_PROTOCOL.md` and the `ptbxl.py` module docstring still say "not run on real data / not downloaded" (docstring left unchanged so the recorded code hash stays valid).
- Persistent test-release log and cluster bootstrap not implemented. Challenge-2021 copy / PTB-XL overlap not established.
