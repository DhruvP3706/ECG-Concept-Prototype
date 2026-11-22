# PTB-XL benchmark protocol (DRAFT — dataset not yet downloaded locally)

Status legend: **[Official]** stated by official documentation (PhysioNet page read 2026-11-11, or stated by the project owner) · **[Pending]** requires the downloaded files · **[Decision]** needs an explicit project decision · **[Implemented]** code exists and is unit-tested on synthetic data only.

**Nothing in this document or in `src/ecg_concept/ptbxl.py` has been run on real PTB-XL records.** No result is reported here.

## 1. Dataset and citation
* PTB-XL v1.0.3, PhysioNet, CC BY 4.0. [Official]
* Wagner, P., Strodthoff, N., Bousseljot, R.-D., et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154 (2020). DOI: 10.1038/s41597-020-0495-6.
* Official facts used by the code [Official]: 21,799 records of 18,869 patients; 10 s, 12 leads; 500 Hz (`records500/`, `…_hr`) and 100 Hz (`records100/`, `…_lr`); 16-bit at 1 µV/LSB; Schiller devices; unique `ecg_id` and `patient_id`; `strat_fold` 1–10, folds 9 and 10 have at least one human evaluation; 71 SCP statements with likelihoods (0 = unknown); no R-peak/amplitude annotations.
* **Patient-preserving split** (folds 1–8 train, 9 validation, 10 test) is [Official]/stated by the project owner. **Local verification is [Pending]**: `validate_records` raises if a patient appears in more than one of train/val/test once `ptbxl_database.csv` is available.

## 2. Intended task
Multi-label ECG statement classification on the 500 Hz records, accuracy-first, with concept/prototype branches evaluated as add-ons and the existing strict concept-bottleneck model kept as a comparator. PTB-XL labels differ from the 45 PhysioNet classes: no PhysioNet head, threshold or class mapping is reused.

## 3. Label choices requiring a decision [Decision]
Implemented, configurable, **no defaults** for the first two (`src/ecg_concept/ptbxl.py`: `derive_vocabulary`, `build_targets`):

| Choice | Options | Note |
|---|---|---|
| Level | `superclass` (diagnostic only), `subclass` (diagnostic only), `statement` (individual SCP codes) | class lists/counts must be read from the downloaded `scp_statements.csv` [Pending] |
| `min_likelihood` | value in (0, 100] | required argument; presence = likelihood ≥ threshold |
| `unknown_as_positive` | True/False | likelihood 0 means unknown; required argument |
| Categories (statement level) | diagnostic / form / rhythm | single category by default; mixing needs `allow_mixed_categories=True` |
| Vocabulary | derived from **training records only**, optional `min_support` | never derived from validation/test |
| Unknown codes | `error` (default) / `ignore` | |

Record the chosen values in the experiment config before fitting.

## 4. Splits and the test-evaluation rule
* `make_splits(records)`: folds 1–8 / 9 / 10; checks fold validity, duplicate `ecg_id`, patient overlap. [Implemented]
* `PtbxlSplits` exposes `train` and `val` only; fold-10 records are released once by `open_test("EVALUATE_TEST_ONCE")`, a second release raises. `load_signal` refuses fold-10 records unless `allow_test=True`. [Implemented; in-process guard only — a persistent release log across sessions is not implemented]
* **One-time rule:** fold 10 is used once, after the model, label level, thresholds (chosen on fold 9 only), metrics and config hash are frozen. No tuning, model selection or threshold fitting on fold 10; no peeking at test signals, predictions or metrics beforehand.
* `check_files` verifies `.hea/.dat` existence without opening files and skips fold 10 by default.

## 5. Signal loading and preprocessing compatibility
* Loader: `load_signal` reads the 500 Hz record as **digital** samples through WFDB (lazy optional import; not installed here), validates the header (500 Hz, 12 leads, 5,000 samples, identical gain/baseline across leads, expected gain 1,000 units/mV, units mV), checks integer dtype and int16 range, and reorders leads **by name** to the model order. [Implemented, tested with a fake WFDB object]
* Model lead order is taken from `src/ecg_concept/preprocessing.py::LEAD_NAMES`: I, II, III, aVR, aVL, aVF, V1–V6 (the backbone is positional). [Verified from source]
* Pending checks on the real files [Pending]: actual header lead names/order (the PhysioNet page text lists "AVL, AVR, AVF", not necessarily the file order); real ADC gain/baseline/units strings; that the wfdb attribute names used (`d_signal`, `sig_name`, `fs`, `sig_len`, `adc_gain`, `baseline`, `units`) match the installed version; prior filtering/denoising history; equivalence of the digital samples to the PhysioNet `val` arrays.
* `backbone_input` applies the existing `preprocess` unchanged. **Equivalence with the PhysioNet pipeline is not claimed.** Whether to keep the int16-truncation quirk when reusing pretrained weights is [Decision].

## 6. Metrics to report (patient-aware)
Primary: macro-AUROC; also micro-AUROC, macro/micro average precision, per-class AUROC/AP with support, F1/precision/recall at thresholds fixed on fold 9 only (or fixed 0.5, stated), exact-match and Hamming loss. Classes with no positives or no negatives in a split are reported as undefined, not zero. Because records are not independent within patient (21,799 records vs 18,869 patients), confidence intervals use a **patient-level (cluster) bootstrap** with a recorded seed, not a record-level bootstrap. Existing `src/ecg_concept/metrics.py` can compute the point metrics; a cluster-bootstrap wrapper is [Pending]. Comparisons to published numbers (e.g., ProtoECGNet) are only valid for identical label sets and splits; otherwise label them non-comparable.

## 7. Reproducibility
Freeze and hash: `ptbxl_database.csv`, `scp_statements.csv`, the role manifest (ecg_id → train/val/test), label config, preprocessing config, code, seeds. Use new run directories; never overwrite earlier results.

## 8. Implemented vs pending
| Item | Status |
|---|---|
| SCP parsing (`ast.literal_eval`), metadata/statements readers | Implemented (synthetic tests) |
| Configurable label construction | Implemented (synthetic tests) |
| Patient-wise split/validation, test lock | Implemented (synthetic tests); real-data verification Pending |
| WFDB loader, lead reorder, header checks | Implemented (fake-WFDB tests); real-file verification Pending |
| Preprocessing equivalence, label-level decision, cluster bootstrap, persistent test-release log | Pending / Decision |

Tests: `.venv\Scripts\python.exe -I tests\test_ptbxl.py` (22 synthetic tests; no download, no WFDB).
