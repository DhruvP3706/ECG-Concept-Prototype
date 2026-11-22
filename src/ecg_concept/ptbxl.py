"""PTB-XL (v1.0.3) ingestion layer: metadata, label targets, patient-wise split guard, and a WFDB signal loader.

Isolated from the PhysioNet pipeline (nothing existing is imported except the lead order and `preprocess`).
Nothing here has been run on real PTB-XL records yet: the dataset is not downloaded locally, so every check that needs
the files (patient separation, WFDB header lead order/gain/baseline, file existence) is implemented but UNVERIFIED on real data.

Design rules
* `scp_codes` are parsed with `ast.literal_eval` only and validated (dict[str, finite number >= 0]).
* Likelihood handling is explicit: `min_likelihood` and `unknown_as_positive` have no defaults.
* Statements (`diagnostic`/`form`/`rhythm`) are never mixed silently; superclass/subclass labels are a separate `level`.
* Folds 1-8 train, 9 validation, 10 test. Test records are not exposed by `PtbxlSplits`; they require an explicit, one-time confirmation,
  and `load_signal` refuses a fold-10 record unless `allow_test=True`.
* Signals are read from the 500 Hz (`filename_hr`) records only, as digital samples, and reordered to the model's lead order by NAME.
* WFDB is imported lazily; it is optional and is not installed by this module.
"""
import ast
import csv
import hashlib
import importlib
import math
import os

import numpy as np

from .preprocessing import LEAD_NAMES, preprocess

MODEL_LEAD_ORDER = tuple(LEAD_NAMES)            # source of truth: src/ecg_concept/preprocessing.py (I, II, III, aVR, aVL, aVF, V1-V6)
TRAIN_FOLDS = tuple(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10
TEST_CONFIRMATION = "EVALUATE_TEST_ONCE"
CATEGORIES = ("diagnostic", "form", "rhythm")
LEVELS = ("statement", "subclass", "superclass")
_LEAD_ALIASES = {n.lower(): n for n in MODEL_LEAD_ORDER}       # "avr" -> "aVR"; matching is case-insensitive
_DB_COLUMNS = ("ecg_id", "patient_id", "scp_codes", "strat_fold", "filename_hr", "filename_lr")
_STATEMENT_COLUMNS = ("diagnostic", "form", "rhythm", "diagnostic_class", "diagnostic_subclass")


# ------------------------------------------------------------------ metadata
class PtbxlRecord:
    """One row of ptbxl_database.csv (only the fields needed here; raw SCP text preserved)."""
    __slots__ = ("ecg_id", "patient_id", "strat_fold", "filename_hr", "filename_lr", "scp_codes", "scp_raw")

    def __init__(self, ecg_id, patient_id, strat_fold, filename_hr, filename_lr, scp_codes, scp_raw=""):
        self.ecg_id, self.patient_id, self.strat_fold = ecg_id, patient_id, strat_fold
        self.filename_hr, self.filename_lr, self.scp_codes, self.scp_raw = filename_hr, filename_lr, dict(scp_codes), scp_raw

    def __repr__(self):
        return f"PtbxlRecord(ecg_id={self.ecg_id}, patient_id={self.patient_id}, fold={self.strat_fold})"


def parse_scp_codes(text):
    """Safely parse an `scp_codes` cell such as "{'NORM': 100.0, 'SR': 0.0}" -> {str: float}. Never uses eval."""
    if text is None or not str(text).strip():
        raise ValueError("empty scp_codes cell")
    try:
        value = ast.literal_eval(str(text).strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError) as e:
        raise ValueError(f"scp_codes is not a literal: {str(text)[:60]!r}") from e
    if not isinstance(value, dict):
        raise ValueError("scp_codes must be a dict")
    out = {}
    for k, v in value.items():
        if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"invalid scp_codes entry {k!r}: {v!r}")
        out[k] = float(v)
    return out


def _to_int(value, name):
    try:
        f = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} is not numeric: {value!r}") from e
    if not math.isfinite(f) or f != int(f):
        raise ValueError(f"{name} is not an integer: {value!r}")
    return int(f)


def read_metadata(db_csv):
    """Read ptbxl_database.csv -> list of PtbxlRecord (order preserved). Duplicate ecg_id or missing columns raise."""
    with open(db_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in _DB_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{db_csv}: missing columns {missing}")
        recs, seen = [], set()
        for row in reader:
            ecg_id = _to_int(row["ecg_id"], "ecg_id")
            if ecg_id in seen:
                raise ValueError(f"duplicate ecg_id {ecg_id}")
            seen.add(ecg_id)
            recs.append(PtbxlRecord(ecg_id, _to_int(row["patient_id"], "patient_id"), _to_int(row["strat_fold"], "strat_fold"),
                                    row["filename_hr"], row["filename_lr"], parse_scp_codes(row["scp_codes"]), row["scp_codes"]))
    return recs


def read_statements(csv_path):
    """Read scp_statements.csv -> {code: {diagnostic, form, rhythm (bool), diagnostic_class, diagnostic_subclass (str|None)}}."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    missing = [c for c in _STATEMENT_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    ix = {c: header.index(c) for c in _STATEMENT_COLUMNS}
    out = {}
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        code = r[0].strip()
        if code in out:
            raise ValueError(f"duplicate statement {code}")
        flag = lambda c: bool(r[ix[c]].strip()) and float(r[ix[c]]) == 1.0
        txt = lambda c: (r[ix[c]].strip() or None)
        out[code] = {"diagnostic": flag("diagnostic"), "form": flag("form"), "rhythm": flag("rhythm"),
                     "diagnostic_class": txt("diagnostic_class"), "diagnostic_subclass": txt("diagnostic_subclass")}
    return out


# ------------------------------------------------------------------ labels
def _present(likelihood, min_likelihood, unknown_as_positive):
    """Explicit presence rule: likelihood > 0 -> present iff >= min_likelihood; likelihood == 0 (unknown) -> `unknown_as_positive`."""
    return unknown_as_positive if likelihood == 0 else likelihood >= min_likelihood


def _targets_of(record, statements, level, categories, unknown_codes):
    """Map one record's statements to target names (before the likelihood rule). Yields (target_name, likelihood)."""
    for code, lik in record.scp_codes.items():
        info = statements.get(code)
        if info is None:
            if unknown_codes == "error":
                raise ValueError(f"record {record.ecg_id}: statement {code!r} not in scp_statements")
            continue
        if level == "statement":
            if any(info[c] for c in categories):
                yield code, lik
        elif info["diagnostic"]:                                      # superclass/subclass use diagnostic statements only
            name = info["diagnostic_class" if level == "superclass" else "diagnostic_subclass"]
            if name:
                yield name, lik


def _check_options(level, categories, allow_mixed_categories, min_likelihood, unknown_codes):
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if unknown_codes not in ("error", "ignore"):
        raise ValueError("unknown_codes must be 'error' or 'ignore'")
    if not (isinstance(min_likelihood, (int, float)) and 0 < min_likelihood <= 100):
        raise ValueError("min_likelihood must be in (0, 100]")
    cats = tuple(categories)
    if not cats or any(c not in CATEGORIES for c in cats):
        raise ValueError(f"categories must be a non-empty subset of {CATEGORIES}")
    if level == "statement" and len(set(cats)) > 1 and not allow_mixed_categories:
        raise ValueError("mixing diagnostic/form/rhythm statements requires allow_mixed_categories=True")
    if level != "statement" and cats != ("diagnostic",):
        raise ValueError("superclass/subclass levels are diagnostic-only; use categories=('diagnostic',)")
    return cats


def derive_vocabulary(records, statements, *, level, min_likelihood, unknown_as_positive, categories=("diagnostic",),
                      allow_mixed_categories=False, min_support=1, unknown_codes="error"):
    """Sorted target names with >= min_support positive records. CALL WITH TRAINING RECORDS ONLY (no hidden derivation elsewhere)."""
    cats = _check_options(level, categories, allow_mixed_categories, min_likelihood, unknown_codes)
    support = {}
    for r in records:
        for name in {n for n, lik in _targets_of(r, statements, level, cats, unknown_codes) if _present(lik, min_likelihood, unknown_as_positive)}:
            support[name] = support.get(name, 0) + 1
    return sorted(n for n, s in support.items() if s >= min_support)


def build_targets(records, statements, vocabulary, *, level, min_likelihood, unknown_as_positive, categories=("diagnostic",),
                  allow_mixed_categories=False, unknown_codes="error"):
    """Multi-label matrix (N, C) uint8 for an EXPLICIT vocabulary. Aggregated levels are 'any member statement present'."""
    cats = _check_options(level, categories, allow_mixed_categories, min_likelihood, unknown_codes)
    vocab = list(vocabulary)
    if not vocab or len(set(vocab)) != len(vocab):
        raise ValueError("vocabulary must be non-empty and unique")
    col = {n: j for j, n in enumerate(vocab)}
    Y = np.zeros((len(records), len(vocab)), np.uint8)
    for i, r in enumerate(records):
        for name, lik in _targets_of(r, statements, level, cats, unknown_codes):
            if name in col and _present(lik, min_likelihood, unknown_as_positive):
                Y[i, col[name]] = 1
    return Y


# ------------------------------------------------------------------ split guard
def record_role(record):
    f = record.strat_fold
    return "train" if f in TRAIN_FOLDS else "val" if f == VAL_FOLD else "test"


def validate_records(records):
    """Metadata-only integrity: valid folds, unique ecg_id, no patient in more than one of train/val/test. Raises ValueError."""
    ids = [r.ecg_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate record IDs")
    bad = [r.ecg_id for r in records if not isinstance(r.strat_fold, int) or isinstance(r.strat_fold, bool) or not 1 <= r.strat_fold <= 10]
    if bad:
        raise ValueError(f"invalid strat_fold for ecg_id {bad[:5]} (must be int 1..10)")
    roles = {}
    for r in records:
        roles.setdefault(r.patient_id, set()).add(record_role(r))
    leaked = sorted(p for p, s in roles.items() if len(s) > 1)
    if leaked:
        raise ValueError(f"{len(leaked)} patients appear in more than one of train/val/test, e.g. {leaked[:5]}")


class PtbxlSplits:
    """Train and validation records are plain attributes. Test records are held privately and released once, with explicit confirmation."""

    def __init__(self, train, val, test):
        self.train, self.val = tuple(train), tuple(val)
        self._test, self._opened = tuple(test), False
        self.n_test = len(self._test)
        self.test_ids_sha256 = hashlib.sha256("\n".join(str(r.ecg_id) for r in self._test).encode()).hexdigest()

    def open_test(self, confirm):
        if confirm != TEST_CONFIRMATION:
            raise PermissionError(f"test records require confirm={TEST_CONFIRMATION!r}")
        if self._opened:
            raise PermissionError("test records were already released once; the one-time test evaluation rule forbids a second release")
        self._opened = True
        return self._test

    def __repr__(self):
        return f"PtbxlSplits(train={len(self.train)}, val={len(self.val)}, test={self.n_test} [locked])"


def make_splits(records):
    validate_records(records)
    by = {"train": [], "val": [], "test": []}
    for r in records:
        by[record_role(r)].append(r)
    return PtbxlSplits(by["train"], by["val"], by["test"])


def check_files(records, root, *, include_test=False):
    """Metadata-only existence check of the 500 Hz .hea/.dat files (never opens them). Returns missing paths."""
    missing = []
    for r in records:
        if record_role(r) == "test" and not include_test:
            continue
        for ext in (".hea", ".dat"):
            p = os.path.join(root, r.filename_hr + ext)
            if not os.path.isfile(p):
                missing.append(p)
    return missing


# ------------------------------------------------------------------ signals
def canonical_lead(name):
    n = str(name).strip().lower()
    if n not in _LEAD_ALIASES:
        raise ValueError(f"unexpected lead name {name!r}")
    return _LEAD_ALIASES[n]


def reorder_leads(d_signal, names, target=MODEL_LEAD_ORDER):
    """(T, n_sig) WFDB-style array + header lead names -> (len(target), T) array in `target` order, by NAME.
    Raises on a wrong count, an unexpected name, a duplicate (after case normalisation) or a missing lead."""
    d = np.asarray(d_signal)
    names = list(names)
    if d.ndim != 2 or d.shape[1] != len(names):
        raise ValueError(f"signal shape {d.shape} does not match {len(names)} header leads (expected (T, n_sig))")
    if len(names) != len(target):
        raise ValueError(f"expected {len(target)} leads, header has {len(names)}")
    canon = [canonical_lead(n) for n in names]
    if len(set(canon)) != len(canon):
        raise ValueError(f"duplicate leads in header: {names}")
    missing = [t for t in target if t not in canon]
    if missing:
        raise ValueError(f"missing leads {missing}")
    return np.ascontiguousarray(d[:, [canon.index(t) for t in target]].T)


class HeaderExpectation:
    """What a 500 Hz PTB-XL header must satisfy. Defaults follow the documented 500 Hz, 10 s, 12-lead, 1 uV/LSB (1000 units/mV) format;
    None skips a check. Baseline must be identical across leads and is reported, not assumed."""

    def __init__(self, fs=500, n_sig=12, sig_len=5000, gain=1000.0, units="mv"):
        self.fs, self.n_sig, self.sig_len, self.gain, self.units = fs, n_sig, sig_len, gain, units


def check_header(info, expected=None):
    """info: dict(names, fs, sig_len, gains, baselines, units). Returns a normalised summary or raises ValueError."""
    e = expected or HeaderExpectation()
    n = len(info["names"])
    for key in ("gains", "baselines", "units"):
        if len(info[key]) != n:
            raise ValueError(f"header field {key} has {len(info[key])} entries for {n} leads")
    if e.n_sig is not None and n != e.n_sig:
        raise ValueError(f"expected {e.n_sig} leads, found {n}")
    if e.fs is not None and info["fs"] != e.fs:
        raise ValueError(f"sampling frequency {info['fs']} != expected {e.fs} (500 Hz records only)")
    if e.sig_len is not None and info["sig_len"] != e.sig_len:
        raise ValueError(f"signal length {info['sig_len']} != expected {e.sig_len}")
    if len(set(float(g) for g in info["gains"])) != 1:
        raise ValueError(f"ADC gain differs across leads: {sorted(set(info['gains']))}")
    gain = float(info["gains"][0])
    if e.gain is not None and not math.isclose(gain, e.gain):
        raise ValueError(f"ADC gain {gain} != expected {e.gain} units/mV")
    if len(set(int(b) for b in info["baselines"])) != 1:
        raise ValueError(f"ADC baseline differs across leads: {sorted(set(info['baselines']))}")
    units = {str(u).strip().lower() for u in info["units"]}
    if len(units) != 1 or (e.units is not None and units != {e.units.lower()}):
        raise ValueError(f"units {sorted(units)} != expected {e.units!r}")
    return {"fs": info["fs"], "sig_len": info["sig_len"], "gain": gain, "baseline": int(info["baselines"][0]), "units": units.pop()}


class LoadedSignal:
    """Digital int16 samples (12, T) in MODEL lead order, plus the header facts used to accept them."""
    __slots__ = ("ecg_id", "signal", "header", "original_lead_names")

    def __init__(self, ecg_id, signal, header, original_lead_names):
        self.ecg_id, self.signal, self.header, self.original_lead_names = ecg_id, signal, header, original_lead_names


def _require_wfdb():
    try:
        return importlib.import_module("wfdb")
    except ImportError as e:
        raise ImportError("PTB-XL signal loading needs the optional 'wfdb' package, which is not installed "
                          "(this module never installs packages)") from e


def load_signal(record, root, *, allow_test=False, expected=None):
    """Read one 500 Hz record as DIGITAL samples, validate the header, reorder leads by name, return `LoadedSignal`."""
    if record_role(record) == "test" and not allow_test:
        raise PermissionError(f"ecg_id {record.ecg_id} is a fold-{TEST_FOLD} test record; waveform access requires allow_test=True")
    wfdb = _require_wfdb()
    rec = wfdb.rdrecord(os.path.join(root, record.filename_hr), physical=False)
    d = rec.d_signal
    if d is None:
        raise ValueError(f"ecg_id {record.ecg_id}: no digital samples returned")
    d = np.asarray(d)
    if not np.issubdtype(d.dtype, np.integer):
        raise ValueError(f"ecg_id {record.ecg_id}: digital samples have dtype {d.dtype}, expected integer")
    if d.size and (d.min() < -32768 or d.max() > 32767):
        raise ValueError(f"ecg_id {record.ecg_id}: samples outside int16 range")
    info = {"names": list(rec.sig_name), "fs": rec.fs, "sig_len": rec.sig_len, "gains": list(rec.adc_gain), "baselines": list(rec.baseline), "units": list(rec.units)}
    header = check_header(info, expected)
    if d.shape != (info["sig_len"], len(info["names"])):
        raise ValueError(f"ecg_id {record.ecg_id}: samples {d.shape} do not match header ({info['sig_len']}, {len(info['names'])})")
    return LoadedSignal(record.ecg_id, reorder_leads(d, info["names"]).astype(np.int16), header, info["names"])


def load_signal_native(record, root, *, allow_test=False):
    """WFDB-free read of one 500 Hz record (see `ptbxl_reader`): digital int16 (12, 5000) in model lead order, checksums verified.
    The fold-10 guard runs BEFORE any file is opened."""
    if record_role(record) == "test" and not allow_test:
        raise PermissionError(f"ecg_id {record.ecg_id} is a fold-{TEST_FOLD} test record; waveform access requires allow_test=True")
    from .ptbxl_reader import read_record
    sig, header = read_record(os.path.join(root, record.filename_hr))
    return LoadedSignal(record.ecg_id, sig, header, header["original_lead_names"])


def backbone_input(loaded):
    """Existing backbone preprocessing applied UNCHANGED to the loaded digital samples. Whether PTB-XL samples are equivalent
    to the PhysioNet `val` arrays (scale, baseline, prior filtering) is NOT verified; do not assume equivalence."""
    return preprocess(loaded.signal)
