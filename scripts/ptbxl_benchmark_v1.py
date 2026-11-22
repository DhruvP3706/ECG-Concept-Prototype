"""PTB-XL benchmark v1: validate the WFDB-free reader on ALL fold 1-9 500 Hz records, then write the manifests.
Never opens a fold-10 .dat/.hea (test records are only read as metadata rows). Refuses to overwrite existing outputs.
Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_benchmark_v1.py
"""
import collections
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import ptbxl as px  # noqa: E402
from ecg_concept.ptbxl_reader import RecordError  # noqa: E402

OUT = os.path.join(ROOT, "benchmarks", "ptbxl_v1")
CFG = os.path.join(OUT, "config.json")
TV, TEST, RES = (os.path.join(OUT, n) for n in ("manifest_train_val.csv", "manifest_test_reserved.csv", "reader_validation_results.json"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    for p in (TV, TEST, RES):
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}")
    cfg = json.load(open(CFG, encoding="utf-8"))
    data = os.path.join(ROOT, cfg["dataset"]["root"])
    # --- source files are the ones the config was frozen against
    for key, name in (("ptbxl_database_csv_sha256", "ptbxl_database.csv"), ("scp_statements_csv_sha256", "scp_statements.csv")):
        if sha(os.path.join(data, name)) != cfg["dataset"][key]:
            sys.exit(f"{name} does not match the frozen checksum")
    sums = {}
    for line in open(os.path.join(data, "SHA256SUMS.txt")):
        h, n = line.strip().split(None, 1)
        sums[n.lstrip("*")] = h
    statements = px.read_statements(os.path.join(data, "scp_statements.csv"))
    records = px.read_metadata(os.path.join(data, "ptbxl_database.csv"))
    splits = px.make_splits(records)                                          # raises on patient overlap
    lab = cfg["labels"]
    vocab = lab["vocabulary"]
    kw = dict(level=lab["level"], min_likelihood=lab["min_likelihood"], unknown_as_positive=lab["unknown_as_positive"])
    derived = px.derive_vocabulary(splits.train, statements, **kw)
    assert sorted(derived) == sorted(vocab), (derived, vocab)

    # --- reader validation: train + validation only (test records are never passed to the reader)
    tv = list(splits.train) + list(splits.val)
    Y = px.build_targets(tv, statements, vocab, **kw)
    status, per_fold, lo, hi, lead_orders = {}, collections.defaultdict(lambda: [0, 0]), 32767, -32768, collections.Counter()
    flat, t0 = 0, time.time()
    for r in tv:
        assert r.strat_fold != px.TEST_FOLD
        try:
            ls = px.load_signal_native(r, data)
            assert ls.signal.shape == (12, 5000) and ls.signal.dtype == np.int16
            lo, hi = min(lo, int(ls.signal.min())), max(hi, int(ls.signal.max()))
            flat += int((ls.signal.std(axis=1) == 0).any())
            lead_orders[tuple(ls.original_lead_names)] += 1
            status[r.ecg_id] = "ok"
            per_fold[r.strat_fold][0] += 1
        except (RecordError, OSError) as e:
            status[r.ecg_id] = f"FAILED: {e}"
            per_fold[r.strat_fold][1] += 1
    failed = {k: v for k, v in status.items() if v != "ok"}
    results = {"n_checked": len(tv), "n_ok": len(tv) - len(failed), "n_failed": len(failed), "failed": failed,
               "per_fold_ok_failed": {str(k): v for k, v in sorted(per_fold.items())}, "sample_min": lo, "sample_max": hi,
               "records_with_a_constant_lead": flat, "header_lead_orders": {",".join(k): v for k, v in lead_orders.items()},
               "seconds": round(time.time() - t0), "fold10_files_opened": 0,
               "code_sha256": {n: sha(os.path.join(ROOT, "src", "ecg_concept", n)) for n in ("ptbxl.py", "ptbxl_reader.py")},
               "config_sha256": sha(CFG)}
    # --- manifests
    base = ["ecg_id", "patient_id", "strat_fold", "role", "filename_hr"]
    tail = vocab + ["n_positive", "excluded", "exclusion_reason"]

    def label_cols(y):
        n = int(y.sum())
        return [int(v) for v in y] + [n, int(n == 0), "" if n else "no_positive_diagnostic_superclass"]

    with open(TV, "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(base + ["scp_codes_raw", "hea_sha256", "dat_sha256", "reader_status"] + tail)
        for r, y in zip(tv, Y):
            p = r.filename_hr
            w.writerow([r.ecg_id, r.patient_id, r.strat_fold, px.record_role(r), p, r.scp_raw, sums[p + ".hea"], sums[p + ".dat"], status[r.ecg_id]] + label_cols(y))
    test = list(splits.open_test(px.TEST_CONFIRMATION))                         # metadata rows only; used to write the reserved manifest
    Yt = px.build_targets(test, statements, vocab, **kw)
    with open(TEST, "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(base + ["scp_codes_raw"] + tail)
        for r, y in zip(test, Yt):
            w.writerow([r.ecg_id, r.patient_id, r.strat_fold, "test_reserved", r.filename_hr, r.scp_raw] + label_cols(y))
    results["manifest_sha256"] = {"train_val": sha(TV), "test_reserved": sha(TEST)}
    results["counts"] = {"train_val": len(tv), "test_reserved": len(test), "excluded_train": int((Y[:len(splits.train)].sum(1) == 0).sum()),
                         "excluded_val": int((Y[len(splits.train):].sum(1) == 0).sum()),
                         "train_support": dict(zip(vocab, Y[:len(splits.train)].sum(0).tolist())), "val_support": dict(zip(vocab, Y[len(splits.train):].sum(0).tolist()))}
    with open(RES, "x", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "failed"}, indent=1))
    print("failed:", list(failed.items())[:10])


if __name__ == "__main__":
    main()
