r"""Build float16 preprocessed signal caches for Phase 1. NOT YET EXECUTED on real data (only --dry-run has been run, which reads manifests but no waveform).
Each cache = x.npy (N,12,5000) float16 [= original V2 `preprocess`], y.npy (N,C) float32, meta.csv (id, group, inner_fold, role), class_names.json, cache_manifest.json.
`inner_fold` >= 0 marks development folds; -1 marks evaluation-only partitions that the training script never loads.

Partitions
  ptbxl/train       folds 1-8, eligible records, inner_fold = patient-grouped inner CV (splits_v3.ptbxl_inner_folds)
  ptbxl/fold9       exposed reference; needs --allow-fold9; build only AFTER a finalist is frozen
  ptbxl/fold10      NEVER built here (refused; the persistent guard would also deny it)
  physionet/paper_cv      original folds 0-8 (paper protocol A1), inner_fold = original fold
  physionet/train_role    Step 3.7 training role (A2), inner_fold = duplicate-aware 3-fold (splits_v3.group_kfold, seed 20261120)
  physionet/paper_test, validation_role, test_reserved   historically exposed; need --allow-exposed; inner_fold = -1
Usage: .venv\Scripts\python.exe -I scripts\build_cache.py --dataset ptbxl --partition train [--dry-run]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import provenance as pv  # noqa: E402
from ecg_concept import splits_v3 as sp  # noqa: E402
from ecg_concept.labels import CLASS_NAMES  # noqa: E402
from ecg_concept.preprocessing import preprocess, read_raw_mat  # noqa: E402
from ecg_concept.ptbxl_guard import Fold10Guard  # noqa: E402

PTBXL_ROOT = os.path.join(ROOT, "data", "raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
PHYS_ROOT = os.path.join(ROOT, "data", "raw", "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0")
PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SEED = 20261120


def read_csv(rel):
    with open(os.path.join(ROOT, rel), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plan(dataset, partition, allow_exposed, allow_fold9):
    """Returns (rows, class_names, loader(row)->raw int16 (12,T) or path info). Metadata only."""
    if dataset == "ptbxl":
        if partition == "fold10":
            raise PermissionError("PTB-XL fold 10 is locked; it is never built into a cache in Phase 1")
        rows = read_csv("benchmarks/ptbxl_v1/manifest_train_val.csv")
        for r in rows:
            r["ecg_id"] = int(r["ecg_id"])
        rows = [r for r in rows if r["excluded"] in ("False", "false", "0", "")]
        if partition == "train":
            inner = sp.ptbxl_inner_folds(rows)
            rows = [r for r in rows if r["ecg_id"] in inner]
            out = [{"id": r["ecg_id"], "group": r["patient_id"], "inner_fold": inner[r["ecg_id"]], "role": "train", "y": [float(r[c]) for c in PTBXL_CLASSES], "path": os.path.join(PTBXL_ROOT, r["filename_hr"]), "kind": "ptbxl"} for r in rows]
            assert all(int(r["strat_fold"]) <= 8 for r in rows)
        elif partition == "fold9":
            if not allow_fold9:
                raise PermissionError("fold 9 is an exposed reference partition: pass --allow-fold9 (only after a finalist is frozen)")
            rows = [r for r in rows if int(r["strat_fold"]) == 9]
            out = [{"id": r["ecg_id"], "group": r["patient_id"], "inner_fold": -1, "role": "fold9", "y": [float(r[c]) for c in PTBXL_CLASSES], "path": os.path.join(PTBXL_ROOT, r["filename_hr"]), "kind": "ptbxl"} for r in rows]
        else:
            raise ValueError(partition)
        return out, PTBXL_CLASSES
    cidx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def yvec(lbl):
        v = np.zeros(len(CLASS_NAMES), np.float32)
        for l in lbl.split("|"):
            if l:
                v[cidx[l]] = 1
        return v.tolist()
    ofm = read_csv("results/step3_7/original_fold_manifest.csv")
    role = {r["record_id"]: r for r in read_csv("results/step3_7/prospective_role_manifest.csv")}
    if partition == "paper_cv":
        out = [{"id": r["record_id"], "group": r["dup_group_id"] or r["record_id"], "inner_fold": int(r["original_fold"]), "role": "paper_cv", "y": yvec(r["labels"]), "path": os.path.join(PHYS_ROOT, r["dataset_rel_path"]), "kind": "physionet"} for r in ofm if int(r["original_fold"]) <= 8]
    elif partition == "train_role":
        sel = [r for r in ofm if role[r["record_id"]]["role"] == "train"]
        groups = [r["dup_group_id"] or f"SINGLE-{r['record_id']}" for r in sel]
        cl = [[cidx[l] for l in r["labels"].split("|") if l] for r in sel]
        fold = sp.group_kfold([r["record_id"] for r in sel], groups, 3, SEED, labels=cl, n_classes=len(CLASS_NAMES))
        out = [{"id": r["record_id"], "group": g, "inner_fold": fold[r["record_id"]], "role": "train_role", "y": yvec(r["labels"]), "path": os.path.join(PHYS_ROOT, r["dataset_rel_path"]), "kind": "physionet"} for r, g in zip(sel, groups)]
    elif partition in ("paper_test", "validation_role", "test_reserved"):
        if not allow_exposed:
            raise PermissionError(f"{partition} is historically exposed: pass --allow-exposed (report-only use)")
        if partition == "paper_test":
            sel = [r for r in ofm if int(r["original_fold"]) == 9]
        else:
            sel = [r for r in ofm if role[r["record_id"]]["role"] == partition]
        out = [{"id": r["record_id"], "group": r["dup_group_id"] or r["record_id"], "inner_fold": -1, "role": partition, "y": yvec(r["labels"]), "path": os.path.join(PHYS_ROOT, r["dataset_rel_path"]), "kind": "physionet"} for r in sel]
    else:
        raise ValueError(partition)
    return out, list(CLASS_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ptbxl", "physionet"])
    ap.add_argument("--partition", required=True)
    ap.add_argument("--allow-exposed", action="store_true")
    ap.add_argument("--allow-fold9", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows, classes = plan(a.dataset, a.partition, a.allow_exposed, a.allow_fold9)
    folds = {}
    for r in rows:
        folds[r["inner_fold"]] = folds.get(r["inner_fold"], 0) + 1
    summary = {"dataset": a.dataset, "partition": a.partition, "n_records": len(rows), "inner_fold_sizes": dict(sorted(folds.items())), "n_classes": len(classes), "waveforms_read": 0 if a.dry_run else len(rows)}
    if a.dry_run:
        print(json.dumps(summary, indent=1))
        return summary
    guard = Fold10Guard().install_hooks()
    out = os.path.join(ROOT, "data", "processed", f"{a.dataset}_{a.partition}_v1")
    if os.path.exists(out):
        raise FileExistsError(out)
    os.makedirs(out)
    x = np.lib.format.open_memmap(os.path.join(out, "x.npy"), mode="w+", dtype=np.float16, shape=(len(rows), 12, 5000))
    t0 = time.time()
    for i, r in enumerate(rows):
        if r["kind"] == "ptbxl":
            from ecg_concept.ptbxl_reader import read_record
            raw = read_record(r["path"])[0]
        else:
            raw = read_raw_mat(r["path"])
        x[i] = preprocess(raw).astype(np.float16)
        if i % 2000 == 0:
            print(i, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    x.flush()
    np.save(os.path.join(out, "y.npy"), np.array([r["y"] for r in rows], np.float32))
    with open(os.path.join(out, "meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "inner_fold", "role"])
        w.writerows([[r["id"], r["group"], r["inner_fold"], r["role"]] for r in rows])
    json.dump(classes, open(os.path.join(out, "class_names.json"), "w"))
    summary.update(preprocessing="ecg_concept.preprocessing.preprocess (original V2 pipeline, int16-truncation quirk kept) stored as float16", x_sha256=pv.sha256_file(os.path.join(out, "x.npy")), fold10_guard_allowed_reads=guard.n_allowed_reads(),
                   code_sha256={f: pv.sha256_file(os.path.join(ROOT, f)) for f in ("scripts/build_cache.py", "src/ecg_concept/preprocessing.py", "src/ecg_concept/splits_v3.py")})
    json.dump(summary, open(os.path.join(out, "cache_manifest.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
