"""Step 3.6: read-only duplicate-signal leakage audit.

Finds byte-identical .mat files and records sharing an identical (non-flat) lead, maps every member onto the
reconstructed original folds (labels.build_index / reproduce_folds), and writes a pair table + JSON summary.
No metrics are computed; nothing in data/, models/ or earlier results is modified.
"""
import csv
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept.labels import CLASS_NAMES, build_index, reproduce_folds, snomed_to_acronym  # noqa: E402

OUT = os.path.join(ROOT, "results", "step3_6")
os.makedirs(OUT, exist_ok=True)
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
# validation fold of each checkpoint (notebook cells 19-21); a checkpoint trains on folds 0-8 minus that fold
VAL = {"ft_fold0": 0, "p1_fold0": 0, "p1_fold1": 1, "p1_fold2": 2}

classes, recs, labs = build_index(DS)
assert classes == CLASS_NAMES
ids = [os.path.basename(r) for r in recs]
folds = reproduce_folds(len(recs))
fold_of = {}
for k, f in enumerate(folds):
    for i in f:
        fold_of[ids[i]] = k
lab_of = {ids[i]: sorted(CLASS_NAMES[j] for j in labs[i]) for i in range(len(ids))}
s2a = snomed_to_acronym(DS)


def hea_info(base):
    d = {"age": "", "sex": "", "dx": ""}
    for l in open(base + ".hea", encoding="utf-8", errors="replace"):
        for k, key in (("#Age", "age"), ("#Sex", "sex"), ("#Dx", "dx")):
            if l.startswith(k):
                d[key] = l.split(":", 1)[1].strip()
    return d


def fingerprint(path):
    b = open(path, "rb").read()
    x = np.frombuffer(b[24:], dtype="<i2").reshape(5000, 12)    # Level-4 .mat: 24-byte header, column-major (12,5000)
    leads = [hashlib.md5(x[:, i].tobytes()).hexdigest() if x[:, i].min() != x[:, i].max() else None for i in range(12)]
    return hashlib.md5(b).hexdigest(), leads


mats = sorted(glob.glob(os.path.join(DS, "WFDBRecords", "*", "*", "*.mat")))
assert len(mats) == 45152
with ThreadPoolExecutor(6) as ex:
    fps = list(ex.map(fingerprint, mats))
rid = [os.path.basename(m)[:-4] for m in mats]
base_of = {r: m[:-4] for r, m in zip(rid, mats)}

by_file, by_lead = defaultdict(list), defaultdict(set)
for r, (fh, lh) in zip(rid, fps):
    by_file[fh].append(r)
    for h in lh:
        if h:
            by_lead[h].add(r)
file_groups = [sorted(v) for v in by_file.values() if len(v) > 1]
exact_ids = {r for g in file_groups for r in g}
# partial duplicates: share >=1 identical non-flat lead but are NOT byte-identical files
partial = defaultdict(set)
for h, s in by_lead.items():
    if 1 < len(s) <= 20:       # a non-flat 5000-sample lead shared by >20 records would be an artefact; none expected
        s = sorted(s)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                partial[(s[i], s[j])].add(h)
file_of = {r: fh for r, (fh, _) in zip(rid, fps)}
partial = {p: hs for p, hs in partial.items() if file_of[p[0]] != file_of[p[1]]}   # byte-identical pairs are reported as "exact"
assert all(len(s) <= 20 for s in by_lead.values()), "unexpectedly widely shared lead hash"


def where(r):
    return fold_of.get(r)          # None => record not in the 44,476 experiment records


def category(a, b):
    fa, fb = where(a), where(b)
    if fa is None or fb is None:
        return "4_not_in_experiment_set"
    if fa == 9 and fb == 9:
        return "1_both_test"
    if (fa == 9) != (fb == 9):
        return "2_cross_test_boundary"
    return "3_both_outside_test"


def pair_row(a, b, kind, nshared=12):
    ia, ib = hea_info(base_of[a]), hea_info(base_of[b])
    return {"kind": kind, "id_a": a, "id_b": b, "fold_a": where(a), "fold_b": where(b), "category": category(a, b),
            "n_identical_leads": nshared,
            "labels_a": "|".join(lab_of.get(a, [])), "labels_b": "|".join(lab_of.get(b, [])),
            "labels_equal": lab_of.get(a) == lab_of.get(b) if a in lab_of and b in lab_of else "",
            "dx_codes_equal": set(ia["dx"].split(",")) == set(ib["dx"].split(",")),
            "age_a": ia["age"], "age_b": ib["age"], "sex_a": ia["sex"], "sex_b": ib["sex"],
            "demographics_equal": (ia["age"], ia["sex"]) == (ib["age"], ib["sex"]),
            "id_gap": abs(int(a[2:]) - int(b[2:]))}


rows = []
for g in file_groups:
    assert len(g) == 2, g                           # (verified below in summary: all groups are pairs)
    rows.append(pair_row(g[0], g[1], "exact_file"))
for (a, b), hs in sorted(partial.items()):
    rows.append(pair_row(a, b, "partial_identical_lead", len(hs)))

with open(os.path.join(OUT, "duplicate_pair_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def summarize(kind):
    rs = [r for r in rows if r["kind"] == kind]
    s = {"pairs": len(rs), "unique_records": len({x for r in rs for x in (r["id_a"], r["id_b"])})}
    for c in sorted({r["category"] for r in rs}):
        s[c] = sum(r["category"] == c for r in rs)
    cross = [r for r in rs if r["category"] == "2_cross_test_boundary"]
    s["test_records_in_cross_pairs"] = len({(r["id_a"] if r["fold_a"] == 9 else r["id_b"]) for r in cross})
    s["test_records_in_any_pair"] = len({x for r in rs for x in (r["id_a"], r["id_b"]) if where(x) == 9})
    s["labels_equal"] = sum(r["labels_equal"] is True for r in rs)
    s["labels_differ"] = sum(r["labels_equal"] is False for r in rs)
    s["demographics_equal"] = sum(r["demographics_equal"] for r in rs)
    s["id_gap_min_median_max"] = [int(v) for v in (min(r["id_gap"] for r in rs), np.median([r["id_gap"] for r in rs]), max(r["id_gap"] for r in rs))] if rs else None
    # per-checkpoint: test records with a duplicate in a fold that checkpoint trained on
    exp = {}
    for ck, v in VAL.items():
        train = set(range(9)) - {v}
        t = set()
        for r in cross:
            tid, oid, of = (r["id_a"], r["id_b"], r["fold_b"]) if r["fold_a"] == 9 else (r["id_b"], r["id_a"], r["fold_a"])
            if of in train:
                t.add(tid)
        exp[ck] = len(t)
    s["test_records_with_duplicate_in_checkpoint_training_folds"] = exp
    cls = defaultdict(int)
    for tid in {(r["id_a"] if r["fold_a"] == 9 else r["id_b"]) for r in cross}:
        for c in lab_of[tid]:
            cls[c] += 1
    s["classes_of_cross_pair_test_records"] = dict(sorted(cls.items(), key=lambda kv: -kv[1]))
    s["cross_pairs_other_fold_distribution"] = dict(sorted(defaultdict(int, {
        str(r["fold_b"] if r["fold_a"] == 9 else r["fold_a"]): 0 for r in cross}).items()))
    for r in cross:
        k = str(r["fold_b"] if r["fold_a"] == 9 else r["fold_a"])
        s["cross_pairs_other_fold_distribution"][k] += 1
    return s


summary = {
    "n_mat_files": len(mats), "n_records_in_experiment_set": len(ids), "test_fold_size": len(folds[9]),
    "n_distinct_file_hashes": len(by_file), "exact_group_sizes": sorted({len(g) for g in file_groups}),
    "exact": summarize("exact_file"), "partial": summarize("partial_identical_lead"),
    "partial_records_not_in_exact": len({x for p in partial for x in p} - exact_ids),
}
json.dump(summary, open(os.path.join(OUT, "duplicate_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
