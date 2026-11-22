"""Step 4.1: header/metadata census of TRAIN-role records only (no waveform samples, no .mat access, no validation/test rows).

Reads the Step 3.7 role manifest, keeps role == 'train' (asserts 35,568), reads only those .hea files, and writes
results/step4_1/inspection_config.json (refuses to overwrite). Validation/test/excluded IDs are never opened.
"""
import csv
import glob
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "step4_1")
os.makedirs(OUT, exist_ok=True)
DEST = os.path.join(OUT, "inspection_config.json")
assert not os.path.exists(DEST), "refusing to overwrite " + DEST
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
ROLE = os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv")
FOLD = os.path.join(ROOT, "results", "step3_7", "original_fold_manifest.csv")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

roles = list(csv.DictReader(open(ROLE)))
train = [r for r in roles if r["role"] == "train"]
assert len(roles) == 44476 and len(train) == 35568, (len(roles), len(train))
train_ids = {r["record_id"] for r in train}
path_of = {}
for r in csv.DictReader(open(FOLD)):                     # join rule: record_id == manifest dataset_rel_path stem
    if r["record_id"] in train_ids:
        path_of[r["record_id"]] = os.path.join(DS, r["dataset_rel_path"][:-4] + ".hea")
assert set(path_of) == train_ids

parse_fail, seen_ids = [], Counter()
base_dt, no_base_dt = [], 0
comment_keys, sig_variants = Counter(), Counter()
fs_c, ns_c, nsig_c, lead_orders = Counter(), Counter(), Counter(), Counter()
age_raw, sex, rx, hx, sx = [], Counter(), Counter(), Counter(), Counter()
ndx = Counter()
hdr = {}
for rid, p in path_of.items():
    try:
        L = open(p, encoding="utf-8", errors="replace").read().replace("\r", "").split("\n")
        t = L[0].split()
        assert len(t) in (4, 6) and t[0] == rid, "record line"      # WFDB allows an optional base time + date after n_samples
        nsig, fs, ns = int(t[1]), int(t[2]), int(t[3])
        if len(t) == 6:
            mt = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})", t[4]) or re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})", t[5])
            md = re.fullmatch(r"(\d{2})-([A-Za-z]{3})-(\d{4})", t[5]) or re.fullmatch(r"(\d{2})-([A-Za-z]{3})-(\d{4})", t[4])
            assert mt and md, "base time/date format"
            base_dt.append((rid, int(md.group(3)), md.group(2)))
        else:
            no_base_dt += 1
        sig = [l.split() for l in L[1:] if l and not l.startswith("#")]
        assert len(sig) == nsig == 12 and all(len(s) == 9 for s in sig), "signal lines"
        nsig_c[nsig] += 1; fs_c[fs] += 1; ns_c[ns] += 1
        lead_orders[tuple(s[8] for s in sig)] += 1
        for s in sig:
            assert s[0] == rid + ".mat"
            sig_variants[(s[1], s[2], s[3], s[4], s[7])] += 1      # format, gain/units, ADC res, ADC zero, block size (not initial value / checksum)
        m = {}
        for l in L:
            if l.startswith("#") and ":" in l:
                k, v = l[1:].split(":", 1)
                comment_keys[k.strip()] += 1
                m[k.strip()] = v.strip()
        hdr[rid] = m
        seen_ids[rid] += 1
        age_raw.append(m.get("Age")); sex[m.get("Sex")] += 1
        rx[m.get("Rx")] += 1; hx[m.get("Hx")] += 1; sx[m.get("Sx")] += 1
        ndx[len([c for c in m.get("Dx", "").split(",") if c.strip()])] += 1
    except Exception as e:
        parse_fail.append([rid, repr(e)[:80]])

num, kinds = [], Counter()
for a in age_raw:
    if a is None:
        kinds["key_absent"] += 1
    elif a.strip().lower() == "nan":
        kinds["literal_NaN"] += 1
    else:
        try:
            v = float(a); num.append(v); kinds["numeric"] += 1
        except ValueError:
            kinds["unparseable"] += 1
num = np.array(num)
age = {"n_total": len(age_raw), "kinds": dict(kinds), "numeric_n": int(num.size), "zero": int((num == 0).sum()), "negative": int((num < 0).sum()),
       "min": float(num.min()), "max": float(num.max()), "mean": float(num.mean()), "median": float(np.median(num)),
       "n_at_max": int((num == num.max()).sum()), "n_ge_90": int((num >= 90).sum()),
       "integer_valued": bool(np.all(num == np.round(num)))}

# label-dictionary-related diagnostic support in TRAIN (labels are diagnoses, NOT measurements)
lab = Counter(c for r in train for c in r["labels"].split("|") if c)
names = {r["Acronym Name"]: (r["Full Name"], r["Snomed_CT"]) for r in csv.DictReader(open(os.path.join(DS, "ConditionNames_SNOMED-CT.csv"), encoding="utf-8-sig"))}

# conflicting values inside training duplicate pairs (identical signals)
by_id = {r["record_id"]: r for r in train}
conf = {"pairs": 0, "age_differs": 0, "sex_differs": 0, "labels_differ": 0}
for r in train:
    p = r["dup_partner_id"]
    if r["within_train_dup_group"] == "True" and r["record_id"] < p and p in hdr:
        conf["pairs"] += 1
        conf["age_differs"] += hdr[r["record_id"]].get("Age") != hdr[p].get("Age")
        conf["sex_differs"] += hdr[r["record_id"]].get("Sex") != hdr[p].get("Sex")
        conf["labels_differ"] += by_id[r["record_id"]]["labels"] != by_id[p]["labels"]

summary = {
    "n_train_records": len(train), "n_headers_parsed": len(hdr), "parse_failures": parse_fail, "duplicate_ids": sum(v > 1 for v in seen_ids.values()),
    "join_coverage": f"{len(path_of)}/{len(train_ids)} train IDs resolved to a .hea via original_fold_manifest dataset_rel_path",
    "record_line": {"n_signals": dict(nsig_c), "fs": dict(fs_c), "n_samples": dict(ns_c)},
    "base_time_date": {"present": len(base_dt), "absent": no_base_dt, "years": dict(sorted(Counter(y for _, y, _ in base_dt).items())),
                       "id_range_with_timestamp": [min(r for r, _, _ in base_dt), max(r for r, _, _ in base_dt)] if base_dt else None,
                       "note": "WFDB base time/date on the record line; meaning (acquisition vs export time) undocumented"},
    "lead_order": {"|".join(k): v for k, v in lead_orders.items()},
    "signal_line_variants_(format,gain,adc_res,adc_zero,block)": {"|".join(k): v for k, v in sig_variants.items()},
    "comment_keys": dict(comment_keys), "age": age, "sex": dict(sex), "Rx": dict(rx), "Hx": dict(hx), "Sx": dict(sx),
    "n_dx_codes_per_record": dict(sorted(ndx.items())),
    "train_diagnostic_label_support": dict(lab.most_common()),
    "train_dup_pairs_conflicts": conf,
    "label_dictionary_entries": {k: list(v) for k, v in names.items()},
}
files = {os.path.relpath(p, ROOT): sha(p) for p in [ROLE, FOLD, os.path.join(DS, "ConditionNames_SNOMED-CT.csv"), os.path.join(DS, "LICENSE.txt"),
         os.path.join(ROOT, "results", "step3_7", "analysis_config.json"), os.path.join(ROOT, "results", "step3_7", "evaluation_protocol.md")]}
cfg = {"step": "4.1 metadata inventory", "restriction": "record-level inspection limited to role==train (35,568); no .mat opened; validation/test/excluded IDs never read",
       "inspected_sources": {"role_manifest": "results/step3_7/prospective_role_manifest.csv", "train_headers": "WFDBRecords/**/<id>.hea for train IDs only",
                             "dataset_docs": ["ConditionNames_SNOMED-CT.csv", "LICENSE.txt", "RECORDS (name list only)"],
                             "dataset_file_types_in_tree": {".hea": 45152, ".mat": 45152, "RECORDS": 453, ".txt": 2, ".csv": 1}},
       "join_rules": {"headers": "record_id equals file stem; path from original_fold_manifest.csv (Step 3.7)", "labels": "role manifest `labels` column (acronyms)"},
       "input_sha256": files, "software": {"python": platform.python_version(), "numpy": np.__version__},
       "command": ".venv\\Scripts\\python.exe -I scripts\\step4_1_metadata_inventory.py", "summary": summary}
json.dump(cfg, open(DEST, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k not in ("label_dictionary_entries", "train_diagnostic_label_support")}, indent=1))
