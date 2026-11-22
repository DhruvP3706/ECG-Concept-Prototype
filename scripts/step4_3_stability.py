"""Step 4.3 DEV-S-only parameter-perturbation stability diagnostic (engineering evidence; no reference annotations).

Re-extracts the 120 DEV-S metric records under perturbed copies of the frozen parameters (the frozen config is not changed)
and reports how often the primary consensus rate / RR statistics and states move. Stability is not accuracy.
Writes results/step4_3/work/stability_dev.json (refuses to overwrite).
"""
import copy
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import rr_concepts as rc  # noqa: E402
from ecg_concept.preprocessing import read_raw_mat  # noqa: E402

OUT = os.path.join(ROOT, "results", "step4_3")
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
dest = os.path.join(OUT, "work", "stability_dev.json")
assert not os.path.exists(dest)
rows = [r for r in csv.DictReader(open(os.path.join(OUT, "pilot_manifest.csv"), encoding="utf-8")) if r["set"] == "DEV-S" and r["member_role"] == "metrics"]
sigs = {r["record_id"]: read_raw_mat(os.path.join(DS, r["rel_path"])) for r in rows}          # DEV-S only


def variant(**kw):
    c = copy.deepcopy(rc.CONFIG)
    for k, v in kw.items():
        if "." in k:
            a, b = k.split(".")
            c[a][b] = v
        else:
            c[k] = v
    return c


VARIANTS = {"fuse_k_min=2": variant(fuse_k_min=2), "fuse_k_min=4": variant(fuse_k_min=4), "fuse_tol_s=0.03": variant(fuse_tol_s=0.03),
            "fuse_tol_s=0.08": variant(fuse_tol_s=0.08), "edge_s=0.05": variant(edge_s=0.05), "edge_s=0.20": variant(edge_s=0.20),
            "A.thr_frac=0.20": variant(**{"A.thr_frac": 0.20}), "A.thr_frac=0.30": variant(**{"A.thr_frac": 0.30}),
            "A.mwi_s=0.12": variant(**{"A.mwi_s": 0.12}), "A.mwi_s=0.18": variant(**{"A.mwi_s": 0.18}),
            "B.beta=0.06": variant(**{"B.beta": 0.06}), "B.beta=0.10": variant(**{"B.beta": 0.10}),
            "loc_halfwin_s=0.06": variant(loc_halfwin_s=0.06), "loc_halfwin_s=0.10": variant(loc_halfwin_s=0.10),
            "match_tol_s=0.03": variant(match_tol_s=0.03), "match_tol_s=0.08": variant(match_tol_s=0.08)}
base = {i: rc.extract_record(s) for i, s in sigs.items()}
out = {"n_records": len(sigs), "variants": {}}
for name, cfg in VARIANTS.items():
    d_rate, d_nr, ch_state, ch_acc = [], [], 0, 0
    for i, s in sigs.items():
        v, b = rc.extract_record(s, cfg), base[i]
        ch_state += v["rate_state"] != b["rate_state"]
        ch_acc += (b["rate_state"] == "ACCEPTED") != (v["rate_state"] == "ACCEPTED")
        if np.isfinite(b["rate_mean_bpm_cons"]) and np.isfinite(v["rate_mean_bpm_cons"]):
            d_rate.append(abs(v["rate_mean_bpm_cons"] - b["rate_mean_bpm_cons"]))
        if np.isfinite(b["rr_nrmssd_cons"]) and np.isfinite(v["rr_nrmssd_cons"]):
            d_nr.append(abs(v["rr_nrmssd_cons"] - b["rr_nrmssd_cons"]))
    d_rate, d_nr = np.array(d_rate), np.array(d_nr)
    out["variants"][name] = {"records_state_changed": int(ch_state), "records_accepted_status_changed": int(ch_acc),
                             "rate_changed_gt_1bpm": int((d_rate > 1).sum()), "rate_changed_gt_5bpm": int((d_rate > 5).sum()), "rate_max_abs_change_bpm": float(d_rate.max()),
                             "nrmssd_changed_gt_0.02": int((d_nr > 0.02).sum()), "n_compared": int(d_rate.size)}
json.dump(out, open(dest, "w"), indent=1)
for k, v in out["variants"].items():
    print(f"{k:20s}", v)
