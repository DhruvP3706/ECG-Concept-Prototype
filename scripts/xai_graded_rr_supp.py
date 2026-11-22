r"""Supplementary analysis for Experiment 7 (same run directory; same record-level bootstrap as xai_graded_rr_analysis.py): absolute group means, changes NET of the 'own' control (CV unchanged,
sequence replaced; paired within record), monotonicity across levels, and comparison with natural records. Usage: .venv\Scripts\python.exe -I scripts\xai_graded_rr_supp.py <run_dir>"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from xai_graded_rr_analysis import PN, bdiff, bm  # noqa: E402

run = os.path.abspath(sys.argv[1])
names = [str(c) for c in np.load(os.path.join(ROOT, "results", "step4_4", "cache", "merged.npz"))["class_names"]]
ci = {n: i for i, n in enumerate(names)}
items = list(csv.DictReader(open(os.path.join(run, "items.csv"))))
L = np.load(os.path.join(run, "logits.npy"))
pop = json.load(open(os.path.join(run, "population.json")))
info = json.load(open(os.path.join(run, "records_beats.json")))
levels = ["own"] + list(pop["levels"])
orig = {it["record_id"]: int(it["logit_row"]) for it in items if it["level"] == "orig"}
T = {(it["record_id"], it["level"], it["pattern"]): it for it in items if it["level"] != "orig"}
grp = {r: v["group"] for r, v in info.items() if v["usable"]}
recs = {g: sorted(r for r in grp if grp[r] == g) for g in ("SR", "AFIB")}


def rec_level(r, lv, c):
    v = [L[int(T[(r, lv, p)]["logit_row"])][ci[c]] for p in PN if (r, lv, p) in T and T[(r, lv, p)]["accepted"] == "True"]
    return float(np.mean(v)) if v else np.nan


res = {"note": "AFIB/SR/STTC changes NET of the own-level control = (level change) - (own-level change) within the same record (paired); absolute means are group means of record-level values"}
res["absolute_means"], res["net_of_own"], res["monotonic"] = {}, {}, {}
for g in ("SR", "AFIB"):
    res["absolute_means"][g] = {"original_untransformed_all_usable_records": {c: bm([L[orig[r]][ci[c]] for r in recs[g]]) for c in ("AFIB", "SR", "STTC")}}
    res["net_of_own"][g] = {}
    for lv in levels[1:]:
        rows = [(r, rec_level(r, lv, "AFIB"), rec_level(r, "own", "AFIB")) for r in recs[g]]
        rr = [r for r, a, b in rows if np.isfinite(a) and np.isfinite(b)]
        if len(rr) < 5:
            continue
        o = {c: np.array([L[orig[r]][ci[c]] for r in rr]) for c in ("AFIB", "SR", "SB", "ST", "STTC", "TWC", "TWO")}
        res["absolute_means"][g][lv] = {c: bm([rec_level(r, lv, c) for r in recs[g] if np.isfinite(rec_level(r, lv, c))]) for c in ("AFIB", "SR", "STTC")}
        res["net_of_own"][g][lv] = {"n_records_with_both_level_and_own": len(rr)}
        for c in o:
            lvv = np.array([rec_level(r, lv, c) for r in rr])
            own = np.array([rec_level(r, "own", c) for r in rr])
            res["net_of_own"][g][lv][c] = {"level_change": bm(lvv - o[c]), "own_change": bm(own - o[c]), "net_level_minus_own": bm(lvv - own)}
    # monotonicity across the levels each record received (accepted), in the direction of increasing target CV
    lvs = [lv for lv in levels[1:] if lv in pop["levels"]]
    mono = {"AFIB_logit": [], "SR_logit": []}
    for r in recs[g]:
        pts = [(pop["levels"][lv], rec_level(r, lv, "AFIB"), rec_level(r, lv, "SR")) for lv in lvs if np.isfinite(rec_level(r, lv, "AFIB"))]
        if len(pts) >= 3:
            pts.sort()
            mono["AFIB_logit"].append(float(all(b[1] >= a[1] for a, b in zip(pts, pts[1:]))))
            mono["SR_logit"].append(float(all(b[2] <= a[2] for a, b in zip(pts, pts[1:]))))
    res["monotonic"][g] = {"n_records_with_>=3_levels": len(mono["AFIB_logit"]), "share_AFIB_logit_nondecreasing_with_target_CV": bm(mono["AFIB_logit"]), "share_SR_logit_nonincreasing_with_target_CV": bm(mono["SR_logit"])}
# natural AFIB records vs SR records transformed to AFIB-like CV
nat = {c: bm([L[orig[r]][ci[c]] for r in recs["AFIB"] if 0.17 <= info[r]["cv"] <= 0.23]) for c in ("AFIB", "SR")}
res["natural_AFIB_records_with_cv_0.17_to_0.23"] = {"n": sum(0.17 <= info[r]["cv"] <= 0.23 for r in recs["AFIB"]), **nat}
res["SR_origin_transformed_to_L3_for_comparison"] = {c: bm([rec_level(r, "L3_high(AFIB median)", c) for r in recs["SR"] if np.isfinite(rec_level(r, "L3_high(AFIB median)", c))]) for c in ("AFIB", "SR")}
res["mean_rr_ms_of_analysed_records"] = {"SR": float(np.mean([info[r]["mean_rr"] * 2 for r in recs["SR"]])), "AFIB_all": float(np.mean([info[r]["mean_rr"] * 2 for r in recs["AFIB"]])),
                                         "AFIB_with_own_level_acceptance>=0.5": float(np.mean([info[r]["mean_rr"] * 2 for r in recs["AFIB"] if np.isfinite(rec_level(r, "own", "AFIB"))]))}
json.dump(res, open(os.path.join(run, "supplementary_results.json"), "x"), indent=1, default=float)
f = lambda v: "%+.2f[%+.2f,%+.2f]" % tuple(v[:3])
print("NET OF OWN CONTROL (level change | own change | net)")
for g in ("SR", "AFIB"):
    for lv, d in res["net_of_own"][g].items():
        print(g, lv[:10].ljust(10), "n", d["n_records_with_both_level_and_own"])
        for c in ("AFIB", "SR", "STTC", "TWC", "TWO", "SB", "ST"):
            print("    %-5s change %s | own %s | net %s" % (c, f(d[c]["level_change"]), f(d[c]["own_change"]), f(d[c]["net_level_minus_own"])))
print("\nABSOLUTE MEANS"); print(json.dumps({g: {lv: {c: [round(x, 2) for x in v[:3]] for c, v in d.items()} for lv, d in res['absolute_means'][g].items() if lv != 'original_untransformed_all_usable_records'} for g in res['absolute_means']}, indent=0)[:2500])
print("\noriginals", {g: {c: [round(x, 2) for x in v[:3]] for c, v in res["absolute_means"][g]["original_untransformed_all_usable_records"].items()} for g in res["absolute_means"]})
print("MONOTONIC", res["monotonic"]); print("natural AFIB 0.17-0.23", res["natural_AFIB_records_with_cv_0.17_to_0.23"]); print("SR@L3", res["SR_origin_transformed_to_L3_for_comparison"]); print(res["mean_rr_ms_of_analysed_records"])
