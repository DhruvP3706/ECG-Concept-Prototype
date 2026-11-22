"""Step 4.4 coverage/failure summaries from the merged cache (train + validation only). Never overwrites."""
import csv, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "step4_4")
for f in ("coverage_summary.csv", "coverage_by_label.csv", "failure_summary.json"):
    assert not os.path.exists(os.path.join(OUT, f))
z = np.load(os.path.join(OUT, "cache", "merged.npz"))
role, lab, cls = z["role"], z["labels"], [str(c) for c in z["class_names"]]
rs, irs = z["rate_state"], z["irr_state"]
name = {2: "ACCEPTED", 1: "UNCERTAIN", 0: "UNAVAILABLE"}
rows = []
for r in ("train", "validation"):
    m = role == r
    for grp, st in (("rate", rs), ("irregularity", irs)):
        for k in (2, 1, 0):
            n = int(((st == k) & m).sum())
            rows.append({"role": r, "concept_group": grp, "state": name[k], "n": n, "fraction": n / int(m.sum())})
with open(os.path.join(OUT, "coverage_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lrows = []
for r in ("train", "validation"):
    m = role == r
    allacc = float((rs[m] == 2).mean())
    for j, c in enumerate(cls):
        s = m & (lab[:, j] == 1)
        if s.sum() == 0:
            continue
        lrows.append({"role": r, "label": c, "n": int(s.sum()), "frac_rate_accepted": float((rs[s] == 2).mean()), "frac_unavailable": float((rs[s] == 0).mean()),
                      "frac_uncertain": float((rs[s] == 1).mean()), "overall_frac_accepted": allacc, "diff_vs_overall": float((rs[s] == 2).mean() - allacc)})
with open(os.path.join(OUT, "coverage_by_label.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(lrows[0])); w.writeheader(); w.writerows(lrows)
names = [str(x) for x in z["reason_names"]]
fs = {"reason_counts": {r: {n: int(z["reason_flags"][role == r][:, i].sum()) for i, n in enumerate(names)} for r in ("train", "validation")},
      "read_failures": int((~z["read_ok"]).sum()), "embedding_failures": int((~z["emb_ok"]).sum()),
      "timing_between_matched_detector_beats": {r: {"records": int((role == r).sum()), "records_with_any_nonzero_dt": int(((z["n_matched_nonzero_dt"] > 0) & (role == r)).sum()),
                                                   "max_abs_dt_ms": float(np.nanmax(z["match_max_abs_dt_ms"][role == r])), "p99_of_record_max_dt_ms": float(np.nanpercentile(z["match_max_abs_dt_ms"][role == r], 99))} for r in ("train", "validation")},
      "within_train_duplicate_records": int(z["within_train_dup"].sum()), "extractor_runtime_median_s": float(np.median(z["runtime_s"])),
      "uncertain_or_unavailable_by_role": {r: int(((rs != 2) & (role == r)).sum()) for r in ("train", "validation")}}
json.dump(fs, open(os.path.join(OUT, "failure_summary.json"), "w"), indent=1)
worst = sorted([x for x in lrows if x["role"] == "train" and x["n"] >= 100], key=lambda x: x["diff_vs_overall"])[:6]
print("lowest-coverage labels (train, n>=100):", [(x["label"], x["n"], round(x["frac_rate_accepted"], 3)) for x in worst])
print(json.dumps(fs["timing_between_matched_detector_beats"]), fs["read_failures"], fs["embedding_failures"])
