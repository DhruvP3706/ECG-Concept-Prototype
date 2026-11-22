"""PTB-XL RR-concept extraction with the FROZEN Step 4.3 extractor (src/ecg_concept/rr_concepts.py, parameters unchanged) on the 18,323 eligible
fold 1-9 records (same ids and order as the frozen-embedding cache). Signals are the DIGITAL int16 samples in model lead order (I, II, III, aVR, aVL, aVF, V1-V6;
the extractor uses leads I, II, V1-V6 by position), read by the validated WFDB-free reader. Fold 10 is never read.

Predeclared compatibility criteria (declared in config.json before extraction; evaluated on TRAIN-role records only; any failure stops the experiment):
  V1 read/extract: every record reads and runs without exception.
  V2 coverage: ACCEPTED fraction >= 0.90 for the rate group AND for the irregularity group (PhysioNet reference: 0.966).
  V3 range: >= 99% of ACCEPTED rate values lie in [30, 250] bpm.
  V4 detector agreement: fraction of records with an UNCERTAIN state (any reason) <= 0.10 (PhysioNet reference: ~0.014).
  V5 directional plausibility against PTB-XL rhythm statements present in scp_codes (metadata, NOT a model input, needs >= 30 records each):
      median accepted rate: SBRAD < SR < STACH ; median accepted rr_cv: AFIB > 2 x SR.
These checks show only that the automatic concepts behave plausibly; they are not clinical validation.
Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_concepts_extract.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import csv
import datetime
import glob
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import beat_detection as bd  # noqa: E402
from ecg_concept import ptbxl as px  # noqa: E402
from ecg_concept import rr_concepts as rc  # noqa: E402

BASE = os.path.join(ROOT, "results", "ptbxl_baseline", "run_20261120_002007")
DATA = os.path.join(ROOT, "data", "raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
FROZEN_PARAMS_SHA = "4f5e41ea05c53bd39bf0267039047a30a668652c2229b83e9ba0b11dfb00403c"
COLS = ["ecg_id", "read_ok", "structural_ok", "n_usable_leads", "n_flat_leads", "n_clip_leads", "n_nonfinite_leads", "n_beats_A", "n_beats_B", "n_beats_cons", "n_rr_cons",
        "n_matched", "n_unmatched_A", "n_unmatched_B", "median_support_A", "median_support_B", "match_median_abs_dt_ms", "rate_mean_bpm_cons", "rate_median_bpm_cons",
        "rr_cv_cons", "rr_nrmssd_cons", "rr_mad_succ_cons", "rate_state", "irr_state", "reason_codes", "accepted_rate_mean_bpm", "accepted_rr_cv", "accepted_rr_nrmssd",
        "accepted_rr_mad_succ", "runtime_s"]
_META = None


def _init():
    global _META
    _META = {r.ecg_id: r for r in px.read_metadata(os.path.join(DATA, "ptbxl_database.csv"))}


def _work(ids):
    out = []
    for i in ids:
        try:
            sig = px.load_signal_native(_META[i], DATA).signal                  # raises PermissionError for fold 10
            res = rc.extract_record(sig, rc.CONFIG)
            res["read_ok"] = True
        except PermissionError:
            raise
        except Exception as e:
            res = rc.extract_record(None, rc.CONFIG)
            res["read_ok"], res["reason_codes"] = False, "READ_ERROR:" + type(e).__name__
        res["ecg_id"] = i
        out.append({c: res.get(c, float("nan")) for c in COLS})
    return out


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert hashlib.sha256(json.dumps(rc.CONFIG, sort_keys=True).encode()).hexdigest() == FROZEN_PARAMS_SHA, "extractor parameters differ from the frozen Step 4.3 set"
    ids = np.concatenate([np.load(f)["ids"] for f in sorted(glob.glob(os.path.join(BASE, "embeddings", "chunk_*.npz")))]).astype(int)
    bench = os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_train_val.csv")
    man = {int(r["ecg_id"]): r for r in csv.DictReader(open(bench, encoding="utf-8"))}
    assert len(ids) == 18323 and all(int(man[i]["strat_fold"]) <= 9 and man[i]["excluded"] == "0" for i in ids)
    run = os.path.join(ROOT, "results", "ptbxl_concepts", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run)
    crit = {"V1": "all records read and extract without exception", "V2": "ACCEPTED fraction >= 0.90 (rate and irregularity), train", "V3": ">= 99% of accepted rate in [30,250] bpm, train",
            "V4": "UNCERTAIN fraction <= 0.10, train", "V5": "median rate SBRAD < SR < STACH; median rr_cv AFIB > 2*SR (each >= 30 records, train, scp_codes presence)"}
    cfg = {"stage": "ptbxl_rr_concept_extraction", "extractor": "src/ecg_concept/rr_concepts.py (frozen Step 4.3 parameters)", "extractor_params_sha256": FROZEN_PARAMS_SHA,
           "extractor_code_sha256": {f: sha_file(os.path.join(ROOT, "src", "ecg_concept", f)) for f in ("rr_concepts.py", "beat_detection.py")},
           "signal": "digital int16 counts, model lead order, WFDB-free reader; no filtering/scaling before extraction", "criteria_declared_before_extraction": crit,
           "baseline_run_for_order": os.path.relpath(BASE, ROOT), "manifest_sha256": sha_file(bench), "n_records": int(len(ids)), "fold10_waveforms_read": 0}
    json.dump(cfg, open(os.path.join(run, "config.json"), "x"), indent=1)
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    t0 = time.time()
    tasks = [ids[i:i + 100].tolist() for i in range(0, len(ids), 100)]
    rows = []
    with ProcessPoolExecutor(10, initializer=_init) as ex:
        for k, part in enumerate(ex.map(_work, tasks)):
            rows += part
            if k % 20 == 0:
                say(f"{len(rows)}/{len(ids)} ({time.time() - t0:.0f}s)")
    assert [r["ecg_id"] for r in rows] == ids.tolist()
    with open(os.path.join(run, "concepts.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([repr(r[c]) if isinstance(r[c], float) else r[c] for c in COLS])
    say(f"extraction finished in {time.time() - t0:.0f}s")

    # ---- predeclared checks on TRAIN records only
    role = np.array([man[i]["role"] for i in ids])
    tr = role == "train"
    col = lambda c: np.array([r[c] for r in rows], dtype=float)
    rate_state = np.array([r["rate_state"] for r in rows])
    irr_state = np.array([r["irr_state"] for r in rows])
    rate = col("accepted_rate_mean_bpm")
    cv = col("accepted_rr_cv")
    meta = {r.ecg_id: r for r in px.read_metadata(os.path.join(DATA, "ptbxl_database.csv"))}
    has = lambda code: np.array([code in meta[i].scp_codes for i in ids])
    chk, res = {}, {}
    res["read_ok_all"] = bool(all(r["read_ok"] for r in rows))
    res["n_read_failures"] = int(sum(not r["read_ok"] for r in rows))
    chk["V1"] = res["read_ok_all"]
    res["train_rate_accepted_frac"] = float((rate_state[tr] == "ACCEPTED").mean())
    res["train_irr_accepted_frac"] = float((irr_state[tr] == "ACCEPTED").mean())
    chk["V2"] = res["train_rate_accepted_frac"] >= 0.90 and res["train_irr_accepted_frac"] >= 0.90
    ok = np.isfinite(rate) & tr
    res["train_accepted_rate_in_30_250_frac"] = float(((rate[ok] >= 30) & (rate[ok] <= 250)).mean())
    chk["V3"] = res["train_accepted_rate_in_30_250_frac"] >= 0.99
    unc = (rate_state == "UNCERTAIN") | (irr_state == "UNCERTAIN")
    res["train_uncertain_frac"] = float(unc[tr].mean())
    chk["V4"] = res["train_uncertain_frac"] <= 0.10
    med = {}
    for name in ("SBRAD", "SR", "STACH", "AFIB"):
        s = has(name) & tr & np.isfinite(rate)
        med[name] = {"n": int(s.sum()), "median_rate": float(np.median(rate[s])) if s.sum() else float("nan"), "median_rr_cv": float(np.nanmedian(cv[s])) if s.sum() else float("nan")}
    res["rhythm_statement_medians_train"] = med
    chk["V5"] = (all(med[k]["n"] >= 30 for k in med) and med["SBRAD"]["median_rate"] < med["SR"]["median_rate"] < med["STACH"]["median_rate"]
                 and med["AFIB"]["median_rr_cv"] > 2 * med["SR"]["median_rr_cv"])
    # descriptive coverage by role / reasons
    res["coverage"] = {r_: {g: {s: float((st[role == r_] == s).mean()) for s in ("ACCEPTED", "UNCERTAIN", "UNAVAILABLE")} for g, st in (("rate", rate_state), ("irregularity", irr_state))}
                       for r_ in ("train", "val")}
    reasons = {}
    for r in rows:
        for c in str(r["reason_codes"]).split(";"):
            if c:
                reasons[c] = reasons.get(c, 0) + 1
    res["reason_counts_all_eligible"] = reasons
    pace = np.array([str(meta[i].scp_raw) and False for i in ids])
    res["accepted_rate_train_percentiles_1_50_99"] = np.percentile(rate[ok], [1, 50, 99]).round(1).tolist()
    res["accepted_rr_cv_train_percentiles_1_50_99"] = np.nanpercentile(cv[tr], [1, 50, 99]).round(4).tolist()
    out = {"criteria": chk, "all_pass": bool(all(chk.values())), "results": res, "concepts_csv_sha256": sha_file(os.path.join(run, "concepts.csv")), "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(run, "validation_checks.json"), "x"), indent=1)
    say(f"criteria: {chk}; ALL PASS: {out['all_pass']}")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
