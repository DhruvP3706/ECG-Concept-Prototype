"""Light-weight worker functions for Experiment 7 (numpy/scipy only: no torch, so spawned worker processes stay small)."""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import rr_concepts as rc  # noqa: E402
from ecg_concept import rr_warp as rw  # noqa: E402
from ecg_concept.preprocessing import preprocess, read_raw_mat  # noqa: E402

DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
SEED = 20261120
PATTERNS = (("iid", 0), ("iid", 1), ("smooth", 0), ("perm_orig", 0))
PNAMES = ["iid_a", "iid_b", "smooth", "perm_orig"]
MARGIN = 0.01
RIGID = {"rigid_post": 150, "rigid_pre": 50}
ACC = {"post": 0.80, "pre": 0.80, "junction": 5.0, "p999": 1.5, "min_rr": 100}


def _extract(task):
    rid, path = task
    raw = read_raw_mat(path)
    res = rc.extract_record(raw, rc.CONFIG, return_beats=True)
    cons = np.round(np.asarray(res["beats"].get("consensus", []), float) * rc.CONFIG["fs"]).astype(int)
    return rid, cons.tolist(), res["rate_state"], res["reason_codes"]


def _work(task):
    rid, group, path, beats, cv0, ridx, levels, keep_raw = task
    raw = read_raw_mat(path)
    b = np.array(beats, int)
    out = [{"record_id": rid, "group": group, "level": "orig", "level_cv": float("nan"), "pattern": "orig", "accepted": True, "reason": "", "cv0": cv0, "x": preprocess(raw), "new_beats": b.tolist()}]
    jobs = [("own", cv0)]
    for li, (name, val) in enumerate(levels.items()):
        if (group == "SR" and val >= cv0 + MARGIN) or (group == "AFIB" and val <= cv0 - MARGIN):
            jobs.append((name, val))
    lvl_index = {n: i for i, n in enumerate(["own"] + list(levels))}
    for lname, target in jobs:
        for pi, ((kind, sub), pname) in enumerate(zip(PATTERNS, PNAMES)):
            rng = np.random.RandomState(SEED + 1000003 * ridx + 1009 * lvl_index[lname] + 17 * pi + 5 * sub)
            new, tries, why = rw.target_cv_beats(b, target, kind, rng)
            item = {"record_id": rid, "group": group, "level": lname, "level_cv": float(target), "pattern": pname, "cv0": cv0, "tries": tries}
            if new is None:
                item.update(accepted=False, reason=why, new_beats=None)
                out.append(item)
                continue
            y = rw.to_int16(rw.warp(raw, b, new, **RIGID))
            qm = rw.quality(raw, y, b, new, pre=50, **RIGID)
            ok = (qm["qrs_windows_identical"] and not (qm["post_qrs_corr_median"] < ACC["post"]) and not (qm["pre_qrs_corr_median"] < ACC["pre"]) and qm["junction_ratio_max"] <= ACC["junction"]
                  and qm["p999_abs_diff_ratio"] <= ACC["p999"])
            reason = "" if ok else ";".join(k for k, bad in (("qrs_changed", not qm["qrs_windows_identical"]), ("post_corr", qm["post_qrs_corr_median"] < ACC["post"]), ("pre_corr", qm["pre_qrs_corr_median"] < ACC["pre"]),
                                                              ("junction", qm["junction_ratio_max"] > ACC["junction"]), ("p999", qm["p999_abs_diff_ratio"] > ACC["p999"])) if bad)
            item.update(qm)
            item.update(accepted=bool(ok), reason=reason, achieved_cv=rw.rr_cv(new), disp_rms=rw.displacement_rms(b, new), new_beats=new.tolist(), n_uniform_fallback=qm["n_uniform_fallback_segments"], n_segments=qm["n_segments"])
            if ok:
                item["x"] = preprocess(y)
                if keep_raw:
                    item["raw_warped"] = y
            out.append(item)
    if keep_raw:
        out[0]["raw_orig"] = raw
    return out


