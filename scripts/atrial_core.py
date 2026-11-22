"""Torch-free worker for Experiment 8 (atrial-activity interventions). One call = one record: builds every condition, runs the quality-control rules and returns accepted signals (preprocessed, float32)."""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import atrial_ops as ao  # noqa: E402
from ecg_concept import rr_concepts as rc  # noqa: E402
from ecg_concept import rr_warp as rw  # noqa: E402
from ecg_concept.preprocessing import preprocess, read_raw_mat  # noqa: E402

DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
SEED = 20261120
MIN_ELIG = 4                                                                   # a record needs >= 4 eligible beats for an atrial intervention
RMS_MIN = 2.0                                                                  # ADC counts: the intervention must actually change the region
DIFF_RATIO_MAX = 1.0                                                           # max |first difference| in/near the regions <= p99.9 of the record's own |first difference|
SHAM_RATIO = (0.8, 1.25)                                                       # sham modification RMS / real modification RMS
TIMING_TOL = 2                                                                 # samples: re-detected beats must equal the intended beats
ATRIAL = [("flat", 160, 0.0), ("g05", 160, 0.5), ("g15", 160, 1.5), ("g20", 160, 2.0), ("flat_short", 120, 0.0), ("flat_long", 200, 0.0)]
RIGID = {"rigid_post": 150, "rigid_pre": 170}                                  # factorial RR warp keeps the whole atrial region rigid
WARP_ACC = {"post": 0.80, "pre": 0.80, "junction": 5.0, "p999": 1.5}
FACTORIAL_CV = 0.20                                                            # AFIB median RR-CV of the validation role (Experiment 7 level L3)


def _timing(y, beats):
    res = rc.extract_record(y, rc.CONFIG, return_beats=True)
    got = np.round(np.asarray(res["beats"].get("consensus", []), float) * rc.CONFIG["fs"]).astype(int)
    b = np.asarray(beats, int)
    if len(got) != len(b):
        return False, float("nan"), len(got)
    return bool(np.abs(got - b).max() <= TIMING_TOL), float(np.abs(got - b).max()), len(got)


def work(task):
    rid, group, path, beats, cv0, ridx, fw_rms, keep_raw = task
    x0 = read_raw_mat(path)
    b = np.array(beats, int)
    mean_rr = float(np.diff(b).mean())
    base = {"record_id": rid, "group": group, "cv0": cv0, "mean_rr": mean_rr, "n_beats": len(b)}
    items = [dict(base, condition="orig", accepted=True, reason="", x=preprocess(x0))]
    if keep_raw:
        items[0]["raw_orig"] = x0

    def item(name, y_int, regs, bts, ref_raw, extra=None, ref_rms=None, strict=True):
        it = dict(base, condition=name, n_eligible=len(regs), coverage=len(regs) / max(len(bts) - 1, 1))
        if extra:
            it.update(extra)
        q = ao.qc(ref_raw, y_int, bts, regs, strict_amp=strict)
        it.update({k: q[k] for k in ("rms_modification", "max_region_diff_ratio", "new_clipped_samples")})
        reasons = []
        if not q["qrs_identical"]:
            reasons.append("qrs_changed")
        if not q["outside_identical"]:
            reasons.append("outside_changed")
        if not q["shape_dtype_ok"]:
            reasons.append("shape")
        if q["max_region_diff_ratio"] > DIFF_RATIO_MAX:
            reasons.append("discontinuity")
        if not q["amplitude_ok"] or q["new_clipped_samples"] > 0:
            reasons.append("amplitude")
        if not name.startswith("rr_only") and q["rms_modification"] < RMS_MIN:
            reasons.append("no_change")
        if ref_rms is not None and ref_rms > 0 and not (SHAM_RATIO[0] <= q["rms_modification"] / ref_rms <= SHAM_RATIO[1]):
            reasons.append("magnitude_mismatch")
        ok_t, shift, n_det = _timing(y_int, bts)
        it.update(timing_max_shift=shift, timing_n_detected=n_det)
        if not ok_t:
            reasons.append("timing_changed")
        if ref_rms is not None:
            it["rms_ratio_to_real"] = q["rms_modification"] / ref_rms if ref_rms > 0 else float("nan")
        it.update(accepted=not reasons, reason=";".join(reasons))
        if not reasons:
            it["x"] = preprocess(y_int)
            if keep_raw:
                it["raw_mod"] = y_int
        return it

    def reject(name, why, regs=()):
        return dict(base, condition=name, accepted=False, reason=why, n_eligible=len(regs), coverage=len(regs) / max(len(b) - 1, 1))

    flat_rms = {}
    for name, a1, s in ATRIAL:
        regs, _ = ao.regions(b, a1)
        if len(regs) < MIN_ELIG:
            items.append(reject(name, "insufficient_eligible_beats", regs))
            continue
        y = ao.to_int16(ao.gain_region(x0, regs, s))
        it = item(name, y, regs, b, x0)
        items.append(it)
        if name == "flat":
            flat_rms["rms"], flat_rms["regs"] = it["rms_modification"], regs
    regs = flat_rms.get("regs")
    for sname, sd in (("sham_a", 1), ("sham_b", 2)):
        if regs is None or len(regs) < MIN_ELIG:
            items.append(reject(sname, "insufficient_eligible_beats", regs or []))
            continue
        rng = np.random.RandomState(SEED + 1000003 * ridx + 31 * sd)
        y = ao.to_int16(ao.random_sham(x0, regs, ao.modification(x0, regs, 0.0), rng))
        items.append(item(sname, y, regs, b, x0, ref_rms=flat_rms["rms"]))
    if group == "SR":
        for fname, sd in (("fwave_a", 11), ("fwave_b", 12), ("fwave_c", 13), ("fwave_d", 14)):
            if regs is None or len(regs) < MIN_ELIG:
                items.append(reject(fname, "insufficient_eligible_beats", regs or []))
                continue
            rng = np.random.RandomState(SEED + 1000003 * ridx + 31 * sd)
            y = ao.to_int16(ao.synthetic_fwave(x0, regs, fw_rms, rng))
            items.append(item(fname, y, regs, b, x0))
        # factorial: RR variability moved to the AFIB median (rigid atrial region), then atrial flatten / flatten + synthetic f-waves
        rng = np.random.RandomState(SEED + 1000003 * ridx + 31 * 21)
        nb, tries, why = rw.target_cv_beats(b, FACTORIAL_CV, "iid", rng)
        if nb is None:
            for nm in ("rr_only", "rr_flat", "rr_flat_fwave"):
                items.append(reject(nm, "rr_target_" + why))
        else:
            x2 = rw.to_int16(rw.warp(x0, b, nb, **RIGID))
            qm = rw.quality(x0, x2, b, nb, pre=170, **RIGID)
            wok = (qm["qrs_windows_identical"] and not (qm["post_qrs_corr_median"] < WARP_ACC["post"]) and not (qm["pre_qrs_corr_median"] < WARP_ACC["pre"]) and qm["junction_ratio_max"] <= WARP_ACC["junction"]
                   and qm["p999_abs_diff_ratio"] <= WARP_ACC["p999"])
            if not wok:
                for nm in ("rr_only", "rr_flat", "rr_flat_fwave"):
                    items.append(reject(nm, "rr_warp_quality"))
            else:
                regs2, _ = ao.regions(nb, 160)
                items.append(item("rr_only", x2, [], nb, x2, extra={"rr_cv_achieved": rw.rr_cv(nb)}))
                if len(regs2) < MIN_ELIG:
                    items.append(reject("rr_flat", "insufficient_eligible_beats", regs2))
                    items.append(reject("rr_flat_fwave", "insufficient_eligible_beats", regs2))
                else:
                    items.append(item("rr_flat", ao.to_int16(ao.gain_region(x2, regs2, 0.0)), regs2, nb, x2, extra={"rr_cv_achieved": rw.rr_cv(nb)}))
                    rng2 = np.random.RandomState(SEED + 1000003 * ridx + 31 * 22)
                    items.append(item("rr_flat_fwave", ao.to_int16(ao.synthetic_fwave(x2, regs2, fw_rms, rng2)), regs2, nb, x2, extra={"rr_cv_achieved": rw.rr_cv(nb)}))
    return items
