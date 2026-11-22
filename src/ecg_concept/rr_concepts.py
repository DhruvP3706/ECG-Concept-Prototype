"""Ventricular rate and "10-second RR statistics" from a raw 12-lead record, with masks and provenance.

IMPORTANT WORDING: the irregularity outputs are descriptive 10-second RR statistics computed on ALL detected RR
intervals. They are not conventional HRV, not NN-interval based, and not expert-validated physiological measurements.
Nothing here uses a diagnosis, age, sex, timestamp or record ID: the interface takes only the signal (and a config).

Concept definitions (RR in seconds between consecutive accepted beats, m = number of RR):
  rate_mean_bpm   = 60 / mean(RR)               (primary)
  rate_median_bpm = 60 / median(RR)             (secondary)
  rr_cv           = sample SD(RR) / mean(RR)    (ddof = 1)
  rr_nrmssd       = sqrt(mean(diff(RR)^2)) / mean(RR)
  rr_mad_succ     = median(|diff(RR)|) / median(RR)
Missing measurements are NaN (never 0) and every state/reason is explicit.

States per concept group (rate, irregularity):
  ACCEPTED    value computed and no uncertainty reason  -> eligible for later supervision
  UNCERTAIN   value computed but one or more uncertainty reasons (kept separate from accepted values)
  UNAVAILABLE no value can be computed (reason codes explain)
Abnormal rhythm/morphology never changes a state by itself; unusual RR intervals are never removed.
"""
import time

import numpy as np

from .beat_detection import DETECTORS, NONREDUNDANT_LEADS, fuse_leads, match_one_to_one

CONFIG = {
    "fs": 500,
    "nonredundant_leads": list(NONREDUNDANT_LEADS),
    "expected_leads": 12,
    "min_samples_s": 2.0,                      # shorter records are structurally rejected
    "refractory_s": 0.2,
    "loc_band": [5.0, 25.0],                   # localisation band for peak time refinement (timing only)
    "loc_halfwin_s": 0.08,
    "A": {"band": [5.0, 15.0], "mwi_s": 0.150, "init_s": 2.0, "thr_frac": 0.25, "searchback_factor": 1.66, "searchback_thr": 0.5},
    "B": {"band": [8.0, 20.0], "w1_s": 0.097, "w2_s": 0.611, "beta": 0.08},
    "fuse_tol_s": 0.05,                        # cross-lead cluster span (relative to first member)
    "fuse_k_min": 3,                           # minimum supporting non-redundant leads for a beat
    "match_tol_s": 0.05,                       # cross-detector one-to-one matching tolerance
    "edge_s": 0.10,                            # beats this close to a record boundary are excluded (possibly truncated QRS)
    "flat_mode_frac": 0.8,                     # lead flat if >= this fraction of samples share one value
    "clip_run": 4,                             # >= this many consecutive samples at the lead's min or max ...
    "clip_edge_frac": 0.02,                    # ... with both flanks at least this fraction of the lead's range away -> clipping suspect
    "int16_limits": [-32768, 32767],
    "min_usable_leads": 3,
    "low_support_median": 4,                   # median lead support below this -> LOW_LEAD_SUPPORT (uncertain)
    "low_usable_leads": 4,                     # fewer usable leads than this -> LOW_USABLE_LEADS (uncertain)
    "rate_min_beats": 2,
    "irr_min_rr": 3,
}


def rr_concepts(times_s, c=CONFIG):
    """Concept values from beat times in seconds. NaN where not computable."""
    t = np.sort(np.asarray(times_s, dtype=float))
    rr = np.diff(t)
    out = {"n_beats": int(t.size), "n_rr": int(rr.size)}
    nan = float("nan")
    out.update(rate_mean_bpm=nan, rate_median_bpm=nan, rr_cv=nan, rr_nrmssd=nan, rr_mad_succ=nan)
    if t.size >= c["rate_min_beats"]:
        out["rate_mean_bpm"] = float(60.0 / rr.mean())
        out["rate_median_bpm"] = float(60.0 / np.median(rr))
    if rr.size >= c["irr_min_rr"]:
        d = np.diff(rr)
        out["rr_cv"] = float(rr.std(ddof=1) / rr.mean())
        out["rr_nrmssd"] = float(np.sqrt(np.mean(d ** 2)) / rr.mean())
        out["rr_mad_succ"] = float(np.median(np.abs(d)) / np.median(rr))
    return out


def lead_quality(x, c=CONFIG):
    """Per-lead indicators for one lead (1-D). Flat and non-finite leads are excluded from detection; clipping is an indicator."""
    x = np.asarray(x)
    q = {"finite": bool(np.isfinite(x).all()), "flat": False, "clip": False, "mode_frac": float("nan")}
    if not q["finite"]:
        return q
    xi = x.astype(np.float64)
    vals, counts = np.unique(xi, return_counts=True)
    q["mode_frac"] = float(counts.max() / xi.size)
    q["flat"] = bool(vals.size == 1 or q["mode_frac"] >= c["flat_mode_frac"])
    lo, hi = c["int16_limits"]
    at_limit = bool(((xi <= lo) | (xi >= hi)).any())
    plateau = False
    if vals.size > 1:
        # clipping plateau: >= clip_run consecutive samples at the lead's max or min AND steep on both sides
        # (a slow baseline minimum that rounds to the same integer has flat sides and is not clipping)
        rng = xi.max() - xi.min()
        ext = (xi == xi.max()) | (xi == xi.min())
        i, n = 0, xi.size
        while i < n and not plateau:
            if not ext[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and ext[j + 1] and xi[j + 1] == xi[i]:
                j += 1
            if j - i + 1 >= c["clip_run"] and i > 0 and j < n - 1:
                steep = min(abs(xi[i - 1] - xi[i]), abs(xi[j + 1] - xi[j])) >= c["clip_edge_frac"] * rng
                plateau = bool(steep)
            i = j + 1
    q["clip"] = bool(at_limit or plateau)
    return q


def _fuse_detector(x, usable, fs, det, c):
    per_lead = {l: DETECTORS[det](x[l], fs, c) for l in usable}
    tol, refr = int(round(c["fuse_tol_s"] * fs)), int(round(c["refractory_s"] * fs))
    times, support = fuse_leads(per_lead, tol, c["fuse_k_min"], refr)
    n = x.shape[1]
    e = c["edge_s"] * fs
    keep = (times >= e) & (times <= n - 1 - e)
    return times[keep], support[keep], int((~keep).sum()), per_lead


def extract_record(signal, c=CONFIG, return_beats=False):
    """Extract concepts from one raw record array (12, N) in header units (any numeric dtype).

    Returns a flat dict of values, states, reason codes and counts (and beat times if return_beats). Deterministic."""
    t_start = time.perf_counter()
    fs = c["fs"]
    res = {"fs": fs, "reasons": [], "uncertainty_reasons": [], "beats": {}}
    nan = float("nan")
    for grp in ("rate", "irr"):
        res[grp + "_state"] = "UNAVAILABLE"
    sig = None if signal is None else np.asarray(signal)
    if sig is None or sig.ndim != 2 or sig.shape[0] != c["expected_leads"] or sig.shape[1] < c["min_samples_s"] * fs:
        res["reasons"].append("STRUCTURAL_SHAPE")
        res["structural_ok"] = False
        res["runtime_s"] = time.perf_counter() - t_start
        return _finish(res, c, return_beats)
    res["structural_ok"] = True
    res["n_samples"] = int(sig.shape[1])
    q = [lead_quality(sig[l], c) for l in range(sig.shape[0])]
    nr = list(c["nonredundant_leads"])
    res["n_nonfinite_leads"] = sum(not q[l]["finite"] for l in nr)
    res["n_flat_leads"] = sum(q[l]["flat"] for l in nr)
    res["n_clip_leads"] = sum(q[l]["clip"] for l in nr)
    usable = [l for l in nr if q[l]["finite"] and not q[l]["flat"]]
    res["n_usable_leads"] = len(usable)
    if res["n_nonfinite_leads"]:
        res["reasons"].append("NONFINITE_LEAD")
    if res["n_flat_leads"]:
        res["reasons"].append("FLAT_LEAD")
    if res["n_clip_leads"]:
        res["reasons"].append("CLIPPING_SUSPECT")
    if len(usable) < c["min_usable_leads"]:
        res["reasons"].append("INSUFFICIENT_USABLE_LEADS")
        res["runtime_s"] = time.perf_counter() - t_start
        return _finish(res, c, return_beats)
    x = sig.astype(np.float64)
    fused = {}
    for det in ("A", "B"):
        tms, sup, n_edge, per_lead = _fuse_detector(x, usable, fs, det, c)
        fused[det] = (tms, sup)
        res["beats"][det] = tms / fs
        res["n_edge_excluded_" + det] = n_edge
        res["median_support_" + det] = float(np.median(sup)) if sup.size else nan
        res["n_lead_detections_" + det] = int(sum(len(v) for v in per_lead.values()))
        res.update({k + "_" + det: v for k, v in rr_concepts(tms / fs, c).items()})
    pairs, ua, ub = match_one_to_one(fused["A"][0], fused["B"][0], c["match_tol_s"] * fs)
    cons = np.array([(fused["A"][0][i] + fused["B"][0][j]) / 2.0 for i, j in pairs]) / fs
    res["beats"]["consensus"] = np.sort(cons)
    res["n_matched"] = len(pairs)
    res["n_unmatched_A"], res["n_unmatched_B"] = len(ua), len(ub)
    res["match_median_abs_dt_ms"] = float(np.median([abs(fused["A"][0][i] - fused["B"][0][j]) for i, j in pairs]) / fs * 1000) if pairs else nan
    res.update({k + "_cons": v for k, v in rr_concepts(cons, c).items()})
    # ---- states and reason codes
    unc = []
    if res["n_unmatched_A"] or res["n_unmatched_B"]:
        unc.append("DETECTOR_DISAGREEMENT")
    meds = [res["median_support_" + d] for d in ("A", "B")]
    if any(np.isfinite(m) and m < c["low_support_median"] for m in meds):
        unc.append("LOW_LEAD_SUPPORT")
    if len(usable) < c["low_usable_leads"]:
        unc.append("LOW_USABLE_LEADS")
    res["uncertainty_reasons"] = unc
    if res["n_beats_cons"] < c["rate_min_beats"]:
        res["reasons"].append("INSUFFICIENT_BEATS")          # insufficient DETECTED information, not physiological absence
    else:
        res["rate_state"] = "UNCERTAIN" if unc else "ACCEPTED"
    if res["n_beats_cons"] >= c["rate_min_beats"] and res["n_rr_cons"] < c["irr_min_rr"]:
        res["reasons"].append("INSUFFICIENT_BEATS_FOR_IRREGULARITY")
    if res["n_rr_cons"] >= c["irr_min_rr"]:
        res["irr_state"] = "UNCERTAIN" if unc else "ACCEPTED"
    res["runtime_s"] = time.perf_counter() - t_start
    return _finish(res, c, return_beats)


def _finish(res, c, return_beats):
    """Fill accepted-only columns (NaN unless ACCEPTED) and normalise missing keys to NaN."""
    nan = float("nan")
    for suffix in ("_A", "_B", "_cons"):
        for k in ("n_beats", "n_rr"):
            res.setdefault(k + suffix, 0)
        for k in ("rate_mean_bpm", "rate_median_bpm", "rr_cv", "rr_nrmssd", "rr_mad_succ"):
            res.setdefault(k + suffix, nan)
    for k in ("n_matched", "n_unmatched_A", "n_unmatched_B"):
        res.setdefault(k, 0)
    res.setdefault("match_median_abs_dt_ms", nan)
    for k in ("median_support_A", "median_support_B"):
        res.setdefault(k, nan)
    for k in ("n_usable_leads", "n_flat_leads", "n_clip_leads", "n_nonfinite_leads"):
        res.setdefault(k, 0)
    for k in ("rate_mean_bpm", "rate_median_bpm"):
        res["accepted_" + k] = res[k + "_cons"] if res["rate_state"] == "ACCEPTED" else nan
    for k in ("rr_cv", "rr_nrmssd", "rr_mad_succ"):
        res["accepted_" + k] = res[k + "_cons"] if res["irr_state"] == "ACCEPTED" else nan
    res["reason_codes"] = ";".join(res["reasons"] + res["uncertainty_reasons"])
    if not return_beats:
        res.pop("beats", None)
    return res


def extract_path(mat_path, c=CONFIG, return_beats=False):
    """Read the RAW integer `val` array of a record .mat (no backbone preprocessing) and extract concepts."""
    from .preprocessing import read_raw_mat
    return extract_record(read_raw_mat(mat_path), c, return_beats)


def extract_many(paths, c=CONFIG, return_beats=False):
    """Extract a list of record .mat paths (later: rows of a role manifest). Failures are returned as unavailable records."""
    out = []
    for p in paths:
        try:
            out.append(extract_path(p, c, return_beats))
        except Exception as e:                       # a read failure must not abort a batch; recorded, not hidden
            r = extract_record(None, c, return_beats)
            r["reasons"] = ["READ_ERROR"]
            r["reason_codes"] = "READ_ERROR:" + type(e).__name__
            out.append(r)
    return out
