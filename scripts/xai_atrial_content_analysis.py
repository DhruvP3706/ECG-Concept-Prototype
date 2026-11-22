r"""Analysis of Experiment 9 (atrial-content substitution). Unit of analysis = ECG record (repeated donors / sham realisations of one record are averaged first); within-record paired logit changes;
record-level bootstrap B=5000 seed 20261120 (bm / bdiff of Experiments 7-8). SR recipients = primary, AFIB recipients = secondary."""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from xai_graded_rr_analysis import bm  # noqa: E402
import xai_ig_figures as fg  # noqa: E402
import xai_figures_v2  # noqa: E402,F401
from ecg_concept import atrial_ops as ao  # noqa: E402

CLS = ["AFIB", "SR", "AF", "SB", "ST", "SA", "STTC", "TWC", "TWO"]
EXP8 = os.path.join(ROOT, "results", "xai_atrial_activity", "run_20261120_152906")
FAM = {"flat": ["flat"], "rev": ["rev"], "shuf": ["shuf"], "dAF": [f"dAF{k}" for k in range(5)], "dAFsc": [f"dAFsc{k}" for k in range(3)], "rawAF": ["rawAF0", "rawAF1"], "dRND": [f"dRND{k}" for k in range(3)],
       "dSR": [f"dSR{k}" for k in range(3)], "sh_flat": ["sh_flat_a", "sh_flat_b"], "sh_rev": ["sh_rev_a", "sh_rev_b"], "sh_shuf": ["sh_shuf_a", "sh_shuf_b"], "shAF": [f"shAF{k}" for k in range(5)], "shSR": [f"shSR{k}" for k in range(3)]}
PAIRS = {"flat": [("flat", "sh_flat_a"), ("flat", "sh_flat_b")], "rev": [("rev", "sh_rev_a"), ("rev", "sh_rev_b")], "shuf": [("shuf", "sh_shuf_a"), ("shuf", "sh_shuf_b")],
         "dAF": [(f"dAF{k}", f"shAF{k}") for k in range(5)], "dSR": [(f"dSR{k}", f"shSR{k}") for k in range(3)]}
ROWS = [("flatten (Exp 8 op.)", "flat", "sh_flat"), ("time reversal", "rev", "sh_rev"), ("block shuffle (5 blocks)", "shuf", "sh_shuf"), ("AFIB donor (5, matched)", "dAF", "shAF"),
        ("AFIB donor, amplitude-scaled", "dAFsc", None), ("AFIB donor, raw samples", "rawAF", None), ("AFIB donor, random (unmatched)", "dRND", None), ("SR donor (3)", "dSR", "shSR")]


def analyze(run):
    names = [str(c) for c in np.load(os.path.join(ROOT, "results", "step4_4", "cache", "merged.npz"))["class_names"]]
    ci = {n: i for i, n in enumerate(names)}
    thr = np.load(os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0_thresh.npy"), allow_pickle=False)
    items = list(csv.DictReader(open(os.path.join(run, "items.csv"))))
    L = np.load(os.path.join(run, "logits.npy"))
    info = json.load(open(os.path.join(run, "records_beats.json")))
    donors = json.load(open(os.path.join(run, "donors.json")))
    orig_row = {it["record_id"]: int(it["logit_row"]) for it in items if it["condition"] == "orig"}
    T = {(it["record_id"], it["condition"]): it for it in items if it["condition"] != "orig"}
    fl = lambda s: float(s) if s not in ("", "nan") else float("nan")
    grp = {r: v["group"] for r, v in info.items() if v["usable"]}
    recs = {g: sorted(r for r in grp if grp[r] == g and r in donors) for g in ("SR", "AFIB")}      # eligible recipients (>=4 eligible beats)
    allrecs = {g: sorted(r for r in grp if grp[r] == g) for g in ("SR", "AFIB")}
    conds = sorted({c for _, c in T})
    pos = lambda lg, k: 1 / (1 + np.exp(-lg)) > thr[ci[k]]

    def acc(r, c):
        it = T.get((r, c))
        return it is not None and it["accepted"] == "True"

    def lgt(r, c, k):
        return L[orig_row[r]][ci[k]] if c == "orig" else L[int(T[(r, c)]["logit_row"])][ci[k]]

    def dval(r, cs, k):
        v = [lgt(r, c, k) - lgt(r, "orig", k) for c in cs if acc(r, c)]
        return float(np.mean(v)) if v else np.nan

    def pval(r, prs, k):
        v = [lgt(r, a, k) - lgt(r, b, k) for a, b in prs if acc(r, a) and acc(r, b)]
        return float(np.mean(v)) if v else np.nan

    def flipv(r, cs, k):
        v = [pos(lgt(r, c, k), k) != pos(lgt(r, "orig", k), k) for c in cs if acc(r, c)]
        return float(np.mean(v)) if v else np.nan

    def stat(vals):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        out = bm(v)
        return {"mean": out[0], "lo": out[1], "hi": out[2], "n": out[3], "median": float(np.median(v)) if len(v) else float("nan"), "p10": float(np.percentile(v, 10)) if len(v) else float("nan"), "p90": float(np.percentile(v, 90)) if len(v) else float("nan")}

    def eff(g, cs):
        rr = [r for r in recs[g] if any(acc(r, c) for c in cs)]
        o = {k: stat([dval(r, cs, k) for r in rr]) for k in CLS}
        o["flip_AFIB"] = bm([flipv(r, cs, "AFIB") for r in rr])
        o["flip_SR"] = bm([flipv(r, cs, "SR") for r in rr])
        o["n_records"] = len(rr)
        return o

    def net(g, prs):
        rr = [r for r in recs[g] if any(acc(r, a) and acc(r, b) for a, b in prs)]
        o = {k: stat([pval(r, prs, k) for r in rr]) for k in CLS}
        o["n_records"] = len(rr)
        return o

    def cmp(g, fa, fb):
        rr = [r for r in recs[g] if any(acc(r, c) for c in fa) and any(acc(r, c) for c in fb)]
        o = {k: stat([dval(r, fa, k) - dval(r, fb, k) for r in rr]) for k in CLS}
        o["n_records"] = len(rr)
        return o

    res = {"n_recipients_eligible": {g: len(v) for g, v in recs.items()}, "n_usable": {g: len(v) for g, v in allrecs.items()}, "bootstrap": "record-level, B=5000, seed 20261120"}

    # ------------------------------------------------ reproduction of Experiment 8
    it8 = list(csv.DictReader(open(os.path.join(EXP8, "items.csv"))))
    L8 = np.load(os.path.join(EXP8, "logits.npy"))
    T8 = {(i["record_id"], i["condition"]): i for i in it8}
    rep = {}
    for c9, c8 in (("orig", "orig"), ("flat", "flat"), ("rr_only", "rr_only"), ("rr_flat", "rr_flat")):
        dd = []
        for r in allrecs["SR"] + allrecs["AFIB"]:
            a = T8.get((r, c8))
            b = T.get((r, c9)) if c9 != "orig" else {"accepted": "True", "logit_row": orig_row[r]}
            if a and b and a["accepted"] == "True" and b["accepted"] == "True":
                dd.append(np.abs(L8[int(a["logit_row"])] - L[int(b["logit_row"])]).max())
        rep[c9] = {"n_common_accepted": len(dd), "max_abs_logit_diff": float(max(dd)) if dd else None}
    res["reproduces_experiment8"] = rep

    # ------------------------------------------------ acceptance and rejection
    acc_t = {}
    for g in ("SR", "AFIB"):
        acc_t[g] = {}
        for c in conds:
            its = [T[(r, c)] for r in allrecs[g] if (r, c) in T]
            ok = sum(i["accepted"] == "True" for i in its)
            elig = [i for i in its if i["reason"] != "insufficient_eligible_beats"]
            reasons = {}
            for i in its:
                if i["accepted"] != "True":
                    for x in i["reason"].split(";"):
                        reasons[x] = reasons.get(x, 0) + 1
            acc_t[g][c] = {"attempted": len(its), "accepted": ok, "rate": ok / max(len(its), 1), "eligible_records": len(elig), "accepted_among_eligible": ok / len(elig) if elig else None, "reasons": reasons}
    res["acceptance"] = acc_t
    dep = {}
    for g in ("SR", "AFIB"):
        dep[g] = {}
        for c in ("flat", "rev", "shuf", "dAF0", "shAF0", "dSR0", "rawAF0", "dRND0", "dAF0_s", "dAF0_l"):
            el = [T[(r, c)] for r in recs[g] if (r, c) in T]
            a_ = [i for i in el if i["accepted"] == "True"]
            r_ = [i for i in el if i["accepted"] != "True"]
            if len(r_) < 3 or len(a_) < 3:
                dep[g][c] = {"n_accepted": len(a_), "n_rejected": len(r_)}
                continue
            f = lambda lst, key: float(np.median([fl(i[key]) for i in lst]))
            bl = lambda lst, k: float(np.median([L[orig_row[i["record_id"]]][ci[k]] for i in lst]))
            dep[g][c] = {"n_accepted": len(a_), "n_rejected": len(r_), "median_mean_rr_samples": [f(a_, "mean_rr"), f(r_, "mean_rr")], "median_rr_cv": [f(a_, "cv0"), f(r_, "cv0")],
                         "median_baseline_AFIB_logit": [bl(a_, "AFIB"), bl(r_, "AFIB")], "median_baseline_SR_logit": [bl(a_, "SR"), bl(r_, "SR")], "note": "[accepted, rejected-among-eligible]"}
    res["rejection_dependence"] = dep

    # ------------------------------------------------ intervention magnitude / sham matching
    mag = {}
    for g in ("SR", "AFIB"):
        mag[g] = {}
        for c in ("flat", "rev", "shuf", "dAF0", "dAFsc0", "rawAF0", "dRND0", "dSR0"):
            v = [fl(T[(r, c)]["rms_modification"]) for r in recs[g] if acc(r, c)]
            d_ = [fl(T[(r, c)]["max_region_diff_ratio"]) for r in recs[g] if acc(r, c)]
            mag[g][c] = {"n": len(v), "modification_rms_counts_median": float(np.median(v)) if v else None, "max_diff_ratio_median": float(np.median(d_)) if d_ else None}
        for c in ("sh_flat_a", "sh_rev_a", "sh_shuf_a", "shAF0", "shAF1", "shSR0"):
            real = {"sh_flat_a": "flat", "sh_rev_a": "rev", "sh_shuf_a": "shuf", "shAF0": "dAF0", "shAF1": "dAF1", "shSR0": "dSR0"}[c]
            rr = [r for r in recs[g] if acc(r, c) and acc(r, real)]
            rt = [fl(T[(r, c)]["rms_ratio_to_real"]) for r in rr]
            sp = [fl(T[(r, c)]["spec_l1_to_real"]) for r in rr]
            dr = [(fl(T[(r, real)]["max_region_diff_ratio"]), fl(T[(r, c)]["max_region_diff_ratio"])) for r in rr]
            mag[g][c] = {"n": len(rr), "rms_ratio_sham_over_real": [float(np.percentile(rt, q)) for q in (5, 50, 95)] if rt else None, "band_energy_L1_sham_vs_real_median": float(np.median(sp)) if sp else None,
                         "max_diff_ratio_median_real_sham": [float(np.median([a for a, _ in dr])), float(np.median([b for _, b in dr]))] if dr else None}
    res["magnitude_matching"] = mag

    # ------------------------------------------------ effects (primary SR recipients, secondary AFIB recipients)
    res["effects"] = {g: {k: eff(g, v) for k, v in FAM.items()} for g in ("SR", "AFIB")}
    res["net_vs_matched_sham"] = {g: {k: net(g, v) for k, v in PAIRS.items()} for g in ("SR", "AFIB")}
    comps = {"dAF_minus_flat": ("dAF", "flat"), "dAF_minus_dSR": ("dAF", "dSR"), "dAF_minus_dRND": ("dAF", "dRND"), "dAF_minus_rev": ("dAF", "rev"), "dAF_minus_shuf": ("dAF", "shuf"), "dAFsc_minus_dAF": ("dAFsc", "dAF"),
             "rawAF_minus_dAF": ("rawAF", "dAF"), "rev_minus_flat": ("rev", "flat"), "shuf_minus_flat": ("shuf", "flat"), "dSR_minus_flat": ("dSR", "flat"), "shAF_minus_sh_flat": ("shAF", "sh_flat")}
    res["comparisons"] = {g: {n: cmp(g, FAM[a], FAM[b]) for n, (a, b) in comps.items()} for g in ("SR", "AFIB")}
    # net-of-sham differences between interventions (paired, record-level)
    nn = {}
    for g in ("SR", "AFIB"):
        nn[g] = {}
        for n, (a, b) in {"net_dAF_minus_net_flat": ("dAF", "flat"), "net_dAF_minus_net_rev": ("dAF", "rev"), "net_dAF_minus_net_shuf": ("dAF", "shuf"), "net_dAF_minus_net_dSR": ("dAF", "dSR")}.items():
            rr = [r for r in recs[g] if any(acc(r, x) and acc(r, y) for x, y in PAIRS[a]) and any(acc(r, x) and acc(r, y) for x, y in PAIRS[b])]
            nn[g][n] = {k: stat([pval(r, PAIRS[a], k) - pval(r, PAIRS[b], k) for r in rr]) for k in ("AFIB", "SR", "AF")}
            nn[g][n]["n_records"] = len(rr)
    res["net_differences"] = nn
    # correlation of AFIB and SR changes within record
    co = {}
    for g in ("SR", "AFIB"):
        rr = [r for r in recs[g] if any(acc(r, c) for c in FAM["dAF"])]
        a_, b_ = np.array([dval(r, FAM["dAF"], "AFIB") for r in rr]), np.array([dval(r, FAM["dAF"], "SR") for r in rr])
        co[g] = {"n": len(rr), "pearson_dAFIB_dSR_donor": float(np.corrcoef(a_, b_)[0, 1]) if len(rr) > 3 else None, "share_records_AFIB_up_and_SR_down": float(np.mean((a_ > 0) & (b_ < 0))) if len(rr) else None, "share_records_AFIB_up": float(np.mean(a_ > 0)) if len(rr) else None}
    res["opposite_direction"] = co

    # ------------------------------------------------ donor / lead / extent consistency
    cons = {"by_donor_rank_net_dAFIB_SR_recipients": {}}
    for k in range(5):
        cons["by_donor_rank_net_dAFIB_SR_recipients"][f"rank{k}"] = {kk: stat([pval(r, [(f"dAF{k}", f"shAF{k}")], kk) for r in recs["SR"] if acc(r, f"dAF{k}") and acc(r, f"shAF{k}")]) for kk in ("AFIB", "SR")}
    per = {}
    for r in recs["SR"]:
        for k in range(5):
            if acc(r, f"dAF{k}") and acc(r, f"shAF{k}"):
                per.setdefault(T[(r, f"dAF{k}")]["donor"], []).append(lgt(r, f"dAF{k}", "AFIB") - lgt(r, f"shAF{k}", "AFIB"))
    dm = np.array([np.mean(v) for v in per.values() if len(v) >= 5])
    cons["by_donor_record"] = {"n_donors_with_>=5_recipients": int(len(dm)), "donor_mean_net_dAFIB_quantiles_p5_25_50_75_95": [float(np.percentile(dm, q)) for q in (5, 25, 50, 75, 95)] if len(dm) else None,
                               "share_donors_positive": float((dm > 0).mean()) if len(dm) else None, "mean_over_donors": bm(dm)[:3] if len(dm) > 2 else None, "recipients_per_donor_median": float(np.median([len(v) for v in per.values()]))}
    cons["donor_per_recipient_sd_of_net_dAFIB_median"] = float(np.median([np.std([lgt(r, f"dAF{k}", "AFIB") - lgt(r, f"shAF{k}", "AFIB") for k in range(5) if acc(r, f"dAF{k}") and acc(r, f"shAF{k}")]) for r in recs["SR"] if sum(acc(r, f"dAF{k}") and acc(r, f"shAF{k}") for k in range(5)) >= 3]))
    leads = {}
    for g in ("SR", "AFIB"):
        leads[g] = {}
        for ln in ("II", "V1", "limb", "chest"):
            leads[g][ln] = {"flat": eff(g, [f"flat_{ln}"]), "dAF0": eff(g, [f"dAF0_{ln}"]), "net_dAF0": net(g, [(f"dAF0_{ln}", f"shAF0_{ln}")]), "net_flat": net(g, [(f"flat_{ln}", f"shAF0_{ln}")])}
        leads[g]["all12"] = {"flat": eff(g, ["flat"]), "dAF0": eff(g, ["dAF0"]), "net_dAF0": net(g, [("dAF0", "shAF0")])}
    res["donor_consistency"] = cons
    res["lead_subsets"] = leads
    ext = {}
    for g in ("SR", "AFIB"):
        ext[g] = {}
        for tag, sfx in (("120", "_s"), ("160", ""), ("200", "_l")):
            ext[g][tag] = {"flat": eff(g, ["flat" + sfx]), "dAF(0,1)": eff(g, [f"dAF0{sfx}", f"dAF1{sfx}"]), "net_dAF0": net(g, [(f"dAF0{sfx}", f"shAF0{sfx}")])}
    res["region_length"] = ext
    # same-record comparison across lengths (records with all three extents accepted)
    same = {}
    for g in ("SR", "AFIB"):
        rr = [r for r in recs[g] if all(acc(r, c) for c in ("flat_s", "flat", "flat_l", "dAF0_s", "dAF0", "dAF0_l", "shAF0_s", "shAF0", "shAF0_l"))]
        same[g] = {"n": len(rr)}
        for tag, sfx in (("120", "_s"), ("160", ""), ("200", "_l")):
            same[g][tag] = {"flat_dAFIB": stat([dval(r, ["flat" + sfx], "AFIB") for r in rr]), "dAF0_dAFIB": stat([dval(r, [f"dAF0{sfx}"], "AFIB") for r in rr]), "net_dAF0_dAFIB": stat([pval(r, [(f"dAF0{sfx}", f"shAF0{sfx}")], "AFIB") for r in rr])}
    res["region_length_same_records"] = same

    # ------------------------------------------------ combined with RR variability (Experiment 7/8 factorial, recomputed here with identical RNG)
    fr = [r for r in recs["SR"] if all(acc(r, c) for c in ("rr_only", "rr_flat", "rr_dAF0", "rr_dAF1", "rr_dAF2", "rr_shAF0"))]
    nat = [r for r in allrecs["AFIB"] if 0.17 <= info[r]["cv"] <= 0.23]
    fa = {"n_records": len(fr), "natural_AFIB_n": len(nat), "natural_AFIB_logit": {k: stat([lgt(r, "orig", k) for r in nat]) for k in ("AFIB", "SR")}}
    cond_abs = {"orig": ["orig"], "rr_only": ["rr_only"], "rr_flat": ["rr_flat"], "rr_dAF": ["rr_dAF0", "rr_dAF1", "rr_dAF2"], "rr_shAF0": ["rr_shAF0"]}
    for nm, cs in cond_abs.items():
        fa[nm] = {k: stat([np.mean([lgt(r, c, k) for c in cs]) for r in fr]) for k in ("AFIB", "SR")}
    gap = float(np.mean([lgt(r, "orig", "AFIB") * 0 for r in fr]) + fa["natural_AFIB_logit"]["AFIB"]["mean"] - fa["rr_only"]["AFIB"]["mean"])
    fa["gap_to_natural_AFIB_after_rr_only"] = gap
    for nm in ("rr_flat", "rr_dAF", "rr_shAF0"):
        d_ = [np.mean([lgt(r, c, "AFIB") for c in cond_abs[nm]]) - lgt(r, "rr_only", "AFIB") for r in fr]
        fa[nm]["change_vs_rr_only_AFIB"] = stat(d_)
        fa[nm]["share_of_gap_closed"] = float(np.mean(d_) / gap)
        fa[nm]["AFIB_positive_rate"] = bm([float(np.mean([pos(lgt(r, c, "AFIB"), "AFIB") for c in cond_abs[nm]])) for r in fr])
    fa["rr_only"]["AFIB_positive_rate"] = bm([float(pos(lgt(r, "rr_only", "AFIB"), "AFIB")) for r in fr])
    fa["rr_dAF_minus_rr_shAF0_AFIB"] = stat([np.mean([lgt(r, c, "AFIB") for c in cond_abs["rr_dAF"]]) - lgt(r, "rr_shAF0", "AFIB") for r in fr])
    fa["rr_dAF_minus_rr_flat_AFIB"] = stat([np.mean([lgt(r, c, "AFIB") for c in cond_abs["rr_dAF"]]) - lgt(r, "rr_flat", "AFIB") for r in fr])
    res["factorial_SR_rr_cv0.2"] = fa
    res["experiment8_reference_values"] = {"SR_flatten_dAFIB": "+8.25 [7.75, 8.77]", "SR_flatten_dSR": "-2.32 [-2.47, -2.19]", "flatten_minus_sine_sham_dAFIB": "+6.51 [6.01, 7.00]", "factorial_rr_only_AFIB": -7.98, "factorial_rr_flat_AFIB": -3.32, "natural_AFIB": 2.46, "gap_closed_by_flat": 0.446,
                                           "source": "results/xai_atrial_activity/run_20261120_152906/analysis_results.json (previously observed; this run re-generates the same conditions and checks bitwise-equal logits above)"}
    json.dump(res, open(os.path.join(run, "analysis_results.json"), "x"), indent=1, default=float)
    print("analysis saved")
    figures(run, res, T, acc, recs, info, lgt, dval, pval, eff, net, orig_row, L, ci, items)


def dotplot(s, x0, y0, w, h, rows, xlim, title, xlabel, key_real, key_sham, cols):
    n = len(rows)
    f = fg.axes(s, x0, y0, w, h, xlim, (0, n), [], [], xlabel, "", title)
    s.line(*f(0, 0), *f(0, n), c="#888", sw=1, dash="3,3")
    for i, (lab, real, sham) in enumerate(rows):
        yy = n - i - 0.5
        s.text(x0 - 8, f(0, yy)[1] + 4, lab, 11, "end")
        for off, rec, hollow in ((0.14, real, False), (-0.14, sham, True)):
            if rec is None:
                continue
            m, lo, hi = rec["mean"], rec["lo"], rec["hi"]
            if not np.isfinite(m):
                continue
            c = cols[0] if not hollow else cols[1]
            s.line(*f(lo, yy + off), *f(hi, yy + off), c=c, sw=2)
            X, Y = f(m, yy + off)
            s.add(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="4.5" fill="{"white" if hollow else c}" stroke="{c}" stroke-width="2"/>')
            s.text(f(xlim[1], 0)[0] + 6, Y + 4, f"{m:+.2f}", 9)
    return f


def figures(run, res, T, acc, recs, info, lgt, dval, pval, eff, net, orig_row, L, ci, items):
    out = os.path.join(run, "figures")
    os.makedirs(out)
    E, N = res["effects"], res["net_vs_matched_sham"]
    C_REAL, C_SHAM = "#0072B2", "#E69F00"
    # ---------- F1: logit change per intervention, real (filled) vs matched sham (hollow)
    s = fg.Svg(1560, 1060)
    s.text(20, 28, "Atrial-content interventions on SR recipients: within-record logit change (modified - original), record-level 95% CI", 14, weight="bold")
    s.text(20, 46, "Filled = real intervention; hollow = matched sham (same location, per-lead energy, spectrum; random phase). Donor/sham realisations averaged within record first.", 11, c="#333")
    for pi, k in enumerate(("AFIB", "SR")):
        rows = [(lab, E["SR"][r][k], E["SR"][sh][k] if sh else None) for lab, r, sh in ROWS]
        allv = [v for _, a, b in rows for v in ((a["lo"], a["hi"]) + ((b["lo"], b["hi"]) if b else ()))]
        lo, hi = min(min(allv), -0.2), max(max(allv), 0.2)
        pad = 0.08 * (hi - lo)
        dotplot(s, 290 + pi * 640, 80, 330, 360, rows, (lo - pad, hi + pad), f"{'A' if k == 'AFIB' else 'B'}. {k} logit change", "change in logit", k, None, (C_REAL, C_SHAM))
    # C: net (real minus sham, paired) for other classes
    classes = ["AF", "SA", "SB", "ST", "STTC", "TWO"]
    ccol = ["#CC79A7", "#999999", "#009E73", "#E69F00", "#555555", "#56B4E9"]
    pairs = [("flat", "flatten"), ("rev", "reversal"), ("shuf", "shuffle"), ("dAF", "AFIB donor"), ("dSR", "SR donor")]
    vals = [N["SR"][p][c] for p, _ in pairs for c in classes]
    lo, hi = min(min(v["lo"] for v in vals), -0.2) * 1.15, max(max(v["hi"] for v in vals), 0.2) * 1.15
    f = fg.axes(s, 80, 540, 700, 300, (0, len(pairs)), (lo, hi), [], np.linspace(lo, hi, 5), "", "net change (real - sham), logit", "C. SR recipients: other classes, real minus matched sham (paired)")
    s.line(*f(0, 0), *f(len(pairs), 0), c="#888", sw=1, dash="3,3")
    for i, (p, lab) in enumerate(pairs):
        for j, c in enumerate(classes):
            v = N["SR"][p][c]
            x1, y1 = f(i + 0.08 + j * 0.14, v["mean"])
            x2, y2 = f(i + 0.08 + j * 0.14 + 0.12, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), ccol[j], 0.9)
            s.line((x1 + x2) / 2, f(0, v["lo"])[1], (x1 + x2) / 2, f(0, v["hi"])[1], "#000", 1)
        s.text(f(i + 0.5, 0)[0], 540 + 300 + 16, lab, 10, "middle")
    for j, c in enumerate(classes):
        s.rect(100 + j * 105, 880, 10, 10, ccol[j])
        s.text(115 + j * 105, 889, c + (" (morph.)" if c in ("STTC", "TWO") else ""), 10)
    # D: AFIB recipients
    rowsA = [(lab, E["AFIB"][r]["AFIB"], E["AFIB"][sh]["AFIB"] if sh else None) for lab, r, sh in ROWS]
    allv = [v for _, a, b in rowsA for v in ((a["lo"], a["hi"]) + ((b["lo"], b["hi"]) if b else ())) if np.isfinite(v)]
    lo, hi = min(min(allv), -0.2), max(max(allv), 0.2)
    pad = 0.08 * (hi - lo)
    dotplot(s, 1110, 560, 300, 280, rowsA, (lo - pad, hi + pad), "D. AFIB recipients (secondary): AFIB logit change", "change in logit", "AFIB", None, (C_REAL, C_SHAM))
    s.text(20, 940, f"n SR recipients (eligible) = {res['n_recipients_eligible']['SR']}; n AFIB recipients = {res['n_recipients_eligible']['AFIB']}. Condition-specific n differ because of quality control (see acceptance).", 11, c="#333")
    s.save(os.path.join(out, "content_effects.svg"))

    # ---------- F2: individual record responses (histograms of record-level AFIB logit change)
    s = fg.Svg(1560, 760)
    s.text(20, 28, "Individual SR-recipient responses: record-level change in AFIB logit", 14, weight="bold")
    s.text(20, 46, "Blue = real intervention, orange outline = matched sham (same records). Dotted line = no change.", 11, c="#333")
    panels = [("flat", "sh_flat", "flatten"), ("rev", "sh_rev", "time reversal"), ("shuf", "sh_shuf", "block shuffle"), ("dAF", "shAF", "AFIB donors"), ("dRND", None, "random AFIB donors"), ("dSR", "shSR", "SR donors")]
    FAMS = {"flat": ["flat"], "rev": ["rev"], "shuf": ["shuf"], "dAF": [f"dAF{k}" for k in range(5)], "dRND": [f"dRND{k}" for k in range(3)], "dSR": [f"dSR{k}" for k in range(3)], "sh_flat": ["sh_flat_a", "sh_flat_b"], "sh_rev": ["sh_rev_a", "sh_rev_b"],
            "sh_shuf": ["sh_shuf_a", "sh_shuf_b"], "shAF": [f"shAF{k}" for k in range(5)], "shSR": [f"shSR{k}" for k in range(3)]}
    allx = []
    for fam, sh, _ in panels:
        for f_ in (fam, sh):
            if f_:
                allx += [dval(r, FAMS[f_], "AFIB") for r in recs["SR"] if any(acc(r, c) for c in FAMS[f_])]
    allx = np.array([v for v in allx if np.isfinite(v)])
    xlo, xhi = np.percentile(allx, 0.5), np.percentile(allx, 99.5)
    bins = np.linspace(xlo, xhi, 31)
    for pi, (fam, sh, lab) in enumerate(panels):
        x0, y0 = 70 + (pi % 3) * 500, 80 + (pi // 3) * 330
        hs = {}
        for nm, f_ in (("real", fam), ("sham", sh)):
            if f_:
                v = np.array([dval(r, FAMS[f_], "AFIB") for r in recs["SR"] if any(acc(r, c) for c in FAMS[f_])])
                hs[nm] = (np.histogram(np.clip(v[np.isfinite(v)], xlo, xhi), bins)[0], int(np.isfinite(v).sum()))
        ymax = max(h[0].max() for h in hs.values()) * 1.1
        f = fg.axes(s, x0, y0, 420, 240, (xlo, xhi), (0, ymax), np.round(np.linspace(xlo, xhi, 5), 1), np.linspace(0, ymax, 4), "change in AFIB logit", "records", f"{lab} (n={hs['real'][1]})")
        s.line(*f(0, 0), *f(0, ymax), c="#444", sw=1, dash="2,3")
        for i in range(len(bins) - 1):
            x1, y1 = f(bins[i], hs["real"][0][i])
            x2, y2 = f(bins[i + 1], 0)
            s.rect(x1, y1, max(x2 - x1 - 1, 1), y2 - y1, C_REAL, 0.85)
        if "sham" in hs:
            pts = []
            for i in range(len(bins) - 1):
                pts += [f(bins[i], hs["sham"][0][i]), f(bins[i + 1], hs["sham"][0][i])]
            s.poly(pts, C_SHAM, 2)
    s.save(os.path.join(out, "individual_responses.svg"))

    # ---------- F3: robustness (donor rank, region length, lead subsets)
    s = fg.Svg(1560, 580)
    s.text(20, 28, "Robustness of the AFIB-donor effect (SR recipients, AFIB-logit change; 95% record-level CI)", 14, weight="bold")
    s.text(20, 46, "All markers are real interventions; net-of-sham values are in analysis_results.json (donor_consistency, lead_subsets, region_length).", 11, c="#333")
    cons = res["by_donor_rank_net_dAFIB_SR_recipients"] if "by_donor_rank_net_dAFIB_SR_recipients" in res else None
    RL, LS = res["region_length"]["SR"], res["lead_subsets"]["SR"]
    rowsA = [(f"donor rank {k}", None, None) for k in range(5)]
    # A: region length
    rowsB = []
    for tag in ("120", "160", "200"):
        rowsB.append((f"flatten, {tag}-sample window", RL[tag]["flat"]["AFIB"], None))
        rowsB.append((f"AFIB donors (0,1), {tag}", RL[tag]["dAF(0,1)"]["AFIB"], None))
    allv = [v for _, a, _b in rowsB for v in (a["lo"], a["hi"])]
    lo, hi = min(allv + [-0.2]), max(allv + [0.2])
    pad = 0.08 * (hi - lo)
    dotplot(s, 300, 80, 300, 300, rowsB, (lo - pad, hi + pad), "A. region length (AFIB logit change)", "change in logit", "AFIB", None, (C_REAL, C_SHAM))
    rowsC = []
    for ln in ("all12", "limb", "chest", "II", "V1"):
        rowsC.append((f"flatten, leads {ln}", LS[ln]["flat"]["AFIB"], None))
        rowsC.append((f"AFIB donor 0, leads {ln}", LS[ln]["dAF0"]["AFIB"], None))
    allv = [v for _, a, _b in rowsC for v in (a["lo"], a["hi"])]
    lo, hi = min(allv + [-0.2]), max(allv + [0.2])
    pad = 0.08 * (hi - lo)
    dotplot(s, 940, 80, 380, 420, rowsC, (lo - pad, hi + pad), "B. lead subsets (AFIB logit change)", "change in logit", "AFIB", None, (C_REAL, C_SHAM))
    rk = res["by_donor_rank_net_dAFIB_SR_recipients"] if "by_donor_rank_net_dAFIB_SR_recipients" in res else None
    s.save(os.path.join(out, "robustness.svg"))

    # ---------- F4: combined with RR variability
    fa = res["factorial_SR_rr_cv0.2"]
    s = fg.Svg(1560, 620)
    s.text(20, 28, "SR records: RR-CV moved to 0.20, then atrial region manipulated (Experiment 8 design + AFIB-donor content)", 14, weight="bold")
    s.text(20, 46, f"{fa['n_records']} records with every condition accepted. Natural AFIB records with RR-CV 0.17-0.23 (n={fa['natural_AFIB_n']}) shown as reference.", 11, c="#333")
    for pi, k in enumerate(("AFIB", "SR")):
        x0 = 90 + pi * 740
        bars = [("original SR", fa["orig"][k], "#777"), ("RR-CV 0.2 only", fa["rr_only"][k], "#0072B2"), ("+ flatten", fa["rr_flat"][k], "#56B4E9"), ("+ AFIB donor (3)", fa["rr_dAF"][k], "#D55E00"), ("+ matched sham", fa["rr_shAF0"][k], "#E69F00"), ("natural AFIB", fa["natural_AFIB_logit"][k], "#009E73")]
        lo, hi = min(b[1]["lo"] for b in bars) * 1.1, max(max(b[1]["hi"] for b in bars) * 1.1, 1)
        f = fg.axes(s, x0, 90, 640, 360, (0, len(bars)), (lo, hi), [], np.linspace(lo, hi, 5), "", f"{k} logit (mean)", f"{'A' if k == 'AFIB' else 'B'}. absolute {k} logit")
        s.line(*f(0, 0), *f(len(bars), 0), c="#888", sw=1, dash="3,3")
        for i, (nm, v, c) in enumerate(bars):
            x1, y1 = f(i + 0.1, v["mean"])
            x2, y2 = f(i + 0.9, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), c, 0.9)
            s.line((x1 + x2) / 2, f(0, v["lo"])[1], (x1 + x2) / 2, f(0, v["hi"])[1], "#000", 1.2)
            s.text((x1 + x2) / 2, 90 + 360 + 15 + (i % 2) * 13, nm, 9, "middle")
    s.text(90, 520, f"share of the remaining gap closed (natural AFIB - RR-only = {fa['gap_to_natural_AFIB_after_rr_only']:.1f} logits): flatten {fa['rr_flat']['share_of_gap_closed']:.2f}, AFIB donor {fa['rr_dAF']['share_of_gap_closed']:.2f}, matched sham {fa['rr_shAF0']['share_of_gap_closed']:.2f}", 11)
    s.text(90, 540, "AFIB-positive records: RR only {:.1f}%, flatten {:.1f}%, AFIB donor {:.1f}%, sham {:.1f}%".format(*[100 * fa[c]["AFIB_positive_rate"][0] for c in ("rr_only", "rr_flat", "rr_dAF", "rr_shAF0")]), 11)
    s.save(os.path.join(out, "factorial_rr_content.svg"))

    # ---------- F5: acceptance
    s = fg.Svg(1560, 600)
    s.text(20, 28, "Acceptance of transformations (all recipient records of the group; ineligible = fewer than 4 eligible beats)", 14, weight="bold")
    sel = ["flat", "rev", "shuf", "dAF0", "dAFsc0", "rawAF0", "dRND0", "dSR0", "sh_flat_a", "sh_rev_a", "sh_shuf_a", "shAF0", "dAF0_s", "dAF0_l", "dAF0_II", "dAF0_limb"]
    f = fg.axes(s, 90, 70, 1380, 360, (0, len(sel)), (0, 1), [], [0, 0.25, 0.5, 0.75, 1], "", "share of records accepted", "")
    for i, c in enumerate(sel):
        for j, g in enumerate(("SR", "AFIB")):
            a_ = res["acceptance"][g].get(c)
            if not a_:
                continue
            x1, y1 = f(i + 0.08 + j * 0.42, a_["rate"])
            x2, y2 = f(i + 0.08 + j * 0.42 + 0.4, 0)
            s.rect(x1, y1, x2 - x1, y2 - y1, ("#1a80bb", "#D55E00")[j], 0.9)
            s.text((x1 + x2) / 2, y1 - 3, f"{a_['accepted']}", 8, "middle")
        s.text(f(i + 0.5, 0)[0], 70 + 360 + 15 + (i % 2) * 12, c, 9, "middle")
    s.rect(110, 500, 10, 10, "#1a80bb"); s.text(125, 509, "SR recipients", 11)
    s.rect(260, 500, 10, 10, "#D55E00"); s.text(275, 509, "AFIB recipients", 11)
    s.save(os.path.join(out, "acceptance.svg"))

    # ---------- F6: waveform examples
    ex = np.load(os.path.join(run, "example_signals.npz"))
    keys = list(ex.keys())
    s = fg.Svg(1560, 1000)
    s.text(20, 28, "Representative SR recipient (lead II, 3 s): original and transformed atrial regions (shaded)", 14, weight="bold")
    rid = sorted({k.split("|")[0] for k in keys if k.endswith("raw_orig") and info[k.split("|")[0]]["group"] == "SR"})[0]
    b = np.array(info[rid]["beats"], int)
    regs = ao.regions(b, 160)[0]
    conds = [("orig", "original"), ("flat", "flatten"), ("rev", "time reversal"), ("shuf", "block shuffle"), ("dAF0", "AFIB donor 0"), ("shAF0", "matched sham of donor 0"), ("rawAF0", "AFIB donor, raw samples")]
    t0, t1 = 600, 2100
    y0 = 70
    conds = [cl for cl in conds if (f"{rid}|{cl[0]}|raw_orig" if cl[0] == "orig" else f"{rid}|{cl[0]}|raw_mod") in ex]
    for ci_, (c, lab) in enumerate(conds):
        key = f"{rid}|{c}|raw_orig" if c == "orig" else f"{rid}|{c}|raw_mod"
        if key not in ex:
            continue
        sig = ex[key][1].astype(float)
        seg = sig[t0:t1]
        base_y = y0 + ci_ * 125
        lo_, hi_ = np.percentile(ex[f"{rid}|orig|raw_orig"][1][t0:t1], [0.5, 99.5])
        sc = 100.0 / (hi_ - lo_ + 1e-9)
        for a, e in regs:
            if a < t1 and e > t0:
                s.rect(230 + (max(a, t0) - t0) / (t1 - t0) * 1250, base_y - 10, (min(e, t1) - max(a, t0)) / (t1 - t0) * 1250, 115, "#FFEBCC", 0.9)
        pts = [(230 + i / (t1 - t0) * 1250, base_y + 100 - (v - lo_) * sc) for i, v in enumerate(seg)]
        s.poly(pts, "#222" if c == "orig" else (C_REAL if c != "shAF0" else C_SHAM), 1.1)
        s.text(220, base_y + 55, lab, 10, "end")
    s.save(os.path.join(out, "waveform_examples.svg"))
