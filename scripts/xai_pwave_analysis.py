r"""Analysis of Experiment 10 (P-wave-like patch transplantation into AFIB recipients). Unit of analysis = AFIB recipient record (donors / sham realisations of one record averaged first); within-record paired logit
changes; record-level bootstrap B=5000 seed 20261120. Reused results of Experiments 7 and 9 are labelled 'reused'."""
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
from xai_atrial_content_analysis import dotplot  # noqa: E402
from ecg_concept import atrial_ops as ao  # noqa: E402

CLS = ["AFIB", "SR", "AF", "SB", "ST", "SA", "STTC", "TWC", "TWO"]
OFFS = [-45, -30, -15, 0, 10]
EXP7 = os.path.join(ROOT, "results", "xai_graded_rr", "run_20261120_145324")
EXP9 = os.path.join(ROOT, "results", "xai_atrial_content", "run_20261120_154832")
K3 = [0, 1, 2]


def off_name(o, k):
    return f"Pal{k}" if o == 0 else f"Pd{o:+d}_{k}"


FAM = {"flat": ["flat"], "Pal": [f"Pal{k}" for k in range(5)], "Pal3": [f"Pal{k}" for k in K3], "shP": [f"shP{k}" for k in range(5)], "shM": [f"shM{k}" for k in K3], "Ctl": [f"Ctl{k}" for k in range(5)], "Full": [f"Full{k}" for k in range(5)],
       "shFull": [f"shFull{k}" for k in K3], "Prnd": [f"Prnd{k}" for k in K3], "PalW": [f"PalW{k}" for k in K3], "PalN": [f"PalN{k}" for k in K3]}
for o in OFFS:
    FAM[f"off{o:+d}"] = [off_name(o, k) for k in K3]
PAIRS = {"Pal": [(f"Pal{k}", f"shP{k}") for k in range(5)], "Pal_vs_shM": [(f"Pal{k}", f"shM{k}") for k in K3], "Full": [(f"Full{k}", f"shFull{k}") for k in K3]}
ROWS = [("flatten (Exp 8 op.)", "flat", None), ("aligned P-like patch (5 donors)", "Pal", "shP"), ("aligned patch (3 donors) vs total-mod. sham", "Pal3", "shM"), ("non-P donor segment (control)", "Ctl", None),
        ("full-region donor segment", "Full", "shFull"), ("aligned patch, random donors", "Prnd", None), ("aligned patch, 50-sample window", "PalW", None), ("aligned patch, 30-sample window", "PalN", None)]


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
    allrec = sorted(orig_row)
    recs = sorted(r for r in allrec if r in donors)
    conds = sorted({c for _, c in T})
    pos = lambda lg, k: 1 / (1 + np.exp(-lg)) > thr[ci[k]]
    mrr = {r: float(np.diff(np.array(info[r]["beats"])).mean()) * 2 for r in allrec}            # ms
    hr_cut = float(np.median([mrr[r] for r in recs]))
    grp_of = {r: ("slow" if mrr[r] >= hr_cut else "fast") for r in recs}

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
        o = bm(v)
        return {"mean": o[0], "lo": o[1], "hi": o[2], "n": o[3], "median": float(np.median(v)) if len(v) else float("nan"), "p10": float(np.percentile(v, 10)) if len(v) else float("nan"), "p90": float(np.percentile(v, 90)) if len(v) else float("nan")}

    def eff(cs, rs=None):
        rr = [r for r in (rs or recs) if any(acc(r, c) for c in cs)]
        o = {k: stat([dval(r, cs, k) for r in rr]) for k in CLS}
        o["flip_AFIB"] = bm([flipv(r, cs, "AFIB") for r in rr])
        o["flip_SR"] = bm([flipv(r, cs, "SR") for r in rr])
        o["n_records"] = len(rr)
        return o

    def net(prs, rs=None):
        rr = [r for r in (rs or recs) if any(acc(r, a) and acc(r, b) for a, b in prs)]
        o = {k: stat([pval(r, prs, k) for r in rr]) for k in CLS}
        o["n_records"] = len(rr)
        return o

    def cmp(fa, fb, rs=None, ks=CLS):
        rr = [r for r in (rs or recs) if any(acc(r, c) for c in fa) and any(acc(r, c) for c in fb)]
        o = {k: stat([dval(r, fa, k) - dval(r, fb, k) for r in rr]) for k in ks}
        o["n_records"] = len(rr)
        return o

    res = {"n_recipients": {"AFIB_usable": len(allrec), "AFIB_eligible": len(recs)}, "bootstrap": "record-level, B=5000, seed 20261120", "heart_rate_split_mean_RR_ms": hr_cut}
    # reproduction of earlier experiments
    rep = {}
    it9 = list(csv.DictReader(open(os.path.join(EXP9, "items.csv"))))
    L9 = np.load(os.path.join(EXP9, "logits.npy"))
    T9 = {(i["record_id"], i["condition"]): i for i in it9}
    for c in ("orig", "flat"):
        dd = []
        for r in allrec:
            a = T9.get((r, c))
            b = T.get((r, c)) if c != "orig" else {"accepted": "True", "logit_row": orig_row[r]}
            if a and b and a["accepted"] == "True" and b["accepted"] == "True":
                dd.append(np.abs(L9[int(a["logit_row"])] - L[int(b["logit_row"])]).max())
        rep[c] = {"n_common": len(dd), "max_abs_logit_diff_vs_Exp9": float(max(dd)) if dd else None}
    res["reproduces_experiment9"] = rep

    # acceptance
    acc_t = {}
    for c in conds:
        its = [T[(r, c)] for r in allrec if (r, c) in T]
        ok = sum(i["accepted"] == "True" for i in its)
        elig = [i for i in its if i["reason"] != "insufficient_eligible_beats"]
        reasons = {}
        for i in its:
            if i["accepted"] != "True":
                for x in i["reason"].split(";"):
                    reasons[x] = reasons.get(x, 0) + 1
        byhr = {g: [sum(T[(r, c)]["accepted"] == "True" for r in recs if grp_of[r] == g and (r, c) in T), sum(grp_of[r] == g for r in recs)] for g in ("slow", "fast")}
        acc_t[c] = {"attempted": len(its), "accepted": ok, "eligible_records": len(elig), "accepted_among_eligible": ok / len(elig) if elig else None, "reasons": reasons, "accepted_by_heart_rate_group_[acc,eligible]": byhr}
    res["acceptance"] = acc_t
    mag = {}
    for c in ("flat", "Pal0", "Ctl0", "Full0", "Pd-45_0", "Pd+10_0", "Prnd0", "PalW0", "PalN0", "shP0", "shP1", "shM0", "shFull0"):
        v = [fl(T[(r, c)]["rms_modification"]) for r in recs if acc(r, c)]
        d_ = [fl(T[(r, c)]["max_region_diff_ratio"]) for r in recs if acc(r, c)]
        o = {"n": len(v), "modification_rms_counts_median": float(np.median(v)) if v else None, "max_diff_ratio_median": float(np.median(d_)) if d_ else None}
        if c.startswith("sh"):
            rt = [fl(T[(r, c)]["rms_ratio_to_real"]) for r in recs if acc(r, c)]
            sp = [fl(T[(r, c)]["spec_l1_to_real"]) for r in recs if acc(r, c)]
            o.update({"rms_ratio_sham_over_real_p5_50_95": [float(np.percentile(rt, q)) for q in (5, 50, 95)] if rt else None, "band_L1_median": float(np.median(sp)) if sp else None})
        mag[c] = o
    res["magnitude"] = mag
    sc = [fl(T[(r, f"Pal{k}")]["scale_ratio"]) for r in recs for k in range(5) if acc(r, f"Pal{k}")]
    res["scale_ratio_patch_over_recipient_region_rms_p5_50_95"] = [float(np.percentile(sc, q)) for q in (5, 50, 95)]
    # rejection dependence
    dep = {}
    for c in ("flat", "Pal0", "Pd-45_0", "Pd+10_0", "shP0", "Ctl0", "Full0"):
        el = [T[(r, c)] for r in recs if (r, c) in T]
        a_ = [i for i in el if i["accepted"] == "True"]
        r_ = [i for i in el if i["accepted"] != "True"]
        if len(r_) < 3 or len(a_) < 3:
            dep[c] = {"n_accepted": len(a_), "n_rejected": len(r_)}
            continue
        f = lambda lst, key: float(np.median([fl(i[key]) for i in lst]))
        bl = lambda lst, k: float(np.median([L[orig_row[i["record_id"]]][ci[k]] for i in lst]))
        dep[c] = {"n_accepted": len(a_), "n_rejected": len(r_), "median_RR_cv_[acc,rej]": [f(a_, "cv0"), f(r_, "cv0")], "median_mean_RR_ms_[acc,rej]": [f(a_, "mean_rr") * 2, f(r_, "mean_rr") * 2], "median_baseline_AFIB_logit_[acc,rej]": [bl(a_, "AFIB"), bl(r_, "AFIB")]}
    res["rejection_dependence"] = dep

    res["effects"] = {k: eff(v) for k, v in FAM.items()}
    res["net_vs_matched_sham"] = {k: net(v) for k, v in PAIRS.items()}
    comps = {"Pal_minus_flat": ("Pal", "flat"), "Pal_minus_Ctl": ("Pal", "Ctl"), "Pal_minus_Full": ("Pal", "Full"), "Pal_minus_Prnd": ("Pal3", "Prnd"), "Pal_minus_shP": ("Pal", "shP"), "Pal_minus_shM": ("Pal3", "shM"), "PalW_minus_Pal": ("PalW", "Pal3"),
             "PalN_minus_Pal": ("PalN", "Pal3"), "Ctl_minus_flat": ("Ctl", "flat"), "Full_minus_flat": ("Full", "flat"), "shP_minus_flat": ("shP", "flat")}
    res["comparisons"] = {n: cmp(FAM[a], FAM[b]) for n, (a, b) in comps.items()}
    # temporal offsets: unit = (record, donor) with every compared offset accepted; unit values are averaged within record
    def off_block(offs, units_name):
        out = {"offsets": offs, "effect": {}, "vs_aligned": {}}
        rr_ok = {}
        for r in recs:
            ks = [k for k in K3 if all(acc(r, off_name(o, k)) for o in offs)]
            if ks:
                rr_ok[r] = ks
        out["n_records"] = len(rr_ok)
        out["n_units_record_donor"] = int(sum(len(v) for v in rr_ok.values()))
        for o in offs:
            out["effect"][str(o)] = {k: stat([np.mean([lgt(r, off_name(o, kk), k) - lgt(r, "orig", k) for kk in ks]) for r, ks in rr_ok.items()]) for k in ("AFIB", "SR", "AF", "SB", "ST")}
            out["vs_aligned"][str(o)] = {k: stat([np.mean([lgt(r, off_name(o, kk), k) - lgt(r, off_name(0, kk), k) for kk in ks]) for r, ks in rr_ok.items()]) for k in ("AFIB", "SR")}
        ref = {}
        for nm, cs in (("flat", ["flat"]), ("shP", [f"shP{kk}" for kk in K3]), ("Ctl", [f"Ctl{kk}" for kk in K3])):
            ref[nm] = {k: stat([dval(r, cs, k) for r in rr_ok if any(acc(r, c) for c in cs)]) for k in ("AFIB", "SR")}
        out["reference_same_records"] = ref
        return out
    ofs = {"main_offsets_-45_to_0": off_block([-45, -30, -15, 0], "main"), "with_+10": off_block([-45, -30, -15, 0, 10], "plus10"), "feasibility_note": "later (positive) shifts are limited by the 2-sample margin before the QRS protection window: +10 accepted only when the donor's lag is <= -78"}
    ofs["unrestricted_effect_per_offset_(all_accepted_records_each)"] = {str(o): {k: eff([off_name(o, k_) for k_ in K3])[k] for k in ("AFIB", "SR")} for o in OFFS}
    res["temporal_offsets"] = ofs
    # offsets per donor (feasibility depends on donor lag): n accepted per offset
    # heart-rate groups
    res["heart_rate_groups"] = {g: {"n": sum(1 for r in recs if grp_of[r] == g), "mean_RR_ms_median": float(np.median([mrr[r] for r in recs if grp_of[r] == g])), "flat": eff(["flat"], [r for r in recs if grp_of[r] == g]), "Pal": eff(FAM["Pal"], [r for r in recs if grp_of[r] == g]),
                                    "net_Pal": net(PAIRS["Pal"], [r for r in recs if grp_of[r] == g]), "Ctl": eff(FAM["Ctl"], [r for r in recs if grp_of[r] == g])} for g in ("slow", "fast")}
    # donors
    cons = {"by_donor_rank_AFIB_and_SR": {}}
    for k in range(5):
        rr = [r for r in recs if acc(r, f"Pal{k}")]
        cons["by_donor_rank_AFIB_and_SR"][f"rank{k}"] = {kk: stat([dval(r, [f"Pal{k}"], kk) for r in rr]) for kk in ("AFIB", "SR")}
    per, snr_of, lag_of = {}, {}, {}
    for r in recs:
        for k in range(5):
            c = f"Pal{k}"
            if acc(r, c):
                did = T[(r, c)]["donor"]
                per.setdefault(did, []).append(lgt(r, c, "AFIB") - lgt(r, "orig", "AFIB"))
                snr_of[did], lag_of[did] = fl(T[(r, c)]["donor_snr"]), fl(T[(r, c)]["donor_lag"])
    dm = {d: float(np.mean(v)) for d, v in per.items() if len(v) >= 3}
    arr = np.array(list(dm.values()))
    cons["by_donor_record"] = {"n_donors_used": len(per), "n_donors_with_>=3_recipients": len(arr), "recipients_per_donor_median": float(np.median([len(v) for v in per.values()])), "donor_mean_dAFIB_quantiles_p5_25_50_75_95": [float(np.percentile(arr, q)) for q in (5, 25, 50, 75, 95)] if len(arr) else None,
                               "share_donors_negative": float((arr < 0).mean()) if len(arr) else None, "mean_over_donors": bm(arr)[:3] if len(arr) > 2 else None}
    sn = np.array([snr_of[d] for d in dm]) if dm else np.array([])
    lg = np.array([lag_of[d] for d in dm]) if dm else np.array([])
    cons["by_donor_snr_and_lag"] = {"snr>=8": bm(arr[sn >= 8])[:4] if (sn >= 8).sum() > 2 else None, "snr<8": bm(arr[sn < 8])[:4] if (sn < 8).sum() > 2 else None, "lag<=-80": bm(arr[lg <= -80])[:4] if (lg <= -80).sum() > 2 else None, "lag>-80": bm(arr[lg > -80])[:4] if (lg > -80).sum() > 2 else None,
                                   "corr_donor_snr_dAFIB": float(np.corrcoef(sn, arr)[0, 1]) if len(arr) > 3 else None}
    res["donor_consistency"] = cons
    # leads
    ld = {}
    for ln in ("II", "V1", "limb", "chest"):
        ld[ln] = {"flat": eff([f"flat_{ln}"]), "Pal0": eff([f"Pal0_{ln}"]), "net_Pal0_vs_shP0": net([(f"Pal0_{ln}", f"shP0_{ln}")])}
    ld["all12"] = {"flat": eff(["flat"]), "Pal0": eff(["Pal0"]), "net_Pal0_vs_shP0": net([("Pal0", "shP0")])}
    res["lead_subsets"] = ld
    # other rhythm logits are in effects/net; opposite-direction share
    rr = [r for r in recs if any(acc(r, c) for c in FAM["Pal"])]
    a_, b_ = np.array([dval(r, FAM["Pal"], "AFIB") for r in rr]), np.array([dval(r, FAM["Pal"], "SR") for r in rr])
    res["opposite_direction_Pal"] = {"n": len(rr), "share_AFIB_down": float((a_ < 0).mean()), "share_AFIB_down_and_SR_up": float(((a_ < 0) & (b_ > 0)).mean()), "pearson": float(np.corrcoef(a_, b_)[0, 1]) if len(rr) > 3 else None}
    base_pos = [r for r in recs if pos(lgt(r, "orig", "AFIB"), "AFIB")]
    res["baseline"] = {"n_AFIB_positive_at_baseline": len(base_pos), "n_eligible": len(recs), "mean_AFIB_logit": stat([lgt(r, "orig", "AFIB") for r in recs]), "mean_SR_logit": stat([lgt(r, "orig", "SR") for r in recs])}

    # combined with RR (Exp 7 reused, Exp 9 reused, new factorial)
    e7 = json.load(open(os.path.join(EXP7, "analysis_results.json")))["dose_response_within_record"]["AFIB"]
    comb = {"reused_Exp7_AFIB_RR-CV_to_SR_median_(L1)": {"n": e7["L1_low(SR median)"]["n_records"], "dAFIB": e7["L1_low(SR median)"]["dlogit"]["AFIB"][:3], "dSR": e7["L1_low(SR median)"]["dlogit"]["SR"][:3], "note": "Experiment 7 population (115 records), different construction (RR-CV 0.02); not paired with this run"}}
    d9 = {}
    for r in recs:
        v = [L9[int(T9[(r, c)]["logit_row"])] - L9[int(T9[(r, "orig")]["logit_row"])] for c in ("dSR0", "dSR1", "dSR2") if (r, c) in T9 and T9[(r, c)]["accepted"] == "True"]
        if v:
            d9[r] = np.mean(v, axis=0)
    comb["reused_Exp9_SR_donor_full_region_(unvalidated_donors)"] = {"n": len(d9), "dAFIB": stat([v[ci["AFIB"]] for v in d9.values()]), "dSR": stat([v[ci["SR"]] for v in d9.values()])}
    both = [r for r in d9 if any(acc(r, c) for c in FAM["Pal"])]
    comb["paired_new_Pal_minus_reused_Exp9_dSR"] = {"n": len(both), "dAFIB": stat([dval(r, FAM["Pal"], "AFIB") - d9[r][ci["AFIB"]] for r in both]), "dSR": stat([dval(r, FAM["Pal"], "SR") - d9[r][ci["SR"]] for r in both])}
    fr = [r for r in recs if all(acc(r, c) for c in ("rr_only", "rr_flat", "rr_shP0")) and any(acc(r, c) for c in ("rr_Pal0", "rr_Pal1", "rr_Pal2"))]
    fct = {"n_records": len(fr)}
    for nm, cs in {"orig": ["orig"], "rr_only": ["rr_only"], "rr_flat": ["rr_flat"], "rr_Pal": ["rr_Pal0", "rr_Pal1", "rr_Pal2"], "rr_shP0": ["rr_shP0"]}.items():
        fct[nm] = {k: stat([np.mean([lgt(r, c, k) for c in cs if (c == "orig" or acc(r, c))]) for r in fr]) for k in ("AFIB", "SR")}
        fct[nm]["AFIB_positive_rate"] = bm([float(np.mean([pos(lgt(r, c, "AFIB"), "AFIB") for c in cs if (c == "orig" or acc(r, c))])) for r in fr])
        fct[nm]["SR_positive_rate"] = bm([float(np.mean([pos(lgt(r, c, "SR"), "SR") for c in cs if (c == "orig" or acc(r, c))])) for r in fr])
    for nm in ("rr_flat", "rr_Pal", "rr_shP0"):
        cs = {"rr_flat": ["rr_flat"], "rr_Pal": ["rr_Pal0", "rr_Pal1", "rr_Pal2"], "rr_shP0": ["rr_shP0"]}[nm]
        fct[nm]["AFIB_change_vs_rr_only"] = stat([np.mean([lgt(r, c, "AFIB") for c in cs if (c == "orig" or acc(r, c))]) - lgt(r, "rr_only", "AFIB") for r in fr])
        fct[nm]["SR_change_vs_rr_only"] = stat([np.mean([lgt(r, c, "SR") for c in cs if (c == "orig" or acc(r, c))]) - lgt(r, "rr_only", "SR") for r in fr])
    fct["rr_Pal_minus_rr_flat"] = {k: stat([np.mean([lgt(r, c, k) for c in ("rr_Pal0", "rr_Pal1", "rr_Pal2") if (c == "orig" or acc(r, c))]) - lgt(r, "rr_flat", k) for r in fr]) for k in ("AFIB", "SR")}
    fct["rr_Pal_minus_rr_shP0"] = {k: stat([np.mean([lgt(r, c, k) for c in ("rr_Pal0", "rr_Pal1", "rr_Pal2") if (c == "orig" or acc(r, c))]) - lgt(r, "rr_shP0", k) for r in fr]) for k in ("AFIB", "SR")}
    fct["same_records_without_RR_change"] = {nm: {k: stat([dval(r, cs, k) for r in fr]) for k in ("AFIB", "SR")} for nm, cs in {"flat": ["flat"], "Pal3": FAM["Pal3"], "shP": FAM["shP"]}.items() if all(any(acc(r, c) for c in cs) for r in fr[:1])}
    comb["factorial_AFIB_RR-CV_0.03_x_atrial_(new)"] = fct
    res["combined"] = comb
    json.dump(res, open(os.path.join(run, "analysis_results.json"), "x"), indent=1, default=float)
    print("analysis saved")
    figures(run, res, T, acc, recs, info, lgt, dval, pval, items, hr_cut, grp_of)


def figures(run, res, T, acc, recs, info, lgt, dval, pval, items, hr_cut, grp_of):
    out = os.path.join(run, "figures")
    os.makedirs(out)
    E, N = res["effects"], res["net_vs_matched_sham"]
    C_REAL, C_SHAM = "#0072B2", "#E69F00"
    n_el = res["n_recipients"]["AFIB_eligible"]
    # ---------- F1 effects
    s = fg.Svg(1560, 1060)
    s.text(20, 28, "SR-donor atrial patches transplanted into AFIB recipients: within-record logit change (modified - original), record-level 95% CI", 14, weight="bold")
    s.text(20, 46, f"Filled = real intervention; hollow = matched sham (flatten + spectrum-matched random patch at the same position, or total-modification surrogate). {n_el} eligible AFIB recipients; donors averaged within record first.", 11, c="#333")
    for pi, k in enumerate(("AFIB", "SR")):
        rows = [(lab, E[r][k], E[sh][k] if sh else None) for lab, r, sh in ROWS]
        allv = [v for _, a, b in rows for v in ((a["lo"], a["hi"]) + ((b["lo"], b["hi"]) if b else ())) if np.isfinite(v)]
        lo, hi = min(min(allv), -0.2), max(max(allv), 0.2)
        pad = 0.1 * (hi - lo)
        dotplot(s, 340 + pi * 630, 80, 300, 380, rows, (lo - pad, hi + pad), f"{'A' if k == 'AFIB' else 'B'}. {k} logit change", "change in logit", k, None, (C_REAL, C_SHAM))
    classes = ["SR", "AF", "SB", "ST", "STTC", "TWO"]
    ccol = ["#0072B2", "#CC79A7", "#009E73", "#E69F00", "#555555", "#56B4E9"]
    vals = [N["Pal"][c] for c in classes] + [res["comparisons"]["Pal_minus_flat"][c] for c in classes]
    lo, hi = min(min(v["lo"] for v in vals), -0.2) * 1.2, max(max(v["hi"] for v in vals), 0.2) * 1.2
    f = fg.axes(s, 80, 560, 520, 280, (0, 2), (lo, hi), [], np.linspace(lo, hi, 5), "", "change in logit", "C. other logits: aligned patch minus matched sham (left) and minus flatten (right)")
    s.line(*f(0, 0), *f(2, 0), c="#888", sw=1, dash="3,3")
    for i, src in enumerate((N["Pal"], res["comparisons"]["Pal_minus_flat"])):
        for j, c in enumerate(classes):
            v = src[c]
            x1, y1 = f(i + 0.06 + j * 0.15, v["mean"])
            x2, y2 = f(i + 0.06 + j * 0.15 + 0.13, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), ccol[j], 0.9)
            s.line((x1 + x2) / 2, f(0, v["lo"])[1], (x1 + x2) / 2, f(0, v["hi"])[1], "#000", 1)
    for j, c in enumerate(classes):
        s.rect(100 + j * 90, 880, 10, 10, ccol[j])
        s.text(115 + j * 90, 889, c + (" (morph.)" if c in ("STTC", "TWO") else ""), 10)
    hg = res["heart_rate_groups"]
    rowsH = []
    for g, lab in (("slow", f"slow AFIB (RR>={hr_cut:.0f} ms)"), ("fast", f"fast AFIB (RR<{hr_cut:.0f} ms)")):
        rowsH += [(f"{lab}: flatten", hg[g]["flat"]["AFIB"], None), (f"{lab}: aligned patch", hg[g]["Pal"]["AFIB"], hg[g]["net_Pal"]["AFIB"] and None)]
    allv = [v for _, a, _b in rowsH for v in (a["lo"], a["hi"]) if np.isfinite(v)]
    lo, hi = min(allv + [-0.2]), max(allv + [0.2])
    pad = 0.1 * (hi - lo)
    dotplot(s, 1090, 580, 300, 260, rowsH, (lo - pad, hi + pad), "D. AFIB logit change by heart-rate group", "change in logit", "AFIB", None, (C_REAL, C_SHAM))
    s.save(os.path.join(out, "pwave_effects.svg"))

    # ---------- F2 offsets
    ofs = res["temporal_offsets"]["main_offsets_-45_to_0"]
    ofs_p = res["temporal_offsets"]["with_+10"]
    s = fg.Svg(1560, 560)
    s.text(20, 28, "Effect of the temporal position of the transplanted P-like patch (AFIB recipients; same records and 3 donors at every offset)", 14, weight="bold")
    s.text(20, 46, f"x = shift of the patch peak relative to the donor's own pre-R lag (0 = natural alignment; negative = earlier / further from QRS). n = {ofs['n_records']} records ({ofs['n_units_record_donor']} record-donor units) for -45..0; the +10 point ({ofs_p['n_records']} records) is limited by the QRS margin. Lines: flatten, matched sham, non-P control (same records).", 11, c="#333")
    for pi, k in enumerate(("AFIB", "SR")):
        x0 = 90 + pi * 760
        ys = [ofs["effect"][str(o)][k] for o in OFFS[:4]] + [ofs_p["effect"]["10"][k]]
        refs = {"flatten": ofs["reference_same_records"]["flat"][k], "matched sham": ofs["reference_same_records"]["shP"][k], "non-P control": ofs["reference_same_records"]["Ctl"][k]}
        allv = [v["lo"] for v in ys] + [v["hi"] for v in ys] + [v["mean"] for v in refs.values()]
        lo, hi = min(allv + [0]), max(allv + [0])
        pad = 0.15 * (hi - lo + 1e-9)
        f = fg.axes(s, x0, 90, 640, 330, (-55, 20), (lo - pad, hi + pad), OFFS, np.linspace(lo - pad, hi + pad, 5), "shift of patch peak (samples; 2 ms each)", f"{k} logit change", f"{'A' if k == 'AFIB' else 'B'}. {k} logit")
        s.line(*f(-55, 0), *f(20, 0), c="#888", sw=1, dash="3,3")
        for (nm, v), col, dash in zip(refs.items(), ("#444444", C_SHAM, "#009E73"), ("6,3", "2,2", "8,3,2,3")):
            s.line(*f(-55, v["mean"]), *f(20, v["mean"]), c=col, sw=1.6, dash=dash)
            s.text(f(20, v["mean"])[0] - 4, f(20, v["mean"])[1] - 4, nm, 10, "end", c=col)
        pts = []
        for o, v in zip(OFFS, ys):
            s.line(*f(o, v["lo"]), *f(o, v["hi"]), c=C_REAL, sw=2)
            X, Y = f(o, v["mean"])
            s.add(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="5" fill="{C_REAL}"/>')
            pts.append((X, Y))
        s.poly(pts, C_REAL, 1.3, op=0.6)
    s.save(os.path.join(out, "temporal_offsets.svg"))

    # ---------- F3 individual responses
    s = fg.Svg(1560, 760)
    s.text(20, 28, "Individual AFIB-recipient responses: record-level change in AFIB logit", 14, weight="bold")
    s.text(20, 46, "Blue = real intervention; orange outline = matched sham (same records) where defined. Dotted line = no change.", 11, c="#333")
    panels = [("flat", "shP", "flatten"), ("Pal", "shP", "aligned P-like patch"), ("off-45", "shP", "patch shifted by -45"), ("Ctl", None, "non-P donor segment"), ("Full", "shFull", "full-region donor segment"), ("Prnd", None, "aligned patch, random donors")]
    FAMS = {**{k: v for k, v in FAM.items()}, "off-45": FAM["off-45"]}
    allx = []
    for fam, sh, _ in panels:
        for f_ in (fam, sh):
            if f_:
                allx += [dval(r, FAMS[f_], "AFIB") for r in recs if any(acc(r, c) for c in FAMS[f_])]
    allx = np.array([v for v in allx if np.isfinite(v)])
    xlo, xhi = np.percentile(allx, 0.5), np.percentile(allx, 99.5)
    bins = np.linspace(xlo, xhi, 26)
    for pi, (fam, sh, lab) in enumerate(panels):
        x0, y0 = 70 + (pi % 3) * 500, 80 + (pi // 3) * 330
        hs = {}
        for nm, f_ in (("real", fam), ("sham", sh)):
            if f_:
                v = np.array([dval(r, FAMS[f_], "AFIB") for r in recs if any(acc(r, c) for c in FAMS[f_])])
                hs[nm] = (np.histogram(np.clip(v[np.isfinite(v)], xlo, xhi), bins)[0], int(np.isfinite(v).sum()))
        ymax = max(h[0].max() for h in hs.values()) * 1.15
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

    # ---------- F4 robustness
    s = fg.Svg(1560, 640)
    s.text(20, 28, "Robustness of the aligned-patch effect on the AFIB logit (95% record-level CI; filled = real)", 14, weight="bold")
    cons, ld, cm = res["donor_consistency"], res["lead_subsets"], res["comparisons"]
    rowsA = [(f"donor rank {k}", cons["by_donor_rank_AFIB_and_SR"][f"rank{k}"]["AFIB"], None) for k in range(5)]
    rowsA += [("50-sample window", E["PalW"]["AFIB"], None), ("40-sample window", E["Pal3"]["AFIB"], None), ("30-sample window", E["PalN"]["AFIB"], None), ("random donors", E["Prnd"]["AFIB"], None)]
    allv = [v for _, a, _b in rowsA for v in (a["lo"], a["hi"]) if np.isfinite(v)]
    lo, hi = min(allv + [-0.2]), max(allv + [0.2])
    pad = 0.1 * (hi - lo)
    dotplot(s, 260, 80, 300, 380, rowsA, (lo - pad, hi + pad), "A. donor rank and localisation window", "change in logit", "AFIB", None, (C_REAL, C_SHAM))
    rowsB = []
    for lnn in ("all12", "limb", "chest", "II", "V1"):
        rowsB += [(f"flatten, leads {lnn}", ld[lnn]["flat"]["AFIB"], None), (f"aligned patch, leads {lnn}", ld[lnn]["Pal0"]["AFIB"], None)]
    allv = [v for _, a, _b in rowsB for v in (a["lo"], a["hi"]) if np.isfinite(v)]
    lo, hi = min(allv + [-0.2]), max(allv + [0.2])
    pad = 0.1 * (hi - lo)
    dotplot(s, 930, 80, 360, 420, rowsB, (lo - pad, hi + pad), "B. lead subsets", "change in logit", "AFIB", None, (C_REAL, C_SHAM))
    s.save(os.path.join(out, "robustness.svg"))

    # ---------- F5 localisation (Stage 1)
    s = fg.Svg(1560, 560)
    s.text(20, 28, "Stage 1: template-based localisation of the phase-locked pre-QRS (P-like) deflection", 14, weight="bold")
    rows = list(csv.DictReader(open(os.path.join(run, "pwave_localization_sr.csv"))))
    lag = np.array([float(x["lag"]) for x in rows])
    snr = np.array([float(x["snr"]) for x in rows])
    ok = np.array([x["excluded_reason"] == "" for x in rows])
    bins = np.arange(-130, -29, 4)
    h_all, h_ok = np.histogram(lag, bins)[0], np.histogram(lag[ok], bins)[0]
    f = fg.axes(s, 80, 80, 400, 300, (-130, -30), (0, h_all.max() * 1.1), [-120, -100, -80, -60, -40], np.linspace(0, h_all.max() * 1.1, 4), "peak lag before R (samples)", "SR records", "A. SR template peak lag")
    for i in range(len(bins) - 1):
        for hh, cc, op in ((h_all, "#999", 0.6), (h_ok, C_REAL, 0.9)):
            x1, y1 = f(bins[i], hh[i])
            x2, y2 = f(bins[i + 1], 0)
            s.rect(x1, y1, max(x2 - x1 - 1, 1), y2 - y1, cc, op)
    s.line(*f(-88, 0), *f(-88, h_all.max() * 1.1), c="#222", sw=1, dash="3,3")
    s.line(*f(-72, 0), *f(-72, h_all.max() * 1.1), c="#222", sw=1, dash="3,3")
    s.text(90, 430, "grey = all SR records with >=4 eligible beats; blue = accepted donors; dashed = donor lag range", 10, c="#333")
    audit = json.load(open(os.path.join(run, "stage1_audit.json")))
    snr_af_p = audit["AFIB_template_snr_percentiles_5_25_50_75_95_(same_localiser)"]
    bn = np.linspace(0, 30, 31)
    hs = np.histogram(np.clip(snr, 0, 30), bn)[0]
    f = fg.axes(s, 580, 80, 400, 300, (0, 30), (0, hs.max() * 1.1), [0, 5, 10, 20, 30], np.linspace(0, hs.max() * 1.1, 4), "template SNR", "records", "B. template SNR: SR records (AFIB: median %.1f)" % snr_af_p[2])
    for i in range(30):
        x1, y1 = f(bn[i], hs[i])
        x2, y2 = f(bn[i + 1], 0)
        s.rect(x1, y1, max(x2 - x1 - 1, 1), y2 - y1, C_REAL, 0.9)
    s.line(*f(5, 0), *f(5, hs.max() * 1.1), c="#D55E00", sw=1.4, dash="3,3")
    s.line(*f(snr_af_p[0], 0), *f(snr_af_p[4], 0), c="#D55E00", sw=4)
    s.text(590, 430, "orange bar = AFIB 5-95% range (same localiser); dashed = acceptance threshold 5", 10, c="#333")
    tz = np.load(os.path.join(run, "sr_templates.npz"))
    meta = json.load(open(os.path.join(run, "bank", "meta.json")))
    tp = np.stack([tz[r][1] for r in tz.files if r in meta])
    tt = np.arange(110) - 160
    mu, q1, q3 = tp.mean(0), np.percentile(tp, 25, axis=0), np.percentile(tp, 75, axis=0)
    lo, hi = float(q1.min()), float(q3.max())
    f = fg.axes(s, 1080, 80, 400, 300, (-160, -50), (lo - 2, hi + 2), [-150, -125, -100, -75, -50], np.round(np.linspace(lo - 2, hi + 2, 5)), "samples relative to R (2 ms each)", "mean deviation (counts)", "C. accepted donors: lead II template")
    s.polygon([f(t, v) for t, v in zip(tt, q3)] + [f(t, v) for t, v in zip(tt[::-1], q1[::-1])], C_REAL, 0.25)
    s.poly([f(t, v) for t, v in zip(tt, mu)], C_REAL, 2)
    s.text(1090, 430, "line = mean over donors, band = inter-quartile range", 10, c="#333")
    s.save(os.path.join(out, "localization.svg"))

    # ---------- F6 combined
    fa = res["combined"]["factorial_AFIB_RR-CV_0.03_x_atrial_(new)"]
    s = fg.Svg(1560, 620)
    s.text(20, 28, "AFIB recipients: RR-CV reduced to 0.03 (SR-like), then atrial region flattened or given an aligned SR P-like patch", 14, weight="bold")
    s.text(20, 46, f"{fa['n_records']} records with every condition accepted (Exp 8/9 design; RR warp and atrial interventions are new in this run).", 11, c="#333")
    for pi, k in enumerate(("AFIB", "SR")):
        x0 = 90 + pi * 740
        bars = [("original AFIB", fa["orig"][k], "#777"), ("RR-CV 0.03 only", fa["rr_only"][k], "#0072B2"), ("+ flatten", fa["rr_flat"][k], "#56B4E9"), ("+ aligned P-like patch (3)", fa["rr_Pal"][k], "#D55E00"), ("+ matched sham", fa["rr_shP0"][k], "#E69F00")]
        lo, hi = min(min(b[1]["lo"] for b in bars), -0.5) * 1.1, max(max(b[1]["hi"] for b in bars) * 1.1, 1)
        f = fg.axes(s, x0, 90, 640, 360, (0, len(bars)), (lo, hi), [], np.linspace(lo, hi, 5), "", f"{k} logit (mean)", f"{'A' if k == 'AFIB' else 'B'}. absolute {k} logit")
        s.line(*f(0, 0), *f(len(bars), 0), c="#888", sw=1, dash="3,3")
        for i, (nm, v, c) in enumerate(bars):
            x1, y1 = f(i + 0.1, v["mean"])
            x2, y2 = f(i + 0.9, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), c, 0.9)
            s.line((x1 + x2) / 2, f(0, v["lo"])[1], (x1 + x2) / 2, f(0, v["hi"])[1], "#000", 1.2)
            s.text((x1 + x2) / 2, 90 + 360 + 15 + (i % 2) * 13, nm, 9, "middle")
    s.text(90, 520, "SR-positive (frozen threshold) records: RR only {:.1f}%, +flatten {:.1f}%, +aligned patch {:.1f}%, +sham {:.1f}%".format(*[100 * fa[c]["SR_positive_rate"][0] for c in ("rr_only", "rr_flat", "rr_Pal", "rr_shP0")]), 11)
    s.text(90, 540, "AFIB-positive records: original {:.1f}%, RR only {:.1f}%, +flatten {:.1f}%, +aligned patch {:.1f}%, +sham {:.1f}%".format(*[100 * fa[c]["AFIB_positive_rate"][0] for c in ("orig", "rr_only", "rr_flat", "rr_Pal", "rr_shP0")]), 11)
    s.save(os.path.join(out, "factorial_rr_pwave.svg"))

    # ---------- F7 acceptance
    s = fg.Svg(1560, 600)
    s.text(20, 28, "Acceptance of transformations among the eligible AFIB recipients, by heart-rate group", 14, weight="bold")
    sel = ["flat", "Pal0", "Pd-45_0", "Pd-30_0", "Pd-15_0", "Pd+10_0", "shP0", "shM0", "Ctl0", "Full0", "shFull0", "Prnd0", "PalW0", "PalN0", "Pal0_II", "Pal0_limb", "rr_Pal0"]
    f = fg.axes(s, 90, 70, 1380, 360, (0, len(sel)), (0, 1), [], [0, 0.25, 0.5, 0.75, 1], "", "share of eligible recipients accepted", "")
    for i, c in enumerate(sel):
        a_ = res["acceptance"].get(c)
        if not a_:
            continue
        for j, g in enumerate(("slow", "fast")):
            ac_, tot = a_["accepted_by_heart_rate_group_[acc,eligible]"][g]
            x1, y1 = f(i + 0.08 + j * 0.42, ac_ / max(tot, 1))
            x2, y2 = f(i + 0.08 + j * 0.42 + 0.4, 0)
            s.rect(x1, y1, x2 - x1, y2 - y1, ("#1a80bb", "#D55E00")[j], 0.9)
            s.text((x1 + x2) / 2, y1 - 3, f"{ac_}/{tot}", 7, "middle")
        s.text(f(i + 0.5, 0)[0], 70 + 360 + 15 + (i % 2) * 12, c, 9, "middle")
    s.rect(110, 500, 10, 10, "#1a80bb"); s.text(125, 509, f"slower AFIB (mean RR >= {hr_cut:.0f} ms)", 11)
    s.rect(380, 500, 10, 10, "#D55E00"); s.text(395, 509, f"faster AFIB (mean RR < {hr_cut:.0f} ms)", 11)
    s.save(os.path.join(out, "acceptance.svg"))

    # ---------- F8 waveforms
    ex = np.load(os.path.join(run, "example_signals.npz"))
    keys = list(ex.keys())
    s = fg.Svg(1560, 1000)
    s.text(20, 28, "Representative AFIB recipient (lead II, 3 s): original and transformed atrial regions (shaded)", 14, weight="bold")
    rid = sorted({k.split("|")[0] for k in keys if k.endswith("raw_orig")})[0]
    b = np.array(info[rid]["beats"], int)
    regs = ao.regions(b, 160)[0]
    conds = [("orig", "original AFIB"), ("flat", "flatten"), ("Pal0", "aligned P-like patch"), ("Pd-45_0", "patch shifted -45 samples"), ("shP0", "matched sham of patch"), ("Ctl0", "non-P donor segment"), ("Full0", "full-region donor segment")]
    conds = [cl for cl in conds if (f"{rid}|{cl[0]}|raw_orig" if cl[0] == "orig" else f"{rid}|{cl[0]}|raw_mod") in ex]
    t0, t1 = 600, 2100
    for ci_, (c, lab) in enumerate(conds):
        key = f"{rid}|{c}|raw_orig" if c == "orig" else f"{rid}|{c}|raw_mod"
        seg = ex[key][1].astype(float)[t0:t1]
        base_y = 70 + ci_ * 125
        lo_, hi_ = np.percentile(ex[f"{rid}|orig|raw_orig"][1][t0:t1], [0.5, 99.5])
        scl = 100.0 / (hi_ - lo_ + 1e-9)
        for a, e in regs:
            if a < t1 and e > t0:
                s.rect(230 + (max(a, t0) - t0) / (t1 - t0) * 1250, base_y - 10, (min(e, t1) - max(a, t0)) / (t1 - t0) * 1250, 115, "#FFEBCC", 0.9)
        s.poly([(230 + i / (t1 - t0) * 1250, base_y + 100 - (v - lo_) * scl) for i, v in enumerate(seg)], "#222" if c == "orig" else (C_SHAM if c == "shP0" else C_REAL), 1.1)
        s.text(220, base_y + 55, lab, 10, "end")
    s.save(os.path.join(out, "waveform_examples.svg"))
