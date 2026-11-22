r"""Analysis of Experiment 8 (atrial-activity interventions). Unit of analysis = ECG record (several patterns of the same record are averaged first, then record-level values are bootstrapped:
B=5000, seed 20261120). All effects are within-record (modified minus original logit); SR-origin and AFIB-origin are analysed separately."""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from xai_graded_rr_analysis import bdiff, bm  # noqa: E402
import xai_ig_figures as fg  # noqa: E402
import xai_figures_v2  # noqa: E402,F401

CLASSES = ["AFIB", "SR", "SB", "ST", "AF", "SA", "STTC", "TWC", "TWO"]
GAINS = [("flat", 0.0), ("g05", 0.5), ("g15", 1.5), ("g20", 2.0)]


def analyze(run):
    names = [str(c) for c in np.load(os.path.join(ROOT, "results", "step4_4", "cache", "merged.npz"))["class_names"]]
    ci = {n: i for i, n in enumerate(names)}
    thr = np.load(os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0_thresh.npy"), allow_pickle=False)
    items = list(csv.DictReader(open(os.path.join(run, "items.csv"))))
    L = np.load(os.path.join(run, "logits.npy"))
    info = json.load(open(os.path.join(run, "records_beats.json")))
    audit = json.load(open(os.path.join(run, "stage1_audit.json")))
    fl = lambda s: float(s) if s not in ("", "nan") else float("nan")
    orig_row = {it["record_id"]: int(it["logit_row"]) for it in items if it["condition"] == "orig"}
    T = {(it["record_id"], it["condition"]): it for it in items if it["condition"] != "orig"}
    grp = {r: v["group"] for r, v in info.items() if v["usable"]}
    recs = {g: sorted(r for r in grp if grp[r] == g) for g in ("SR", "AFIB")}
    conds = sorted({it["condition"] for it in items if it["condition"] != "orig"})
    res = {"n_records": {g: len(v) for g, v in recs.items()}, "bootstrap": "record-level, B=5000, seed 20261120", "stage1_audit_ref": "stage1_audit.json"}

    def acc(r, c):
        it = T.get((r, c))
        return it is not None and it["accepted"] == "True"

    def val(r, conds_, k):
        v = [L[int(T[(r, c)]["logit_row"])][ci[k]] for c in conds_ if acc(r, c)]
        return float(np.mean(v)) if v else np.nan

    # ------------------------------------------------ acceptance / rejection / selection
    rej = {}
    for g in ("SR", "AFIB"):
        rej[g] = {}
        for c in conds:
            its = [T[(r, c)] for r in recs[g] if (r, c) in T]
            if not its:
                continue
            reasons = {}
            for it in its:
                if it["accepted"] != "True":
                    for x in it["reason"].split(";"):
                        reasons[x] = reasons.get(x, 0) + 1
            n_ok = sum(it["accepted"] == "True" for it in its)
            elig = sum(it["reason"] != "insufficient_eligible_beats" and not it["reason"].startswith("rr_") for it in its)
            rej[g][c] = {"records_attempted": len(its), "accepted": n_ok, "acceptance_rate": n_ok / len(its), "records_with_enough_eligible_beats": elig, "accepted_among_eligible": n_ok / elig if elig else None, "reasons": reasons}
    res["acceptance"] = rej
    sel = {}
    for g in ("SR", "AFIB"):
        ok = np.array([acc(r, "flat") for r in recs[g]])
        inel = np.array([T[(r, "flat")]["reason"] == "insufficient_eligible_beats" for r in recs[g]])
        mrr = np.array([info[r]["mean_rr"] * 2 for r in recs[g]])
        cv = np.array([info[r]["cv"] for r in recs[g]])
        b = lambda k, m: float(np.mean(k[m])) if m.any() else None
        sel[g] = {"n_records": len(ok), "accepted_flat": int(ok.sum()), "ineligible": int(inel.sum()), "rejected_other": int((~ok & ~inel).sum()),
                  "mean_rr_ms": {"accepted": b(mrr, ok), "ineligible": b(mrr, inel), "rejected_other": b(mrr, ~ok & ~inel)}, "rr_cv0": {"accepted": b(cv, ok), "ineligible": b(cv, inel), "rejected_other": b(cv, ~ok & ~inel)},
                  "baseline_AFIB_logit": {"accepted": b(np.array([L[orig_row[r]][ci['AFIB']] for r in recs[g]]), ok), "ineligible": b(np.array([L[orig_row[r]][ci['AFIB']] for r in recs[g]]), inel)},
                  "baseline_SR_logit": {"accepted": b(np.array([L[orig_row[r]][ci['SR']] for r in recs[g]]), ok), "ineligible": b(np.array([L[orig_row[r]][ci['SR']] for r in recs[g]]), inel)},
                  "corr_accepted_with_mean_rr": float(np.corrcoef(ok.astype(float), mrr)[0, 1]) if ok.any() and (~ok).any() else None}
    res["selection"] = sel
    mag = {}
    for g in ("SR", "AFIB"):
        ratios = [fl(T[(r, "sham_a")]["rms_ratio_to_real"]) for r in recs[g] if acc(r, "sham_a")] + [fl(T[(r, "sham_b")]["rms_ratio_to_real"]) for r in recs[g] if acc(r, "sham_b")]
        flat_rms = [fl(T[(r, "flat")]["rms_modification"]) for r in recs[g] if acc(r, "flat")]
        mag[g] = {"sham_over_real_modification_rms": {"median": float(np.median(ratios)), "p5": float(np.percentile(ratios, 5)), "p95": float(np.percentile(ratios, 95)), "n": len(ratios)},
                  "flat_modification_rms_counts_median": float(np.median(flat_rms)) if flat_rms else None, "flat_coverage_median": float(np.median([fl(T[(r, 'flat')]['coverage']) for r in recs[g] if acc(r, 'flat')])) if flat_rms else None,
                  "timing_max_shift_samples_max_over_accepted_flat": float(np.nanmax([fl(T[(r, 'flat')]['timing_max_shift']) for r in recs[g] if acc(r, 'flat')])) if flat_rms else None}
    res["magnitude_matching"] = mag

    # ------------------------------------------------ within-record effects
    def rec_arrays(g, conds_, need=None):
        rr = [r for r in recs[g] if any(acc(r, c) for c in conds_) and (need is None or all(any(acc(r, c) for c in cs) for cs in need))]
        return rr

    eff = {}
    for g in ("SR", "AFIB"):
        eff[g] = {}
        for name, group_conds in (("flat", ["flat"]), ("g05", ["g05"]), ("g15", ["g15"]), ("g20", ["g20"]), ("flat_short", ["flat_short"]), ("flat_long", ["flat_long"]), ("sham(matched to flat)", ["sham_a", "sham_b"]),
                                  ("sham_a", ["sham_a"]), ("sham_b", ["sham_b"])) + ((("fwave(mean of 4 patterns)", ["fwave_a", "fwave_b", "fwave_c", "fwave_d"]),) + tuple((f"fwave_{p}", [f"fwave_{p}"]) for p in "abcd") if g == "SR" else ()):
            rr = rec_arrays(g, group_conds)
            if len(rr) < 5:
                continue
            row = {"n_records": len(rr), "patterns_per_record_mean": float(np.mean([sum(acc(r, c) for c in group_conds) for r in rr])), "dlogit": {}, "abs_logit": {}}
            for k in CLASSES:
                d = np.array([val(r, group_conds, k) - L[orig_row[r]][ci[k]] for r in rr])
                row["dlogit"][k] = bm(d)
            for k in ("AFIB", "SR"):
                row["abs_logit"][k] = bm([val(r, group_conds, k) for r in rr])
                row["orig_logit"] = row.get("orig_logit", {})
                row["orig_logit"][k] = bm([L[orig_row[r]][ci[k]] for r in rr])
            row["label_flip_rate"] = {k: bm([np.mean([(1 / (1 + np.exp(-L[int(T[(r, c)]["logit_row"])][ci[k]])) > thr[ci[k]]) != (1 / (1 + np.exp(-L[orig_row[r]][ci[k]])) > thr[ci[k]]) for c in group_conds if acc(r, c)]) for r in rr]) for k in ("AFIB", "SR")}
            row["share_dAFIB>0"] = float(np.mean([val(r, group_conds, "AFIB") > L[orig_row[r]][ci["AFIB"]] for r in rr]))
            row["share_dSR>0"] = float(np.mean([val(r, group_conds, "SR") > L[orig_row[r]][ci["SR"]] for r in rr]))
            eff[g][name] = row
    res["within_record_effects"] = eff
    # real vs sham, paired within record (records where flat and at least one sham are accepted)
    pair = {}
    for g in ("SR", "AFIB"):
        rr = rec_arrays(g, ["flat"], need=[["sham_a", "sham_b"]])
        pair[g] = {"n_records": len(rr)}
        for k in CLASSES:
            dflat = np.array([val(r, ["flat"], k) - L[orig_row[r]][ci[k]] for r in rr])
            dsh = np.array([val(r, ["sham_a", "sham_b"], k) - L[orig_row[r]][ci[k]] for r in rr])
            pair[g][k] = {"flat": bm(dflat), "sham": bm(dsh), "flat_minus_sham": bm(dflat - dsh)}
        # per-pattern consistency
        pair[g]["per_pattern_flat_minus_each_sham_dAFIB"] = {}
        for sc in ("sham_a", "sham_b"):
            r2 = rec_arrays(g, ["flat"], need=[[sc]])
            pair[g]["per_pattern_flat_minus_each_sham_dAFIB"][sc] = {"n": len(r2), "dAFIB": bm([val(r, ["flat"], "AFIB") - val(r, [sc], "AFIB") for r in r2]), "dSR": bm([val(r, ["flat"], "SR") - val(r, [sc], "SR") for r in r2])}
    res["real_vs_sham_paired"] = pair
    # dose-response in gain, using records with all four gains accepted
    dose = {}
    for g in ("SR", "AFIB"):
        rr = rec_arrays(g, ["flat"], need=[["g05"], ["g15"], ["g20"]])
        dose[g] = {"n_records_all_gains": len(rr), "gain_vs_dlogit": {}}
        for nm, s in GAINS:
            dose[g]["gain_vs_dlogit"][f"s={s}"] = {k: bm([val(r, [nm], k) - L[orig_row[r]][ci[k]] for r in rr]) for k in ("AFIB", "SR")}
    res["dose_response_gain"] = dose
    # dependence on coverage / rate within the flat effect
    dep = {}
    for g in ("SR", "AFIB"):
        rr = rec_arrays(g, ["flat"])
        d = np.array([val(r, ["flat"], "AFIB") - L[orig_row[r]][ci["AFIB"]] for r in rr])
        ds = np.array([val(r, ["flat"], "SR") - L[orig_row[r]][ci["SR"]] for r in rr])
        cov = np.array([fl(T[(r, "flat")]["coverage"]) for r in rr])
        mrr = np.array([info[r]["mean_rr"] * 2 for r in rr])
        base = np.array([L[orig_row[r]][ci["AFIB"]] for r in rr])
        dep[g] = {"corr_dAFIB_with_coverage": float(np.corrcoef(d, cov)[0, 1]), "corr_dSR_with_coverage": float(np.corrcoef(ds, cov)[0, 1]), "corr_dAFIB_with_mean_rr": float(np.corrcoef(d, mrr)[0, 1]),
                  "corr_dAFIB_with_baseline_AFIB_logit": float(np.corrcoef(d, base)[0, 1]), "dAFIB_by_coverage_tercile": {}}
        q = np.percentile(cov, [33.3, 66.7])
        for t in range(3):
            m = np.digitize(cov, q) == t
            dep[g]["dAFIB_by_coverage_tercile"][f"T{t + 1}"] = {"n": int(m.sum()), "coverage_mean": float(cov[m].mean()), "dAFIB": bm(d[m]), "dSR": bm(ds[m])}
    res["dependence_flat_effect"] = dep
    # correctness
    cor = {}
    for g in ("SR", "AFIB"):
        own = ci[g]
        rr = rec_arrays(g, ["flat"])
        ok = np.array([L[orig_row[r]][own] > np.log(thr[own] / (1 - thr[own])) for r in rr])
        cor[g] = {"correct_n": int(ok.sum()), "incorrect_n": int((~ok).sum()), "correct_dAFIB": bm([val(r, ["flat"], "AFIB") - L[orig_row[r]][ci["AFIB"]] for r, o in zip(rr, ok) if o]),
                  "correct_dSR": bm([val(r, ["flat"], "SR") - L[orig_row[r]][ci["SR"]] for r, o in zip(rr, ok) if o]),
                  "incorrect_dSR": bm([val(r, ["flat"], "SR") - L[orig_row[r]][ci["SR"]] for r, o in zip(rr, ok) if not o])}
    res["by_original_correctness"] = cor
    # E and factorial (SR)
    fw = rec_arrays("SR", ["fwave_a", "fwave_b", "fwave_c", "fwave_d"], need=[["flat"]])
    res["E_synthetic_fwave_vs_flat_SR"] = {"n_records": len(fw), **{k: {"fwave": bm([val(r, ["fwave_a", "fwave_b", "fwave_c", "fwave_d"], k) - L[orig_row[r]][ci[k]] for r in fw]), "flat": bm([val(r, ["flat"], k) - L[orig_row[r]][ci[k]] for r in fw]),
                                                                  "fwave_minus_flat": bm([val(r, ["fwave_a", "fwave_b", "fwave_c", "fwave_d"], k) - val(r, ["flat"], k) for r in fw])} for k in ("AFIB", "SR", "SB", "ST")}}
    fr = [r for r in recs["SR"] if acc(r, "rr_only") and acc(r, "rr_flat") and acc(r, "rr_flat_fwave")]
    nat = [L[orig_row[r]][ci["AFIB"]] for r in recs["AFIB"] if 0.17 <= info[r]["cv"] <= 0.23]
    natS = [L[orig_row[r]][ci["SR"]] for r in recs["AFIB"] if 0.17 <= info[r]["cv"] <= 0.23]
    fact = {"n_records": len(fr), "natural_AFIB_records_cv_0.17_0.23": {"n": len(nat), "AFIB_logit": bm(nat), "SR_logit": bm(natS)}}
    for k in ("AFIB", "SR"):
        o = np.array([L[orig_row[r]][ci[k]] for r in fr])
        a = {c: np.array([val(r, [c], k) for r in fr]) for c in ("rr_only", "rr_flat", "rr_flat_fwave")}
        fact[k] = {"orig_abs": bm(o), **{c + "_abs": bm(v) for c, v in a.items()}, **{c + "_change": bm(v - o) for c, v in a.items()}, "rr_flat_minus_rr_only": bm(a["rr_flat"] - a["rr_only"]), "rr_flat_fwave_minus_rr_only": bm(a["rr_flat_fwave"] - a["rr_only"]),
                   "rr_flat_fwave_minus_rr_flat": bm(a["rr_flat_fwave"] - a["rr_flat"])}
    nat_m = float(np.mean(nat))
    gap = nat_m - float(np.mean([val(r, ["rr_only"], "AFIB") for r in fr]))
    fact["AFIB_logit_gap_to_natural_AFIB_after_rr_only"] = gap
    fact["share_of_gap_closed"] = {c: float((np.mean([val(r, [c], "AFIB") for r in fr]) - np.mean([val(r, ["rr_only"], "AFIB") for r in fr])) / gap) for c in ("rr_flat", "rr_flat_fwave")}
    fact["label_flips_AFIB_positive_rate"] = {c: float(np.mean([1 / (1 + np.exp(-val(r, [c], "AFIB"))) > thr[ci["AFIB"]] for r in fr])) for c in ("rr_only", "rr_flat", "rr_flat_fwave")}
    fact["RR_cv_achieved_mean"] = float(np.mean([fl(T[(r, "rr_only")]["rr_cv_achieved"]) for r in fr]))
    res["factorial_SR_rr_cv0.2_x_atrial"] = fact
    if not os.path.exists(os.path.join(run, "analysis_results.json")):
        json.dump(res, open(os.path.join(run, "analysis_results.json"), "x"), indent=1, default=float)
    print("analysis saved")
    figures(run, res, items, T, info, L, ci, recs, acc, val, orig_row, conds)


def figures(run, res, items, T, info, L, ci, recs, acc, val, orig_row, conds):
    out = os.path.join(run, "figures_v2")
    os.makedirs(out)
    col = {"SR": "#0072B2", "AFIB": "#D55E00"}
    eff = res["within_record_effects"]
    # F1 dose-response in gain + real-vs-sham
    s = fg.Svg(1500, 1000)
    s.text(20, 28, "Atrial-region interventions: within-record logit change (modified - original), record-level 95% CI", 14, weight="bold")
    s.text(20, 46, "Gain s scales the deviation of the atrial region (r-160..r-51) from its straight baseline: s=0 flatten, 0.5 attenuate, 1 unchanged, 1.5 / 2 amplify. QRS windows, RR intervals and all other samples are bit-identical.", 11, c="#333")
    for pi, (k, ttl) in enumerate((("AFIB", "A. AFIB logit"), ("SR", "B. SR logit"))):
        x0, y0 = 90 + pi * 730, 90
        vals = [eff[g][n]["dlogit"][k] for g in ("SR", "AFIB") for n in ("flat", "g05", "g15", "g20", "sham(matched to flat)") if n in eff[g]]
        lo, hi = min(min(v[1] for v in vals), -0.2) * 1.2, max(max(v[2] for v in vals), 0.2) * 1.2
        f = fg.axes(s, x0, y0, 620, 330, (-0.2, 2.3), (lo, hi), [0, 0.5, 1, 1.5, 2], np.linspace(lo, hi, 5), "gain s on the atrial-region deviation", "change in logit", ttl)
        s.line(*f(-0.2, 0), *f(2.3, 0), c="#888", sw=1, dash="3,3")
        for g, off in (("SR", -0.03), ("AFIB", 0.03)):
            pts = [f(1 + off, 0)]
            for n, sg in (("flat", 0.0), ("g05", 0.5), ("g15", 1.5), ("g20", 2.0)):
                if n not in eff[g]:
                    continue
                m, l_, h_, nn = eff[g][n]["dlogit"][k]
                s.line(*f(sg + off, l_), *f(sg + off, h_), c=col[g], sw=1.6)
                s.add(f'<circle cx="{f(sg + off, m)[0]:.1f}" cy="{f(sg + off, m)[1]:.1f}" r="4" fill="{col[g]}"/>')
                pts.append(f(sg + off, m))
            pts.sort()
            s.poly(pts, col[g], 1.1, op=0.5)
            shn = "sham(matched to flat)"
            if shn in eff[g]:
                m, l_, h_, nn = eff[g][shn]["dlogit"][k]
                xx = -0.12 + off
                s.line(*f(xx, l_), *f(xx, h_), c=col[g], sw=1.6)
                s.add(f'<circle cx="{f(xx, m)[0]:.1f}" cy="{f(xx, m)[1]:.1f}" r="4" fill="white" stroke="{col[g]}" stroke-width="2"/>')
    s.text(90, 478, "hollow markers at x = -0.12: matched random-perturbation sham (same RMS modification per lead and region as s = 0)", 11, c="#333")
    # panel C/D: other classes real vs sham
    classes = ["SB", "ST", "AF", "SA", "STTC", "TWC", "TWO"]
    ccol = ["#009E73", "#E69F00", "#CC79A7", "#999", "#555", "#222", "#56B4E9"]
    for pi, g in enumerate(("SR", "AFIB")):
        x0, y0 = 90 + pi * 730, 520
        vals = [eff[g][n]["dlogit"][c] for n in ("flat", "sham(matched to flat)") for c in classes if n in eff[g]]
        lo, hi = min(min(v[1] for v in vals), -0.2) * 1.2, max(max(v[2] for v in vals), 0.2) * 1.2
        f = fg.axes(s, x0, y0, 620, 290, (0, 2), (lo, hi), [], np.linspace(lo, hi, 5), "", "change in logit", f"{'C' if g == 'SR' else 'D'}. other classes, {g}-origin: flatten (left) vs matched sham (right)")
        s.line(*f(0, 0), *f(2, 0), c="#888", sw=1, dash="3,3")
        for li, n in enumerate(("flat", "sham(matched to flat)")):
            if n not in eff[g]:
                continue
            for j, c in enumerate(classes):
                m, l_, h_, nn = eff[g][n]["dlogit"][c]
                x1, y1 = f(li + 0.06 + j * 0.13, m)
                x2, y2 = f(li + 0.06 + j * 0.13 + 0.11, 0)
                s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), ccol[j], 0.9)
                s.line((x1 + x2) / 2, f(0, l_)[1], (x1 + x2) / 2, f(0, h_)[1], "#000", 1)
    for i, c in enumerate(classes):
        s.rect(110 + i * 125, 880, 10, 10, ccol[i])
        s.text(125 + i * 125, 889, c + (" (morph.)" if c in ("STTC", "TWC", "TWO") else ""), 10)
    s.rect(110, 920, 10, 10, col["SR"])
    s.text(125, 929, "SR-origin", 11)
    s.rect(240, 920, 10, 10, col["AFIB"])
    s.text(255, 929, "AFIB-origin", 11)
    s.save(os.path.join(out, "atrial_gain_and_sham.svg"))
    # F2 factorial SR + synthetic f-wave
    fa = res["factorial_SR_rr_cv0.2_x_atrial"]
    s = fg.Svg(1500, 620)
    s.text(20, 28, "SR records: RR variability moved to CV 0.2, then atrial region flattened or replaced by synthetic f-wave activity", 14, weight="bold")
    s.text(20, 46, f"{fa['n_records']} records. Natural AFIB records with CV 0.17-0.23 shown for reference (n={fa['natural_AFIB_records_cv_0.17_0.23']['n']}). Synthetic f-waves are an unrealistic sensitivity test (independent across leads).", 11, c="#333")
    for pi, k in enumerate(("AFIB", "SR")):
        x0 = 90 + pi * 730
        bars = [("original", fa[k]["orig_abs"], "#777"), ("RR only (CV 0.2)", fa[k]["rr_only_abs"], "#0072B2"), ("RR + flat atrial", fa[k]["rr_flat_abs"], "#56B4E9"), ("RR + flat + f-wave", fa[k]["rr_flat_fwave_abs"], "#009E73"),
                ("natural AFIB (CV~0.2)", fa["natural_AFIB_records_cv_0.17_0.23"][k + "_logit"], "#D55E00")]
        lo, hi = min(b[1][1] for b in bars) * 1.1, max(max(b[1][2] for b in bars) * 1.1, 1)
        f = fg.axes(s, x0, 90, 620, 360, (0, len(bars)), (lo, hi), [], np.linspace(lo, hi, 5), "", f"{k} logit (mean)", f"{'A' if k == 'AFIB' else 'B'}. absolute {k} logit")
        s.line(*f(0, 0), *f(len(bars), 0), c="#888", sw=1, dash="3,3")
        for i, (nm, v, c) in enumerate(bars):
            x1, y1 = f(i + 0.1, v[0])
            x2, y2 = f(i + 0.9, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), c, 0.9)
            s.line((x1 + x2) / 2, f(0, v[1])[1], (x1 + x2) / 2, f(0, v[2])[1], "#000", 1.2)
            s.text((x1 + x2) / 2, 90 + 360 + 15 + (i % 2) * 13, nm, 9, "middle")
    s.text(90, 520, f"share of the AFIB-logit gap (natural AFIB minus RR-only) closed by: flat atrial {fa['share_of_gap_closed']['rr_flat']:.2f}, flat + synthetic f-wave {fa['share_of_gap_closed']['rr_flat_fwave']:.2f}; AFIB-positive records: "+", ".join(f"{k} {100*v:.1f}%" for k,v in fa['label_flips_AFIB_positive_rate'].items()), 11)
    s.save(os.path.join(out, "factorial_rr_by_atrial.svg"))
    # F3 acceptance + selection by mean RR
    s = fg.Svg(1500, 560)
    s.text(20, 28, "Acceptance of atrial transformations by group and condition", 14, weight="bold")
    cs = ["flat", "g20", "flat_short", "flat_long", "sham_a", "fwave_a", "rr_flat"]
    f = fg.axes(s, 90, 80, 1340, 330, (0, len(cs)), (0, 1), [], [0, 0.25, 0.5, 0.75, 1], "", "share of records accepted", "A. accepted / records (insufficient eligible beats = the dominant rejection reason)")
    for ci_, c in enumerate(cs):
        for j, g in enumerate(("SR", "AFIB")):
            d = res["acceptance"][g].get(c)
            if not d:
                continue
            x1, y1 = f(ci_ + 0.1 + j * 0.4, d["acceptance_rate"])
            x2, y2 = f(ci_ + 0.1 + j * 0.4 + 0.35, 0)
            s.rect(x1, y1, x2 - x1, y2 - y1, col[g], 0.9)
            s.text((x1 + x2) / 2, y1 - 3, f"{d['accepted']}/{d['records_attempted']}", 8, "middle")
        s.text(f(ci_ + 0.5, 0)[0], 80 + 330 + 15, c, 10, "middle")
    sl = res["selection"]
    s.text(90, 470, "SR: mean RR (ms) accepted %.0f / ineligible %s;  AFIB: accepted %.0f / ineligible %.0f. Baseline AFIB logit AFIB accepted %.2f vs ineligible %.2f." % (
        sl["SR"]["mean_rr_ms"]["accepted"], "%.0f" % sl["SR"]["mean_rr_ms"]["ineligible"] if sl["SR"]["mean_rr_ms"]["ineligible"] else "n/a", sl["AFIB"]["mean_rr_ms"]["accepted"], sl["AFIB"]["mean_rr_ms"]["ineligible"],
        sl["AFIB"]["baseline_AFIB_logit"]["accepted"], sl["AFIB"]["baseline_AFIB_logit"]["ineligible"]), 11)
    s.rect(110, 520, 10, 10, col["SR"])
    s.text(125, 529, "SR", 11)
    s.rect(180, 520, 10, 10, col["AFIB"])
    s.text(195, 529, "AFIB", 11)
    s.save(os.path.join(out, "acceptance.svg"))
    # F4 example waveforms
    ex = np.load(os.path.join(run, "example_signals.npz"))
    recs_ex = sorted({k.split("|")[0] for k in ex.files})
    s = fg.Svg(1500, 1000)
    s.text(20, 28, "Representative ECGs (lead II, 3 s): original and modified; shaded = atrial regions (r-160..r-51); dotted = R peaks", 14, weight="bold")
    y = 60
    for r in recs_ex[:4]:
        g = info[r]["group"]
        b = np.array(info[r]["beats"], int)
        from ecg_concept import atrial_ops as ao
        regs, _ = ao.regions(b, 160)
        regs = [(a, e) for a, e in regs if e <= 1500]
        s.text(90, y, f"{r} ({g}); RR-CV {info[r]['cv']:.3f}; mean RR {info[r]['mean_rr'] * 2:.0f} ms", 11, weight="bold")
        for j, (cond, c_) in enumerate((("orig", "#000"), ("flat", "#0072B2"), ("sham_a", "#999"), ("g20", "#D55E00"))):
            k = f"{r}|{cond}|raw_orig" if cond == "orig" else f"{r}|{cond}|raw_mod"
            if k not in ex.files:
                continue
            sig = ex[k][1, :1500].astype(float)
            sc = 26 / (np.abs(sig).max() + 1e-9)
            yy = y + 10 + j * 52
            for a, e in regs:
                s.rect(90 + a / 1500 * 1300, yy, (e - a) / 1500 * 1300, 48, "#FFE0B2", 0.45)
            s.poly([(90 + i / 1500 * 1300, yy + 26 - sig[i] * sc) for i in range(0, 1500)], c_, 0.9)
            for bp in b:
                if bp < 1500:
                    s.line(90 + bp / 1500 * 1300, yy, 90 + bp / 1500 * 1300, yy + 48, "#aaa", 0.5, dash="2,3")
            s.text(82, yy + 28, cond, 9, "end")
        y += 235
    s.save(os.path.join(out, "waveform_examples.svg"))
    print(sorted(os.listdir(out)))
