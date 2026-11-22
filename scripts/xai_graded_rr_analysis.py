r"""Analysis of Experiment 7 (graded RR-variability interventions; see xai_graded_rr.py). Reads items.csv / logits.npy of a run directory; writes analysis_results.json, tables and SVG figures.
Unit of analysis = the ECG record: accepted patterns of one record at one level are averaged FIRST, then record-level values are bootstrapped (B=5000, seed 20261120), so the several synthetic patterns of one ECG
are never treated as independent. Two-sample differences (SR-origin vs AFIB-origin) resample the two disjoint record sets independently. Changes are always within-record (transformed - original logit)."""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import xai_figures_v2 as v2  # noqa: E402,F401  (figure-layer fixes)
import xai_ig_figures as fg  # noqa: E402

B, SEED = 5000, 20261120
RHY = ["AFIB", "SR", "SB", "ST", "AF", "SA", "AT", "SVT"]
MORPH = ["STTC", "TWC", "TWO"]
PN = ["iid_a", "iid_b", "smooth", "perm_orig"]


def bm(v, seed=SEED):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float("nan")] * 3 + [len(v)]
    idx = np.random.RandomState(seed).randint(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(v))]


def bdiff(a, b, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return [float("nan")] * 3 + [len(a), len(b)]
    rng = np.random.RandomState(seed)
    ma = a[rng.randint(0, len(a), (B, len(a)))].mean(1)
    mb = b[rng.randint(0, len(b), (B, len(b)))].mean(1)
    d = ma - mb
    return [float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(a), len(b)]


def analyze(run):
    names = [str(c) for c in np.load(os.path.join(ROOT, "results", "step4_4", "cache", "merged.npz"))["class_names"]]
    ci = {n: i for i, n in enumerate(names)}
    thr = np.load(os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0_thresh.npy"), allow_pickle=False)
    items = list(csv.DictReader(open(os.path.join(run, "items.csv"))))
    L = np.load(os.path.join(run, "logits.npy"))
    pop = json.load(open(os.path.join(run, "population.json")))
    info = json.load(open(os.path.join(run, "records_beats.json")))
    levels = ["own"] + list(pop["levels"])
    lvl_cv = pop["levels"]
    f = lambda s: float(s) if s not in ("", "nan") else float("nan")
    orig_row = {it["record_id"]: int(it["logit_row"]) for it in items if it["level"] == "orig"}
    grp = {r: v["group"] for r, v in info.items() if v["usable"]}
    recs = {g: sorted(r for r in grp if grp[r] == g) for g in ("SR", "AFIB")}
    T = {}
    for it in items:
        if it["level"] == "orig":
            continue
        T[(it["record_id"], it["level"], it["pattern"])] = it
    res = {"run": os.path.relpath(run, ROOT), "bootstrap": f"record-level (patterns averaged within record first), B={B}, seed={SEED}", "levels": pop["levels"], "population": pop["groups"], "n_records_usable": {g: len(v) for g, v in recs.items()}}

    # ---------------- selection / rejection analysis
    rej = {}
    for g in ("SR", "AFIB"):
        rej[g] = {}
        for lv in levels:
            its = [it for it in items if it["group"] == g and it["level"] == lv]
            att = len(its)
            acc = sum(it["accepted"] == "True" for it in its)
            reasons = {}
            for it in its:
                if it["accepted"] != "True":
                    reasons[it["reason"]] = reasons.get(it["reason"], 0) + 1
            rr = sorted({it["record_id"] for it in its})
            rec_any = sum(any(T[(r, lv, p)]["accepted"] == "True" for p in PN if (r, lv, p) in T) for r in rr)
            rej[g][lv] = {"records_eligible": len(rr), "transformations_attempted": att, "accepted": acc, "rejected": att - acc, "acceptance_rate": acc / att if att else None, "records_with_>=1_accepted_pattern": rec_any,
                          "reject_reasons": reasons}
    # acceptance by original-CV tercile / mean RR within each group (own level: no CV change, isolates the geometry of the record)
    sel = {}
    for g in ("SR", "AFIB"):
        cv0 = np.array([info[r]["cv"] for r in recs[g]])
        mrr = np.array([info[r]["mean_rr"] * 2 for r in recs[g]])
        frac = np.array([np.mean([T[(r, "own", p)]["accepted"] == "True" for p in PN if (r, "own", p) in T]) if any((r, "own", p) in T for p in PN) else np.nan for r in recs[g]])
        q = np.percentile(cv0, [33.3, 66.7])
        bins = np.digitize(cv0, q)
        sel[g] = {"own_level_acceptance_by_cv_tercile": {f"T{b + 1} (cv<={q[min(b, 1)]:.3f})" if b < 2 else f"T3 (cv>{q[1]:.3f})": float(np.nanmean(frac[bins == b])) for b in range(3)},
                  "own_level_acceptance_corr_with_mean_rr": float(np.corrcoef(frac[np.isfinite(frac)], mrr[np.isfinite(frac)])[0, 1]),
                  "mean_rr_ms_accepted_records(frac>=0.5)": float(np.mean(mrr[frac >= 0.5])), "mean_rr_ms_rejected_records(frac<0.5)": float(np.mean(mrr[frac < 0.5])) if (frac < 0.5).any() else None,
                  "cv0_accepted_records": float(np.mean(cv0[frac >= 0.5])), "cv0_rejected_records": float(np.mean(cv0[frac < 0.5])) if (frac < 0.5).any() else None,
                  "baseline_AFIB_logit_accepted": float(np.mean([L[orig_row[r]][ci['AFIB']] for r, fr in zip(recs[g], frac) if fr >= 0.5])),
                  "baseline_AFIB_logit_rejected": float(np.mean([L[orig_row[r]][ci['AFIB']] for r, fr in zip(recs[g], frac) if fr < 0.5])) if (frac < 0.5).any() else None,
                  "n_records_with_own_level_acceptance>=0.5": int((frac >= 0.5).sum()), "n_records": len(frac)}
    res["rejection"], res["selection_bias"] = rej, sel

    # ---------------- record-level arrays per (group, level)
    def rl(g, lv, patterns=PN):
        rows = []
        for r in recs[g]:
            ps = [p for p in patterns if (r, lv, p) in T and T[(r, lv, p)]["accepted"] == "True"]
            if ps:
                rows.append((r, ps))
        out = {"rec": [r for r, _ in rows], "n_pat": np.array([len(p) for _, p in rows]), "orig": np.array([L[orig_row[r]] for r, _ in rows]), "cv": np.array([np.mean([f(T[(r, lv, p)]["achieved_cv"]) for p in ps]) for r, ps in rows]),
               "cv0": np.array([info[r]["cv"] for r, _ in rows])}
        out["abs"] = np.array([np.mean([L[int(T[(r, lv, p)]["logit_row"])] for p in ps], axis=0) for r, ps in rows]) if rows else np.zeros((0, 45))
        out["d"] = out["abs"] - out["orig"] if rows else np.zeros((0, 45))
        out["flip"] = {c: np.array([np.mean([(1 / (1 + np.exp(-L[int(T[(r, lv, p)]['logit_row'])][ci[c]])) > thr[ci[c]]) != (1 / (1 + np.exp(-L[orig_row[r]][ci[c]])) > thr[ci[c]]) for p in ps]) for r, ps in rows]) for c in ("AFIB", "SR")}
        return out

    A = {(g, lv): rl(g, lv) for g in ("SR", "AFIB") for lv in levels}
    dose = {}
    for g in ("SR", "AFIB"):
        dose[g] = {}
        for lv in levels:
            a = A[(g, lv)]
            if len(a["rec"]) < 2:
                continue
            row = {"n_records": len(a["rec"]), "mean_accepted_patterns_per_record": float(a["n_pat"].mean()), "n_accepted_transformations": int(a["n_pat"].sum()), "target_cv": (lvl_cv.get(lv) if lv != "own" else "own"),
                   "achieved_cv_mean": float(a["cv"].mean()), "achieved_cv_sd": float(a["cv"].std()), "original_cv_mean": float(a["cv0"].mean()),
                   "dlogit": {c: bm(a["d"][:, ci[c]]) for c in RHY + MORPH}, "median_dlogit": {c: float(np.median(a["d"][:, ci[c]])) for c in ("AFIB", "SR", "SB", "ST")}, "abs_logit": {c: bm(a["abs"][:, ci[c]]) for c in ("AFIB", "SR")},
                   "orig_logit": {c: bm(a["orig"][:, ci[c]]) for c in ("AFIB", "SR")}, "label_flip_rate": {c: bm(a["flip"][c]) for c in ("AFIB", "SR")},
                   "share_records_dlogit_AFIB>0": float((a["d"][:, ci["AFIB"]] > 0).mean()), "share_records_dlogit_SR<0": float((a["d"][:, ci["SR"]] < 0).mean()), "share_records_dlogit_AFIB<0": float((a["d"][:, ci["AFIB"]] < 0).mean()),
                   "share_records_dlogit_SR>0": float((a["d"][:, ci["SR"]] > 0).mean())}
            own = {"SR": ci["SR"], "AFIB": ci["AFIB"]}[g]
            ok = a["orig"][:, own] > np.log(thr[own] / (1 - thr[own]))
            row["by_original_correctness(own label predicted positive)"] = {"correct": {"n": int(ok.sum()), "dlogit_AFIB": bm(a["d"][ok, ci["AFIB"]]), "dlogit_SR": bm(a["d"][ok, ci["SR"]])},
                                                                               "incorrect": {"n": int((~ok).sum()), "dlogit_AFIB": bm(a["d"][~ok, ci["AFIB"]]), "dlogit_SR": bm(a["d"][~ok, ci["SR"]])}}
            dose[g][lv] = row
    res["dose_response_within_record"] = dose
    # ---------------- matched CV: SR-origin vs AFIB-origin at the same level
    matched = {}
    for lv in levels:
        a, b = A[("SR", lv)], A[("AFIB", lv)]
        if len(a["rec"]) < 5 or len(b["rec"]) < 5:
            continue
        matched[lv] = {"n_SR": len(a["rec"]), "n_AFIB": len(b["rec"]), "achieved_cv_SR": float(a["cv"].mean()), "achieved_cv_AFIB": float(b["cv"].mean())}
        for c in ("AFIB", "SR", "SB", "ST") + tuple(MORPH):
            matched[lv][c] = {"abs_logit_SRorigin_minus_AFIBorigin_transformed": bdiff(a["abs"][:, ci[c]], b["abs"][:, ci[c]]), "baseline_logit_SRorigin_minus_AFIBorigin_original": bdiff(a["orig"][:, ci[c]], b["orig"][:, ci[c]]),
                              "dlogit_SRorigin_minus_AFIBorigin": bdiff(a["d"][:, ci[c]], b["d"][:, ci[c]])}
    res["matched_cv_between_groups"] = matched
    # ---------------- pattern consistency
    pat = {}
    for g in ("SR", "AFIB"):
        pat[g] = {}
        for lv in levels:
            pat[g][lv] = {}
            for p in PN:
                a = rl(g, lv, [p])
                if len(a["rec"]) >= 5:
                    pat[g][lv][p] = {"n_records": len(a["rec"]), "dlogit_AFIB": bm(a["d"][:, ci["AFIB"]]), "dlogit_SR": bm(a["d"][:, ci["SR"]]), "achieved_cv": float(a["cv"].mean())}
            for c in ("dlogit_AFIB", "dlogit_SR"):
                v = [x[c][0] for x in pat[g][lv].values() if isinstance(x, dict)]
                if len(v) > 1:
                    pat[g][lv][f"spread_of_pattern_means_{c}"] = float(max(v) - min(v))
    res["pattern_consistency"] = pat
    # ---------------- balanced-record sensitivity: records with >=1 accepted pattern at EVERY level they are eligible for
    bal = {}
    for g in ("SR", "AFIB"):
        elig = {r: [lv for lv in levels if any((r, lv, p) in T for p in PN)] for r in recs[g]}
        okr = [r for r in recs[g] if elig[r] and all(any(T[(r, lv, p)]["accepted"] == "True" for p in PN if (r, lv, p) in T) for lv in elig[r])]
        bal[g] = {"n_records_balanced": len(okr), "n_records_eligible": len(recs[g]), "dlogit": {}}
        for lv in levels:
            a = A[(g, lv)]
            m = np.array([r in set(okr) for r in a["rec"]])
            if m.sum() >= 5:
                bal[g]["dlogit"][lv] = {"n": int(m.sum()), "AFIB": bm(a["d"][m, ci["AFIB"]]), "SR": bm(a["d"][m, ci["SR"]])}
    res["balanced_records_sensitivity"] = bal
    json.dump(res, open(os.path.join(run, "analysis_results.json"), "x"), indent=1, default=float)
    print("analysis saved")
    figures(run, res, A, T, info, items, L, ci, names, pop, levels)


def figures(run, res, A, T, info, items, L, ci, names, pop, levels):
    out = os.path.join(run, "figures")
    os.makedirs(out)
    lv_cv = [pop["levels"][l] for l in levels[1:]]
    col = {"SR": "#0072B2", "AFIB": "#D55E00"}
    # F1: dose-response, four panels (within-record changes)
    s = fg.Svg(1500, 980)
    s.text(20, 28, "Within-record logit change as RR variability is moved to common levels (SR records: increased; AFIB records: reduced)", 14, weight="bold")
    s.text(20, 46, "Points = mean change of the original model's logit over records (patterns averaged within record first); whiskers = record-level bootstrap 95% CI. x = target RR-CV level; 'own' = CV unchanged, RR sequence replaced.", 11, c="#333")
    xs = {"own": -0.03, **{l: pop["levels"][l] for l in levels[1:]}}
    for pi, (cls_, title) in enumerate((("AFIB", "A. AFIB logit"), ("SR", "B. SR logit"), ("SB", "C. SB logit"), ("ST", "D. ST logit"))):
        x0, y0 = 90 + (pi % 2) * 730, 90 + (pi // 2) * 400
        allv = [A_ for g in ("SR", "AFIB") for lv in levels if lv in res["dose_response_within_record"][g] for A_ in res["dose_response_within_record"][g][lv]["dlogit"][cls_][1:3]]
        lo, hi = min(min(allv), -0.3) * 1.15, max(max(allv), 0.3) * 1.15
        f = fg.axes(s, x0, y0, 620, 290, (-0.06, 0.28), (lo, hi), [0, 0.02, 0.09, 0.2, 0.244], np.linspace(lo, hi, 5), "target RR-CV level (own = -0.03)", "change in logit", title)
        s.line(*f(-0.06, 0), *f(0.28, 0), c="#888", sw=1, dash="3,3")
        for g, off in (("SR", -0.0035), ("AFIB", 0.0035)):
            pts = []
            for lv in levels:
                d = res["dose_response_within_record"][g].get(lv)
                if not d:
                    continue
                m, l_, h_, n = d["dlogit"][cls_]
                xx = xs[lv] + off
                pts.append(f(xx, m))
                s.line(*f(xx, l_), *f(xx, h_), c=col[g], sw=1.6)
                s.add(f'<circle cx="{f(xx, m)[0]:.1f}" cy="{f(xx, m)[1]:.1f}" r="4" fill="{col[g]}"/>')
                s.text(f(xx, lo)[0], y0 + 290 - 4, f"n={n}", 7, "middle", c=col[g])
            s.poly(pts, col[g], 1.2, op=0.6)
    s.rect(110, 900, 10, 10, col["SR"])
    s.text(125, 909, "SR-origin records (variability increased)", 11)
    s.rect(420, 900, 10, 10, col["AFIB"])
    s.text(435, 909, "AFIB-origin records (variability reduced)", 11)
    s.save(os.path.join(out, "dose_response.svg"))
    # F2: matched-CV absolute logits for AFIB & SR classes, both groups
    s = fg.Svg(1500, 560)
    s.text(20, 28, "Absolute logits at matched RR-CV (transformed records) and at each record's original CV", 14, weight="bold")
    s.text(20, 46, "Between-group differences at the same CV combine the intervention and baseline differences between records; the within-record changes are in dose_response.svg.", 11, c="#333")
    for pi, (cls_, title) in enumerate((("AFIB", "AFIB logit"), ("SR", "SR logit"))):
        x0 = 90 + pi * 730
        vals = [res["dose_response_within_record"][g][lv]["abs_logit"][cls_] for g in ("SR", "AFIB") for lv in levels if lv in res["dose_response_within_record"][g]]
        orig = [res["dose_response_within_record"][g]["own"]["orig_logit"][cls_] for g in ("SR", "AFIB")]
        lo, hi = min(min(v[1] for v in vals), min(o[1] for o in orig)) * 1.1, max(max(v[2] for v in vals), max(o[2] for o in orig)) * 1.1
        f = fg.axes(s, x0, 90, 620, 330, (-0.06, 0.28), (lo, hi), [0, 0.02, 0.09, 0.2, 0.244], np.linspace(lo, hi, 5), "RR-CV (target level; 'own'/original at -0.03)", "logit", title)
        for g, off in (("SR", -0.0035), ("AFIB", 0.0035)):
            for lv in levels:
                d = res["dose_response_within_record"][g].get(lv)
                if not d:
                    continue
                m, l_, h_, n = d["abs_logit"][cls_]
                xx = xs[lv] + off
                s.line(*f(xx, l_), *f(xx, h_), c=col[g], sw=1.6)
                s.add(f'<circle cx="{f(xx, m)[0]:.1f}" cy="{f(xx, m)[1]:.1f}" r="4" fill="{col[g]}"/>')
            m, l_, h_, n = res["dose_response_within_record"][g]["own"]["orig_logit"][cls_]
            xo = -0.05 + off
            s.line(*f(xo, l_), *f(xo, h_), c=col[g], sw=1.6)
            s.add(f'<circle cx="{f(xo, m)[0]:.1f}" cy="{f(xo, m)[1]:.1f}" r="4" fill="white" stroke="{col[g]}" stroke-width="2"/>')
    s.text(110, 470, "filled = transformed (own: same CV, replaced sequence); hollow at x = -0.05 = untransformed originals (each group at its natural CV: SR ~0.02, AFIB ~0.20)", 10, c="#333")
    s.rect(110, 500, 10, 10, col["SR"])
    s.text(125, 509, "SR-origin", 11)
    s.rect(240, 500, 10, 10, col["AFIB"])
    s.text(255, 509, "AFIB-origin", 11)
    s.save(os.path.join(out, "matched_cv_absolute_logits.svg"))
    # F3: negative controls and other rhythm classes
    s = fg.Svg(1500, 640)
    s.text(20, 28, "Other rhythm classes and morphology negative controls: within-record logit change by level", 14, weight="bold")
    classes = ["SB", "ST", "AF", "SA", "STTC", "TWC", "TWO"]
    ccol = ["#009E73", "#E69F00", "#CC79A7", "#999", "#555", "#222", "#56B4E9"]
    for pi, g in enumerate(("SR", "AFIB")):
        x0 = 90 + pi * 730
        lvs = [lv for lv in levels if lv in res["dose_response_within_record"][g]]
        vals = [res["dose_response_within_record"][g][lv]["dlogit"][c][1:3] for lv in lvs for c in classes]
        lo, hi = min(min(v[0] for v in vals), -0.3) * 1.15, max(max(v[1] for v in vals), 0.3) * 1.15
        f = fg.axes(s, x0, 90, 620, 360, (0, len(lvs)), (lo, hi), [], np.linspace(lo, hi, 5), "", "change in logit", f"{g}-origin records")
        s.line(*f(0, 0), *f(len(lvs), 0), c="#888", sw=1, dash="3,3")
        for li, lv in enumerate(lvs):
            for j, c in enumerate(classes):
                m, l_, h_, n = res["dose_response_within_record"][g][lv]["dlogit"][c]
                xx = li + 0.06 + j * 0.13
                x1, y1 = f(xx, m)
                x2, y2 = f(xx + 0.11, 0)
                s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), ccol[j], 0.9)
                s.line((x1 + x2) / 2, f(0, l_)[1], (x1 + x2) / 2, f(0, h_)[1], "#000", 1)
            s.text(f(li + 0.5, 0)[0], 90 + 360 + 15, lv.split("(")[0] if lv != "own" else "own", 10, "middle")
    for i, c in enumerate(classes):
        s.rect(110 + i * 120, 520, 10, 10, ccol[i])
        s.text(125 + i * 120, 529, c + (" (morphology)" if c in MORPH else ""), 10)
    s.save(os.path.join(out, "other_classes_and_controls.svg"))
    # F4: RR sequences (original vs transformed) and rejection rates
    seqs = json.load(open(os.path.join(run, "all_beat_sequences.json")))
    ex = np.load(os.path.join(run, "example_signals.npz"))
    exrecs = sorted({k.split("|")[0] for k in ex.files})
    s = fg.Svg(1500, 1100)
    s.text(20, 28, "RR sequences before/after the interventions (examples), and rejection of transformations", 14, weight="bold")
    pat_col = {"iid_a": "#D55E00", "iid_b": "#E69F00", "smooth": "#009E73", "perm_orig": "#0072B2"}
    shown = 0
    for r in exrecs:
        g = info[r]["group"]
        lv = "L3_high(AFIB median)" if g == "SR" else "L1_low(SR median)"
        base = np.diff(np.array(seqs[f"{r}|orig|orig"])) * 2
        x0, y0 = 90 + (shown % 2) * 730, 80 + (shown // 2) * 250
        mx = max(base.max(), 1500) * 1.1
        f = fg.axes(s, x0, y0, 620, 190, (0, len(base) - 1), (0, mx), [0, 5, 10, 15], np.linspace(0, mx, 4).round(0), "interval index", "RR (ms)", f"{r} ({g}): original (black) vs {lv.split('(')[0]} (target CV {pop['levels'][lv]})")
        s.poly([f(i, v) for i, v in enumerate(base)], "#000", 2.2)
        for p, c_ in pat_col.items():
            k = f"{r}|{lv}|{p}"
            if k in seqs and seqs[k]:
                s.poly([f(i, v) for i, v in enumerate(np.diff(np.array(seqs[k])) * 2)], c_, 1.2)
        shown += 1
        if shown >= 4:
            break
    for i, (p, c_) in enumerate(pat_col.items()):
        s.line(110 + i * 130, 590, 130 + i * 130, 590, c_, 3)
        s.text(135 + i * 130, 594, p, 10)
    rj = res["rejection"]
    lvs = levels
    f = fg.axes(s, 90, 680, 1340, 280, (0, len(lvs)), (0, 1), [], [0, 0.25, 0.5, 0.75, 1], "", "acceptance rate of transformations", "Rejection by group and level (all rejected items are kept in items.csv; reasons in analysis_results.json)")
    for li, lv in enumerate(lvs):
        for j, g in enumerate(("SR", "AFIB")):
            d = rj[g].get(lv)
            if not d or d["acceptance_rate"] is None:
                continue
            x1, y1 = f(li + 0.1 + j * 0.4, d["acceptance_rate"])
            x2, y2 = f(li + 0.1 + j * 0.4 + 0.35, 0)
            s.rect(x1, y1, x2 - x1, y2 - y1, col[g], 0.9)
            s.text((x1 + x2) / 2, y1 - 3, f"{d['accepted']}/{d['transformations_attempted']}", 9, "middle")
        s.text(f(li + 0.5, 0)[0], 680 + 280 + 15, lv.split("(")[0], 10, "middle")
    s.rect(110, 1010, 10, 10, col["SR"])
    s.text(125, 1019, "SR", 11)
    s.rect(180, 1010, 10, 10, col["AFIB"])
    s.text(195, 1019, "AFIB", 11)
    s.save(os.path.join(out, "rr_sequences_and_rejection.svg"))
    # F5: representative waveforms (lead II, first 5 s)
    s = fg.Svg(1500, 900)
    s.text(20, 28, "Representative ECGs: original and transformed (lead II, first 5 s; dashed lines = beat positions)", 14, weight="bold")
    y = 70
    for r in exrecs[:4]:
        g = info[r]["group"]
        lv = "L3_high(AFIB median)" if g == "SR" else "L1_low(SR median)"
        keys = [("orig", f"{r}|orig|orig|raw_orig")] + [(p, f"{r}|{lv}|{p}|raw_warped") for p in ("iid_a", "perm_orig")]
        s.text(90, y, f"{r} ({g}); CV original {info[r]['cv']:.3f}; transformed to target {pop['levels'][lv]}", 11, weight="bold")
        for j, (nm, k) in enumerate(keys):
            if k not in ex.files:
                continue
            sig = ex[k][1, :2500].astype(float)
            sc = 30 / (np.abs(sig).max() + 1e-9)
            yy = y + 12 + j * 60
            s.poly([(90 + i / 2500 * 1300, yy + 30 - sig[i] * sc) for i in range(0, 2500, 2)], "#000" if nm == "orig" else pat_col[nm], 0.9)
            bs = seqs[f"{r}|orig|orig"] if nm == "orig" else seqs.get(f"{r}|{lv}|{nm}")
            for bp in (bs or []):
                if bp < 2500:
                    s.line(90 + bp / 2500 * 1300, yy, 90 + bp / 2500 * 1300, yy + 55, "#aaa", 0.5, dash="2,3")
            s.text(82, yy + 32, nm, 9, "end")
        y += 205
    s.save(os.path.join(out, "waveform_examples.svg"))
    print(sorted(os.listdir(out)))
