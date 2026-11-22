r"""Evaluation stage of Experiment 6 (see xai_rr_timing.py): unchanged frozen LightECGNetV2 + original preprocessing on every ACCEPTED transformed record; the explained class is the original
predicted class of each pair (fixed). Statistics: paired, record-clustered bootstrap (157 pairs / 154 records; a record can carry several class pairs)."""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.preprocessing import preprocess  # noqa: E402
import xai_attention_analysis as xa  # noqa: E402
import xai_ig_faithfulness as xf  # noqa: E402
import xai_ig_figures as fg  # noqa: E402
import xai_rr_timing as xt  # noqa: E402

CONDS, RHYTHM, MORPH = xt.CONDS, xt.RHYTHM, xt.MORPH
sha_file = xf.sha_file


def wcorr(x, y, w):
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3 or w.sum() == 0:
        return np.nan
    mx, my = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    c = (w * (x - mx) * (y - my)).sum()
    d = np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())
    return float(c / d) if d > 0 else np.nan


def evaluate(run, outd=None):
    outd = run if outd is None else outd
    os.makedirs(outd, exist_ok=True)
    log = open(os.path.join(outd, "eval_log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    pairs, beats = xt.load_inputs()
    z = np.load(os.path.join(run, "warped_raw_int16.npz"))
    recs, W = [str(r) for r in z["records"]], z["data"]
    q = {}
    for r in csv.DictReader(open(os.path.join(run, "quality.csv"))):
        q[(r["record_id"], r["condition"])] = r
    keep = [i for i, p in enumerate(pairs) if p["record_id"] in set(recs)]
    pairs = [pairs[i] for i in keep]
    n = len(pairs)
    rec = np.array([p["record_id"] for p in pairs])
    cls = np.array([p["class"] for p in pairs])
    cidx = np.array([p["class_idx"] for p in pairs])
    corr = np.array([p["correct"] for p in pairs])
    cl = xa.Clust(rec)
    rix = {r: i for i, r in enumerate(recs)}
    assert xf.sha_file(xf.CKPT) == xf.CKPT_SHA
    dev = "cuda"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = load_model(xf.CKPT, dev)
    h0 = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    thr = np.load(xf.THR, allow_pickle=False)
    # ---- logits for every (record, condition): accepted ones only are used later
    LG = np.full((len(recs), len(CONDS), 45), np.nan)
    t0 = time.time()
    with torch.no_grad():
        for ri in range(len(recs)):
            xs = np.stack([preprocess(W[ri, ci]) for ci in range(len(CONDS))])
            LG[ri] = model(torch.from_numpy(xs).to(dev)).cpu().numpy().astype(np.float64)
    say(f"model run on {len(recs) * len(CONDS)} transformed/untransformed records in {time.time() - t0:.0f}s")
    assert h0 == hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest() and sha_file(xf.CKPT) == xf.CKPT_SHA
    f0 = np.array([LG[rix[rec[k]], 0, cidx[k]] for k in range(n)])
    assert np.abs(f0 - np.array([p["f_x"] for p in pairs])).max() < 1e-3, "original condition does not reproduce the Experiment 1 logits"
    say(f"condition 'orig' reproduces the stored logits (max abs diff {np.abs(f0 - np.array([p['f_x'] for p in pairs])).max():.1e})")
    # ---- per pair/condition quantities
    acc = {c: np.array([q[(rec[k], c)]["accepted"] == "True" for k in range(n)]) for c in CONDS}
    inf = {c: np.array([q[(rec[k], c)]["informative"] == "True" for k in range(n)]) for c in CONDS}
    cv_b = np.array([float(q[(rec[k], "orig")]["cv_before"]) for k in range(n)])
    cv_a = {c: np.array([float(q[(rec[k], c)].get("cv_after") or np.nan) for k in range(n)]) for c in CONDS}
    prob0 = 1 / (1 + np.exp(-f0))
    pos0 = prob0 > thr[cidx]
    d_logit, d_prob, flip, d_rank, f_c = {}, {}, {}, {}, {}
    for c, ci in zip(CONDS, range(len(CONDS))):
        f = np.array([LG[rix[rec[k]], ci, cidx[k]] for k in range(n)])
        f_c[c] = f
        d_logit[c] = f - f0
        d_prob[c] = 1 / (1 + np.exp(-f)) - prob0
        flip[c] = (pos0 & ~(1 / (1 + np.exp(-f)) > thr[cidx])).astype(float)
        rank0 = np.array([1 + (LG[rix[rec[k]], 0] > LG[rix[rec[k]], 0, cidx[k]]).sum() for k in range(n)])
        rankc = np.array([1 + (LG[rix[rec[k]], ci] > LG[rix[rec[k]], ci, cidx[k]]).sum() for k in range(n)])
        d_rank[c] = (rankc - rank0).astype(float)
    use = {c: acc[c] & inf[c] for c in CONDS if c != "orig"}                                    # accepted AND informative (the beats actually moved)
    groups = {"ALL": np.ones(n, bool), "correct": corr, "incorrect": ~corr, "rhythm(AFIB,AF,SR,SB,ST)": np.isin(cls, RHYTHM), "morphology(STTC,TWC,TWO)": np.isin(cls, MORPH), **{c: cls == c for c in sorted(set(cls))}}
    res = {"n_pairs": n, "n_unique_records": int(cl.R), "bootstrap": f"record-clustered, B={xa.BOOT_B}, seed={xa.BOOT_SEED}", "conditions": {}, "contrasts": {}, "dose_response": {}}
    for c in CONDS[1:]:
        sel0 = use[c]
        row = {"n_pairs_accepted": int(acc[c].sum()), "n_pairs_accepted_informative": int(sel0.sum()), "n_records_accepted_informative": int(len(set(rec[sel0]))), "n_pairs_rejected": int((~acc[c]).sum()), "groups": {}}
        for g, gm in groups.items():
            sel = sel0 & gm
            if sel.sum() < 2:
                continue
            row["groups"][g] = {"n_pairs": int(sel.sum()), "n_records": int(len(set(rec[sel]))), "mean_dlogit": cl.mean_ci(d_logit[c], sel), "median_dlogit": float(np.median(d_logit[c][sel])), "mean_abs_dlogit": cl.mean_ci(np.abs(d_logit[c]), sel),
                                "mean_dprob": cl.mean_ci(d_prob[c], sel), "label_flip_rate": cl.mean_ci(flip[c], sel), "mean_rank_change": cl.mean_ci(d_rank[c], sel), "share_logit_decrease": float((d_logit[c][sel] < 0).mean()),
                                "mean_cv_before": float(np.mean(cv_b[sel])), "mean_cv_after": float(np.nanmean(cv_a[c][sel]))}
        res["conditions"][c] = row
    pairs_contrasts = [("reg0", "sham_smooth_B"), ("reg0", "sham_noise_B"), ("shuffle", "sham_smooth_C"), ("shuffle", "sham_noise_C"), ("reg0", "shuffle"), ("reg50", "reg0"), ("reg50", "sham_smooth_B")]
    for a_, b_ in pairs_contrasts:
        sel0 = use[a_] & use[b_]
        r_ = {"n_pairs": int(sel0.sum()), "n_records": int(len(set(rec[sel0]))), "groups": {}}
        for g in ("ALL", "correct", "incorrect", "rhythm(AFIB,AF,SR,SB,ST)", "morphology(STTC,TWC,TWO)", "AFIB", "AF", "SR", "SB", "ST", "STTC", "TWC", "TWO"):
            sel = sel0 & groups[g]
            if sel.sum() < 3:
                continue
            r_["groups"][g] = {"n_pairs": int(sel.sum()), "signed_dlogit_difference(a-b)": cl.mean_ci(d_logit[a_] - d_logit[b_], sel), "abs_dlogit_difference(|a|-|b|)": cl.mean_ci(np.abs(d_logit[a_]) - np.abs(d_logit[b_]), sel),
                               "label_flip_rate_difference": cl.mean_ci(flip[a_] - flip[b_], sel)}
        res["contrasts"][f"{a_} vs {b_}"] = r_
    # dose-response: does a larger change in RR variability / larger baseline irregularity go with a larger logit change?
    for c in ("reg0", "reg50"):
        sel = use[c]
        x_cv = cv_a[c] - cv_b
        for nm, xx, sel_g in (("dCV(after-before) vs dlogit, ALL", x_cv, sel), ("dCV vs dlogit, rhythm classes", x_cv, sel & np.isin(cls, RHYTHM)), ("dCV vs dlogit, AF+AFIB", x_cv, sel & np.isin(cls, ("AF", "AFIB"))),
                              ("baseline CV vs |dlogit|, ALL", cv_b, sel)):
            yy = d_logit[c] if "|dlogit|" not in nm else np.abs(d_logit[c])
            w0 = sel_g.astype(float)
            ests = [wcorr(xx, yy, cl.counts[b][cl.inv] * w0) for b in range(0, len(cl.counts), 5)]
            res["dose_response"][f"{c}: {nm}"] = {"n_pairs": int(sel_g.sum()), "pearson": wcorr(xx, yy, w0), "ci95": [float(np.nanpercentile(ests, 2.5)), float(np.nanpercentile(ests, 97.5))]}
    res["notes"] = ("'informative' = RMS beat displacement >= 3 samples; accepted = morphology/discontinuity checks of xai_rr_timing.py passed. dlogit = f_c(transformed) - f_c(original) for the ORIGINAL predicted class. "
                    "Accepted AF/AFIB transformations exclude records whose short RR intervals force uniform stretching (selection effect).")
    json.dump(res, open(os.path.join(outd, "eval_results.json"), "x"), indent=1, default=float)
    with open(os.path.join(outd, "eval_pairs.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "record_id", "class", "correct", "condition", "accepted", "informative", "logit", "dlogit", "dprob", "label_flip", "drank", "cv_before", "cv_after"])
        for k in range(n):
            for c in CONDS:
                w.writerow([k, rec[k], cls[k], corr[k], c, bool(acc[c][k]), bool(inf[c][k]), f_c[c][k], d_logit[c][k], d_prob[c][k], flip[c][k], d_rank[c][k], cv_b[k], cv_a[c][k]])
    json.dump({"checkpoint_sha256": xf.CKPT_SHA, "build_quality_summary_sha256": sha_file(os.path.join(run, "quality_summary.json")), "warped_sha256": sha_file(os.path.join(run, "warped_raw_int16.npz")),
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("scripts/xai_rr_timing_eval.py", "scripts/xai_rr_timing.py", "src/ecg_concept/rr_warp.py")}}, open(os.path.join(outd, "eval_config.json"), "x"), indent=1)
    say(f"eval saved; results sha256 {sha_file(os.path.join(outd, 'eval_results.json'))[:16]}")
    figures(run, outd, res, rec, cls, corr, use, d_logit, f_c, cv_b, cv_a, recs, W, beats, pairs, n, groups)


def figures(run, outd, res, rec, cls, corr, use, d_logit, f_c, cv_b, cv_a, recs, W, beats, pairs, n, groups):
    out = os.path.join(outd, "figures_eval")
    os.makedirs(out)
    cl_ = [c for c in sorted(set(cls))]
    cc = {"reg0": "#0072B2", "shuffle": "#D55E00", "sham_smooth_B": "#999", "sham_noise_B": "#444"}
    # 1. per-class mean dlogit and mean |dlogit|
    s = fg.Svg(1500, 1000)
    s.text(20, 28, "Change of the explained class logit under RR-timing interventions (original predicted class fixed; accepted, informative transformations)", 14, weight="bold")
    s.text(20, 46, f"{res['n_pairs']} class-record pairs / {res['n_unique_records']} records in total; bars = pair means per class, whiskers = record-clustered 95% CI; sham_*_B is magnitude-matched to reg0.", 11, c="#333")
    for pi, (key, title) in enumerate((("mean_dlogit", "A. mean signed change in logit"), ("mean_abs_dlogit", "B. mean absolute change in logit"))):
        vals = {(c, g): res["conditions"][c]["groups"][g][key] for c in cc for g in cl_ if g in res["conditions"][c]["groups"]}
        lo, hi = min(0, min(v[1] for v in vals.values())) * 1.15, max(0.05, max(v[2] for v in vals.values())) * 1.15
        f = fg.axes(s, 90, 100 + pi * 420, 1340, 300, (0, len(cl_)), (lo, hi), [], np.linspace(lo, hi, 5), "", "logit units", title)
        for gi, g in enumerate(cl_):
            for j, c in enumerate(cc):
                if (c, g) not in vals:
                    continue
                m, l_, h_ = vals[(c, g)]
                x1, y1 = f(gi + 0.05 + j * 0.22, m)
                x2, y2 = f(gi + 0.05 + j * 0.22 + 0.2, 0)
                s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), cc[c], 0.9)
                s.line((x1 + x2) / 2, f(0, l_)[1], (x1 + x2) / 2, f(0, h_)[1], "#000", 1.1)
            s.text(f(gi + 0.5, 0)[0], 100 + pi * 420 + 316, g + (" (rhythm)" if g in RHYTHM else " (morph.)"), 10, "middle")
    for i, c in enumerate(cc):
        s.rect(110 + i * 200, 945, 12, 12, cc[c])
        s.text(128 + i * 200, 956, c, 11)
    s.save(os.path.join(out, "per_class_logit_change.svg"))
    # 2. actual vs sham, rhythm vs morphology groups (signed and absolute)
    s = fg.Svg(1500, 560)
    s.text(20, 28, "Actual interventions vs magnitude-matched shams: rhythm classes vs morphology classes", 14, weight="bold")
    s.text(20, 46, "Bars: mean |change in logit| (record-clustered 95% CI). Shams change beat positions by a comparable amount but do not specifically regularise (reg0) or reorder (shuffle) the RR sequence; they do raise RR variability somewhat.", 11, c="#333")
    conds = ["reg50", "reg0", "sham_smooth_B", "sham_noise_B", "shuffle", "sham_smooth_C", "sham_noise_C"]
    ccol = {"reg50": "#56B4E9", "reg0": "#0072B2", "sham_smooth_B": "#999", "sham_noise_B": "#444", "shuffle": "#D55E00", "sham_smooth_C": "#BBB", "sham_noise_C": "#222"}
    gl = ["rhythm(AFIB,AF,SR,SB,ST)", "morphology(STTC,TWC,TWO)", "AFIB", "AF"]
    vals = {(c, g): res["conditions"][c]["groups"][g]["mean_abs_dlogit"] for c in conds for g in gl if g in res["conditions"][c]["groups"]}
    hi = max(v[2] for v in vals.values()) * 1.15
    f = fg.axes(s, 90, 90, 1340, 330, (0, len(gl)), (0, hi), [], np.linspace(0, hi, 5), "", "mean |change in logit|", None)
    for gi, g in enumerate(gl):
        for j, c in enumerate(conds):
            if (c, g) not in vals:
                continue
            m, l_, h_ = vals[(c, g)]
            x1, y1 = f(gi + 0.04 + j * 0.135, m)
            x2, y2 = f(gi + 0.04 + j * 0.135 + 0.12, 0)
            s.rect(x1, y1, x2 - x1, y2 - y1, ccol[c], 0.9)
            s.line((x1 + x2) / 2, f(0, l_)[1], (x1 + x2) / 2, f(0, h_)[1], "#000", 1.1)
        s.text(f(gi + 0.5, 0)[0], 90 + 346, g, 11, "middle")
    for i, c in enumerate(conds):
        s.rect(110 + i * 190, 470, 12, 12, ccol[c])
        s.text(128 + i * 190, 481, c, 11)
    s.save(os.path.join(out, "actual_vs_sham.svg"))
    # 3. dose-response and example waveforms
    s = fg.Svg(1500, 1150)
    s.text(20, 28, "Dependence on baseline RR irregularity, and examples", 14, weight="bold")
    f = fg.axes(s, 90, 90, 560, 330, (0, 0.5), (-4, 3), [0, 0.1, 0.2, 0.3, 0.4, 0.5], [-4, -3, -2, -1, 0, 1, 2, 3], "baseline RR CV of the record", "change in logit", "C. change in logit vs baseline RR variability (accepted, informative)")
    s.line(*f(0, 0), *f(0.5, 0), c="#888", sw=1, dash="3,3")
    for c in ("reg0", "shuffle"):
        sel = use[c]
        for k in np.where(sel)[0]:
            x, y = f(min(cv_b[k], 0.5), np.clip(d_logit[c][k], -4, 3))
            s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{3.3 if cls[k] in ("AF", "AFIB") else 2.2}" fill="{cc[c]}" fill-opacity="0.6" stroke="{"#000" if cls[k] in ("AF", "AFIB") else "none"}" stroke-width="0.6"/>')
    s.rect(110, 462, 10, 10, cc["reg0"])
    s.text(125, 471, "reg0", 10)
    s.rect(180, 462, 10, 10, cc["shuffle"])
    s.text(195, 471, "shuffle", 10)
    s.text(260, 471, "outlined = AF (flutter) / AFIB (fibrillation) pairs", 10)
    s._clip = None
    dr = res["dose_response"]
    for i, (k, v) in enumerate(dr.items()):
        s.text(700, 110 + i * 18, f"{k}: r = {v['pearson']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] (n = {v['n_pairs']})", 11)
    # example waveforms: largest |dlogit| under reg0 among AFIB/AF, and one SR
    cand = [k for k in np.where(use["reg0"] & np.isin(cls, ("AFIB", "AF")))[0]]
    ex = [max(cand, key=lambda k: abs(d_logit["reg0"][k]))] if cand else []
    cand = [k for k in np.where(use["reg0"] & (cls == "SR"))[0]]
    ex += [min(cand, key=lambda k: abs(d_logit["reg0"][k] - np.median(d_logit["reg0"][cand])))] if cand else []
    y0 = 540
    for e_ in ex:
        ri = recs.index(rec[e_])
        s.text(90, y0 - 8, f"{rec[e_]} ({cls[e_]}, label {'present' if corr[e_] else 'absent'}): lead II, first 5 s. logit original {f_c['orig'][e_]:.2f}; reg0 {f_c['reg0'][e_]:.2f}; shuffle {f_c['shuffle'][e_]:.2f}"
               + (f"; RR CV {cv_b[e_]:.2f} -> {cv_a['reg0'][e_]:.3f} (reg0)" if np.isfinite(cv_a['reg0'][e_]) else ""), 11, weight="bold")
        for j, c in enumerate(("orig", "reg0", "shuffle")):
            ci = xt.CONDS.index(c)
            sig = W[ri, ci, 1, :2500].astype(float)
            sc = 38 / (np.abs(sig).max() + 1e-9)
            yy = y0 + 20 + j * 62
            s.poly([(90 + i / 2500 * 1300, yy + 28 - sig[i] * sc) for i in range(0, 2500, 2)], {"orig": "#000", "reg0": "#0072B2", "shuffle": "#D55E00"}[c], 0.9)
            nb = json.load(open(os.path.join(run, "new_beats.json")))[rec[e_]][c]
            for bp in nb:
                if bp < 2500:
                    s.line(90 + bp / 2500 * 1300, yy - 4, 90 + bp / 2500 * 1300, yy + 60, "#999", 0.5, dash="2,3")
            s.text(82, yy + 30, c, 10, "end")
        y0 += 230
    s.save(os.path.join(out, "dose_response_and_examples.svg"))
    print(sorted(os.listdir(out)))
