r"""Experiment 6: controlled RR-timing interventions on the frozen original LightECGNetV2 (counterfactual sensitivity; not clinical simulation).
Stages:  build <>   -> creates results/xai_rr_timing/run_<stamp>/: beat-anchored time warps of the RAW int16 records (src/ecg_concept/rr_warp.py), quality metrics, accept/reject, figures.
         evaluate <run_dir> -> runs the unchanged model (original preprocessing) on every accepted transformed record and compares logits/probabilities/labels.
Conditions per record (8): orig | reg50 (RR deviations from the mean halved) | reg0 (all RR equal to the mean) | shuffle (RR order permuted) | sham_smooth_B / sham_noise_B (magnitude-matched to reg0:
smooth beat displacement / random RR perturbation of the same relative size) | sham_smooth_C / sham_noise_C (matched to shuffle). The first and last beat never move: beat count, order and total duration are preserved.
Predeclared acceptance (per record and condition): QRS windows bit-identical; median post-QRS (150 samples) and pre-QRS (50 samples = the rigid P-wave region) segment correlation >= 0.80; max junction ratio (|second difference| / p99 of the
original) <= 5; 99.9th percentile of |first difference| not above 1.5x the original; all new RR >= 100 samples. "Informative" = RMS beat displacement >= 3 samples (6 ms).
Beats: Experiment 5 detections (frozen Step 4.3 extractor; automatic, not expert-validated). Population: same 160 pairs; records with >= 3 detected beats. Validation role only.
Usage: .venv\Scripts\python.exe -I scripts\xai_rr_timing.py build   |   ... evaluate <run_dir>
"""
import csv
import datetime
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ecg_concept import rr_warp as rw  # noqa: E402
from ecg_concept.preprocessing import preprocess, read_raw_mat  # noqa: E402
import xai_attention_analysis as xa  # noqa: E402
import xai_figures_v2 as v2  # noqa: E402,F401  (applies the figure-layer fixes)
import xai_ig_faithfulness as xf  # noqa: E402
import xai_ig_figures as fg  # noqa: E402

CONDS = ["orig", "reg50", "reg0", "shuffle", "sham_smooth_B", "sham_noise_B", "sham_smooth_C", "sham_noise_C"]
SEED = 20261120
RHYTHM, MORPH = ("AFIB", "AF", "SR", "SB", "ST"), ("STTC", "TWC", "TWO")
EXP5 = sorted(glob.glob(os.path.join(ROOT, "results", "xai_beats", "run_*")))[-1]
sha_file = xf.sha_file
RIGID = {"rigid_post": 150, "rigid_pre": 50}
ACC = {"post": 0.80, "pre": 0.80, "junction": 5.0, "p999": 1.5, "min_rr": 100}


def load_inputs():
    pairs = list(csv.DictReader(open(os.path.join(xa.PREV, "pairs.csv"))))
    for p in pairs:
        p["class_idx"] = int(p["class_idx"])
        p["correct"] = p["correct"] == "True"
        p["f_x"] = float(p["f_x"])
    bj = json.load(open(os.path.join(EXP5, "beats.json")))
    beats = {r: np.array(v["beat_samples"], int) for r, v in bj.items()}
    return pairs, beats


def build():
    pairs, beats = load_inputs()
    recs = sorted({p["record_id"] for p in pairs if len(beats[p["record_id"]]) >= 3})
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(xf.CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    run = os.path.join(ROOT, "results", "xai_rr_timing", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run)
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    W = np.zeros((len(recs), len(CONDS), 12, 5000), np.int16)
    rows, newbeats = [], {}
    for ri, r in enumerate(recs):
        raw = read_raw_mat(os.path.join(xf.DS, rel[r]))
        assert raw.shape == (12, 5000)
        b = beats[r]
        nb = {"orig": b, "reg50": rw.regularise(b, 0.5), "reg0": rw.regularise(b, 0.0), "shuffle": rw.shuffle_rr(b, np.random.RandomState(SEED + 100 * ri + 3))}
        tB = (rw.displacement_rms(b, nb["reg0"]), rw.rel_rr_change_rms(b, nb["reg0"]))
        tC = (rw.displacement_rms(b, nb["shuffle"]), rw.rel_rr_change_rms(b, nb["shuffle"]))
        for tag, (td, tr_) in (("B", tB), ("C", tC)):
            nb[f"sham_smooth_{tag}"] = b.copy() if td < 0.5 else rw.sham_smooth(b, td, np.random.RandomState(SEED + 100 * ri + (5 if tag == "B" else 7)))
            nb[f"sham_noise_{tag}"] = b.copy() if tr_ < 1e-4 else rw.sham_noise(b, tr_, np.random.RandomState(SEED + 100 * ri + (6 if tag == "B" else 8)))
        newbeats[r] = {c: (None if nb[c] is None else nb[c].tolist()) for c in CONDS}
        s0 = rw.rr_stats(b)
        for ci, c in enumerate(CONDS):
            row = {"record_id": r, "condition": c, "n_beats": len(b), "cv_before": s0["cv"], "mean_rr_before": s0["mean"]}
            if nb[c] is None or not rw.valid_beats(nb[c]):
                row.update(accepted=False, reason="invalid_beat_positions" if nb[c] is not None else "no_valid_sham_found", informative=False)
                W[ri, ci] = raw
                rows.append(row)
                continue
            y = rw.to_int16(rw.warp(raw, b, nb[c], **RIGID)) if c != "orig" else raw
            W[ri, ci] = y
            q = rw.quality(raw, y, b, nb[c], pre=50, **RIGID) if c != "orig" else rw.quality(raw, raw, b, b, pre=50, **RIGID)
            s1 = rw.rr_stats(nb[c])
            ok = (q["qrs_windows_identical"] and not (q["post_qrs_corr_median"] < ACC["post"]) and not (q["pre_qrs_corr_median"] < ACC["pre"]) and q["junction_ratio_max"] <= ACC["junction"]
                  and q["p999_abs_diff_ratio"] <= ACC["p999"] and np.diff(nb[c]).min() >= ACC["min_rr"])
            reason = "" if ok else ";".join(k for k, bad in (("qrs_changed", not q["qrs_windows_identical"]), ("post_corr", q["post_qrs_corr_median"] < ACC["post"]), ("pre_corr", q["pre_qrs_corr_median"] < ACC["pre"]),
                                                              ("junction", q["junction_ratio_max"] > ACC["junction"]), ("p999", q["p999_abs_diff_ratio"] > ACC["p999"]), ("min_rr", np.diff(nb[c]).min() < ACC["min_rr"])) if bad)
            disp = rw.displacement_rms(b, nb[c])
            row.update(q, cv_after=s1["cv"], mean_rr_after=s1["mean"], rmssd_before=s0["rmssd"], rmssd_after=s1["rmssd"], disp_rms=disp, rel_rr_rms=rw.rel_rr_change_rms(b, nb[c]), accepted=bool(ok), reason=reason,
                       informative=bool(disp >= 3.0), first_last_fixed=bool(nb[c][0] == b[0] and nb[c][-1] == b[-1]),
                       rr_multiset_equal=bool(sorted(np.diff(nb[c]).tolist()) == sorted(np.diff(b).tolist())), n_beats_equal=bool(len(nb[c]) == len(b)))
            rows.append(row)
        if ri % 30 == 0:
            say(f"built {ri}/{len(recs)} records")
    np.savez_compressed(os.path.join(run, "warped_raw_int16.npz"), data=W, records=np.array(recs), conditions=np.array(CONDS))
    json.dump(newbeats, open(os.path.join(run, "new_beats.json"), "x"))
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(run, "quality.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summ = {"n_records": len(recs), "n_pairs": int(sum(p["record_id"] in set(recs) for p in pairs)), "acceptance_rule": ACC, "conditions": {}}
    for c in CONDS:
        rc = [r for r in rows if r["condition"] == c]
        acc = [r for r in rc if r["accepted"]]
        inf = [r for r in acc if r["informative"]]
        d = {"n_records": len(rc), "accepted": len(acc), "rejected": len(rc) - len(acc), "informative_accepted": len(inf), "reject_reasons": {}}
        for r in rc:
            if not r["accepted"]:
                for x_ in str(r["reason"]).split(";"):
                    d["reject_reasons"][x_] = d["reject_reasons"].get(x_, 0) + 1
        if inf:
            g = lambda k: [r[k] for r in inf if np.isfinite(r[k])]
            d.update({"cv_before_mean": float(np.mean(g("cv_before"))), "cv_after_mean": float(np.mean(g("cv_after"))), "share_cv_reduced": float(np.mean([r["cv_after"] < r["cv_before"] - 1e-9 for r in inf if np.isfinite(r["cv_after"])])),
                      "share_cv_increased": float(np.mean([r["cv_after"] > r["cv_before"] + 1e-9 for r in inf if np.isfinite(r["cv_after"])])),
                      "mean_rr_abs_change_samples_max": float(np.max([abs(r["mean_rr_after"] - r["mean_rr_before"]) for r in inf])), "first_last_fixed_all": bool(all(r["first_last_fixed"] for r in inf)),
                      "n_beats_equal_all": bool(all(r["n_beats_equal"] for r in inf)), "rr_multiset_equal_share": float(np.mean([r["rr_multiset_equal"] for r in inf])),
                      "disp_rms_median_samples": float(np.median(g("disp_rms"))), "rel_rr_rms_median": float(np.median(g("rel_rr_rms"))), "post_qrs_corr_median": float(np.median(g("post_qrs_corr_median"))),
                      "post_qrs_corr_p5": float(np.percentile(g("post_qrs_corr_median"), 5)), "pre_qrs_corr_median": float(np.median(g("pre_qrs_corr_median"))), "junction_ratio_max_median": float(np.median(g("junction_ratio_max"))),
                      "junction_ratio_max_p95": float(np.percentile(g("junction_ratio_max"), 95)), "std_ratio_range": [float(np.min(g("std_ratio_min"))), float(np.max(g("std_ratio_max")))],
                      "rms_log_stretch_median": float(np.median(g("rms_log_stretch"))), "stretch_factor_range": [float(np.min(g("stretch_factor_min"))), float(np.max(g("stretch_factor_max")))], "share_segments_uniform_fallback": float(np.sum([r["n_uniform_fallback_segments"] for r in inf]) / np.sum([r["n_segments"] for r in inf]))})
        summ["conditions"][c] = d
        say(f"{c}: accepted {len(acc)}/{len(rc)} (informative {len(inf)}), reasons {d['reject_reasons']}" + (f"; CV {d['cv_before_mean']:.3f}->{d['cv_after_mean']:.3f}, post-QRS corr median {d['post_qrs_corr_median']:.3f}" if inf else ""))
    json.dump(summ, open(os.path.join(run, "quality_summary.json"), "x"), indent=1, default=float)
    json.dump({"previous_runs": [os.path.relpath(xa.PREV, ROOT), os.path.relpath(EXP5, ROOT)], "seed": SEED, "half_width": rw.HALF, "rigid_regions_samples": RIGID, "acceptance": ACC, "checkpoint_sha256": xf.CKPT_SHA,
               "beats_json_sha256": sha_file(os.path.join(EXP5, "beats.json")), "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("src/ecg_concept/rr_warp.py", "scripts/xai_rr_timing.py")}},
              open(os.path.join(run, "config.json"), "x"), indent=1)
    quality_figure(run, recs, beats, newbeats, W, rows, pairs)
    say(f"build done: {run}")
    print(run)


def quality_figure(run, recs, beats, newbeats, W, rows, pairs):
    out = os.path.join(run, "figures")
    os.makedirs(out)
    cls_of = {}
    for p in pairs:
        cls_of.setdefault(p["record_id"], p["class"])
    s = fg.Svg(1500, 1020)
    s.text(20, 28, "Quality of the RR-timing interventions (beat-anchored warps of the raw record; QRS windows copied rigidly)", 15, weight="bold")
    # A: CV before/after
    show = [c for c in CONDS if c != "orig"]
    colr = {"reg50": "#56B4E9", "reg0": "#0072B2", "shuffle": "#D55E00", "sham_smooth_B": "#999", "sham_noise_B": "#555", "sham_smooth_C": "#CCC", "sham_noise_C": "#222"}
    f = fg.axes(s, 90, 80, 420, 330, (0, 0.6), (0, 0.6), [0, 0.2, 0.4, 0.6], [0, 0.2, 0.4, 0.6], "RR CV before", "RR CV after", "A. RR variability before vs after")
    s.line(*f(0, 0), *f(0.6, 0.6), c="#888", sw=1, dash="3,3")
    for c in show:
        for r in rows:
            if r["condition"] == c and r.get("accepted") and np.isfinite(r.get("cv_after", np.nan)):
                x, y = f(min(r["cv_before"], 0.6), min(r["cv_after"], 0.6))
                s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{colr[c]}" fill-opacity="0.55"/>')
    for i, c in enumerate(show):
        s.rect(540 + (i % 2) * 150, 100 + (i // 2) * 18, 10, 10, colr[c])
        s.text(555 + (i % 2) * 150, 109 + (i // 2) * 18, c, 10)
    # B: post-QRS corr by condition (strip)
    f = fg.axes(s, 90 + 720, 80, 640, 330, (0, len(show)), (0.5, 1.02), [], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "", "post-QRS segment correlation", "B. Morphology preservation of the stretched segments (per record, median over beats)")
    rng = np.random.RandomState(0)
    for i, c in enumerate(show):
        for r in rows:
            if r["condition"] == c and r.get("accepted") is not None and np.isfinite(r.get("post_qrs_corr_median", np.nan)):
                x, y = f(i + 0.5 + rng.uniform(-0.3, 0.3), max(0.5, r["post_qrs_corr_median"]))
                s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{colr[c]}" fill-opacity="{0.7 if r["accepted"] else 0.9}" stroke="{"none" if r["accepted"] else "#c00"}"/>')
        s.text(f(i + 0.5, 0.5)[0], 80 + 330 + 16, c, 8, "middle")
    s.line(*f(0, 0.8), *f(len(show), 0.8), c="#c00", sw=1, dash="4,3")
    s.text(f(0.1, 0.8)[0], f(0, 0.8)[1] - 4, "acceptance threshold 0.80 (red outline = rejected)", 9, c="#c00")
    # C: tachogram of an AFIB record, D: waveform
    ex = next((r for r in recs if cls_of.get(r) == "AFIB" and all(newbeats[r][c] for c in ("reg0", "shuffle"))), recs[0])
    ri = recs.index(ex)
    b = np.array(beats[ex])
    rrs = {c: np.diff(np.array(newbeats[ex][c])) * 2 for c in ("orig", "reg0", "shuffle", "sham_smooth_B") if newbeats[ex][c]}
    f = fg.axes(s, 90, 480, 420, 230, (0, len(b) - 1), (0, max(max(v) for v in rrs.values()) * 1.1), [0, 5, 10], np.linspace(0, max(max(v) for v in rrs.values()) * 1.1, 5).round(0), "beat interval index", "RR (ms)", f"C. RR tachogram of {ex} (AFIB)", yfmt="{:.0f}")
    for c, col in (("orig", "#000"), ("reg0", "#0072B2"), ("shuffle", "#D55E00"), ("sham_smooth_B", "#999")):
        if c in rrs:
            s.poly([f(i, v) for i, v in enumerate(rrs[c])], col, 1.8)
    for i, (c, col) in enumerate((("original", "#000"), ("reg0", "#0072B2"), ("shuffle", "#D55E00"), ("sham_smooth_B", "#999"))):
        s.line(110 + i * 105, 752, 130 + i * 105, 752, col, 3)
        s.text(135 + i * 105, 756, c, 10)
    s._clip = None                                                                      # panel D is drawn outside the axes of panel C
    x0, w_ = 90 + 720, 640
    t0, t1 = 0, 2500
    s.text(x0, 470, f"D. Lead II of {ex}: original (black) and transformed (colour), first 5 s; markers = beat positions", 12, weight="bold")
    for k, (c, col) in enumerate((("orig", "#000"), ("reg0", "#0072B2"), ("shuffle", "#D55E00"))):
        y0 = 500 + k * 80
        sig = W[ri, CONDS.index(c), 1, t0:t1].astype(float)
        sc = 60 / (np.abs(sig).max() + 1e-9)
        s.poly([(x0 + i / (t1 - t0) * w_, y0 + 35 - sig[i] * sc * 0.5) for i in range(0, t1 - t0, 2)], col, 0.9)
        for bp in newbeats[ex][c]:
            if bp < t1:
                s.line(x0 + bp / (t1 - t0) * w_, y0, x0 + bp / (t1 - t0) * w_, y0 + 70, col, 0.6, dash="2,3")
        s.text(x0 - 5, y0 + 38, c, 10, "end")
    s.text(x0, 760, "x axis: 0-5 s of the transformed record; the QRS windows are identical samples, only the spacing between them changes", 10, c="#333")
    summ = json.load(open(os.path.join(run, "quality_summary.json")))
    y = 800
    s.text(90, y, "Quality summary (informative = RMS beat displacement >= 3 samples)", 12, weight="bold")
    for i, c in enumerate(CONDS[1:]):
        d = summ["conditions"][c]
        if "cv_after_mean" in d:
            s.text(90, y + 22 + i * 17, f"{c}: accepted {d['accepted']}/{d['n_records']} (informative {d['informative_accepted']}); CV {d['cv_before_mean']:.3f} -> {d['cv_after_mean']:.3f}; post-QRS corr median {d['post_qrs_corr_median']:.2f} (5th pct {d['post_qrs_corr_p5']:.2f}); "
                   f"junction ratio median {d['junction_ratio_max_median']:.2f}; rejected {d['rejected']}", 11)
        else:
            s.text(90, y + 22 + i * 17, f"{c}: accepted {d['accepted']}/{d['n_records']}, no informative transformations", 11)
    s.save(os.path.join(out, "quality_overview.svg"))


def regen_quality_figure(run):
    """rebuild figures/quality_overview.svg (into <run>/v2/figures) from the SAVED build outputs."""
    pairs, beats = load_inputs()
    z = np.load(os.path.join(run, "warped_raw_int16.npz"))
    recs, W = [str(r) for r in z["records"]], z["data"]
    newbeats = json.load(open(os.path.join(run, "new_beats.json")))
    rows = []
    for r in csv.DictReader(open(os.path.join(run, "quality.csv"))):
        d = {}
        for k, val in r.items():
            if val in ("True", "False"):
                d[k] = val == "True"
            else:
                try:
                    d[k] = float(val)
                except ValueError:
                    d[k] = val
        rows.append(d)
    view = os.path.join(run, "v2")
    os.makedirs(view)
    import shutil
    shutil.copy(os.path.join(run, "quality_summary.json"), os.path.join(view, "quality_summary.json"))
    quality_figure(view, recs, beats, newbeats, W, rows, pairs)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    elif len(sys.argv) > 2 and sys.argv[1] == "regen_quality_figure":
        regen_quality_figure(os.path.abspath(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "evaluate":
        import xai_rr_timing_eval as ev
        ev.evaluate(os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        sys.exit(__doc__)
