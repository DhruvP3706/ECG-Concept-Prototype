"""Experiment 5: does single-class IG (Experiment 1, same 160 pairs / 157 records) line up with ALGORITHMICALLY detected beats?
Beats: frozen Step 4.3 extractor (src/ecg_concept/rr_concepts.py, parameters unchanged), run on the RAW int16 record exactly as in the PhysioNet pipeline; consensus beat times (mean of the two
detectors' matched times) x 500 Hz = sample indices into the same 5000-sample record the IG maps use (the preprocessing neither crops nor pads and its filter is zero-phase).
Regions (see src/ecg_concept/beat_regions.py): QRS-centred windows [r-40, r+40] (+-80 ms) around each detected R peak; RR regions = inter-beat samples outside those windows; outside = before the
first / after the last detected beat. These are regions around AUTOMATIC detections, not verified QRS complexes. Controls: circular within-record shifts of the region masks (same sizes, same
spacing) and random duration-matched windows. Perturbation: deletion of 81-sample tiles (top-attributed QRS tiles vs top-attributed RR tiles vs random matched tiles) measuring the target logit.
Statistics: record-clustered bootstrap. Validation role only; no model is trained or modified. Run: .venv\\Scripts\\python.exe -I scripts\\xai_beat_attribution.py -> results/xai_beats/run_<stamp>/
"""
import csv
import datetime
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
from ecg_concept import attribution as at  # noqa: E402
from ecg_concept import beat_regions as br  # noqa: E402
from ecg_concept import rr_concepts as rc  # noqa: E402
from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.preprocessing import load_ecg, read_raw_mat  # noqa: E402
import xai_attention_analysis as xa  # noqa: E402
import xai_ig_faithfulness as xf  # noqa: E402

PREV = xa.PREV
HALF, N_CTRL, MIN_BEATS = 40, 200, 3
M_LIST, N_RAND_PERT = (2, 4, 8), 20
FROZEN = os.path.join(ROOT, "results", "step4_3", "extraction_config.json")
sha_file = xf.sha_file
RHYTHM = ("AFIB", "AF", "SR", "SB", "ST")


def verify_extractor():
    import json as _j
    fz = _j.load(open(FROZEN))
    for k, v in fz["source_sha256"].items():
        assert sha_file(os.path.join(ROOT, k)) == v, f"extractor source changed since freeze: {k}"
    assert hashlib.sha256(_j.dumps(rc.CONFIG, sort_keys=True).encode()).hexdigest() == fz["parameters_sha256"], "extractor parameters changed"
    return fz["parameters_sha256"]


def main():
    cfg = json.load(open(os.path.join(PREV, "config.json")))
    assert sha_file(os.path.join(PREV, "results.json")).startswith(xa.PREV_RESULTS_PREFIX)
    params_sha = verify_extractor()
    run = os.path.join(ROOT, "results", "xai_beats", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run)
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    pairs = list(csv.DictReader(open(os.path.join(PREV, "pairs.csv"))))
    for p in pairs:
        p["class_idx"] = int(p["class_idx"])
        p["correct"] = p["correct"] == "True"
    A0 = np.load(os.path.join(PREV, "attributions.npz"))["attr"].astype(np.float64)
    rec = np.array([p["record_id"] for p in pairs])
    cls = np.array([p["class"] for p in pairs])
    cidx = np.array([p["class_idx"] for p in pairs])
    corr = np.array([p["correct"] for p in pairs])
    n = len(pairs)
    cl = xa.Clust(rec)
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(xf.CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    assert all(r in rel for r in rec)
    # ---- Step 2: beats for the unique records (raw record -> extractor)
    ext = {}
    for r in cl.u:
        raw = read_raw_mat(os.path.join(xf.DS, rel[str(r)]))
        assert raw.shape == (12, 5000), (r, raw.shape)
        res = rc.extract_record(raw, rc.CONFIG, return_beats=True)
        cons = np.asarray(res["beats"].get("consensus", []), float) * rc.CONFIG["fs"]
        ext[str(r)] = {"beats": np.round(cons).astype(int), "n_A": len(res["beats"].get("A", [])), "n_B": len(res["beats"].get("B", [])), "rate_state": res["rate_state"], "irr_state": res["irr_state"],
                       "reasons": res["reason_codes"], "rate_mean_bpm": float(res.get("rate_mean_bpm_cons", np.nan))}
    beats = {r: v["beats"] for r, v in ext.items()}
    usable_rec = {r for r, b in beats.items() if len(b) >= MIN_BEATS}
    inrange = all(((b >= 0) & (b < 5000)).all() for b in beats.values())
    rrs = np.concatenate([np.diff(b) for b in beats.values() if len(b) >= 2]) if beats else np.array([])
    ext_summary = {"n_unique_records": len(ext), "usable_records_(>=3_consensus_beats)": len(usable_rec), "records_with_0_or_1_beat": int(sum(len(b) < 2 for b in beats.values())),
                   "rate_state_counts": {s: int(sum(v["rate_state"] == s for v in ext.values())) for s in ("ACCEPTED", "UNCERTAIN", "UNAVAILABLE")},
                   "irr_state_counts": {s: int(sum(v["irr_state"] == s for v in ext.values())) for s in ("ACCEPTED", "UNCERTAIN", "UNAVAILABLE")},
                   "reason_counts": {}, "all_beats_within_record": bool(inrange), "all_rr_positive": bool((rrs > 0).all()),
                   "rr_ms_percentiles_0_1_50_99_100": (np.percentile(rrs * 2, [0, 1, 50, 99, 100]).round(0).tolist() if len(rrs) else None), "rr_below_300ms": int((rrs * 2 < 300).sum()),
                   "rr_above_2000ms": int((rrs * 2 > 2000).sum()), "n_rr_total": int(len(rrs)), "extractor_params_sha256": params_sha}
    for v in ext.values():
        for c_ in str(v["reasons"]).split(";"):
            if c_:
                ext_summary["reason_counts"][c_] = ext_summary["reason_counts"].get(c_, 0) + 1
    usable = np.array([str(r) in usable_rec for r in rec])
    accepted = np.array([ext[str(r)]["rate_state"] == "ACCEPTED" and str(r) in usable_rec for r in rec])
    ext_summary.update({"usable_pairs": int(usable.sum()), "usable_pairs_by_class": {c: int((usable & (cls == c)).sum()) for c in sorted(set(cls))}, "accepted_and_usable_pairs": int(accepted.sum()),
                        "usable_pairs_correct": int((usable & corr).sum()), "usable_pairs_incorrect": int((usable & ~corr).sum())})
    say(f"extractor: {len(usable_rec)}/{len(ext)} records with >= {MIN_BEATS} consensus beats; usable pairs {int(usable.sum())}/{n}; states {ext_summary['rate_state_counts']}; RR percentiles(ms) {ext_summary['rr_ms_percentiles_0_1_50_99_100']}")
    # ---- signals / model
    sc_ = {}
    sig = lambda r: sc_.setdefault(r, load_ecg(os.path.join(xf.DS, rel[r])))
    X = np.stack([sig(r) for r in rec])
    assert X.shape == (n, 12, 5000)
    donor_ids = cfg["donor_record_ids"]
    D = np.stack([sig(i) for i in donor_ids])
    donor_of = [D[(k + 1) % len(D)] if donor_ids[(k + 1) % len(D)] != rec[k] else D[(k + 2) % len(D)] for k in range(n)]
    dev = "cuda"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    assert sha_file(xf.CKPT) == xf.CKPT_SHA
    model = load_model(xf.CKPT, dev)
    h0 = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    fx = np.array([float(p["f_x"]) for p in pairs])
    # ---- Step 4/5: attribution distribution and controls
    keys = ("abs", "pos", "neg")
    obs = {k: {r: np.full(n, np.nan) for r in ("qrs", "rr", "out")} for k in keys}
    shf = {k: {r: np.full(n, np.nan) for r in ("qrs", "rr", "out")} for k in keys}
    rnd = {k: np.full(n, np.nan) for k in keys}
    dur = {r: np.full(n, np.nan) for r in ("qrs", "rr", "out")}
    sigsh = {r: np.full(n, np.nan) for r in ("qrs", "rr", "out")}
    per_beat, per_rr, nb = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    for k in range(n):
        if not usable[k]:
            continue
        b = beats[str(rec[k])]
        reg = br.regions_from_beats(b, 5000, HALF)
        absm, pos, neg = br.time_mass(A0[k])
        mass = {"abs": absm, "pos": pos, "neg": -neg}
        d = br.duration_shares(reg)
        for r in d:
            dur[r][k] = d[r]
        sx = np.abs(X[k]).sum(0)
        s_ = br.shares(sx, reg)
        for r in s_:
            sigsh[r][k] = s_[r]
        for kk in keys:
            s = br.shares(mass[kk], reg)
            for r in s:
                obs[kk][r][k] = s[r]
        rng = np.random.RandomState(5000 + k)
        shifts = rng.randint(250, 4750, N_CTRL)
        acc = {kk: {r: [] for r in ("qrs", "rr", "out")} for kk in keys}
        for sft in shifts:
            sh = br.shifted(reg, int(sft))
            for kk in keys:
                s = br.shares(mass[kk], sh)
                for r in s:
                    acc[kk][r].append(s[r])
        for kk in keys:
            for r in ("qrs", "rr", "out"):
                shf[kk][r][k] = float(np.mean(acc[kk][r]))
        rw = [br.random_windows(len(b), 2 * HALF + 1, 5000, rng) for _ in range(N_CTRL)]
        for kk in keys:
            rnd[kk][k] = float(np.mean([mass[kk][m].sum() / mass[kk].sum() for m in rw]))
        nb[k] = len(b)
        per_beat[k] = float(absm[reg["qrs"]].sum() / len(b))
        n_rr = max(len(b) - 1, 1)
        per_rr[k] = float(absm[reg["rr"]].sum() / n_rr)
    groups = {"ALL": np.ones(n, bool), "correct": corr, "incorrect": ~corr, **{c: cls == c for c in sorted(set(cls))}, "rhythm_classes(AFIB,AF,SR,SB,ST)": np.isin(cls, RHYTHM), "ST-T/morphology(STTC,TWC,TWO)": np.isin(cls, ("STTC", "TWC", "TWO"))}
    dist = {}
    for gname, gm in groups.items():
        sel = usable & gm
        if not sel.any():
            continue
        row = {"n_pairs": int(sel.sum()), "n_records": int(len(set(rec[sel]))), "duration_share": {r: float(np.nanmean(dur[r][sel])) for r in dur}, "signal_abs_amplitude_share": {r: float(np.nanmean(sigsh[r][sel])) for r in sigsh}}
        for kk in keys:
            row[f"{kk}_share"] = {r: cl.mean_ci(obs[kk][r], sel) for r in ("qrs", "rr", "out")}
            row[f"{kk}_share_shift_control"] = {r: cl.mean_ci(shf[kk][r], sel) for r in ("qrs", "rr", "out")}
            row[f"{kk}_share_minus_shift_control"] = {r: cl.mean_ci(obs[kk][r] - shf[kk][r], sel) for r in ("qrs", "rr", "out")}
            row[f"{kk}_qrs_share_minus_random_windows"] = cl.mean_ci(obs[kk]["qrs"] - rnd[kk], sel)
        row["qrs_enrichment_abs(share/duration share)"] = cl.mean_ci(obs["abs"]["qrs"] / dur["qrs"], sel)
        row["qrs_amplitude_adjusted_enrichment(attr share/|signal| share)"] = cl.mean_ci(obs["abs"]["qrs"] / sigsh["qrs"], sel)
        row["rr_enrichment_abs"] = cl.mean_ci(obs["abs"]["rr"] / dur["rr"], sel)
        row["mean_abs_mass_per_beat"] = float(np.nanmean(per_beat[sel]))
        row["mean_abs_mass_per_RR_interval"] = float(np.nanmean(per_rr[sel]))
        row["mean_beats"] = float(nb[sel].mean())
        dist[gname] = row
    say("attribution distribution done")
    sens = {}
    for half in (25, 60):                                                                   # region-width sensitivity (QRS abs share minus shift control)
        diffs = np.full(n, np.nan)
        for k in range(n):
            if usable[k]:
                reg = br.regions_from_beats(beats[str(rec[k])], 5000, half)
                absm = br.time_mass(A0[k])[0]
                rng = np.random.RandomState(7000 + k)
                c_ = np.mean([br.shares(absm, br.shifted(reg, int(s)))["qrs"] for s in rng.randint(250, 4750, 100)])
                diffs[k] = br.shares(absm, reg)["qrs"] - c_
        sens[f"half_width_{half}_samples"] = cl.mean_ci(diffs, usable)
    # ---- Step 6: targeted deletions
    zero = np.zeros((12, 5000), np.float32)
    fams = ("top_qrs", "top_rr", "top_any", "rand_qrs", "rand_rr", "rand_any")
    pert = {(m, b): {f: np.full(n, np.nan) for f in fams} for m in M_LIST for b in ("zero", "donor")}
    tiles_any = [(s, s + 2 * HALF + 1) for s in range(0, 5000 - 2 * HALF, 2 * HALF + 1)]
    t1 = time.time()
    for k in range(n):
        if not usable[k]:
            continue
        b = beats[str(rec[k])]
        tq, tr = br.qrs_tiles(b, 5000, HALF), br.rr_tiles(b, 5000, HALF)
        sc = lambda tl: np.array([A0[k][:, s:e].sum() for s, e in tl])
        sq, sr, sa = sc(tq), sc(tr), sc(tiles_any)
        prng = np.random.RandomState(9000 + k)
        for m in M_LIST:
            if m > min(len(tq), len(tr)):
                continue

            def mk(tl, idx):
                mt = np.zeros(5000, bool)
                for i in idx:
                    mt[tl[i][0]:tl[i][1]] = True
                return mt

            sets = {"top_qrs": [mk(tq, np.argsort(-sq)[:m])], "top_rr": [mk(tr, np.argsort(-sr)[:m])], "top_any": [mk(tiles_any, np.argsort(-sa)[:m])],
                    "rand_qrs": [mk(tq, prng.choice(len(tq), m, replace=False)) for _ in range(N_RAND_PERT)], "rand_rr": [mk(tr, prng.choice(len(tr), m, replace=False)) for _ in range(N_RAND_PERT)],
                    "rand_any": [mk(tiles_any, prng.choice(len(tiles_any), m, replace=False)) for _ in range(N_RAND_PERT)]}
            order = [(f, i) for f in fams for i in range(len(sets[f]))]
            masks = np.stack([np.broadcast_to(sets[f][i][None, :], (12, 5000)) for f, i in order])
            for bname, base in (("zero", zero), ("donor", donor_of[k])):
                lg = at.perturbed_logits(model, X[k], base, int(cidx[k]), masks, "delete") - fx[k]
                for f in fams:
                    pert[(m, bname)][f][k] = float(np.mean([lg[j] for j, (ff, _) in enumerate(order) if ff == f]))
        if k % 40 == 0:
            say(f"  deletions: pair {k}/{n} ({time.time() - t1:.0f}s)")
    assert h0 == hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest() and sha_file(xf.CKPT) == xf.CKPT_SHA
    say("model hash unchanged")
    pres = {}
    for (m, bname), P in pert.items():
        sel = np.isfinite(P["top_qrs"])
        d = lambda a, b: P[a] - P[b]
        pres[f"m={m}|{bname}"] = {"n_pairs": int(sel.sum()), "n_records": int(len(set(rec[sel]))), "mean_delta_logit": {f: cl.mean_ci(P[f], sel) for f in fams},
                                  "top_qrs_minus_rand_qrs": cl.mean_ci(d("top_qrs", "rand_qrs"), sel), "top_rr_minus_rand_rr": cl.mean_ci(d("top_rr", "rand_rr"), sel),
                                  "top_qrs_minus_top_rr": cl.mean_ci(d("top_qrs", "top_rr"), sel), "top_qrs_minus_rand_any": cl.mean_ci(d("top_qrs", "rand_any"), sel),
                                  "top_any_minus_rand_any": cl.mean_ci(d("top_any", "rand_any"), sel), "top_qrs_minus_top_any": cl.mean_ci(d("top_qrs", "top_any"), sel),
                                  "per_class_top_qrs_minus_rand_qrs": {c: float(np.nanmean(d("top_qrs", "rand_qrs")[sel & (cls == c)])) for c in sorted(set(cls)) if (sel & (cls == c)).any()},
                                  "per_class_top_rr_minus_rand_rr": {c: float(np.nanmean(d("top_rr", "rand_rr")[sel & (cls == c)])) for c in sorted(set(cls)) if (sel & (cls == c)).any()},
                                  "per_class_top_qrs_minus_top_rr": {c: float(np.nanmean(d("top_qrs", "top_rr")[sel & (cls == c)])) for c in sorted(set(cls)) if (sel & (cls == c)).any()},
                                  "correct_top_qrs_minus_rand_qrs": cl.mean_ci(d("top_qrs", "rand_qrs"), sel & corr), "incorrect_top_qrs_minus_rand_qrs": cl.mean_ci(d("top_qrs", "rand_qrs"), sel & ~corr)}
    res = {"extractor": ext_summary, "regions": {"qrs_half_width_samples": HALF, "qrs_window_samples": 2 * HALF + 1, "tile_samples": 2 * HALF + 1, "controls": f"{N_CTRL} circular shifts in [250, 4750) and {N_CTRL} random non-overlapping windows per pair",
                                                "n_perturbation_random_draws": N_RAND_PERT}, "bootstrap": f"record-clustered, B={xa.BOOT_B}, seed={xa.BOOT_SEED}, percentile 95% CI",
           "attribution_distribution": dist, "width_sensitivity_abs_qrs_share_minus_shift_control": sens, "targeted_deletion": pres,
           "notes": "abs share = share of sum_l |IG|; pos/neg shares use the positive part and the magnitude of the negative part separately; enrichment = share / duration share (1 = proportional)."}
    json.dump(res, open(os.path.join(run, "results.json"), "x"), indent=1, default=float)
    with open(os.path.join(run, "beats.json"), "x") as f:
        json.dump({r: {"beat_samples": v["beats"].tolist(), "n_A": v["n_A"], "n_B": v["n_B"], "rate_state": v["rate_state"], "irr_state": v["irr_state"], "reasons": v["reasons"]} for r, v in ext.items()}, f)
    with open(os.path.join(run, "pair_table.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "record_id", "class", "correct", "usable", "n_beats", "abs_share_qrs", "abs_share_rr", "abs_share_out", "dur_share_qrs", "dur_share_rr", "pos_share_qrs", "neg_share_qrs"])
        for k in range(n):
            w.writerow([k, rec[k], cls[k], corr[k], bool(usable[k]), nb[k], obs["abs"]["qrs"][k], obs["abs"]["rr"][k], obs["abs"]["out"][k], dur["qrs"][k], dur["rr"][k], obs["pos"]["qrs"][k], obs["neg"]["qrs"][k]])
    json.dump({"previous_run": os.path.relpath(PREV, ROOT), "checkpoint_sha256": xf.CKPT_SHA, "extractor_params_sha256": params_sha, "ig_source": "Experiment 1 attributions.npz (not recomputed)",
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("scripts/xai_beat_attribution.py", "src/ecg_concept/beat_regions.py", "src/ecg_concept/rr_concepts.py", "src/ecg_concept/beat_detection.py")}},
              open(os.path.join(run, "config.json"), "x"), indent=1)
    say(f"saved; results sha256 {sha_file(os.path.join(run, 'results.json'))[:16]}")
    figures(run, X, A0, rec, cls, corr, usable, accepted, beats, ext, res, obs, shf, dur, groups, pert, cl, pairs, fx)
    print(run)


def figures(run, X, A0, rec, cls, corr, usable, accepted, beats, ext, res, obs, shf, dur, groups, pert, cl, pairs, fx):
    import xai_figures_v2 as v2                                                             # figure-layer fixes found by rendering SVGs
    fg = v2.fg
    out = os.path.join(run, "figures")
    os.makedirs(out)
    L, Rr = 70, 1010
    chosen = []
    for c in RHYTHM:
        k = next((i for i in range(len(rec)) if cls[i] == c and corr[i] and accepted[i]), None)
        if k is not None:
            chosen.append((k, f"first correct pair of {c} with ACCEPTED beat detection"))
    k = next((i for i in range(len(rec)) if (not corr[i]) and usable[i]), None)
    if k is not None:
        chosen.append((k, "first incorrect prediction with usable beats"))
    for k, note in chosen:
        b = beats[str(rec[k])]
        reg = br.regions_from_beats(b, 5000, HALF)
        s = fg.Svg(1400, 800)
        s.text(20, 26, f"{rec[k]}: IG for {cls[k]} (model-predicted; label {'PRESENT' if corr[k] else 'ABSENT'}); {note}", 14, weight="bold")
        absm, pos, neg = br.time_mass(A0[k])
        sh = br.shares(absm, reg)
        dd = br.duration_shares(reg)
        s.text(20, 45, f"{len(b)} detected R peaks (automatic; extractor rate state {ext[str(rec[k])]['rate_state']}); |IG| share: QRS-centred {100 * sh['qrs']:.0f}% (duration {100 * dd['qrs']:.0f}%), "
               f"RR regions {100 * sh['rr']:.0f}% ({100 * dd['rr']:.0f}%), outside {100 * sh['out']:.0f}% ({100 * dd['out']:.0f}%)", 11, c="#333")
        top, rowh = 100, 34
        xs = lambda smp: L + smp / 5000 * (Rr - L)
        # strip: QRS windows, R markers, RR labels
        s.text(L, 64, "Detected beats: dark = QRS-centred window (+-80 ms), triangle = automatic R-peak, number = RR interval (ms)", 11, weight="bold")
        for (a, e) in br.qrs_tiles(b, 5000, HALF):
            s.rect(xs(a), 84, xs(e) - xs(a), 8, "#444", 0.85)
        for r0, r1 in zip(b[:-1], b[1:]):
            if xs(r1) - xs(r0) > 26:
                s.text((xs(r0) + xs(r1)) / 2, 90, f"{int((r1 - r0) * 2)}", 8, "middle", c="#0072B2")
        for r_ in b:
            s.poly([(xs(r_) - 3, 78), (xs(r_) + 3, 78), (xs(r_), 83)], "#D55E00", 1, "#D55E00")
        blocks = A0[k].reshape(12, 250, 20).sum(2)
        vmax = np.percentile(np.abs(blocks), 99.5) + 1e-12
        cw = (Rr - L) / 250
        for l in range(12):
            y0 = top + rowh * l
            for i in range(250):
                s.rect(L + i * cw, y0, cw + 0.3, rowh, fg.div_color(blocks[l, i] / vmax), 0.55)
            s.text(L - 8, y0 + rowh / 2 + 4, fg.LEADS[l], 11, "end", weight="bold")
            seg = X[k][l].astype(float).reshape(1250, -1)
            pts = []
            for i in range(1250):
                xx = L + i / 1250 * (Rr - L)
                pts += [(xx, y0 + rowh / 2 - np.clip(seg[i].min(), -4, 4) * rowh * 0.1), (xx, y0 + rowh / 2 - np.clip(seg[i].max(), -4, 4) * rowh * 0.1)]
            s.poly(pts, "#111", 0.8)
        for r_ in b:
            s.line(xs(r_), top, xs(r_), top + 12 * rowh, "#D55E00", 0.5, dash="2,3", op=0.7)
        s.rect(L, top, Rr - L, 12 * rowh, "none", stroke="#444")
        yb = top + 12 * rowh
        for sec in range(11):
            xx = L + sec / 10 * (Rr - L)
            s.line(xx, yb, xx, yb + 4, "#444")
            s.text(xx, yb + 16, f"{sec}", 10, "middle")
        s.text((L + Rr) / 2, yb + 30, "time (s); background colour = signed single-class IG (red raises the logit, blue lowers it), same scale in all leads", 11, "middle")
        # profile
        py = yb + 70
        s.text(L, py - 14, "Lead-summed IG per 20-sample block: positive (red, up) / negative (blue, down); grey = QRS-centred windows", 12, weight="bold")
        prof = A0[k].sum(0).reshape(250, 20).sum(1)
        pm = np.abs(prof).max() + 1e-12
        mid, hgt = py + 90, 80
        for (a, e) in br.qrs_tiles(b, 5000, HALF):
            s.rect(xs(a), py, xs(e) - xs(a), 2 * hgt + 10, "#999", 0.22)
        for i in range(250):
            hh = prof[i] / pm * hgt
            s.rect(L + i * cw, mid - max(hh, 0), cw + 0.3, abs(hh), "#D55E00" if hh >= 0 else "#0072B2", 0.9)
        s.line(L, mid, Rr, mid, "#444", 1)
        s.text(L, H_ := py + 2 * hgt + 40, "Regions surround AUTOMATIC detections from the Step 4.3 extractor (not expert-validated); IG describes model sensitivity, not a physiological cause.", 10, c="#555")
        s.save(os.path.join(out, f"example_{cls[k]}_{'correct' if corr[k] else 'incorrect'}_{rec[k]}.svg"))
    # ---- summary figure
    s = fg.Svg(1500, 980)
    d = res["attribution_distribution"]
    cl_ = [c for c in sorted(set(cls)) if c in d]
    s.text(20, 28, "Where does single-class IG put its mass relative to automatically detected beats?", 15, weight="bold")
    s.text(20, 46, f"{d['ALL']['n_pairs']} usable pairs / {d['ALL']['n_records']} records. Regions: QRS-centred +-80 ms windows, inter-beat (RR) segments, outside the detected-beat span.", 11, c="#333")
    # A: stacked shares per class (abs) with duration markers
    ax0, ay0, aw, ah = 90, 100, 640, 300
    f = fg.axes(s, ax0, ay0, aw, ah, (0, len(cl_) + 1), (0, 1), [], [0, 0.25, 0.5, 0.75, 1], "", "share of sum|IG|", "A. |IG| share by region (bars) vs duration share (ticks)")
    for i, c in enumerate(cl_ + ["ALL"]):
        row = d[c]
        yb_ = 0
        for r_, col in (("qrs", "#D55E00"), ("rr", "#0072B2"), ("out", "#999")):
            v = row["abs_share"][r_][0]
            x1, y1 = f(i + 0.1, yb_ + v)
            x2, y2 = f(i + 0.9, yb_)
            s.rect(x1, y1, x2 - x1, y2 - y1, col, 0.9)
            yd = yb_ + v
            dsh = row["duration_share"][r_]
            s.line(x1, f(0, np.cumsum([row['duration_share'][q] for q in ('qrs', 'rr', 'out')])[('qrs', 'rr', 'out').index(r_)])[1], x2, f(0, np.cumsum([row['duration_share'][q] for q in ('qrs', 'rr', 'out')])[('qrs', 'rr', 'out').index(r_)])[1], "#000", 1.4)
            yb_ += v
        s.text(f(i + 0.5, 0)[0], ay0 + ah + 16, f"{c}", 10, "middle")
    for i, (nm, col) in enumerate((("QRS-centred", "#D55E00"), ("RR segments", "#0072B2"), ("outside beat span", "#999"))):
        s.rect(100 + i * 150, 440, 12, 12, col)
        s.text(118 + i * 150, 451, nm, 10)
    s.text(520, 451, "black ticks: cumulative duration share", 10)
    # B: observed - shift control by class
    bx0 = 800
    vals = {c: d[c]["abs_share_minus_shift_control"]["qrs"] for c in cl_ + ["ALL"]}
    lo = min(0, min(v[1] for v in vals.values())) * 1.2
    hi = max(v[2] for v in vals.values()) * 1.2
    f = fg.axes(s, bx0, ay0, 600, ah, (0, len(vals)), (lo, hi), [], np.linspace(lo, hi, 5), "", "QRS share minus shift control", "B. QRS-centred |IG| share minus circular-shift control (95% CI)")
    for i, (c, v) in enumerate(vals.items()):
        x1, y1 = f(i + 0.15, v[0])
        x2, y2 = f(i + 0.85, 0)
        s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), "#D55E00", 0.9)
        xm = (x1 + x2) / 2
        s.line(xm, f(0, v[1])[1], xm, f(0, v[2])[1], "#000", 1.3)
        s.text(xm, ay0 + ah + 16, c, 9, "middle")
    # C: deletion
    cy0 = 560
    pd = res["targeted_deletion"]
    fams = [("top_qrs", "#D55E00", "top-attributed QRS tiles"), ("rand_qrs", "#E69F00", "random QRS tiles"), ("top_rr", "#0072B2", "top-attributed RR tiles"), ("rand_rr", "#56B4E9", "random RR tiles"),
            ("top_any", "#CC79A7", "top-attributed tiles anywhere"), ("rand_any", "#555", "random tiles anywhere")]
    groups_ = [f"m={m}|zero" for m in M_LIST]
    lo = min(pd[g]["mean_delta_logit"][fm][1] for g in groups_ for fm, _, _ in fams) * 1.15
    f = fg.axes(s, 90, cy0, 1350, 300, (0, len(groups_)), (lo, 0.1), [], np.linspace(lo, 0, 5), "", "change in target logit", "C. Deleting m 81-sample tiles (zero baseline): change in the explained class logit; whiskers = 95% CI")
    for gi, g in enumerate(groups_):
        for j, (fm, col, nm) in enumerate(fams):
            v = pd[g]["mean_delta_logit"][fm]
            x1, y1 = f(gi + 0.04 + j * 0.15, v[0])
            x2, y2 = f(gi + 0.04 + j * 0.15 + 0.13, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), col, 0.9)
            xm = (x1 + x2) / 2
            s.line(xm, f(0, v[1])[1], xm, f(0, v[2])[1], "#000", 1.1)
        s.text(f(gi + 0.5, 0)[0], cy0 + 300 + 16, f"m = {M_LIST[gi]} tiles deleted (n = {pd[g]['n_pairs']} pairs)", 10, "middle")
    for i, (fm, col, nm) in enumerate(fams):
        s.rect(100 + i * 220, cy0 + 340, 12, 12, col)
        s.text(118 + i * 220, cy0 + 351, nm, 10)
    s.save(os.path.join(out, "summary_attribution_vs_beats.svg"))
    print(sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
