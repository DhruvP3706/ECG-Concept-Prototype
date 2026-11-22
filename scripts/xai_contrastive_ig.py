"""Experiment 3: class-contrastive Integrated Gradients on the frozen original LightECGNetV2.
Same 160 (record, class) pairs as Experiment 1. For target class c (the model's own predicted class) a competitor k is chosen deterministically from the model's OUTPUTS only (never ground truth):
  candidates = classes != c whose probability is NOT above their saved threshold (i.e. not predicted positive); competitor = the candidate with the highest probability (tie -> lowest index);
  if every other class is predicted positive, the strongest non-target class is used and the pair is flagged `no_nonpositive_competitor`.
Target function: M(x) = f_c(x) - f_k(x) (pre-sigmoid logit margin, not exclusive-class). IG of M along the same zero-baseline path equals IG(f_c) - IG(f_k) by linearity; it is computed directly
(64 midpoint steps, as in Experiment 1) and the identity is checked numerically on a subset. Perturbation tests (deletion / insertion, 250 ms windows, lead x window cells, zero and donor baselines)
measure M(x). All statistics are record-clustered bootstraps. Population: frozen validation role only; test_reserved/train never read; nothing is trained or modified.
Run: .venv\\Scripts\\python.exe -I scripts\\xai_contrastive_ig.py        -> results/xai_contrastive/run_<stamp>/
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
from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.preprocessing import LEAD_NAMES, load_ecg  # noqa: E402
import xai_attention_analysis as xa  # noqa: E402
import xai_ig_faithfulness as xf  # noqa: E402
import xai_ig_figures as fg  # noqa: E402

PREV = xa.PREV
KS, WIN, N_RAND, IG_STEPS = xf.KS, xf.WIN, xf.N_RAND, xf.IG_STEPS
ORDS = ("pos", "abs", "low_abs", "neg", "pos_part", "neg_part", "single_pos", "single_abs", "random")
TESTS = [("time", "zero", "delete"), ("time", "zero", "insert"), ("cell", "zero", "delete"), ("cell", "zero", "insert"), ("time", "donor", "delete"), ("time", "donor", "insert"), ("lead", "zero", "delete")]
sha_file = xf.sha_file


def pick_competitor(logits, thr, c):
    """Deterministic competitor rule (outputs only). Returns (k, flagged)."""
    logits = np.asarray(logits, float)
    prob = 1 / (1 + np.exp(-logits))
    others = [i for i in range(len(logits)) if i != c]
    cand = [i for i in others if prob[i] <= thr[i]]
    flag = False
    if not cand:
        cand, flag = others, True
    best = max(cand, key=lambda i: (prob[i], -i))                                         # highest probability; tie -> lowest index
    return int(best), flag


def build_orders(aM, a0, r_id, nreg, prng):
    ss, sa = at.region_scores(aM, r_id, nreg, "signed"), at.region_scores(aM, r_id, nreg, "abs")
    pp = at.region_scores(np.maximum(aM, 0), r_id, nreg, "signed")
    nn_ = at.region_scores(np.minimum(aM, 0), r_id, nreg, "signed")
    s0, a0s = at.region_scores(a0, r_id, nreg, "signed"), at.region_scores(a0, r_id, nreg, "abs")
    od = at.orderings(ss, sa, prng, N_RAND)
    od["pos_part"], od["neg_part"] = np.argsort(-pp, kind="stable"), np.argsort(nn_, kind="stable")                  # neg_part: largest negative (pro-competitor) evidence first
    od["single_pos"], od["single_abs"] = np.argsort(-s0, kind="stable"), np.argsort(-a0s, kind="stable")
    return od


def suite(model, X, AM, A0, target, donor_of, say):
    rid = {g: at.region_ids(g, 12, 5000, WIN) for g in ("time", "cell", "lead")}
    curves, lead_single = {}, np.zeros((len(X), 12))
    zero = np.zeros((12, 5000), np.float32)
    t1 = time.time()
    for k in range(len(X)):
        prng = np.random.RandomState(1000 + k)
        a, a0 = AM[k].astype(np.float64), A0[k].astype(np.float64)
        for gran in ("time", "cell", "lead"):
            r_id, nreg = rid[gran]
            od = build_orders(a, a0, r_id, nreg, prng)
            ks = KS[gran]
            plan = [(zero, "zero", d) for d in ("delete", "insert")] if gran != "lead" else [(zero, "zero", "delete")]
            if gran == "time":
                plan += [(donor_of[k], "donor", d) for d in ("delete", "insert")]
            for base, bname, direction in plan:
                masks, tags = [], []
                for o in ORDS:
                    for perm in (od[o] if o == "random" else [od[o]]):
                        masks.append(at.prefix_masks(perm, r_id, ks))
                        tags.append(o)
                lg = at.perturbed_logits(model, X[k], base, target[k], np.concatenate(masks), direction).reshape(len(tags), len(ks))
                for o in ORDS:
                    curves.setdefault((gran, bname, direction), {}).setdefault(o, np.zeros((len(X), len(ks))))[k] = lg[[i for i, t in enumerate(tags) if t == o]].mean(0)
        lm = (np.arange(12)[:, None, None] == np.arange(12)[None, :, None]) & np.ones((12, 12, 5000), bool)
        lead_single[k] = at.perturbed_logits(model, X[k], zero, target[k], lm, "delete")
        if k % 40 == 0:
            say(f"  perturbation tests: pair {k}/{len(X)} ({time.time() - t1:.0f}s)")
    return curves, lead_single


def raw_gaps(curves):
    """per pair and ordering: AUC(ordering) - AUC(random) of the change of the margin from its starting value (deletion: from M(x); insertion: from the baseline margin).
    Deletion: negative = margin falls faster than random. Insertion: positive = margin recovers faster than random."""
    out = {}
    for (gran, bname, direction), cv in curves.items():
        fr = KS[gran] / KS[gran][-1]
        start = cv["random"][:, 0]
        auc = {o: np.array([at.curve_auc(cv[o][i] - start[i], fr) for i in range(len(start))]) for o in ORDS}
        out[(gran, bname, direction)] = {"raw": {o: auc[o] - auc["random"] for o in ORDS if o != "random"}, "auc": auc, "start": start}
    return out


def example_figure(path, x, a0, aM, meta):
    W, H = 1400, 1230
    s = fg.Svg(W, H)
    s.text(20, 28, f"{meta['record_id']}: target {meta['class']} vs competitor {meta['competitor']} (ground-truth label for {meta['class']} {'PRESENT' if meta['correct'] else 'ABSENT'}; {meta['note']})", 15, weight="bold")
    s.text(20, 48, f"f_c(x) = {meta['f_c']:.2f}, f_k(x) = {meta['f_k']:.2f}, margin M(x) = {meta['Mx']:.2f}; zero-baseline margin {meta['M0']:.2f}; contrastive IG completeness error {100 * meta['comp_rel']:.1f}% of M(x)-M(0)", 11, c="#333")
    L, Rr, top, rowh = 70, 1010, 80, 36
    T = x.shape[1]
    nw = 40
    wM = aM.reshape(12, nw, T // nw).sum((0, 2))
    top4 = np.argsort(-wM)[:4]
    s.text(L, top - 8, "A. Input (filtered, z-scored); shaded: 4 windows with the highest CONTRASTIVE (margin) attribution", 12, weight="bold")
    for l in range(12):
        yc = top + rowh * (l + 0.5)
        s.line(L, yc, Rr, yc, "#eee", 0.6)
        s.text(L - 8, yc + 4, fg.LEADS[l], 11, "end", weight="bold")
    for wi in top4:
        s.rect(L + wi / nw * (Rr - L), top, (Rr - L) / nw, 12 * rowh, "#D55E00", 0.18)
    n = 1250
    for l in range(12):
        yc = top + rowh * (l + 0.5)
        seg = x[l].astype(float).reshape(n, -1)
        mx, mn = seg.max(1), seg.min(1)
        pts = []
        for i in range(n):
            xx = L + i / n * (Rr - L)
            pts += [(xx, yc - np.clip(mn[i], -4, 4) * rowh * 0.11), (xx, yc - np.clip(mx[i], -4, 4) * rowh * 0.11)]
        s.poly(pts, "#222", 0.8)
    yb = top + 12 * rowh
    for sec in range(11):
        xx = L + sec / 10 * (Rr - L)
        s.line(xx, yb, xx, yb + 4, "#444")
        s.text(xx, yb + 16, f"{sec}", 10, "middle")
    s.text((L + Rr) / 2, yb + 30, "time (s)", 11, "middle")
    b0, bM = a0.reshape(12, 250, 20).sum(2), aM.reshape(12, 250, 20).sum(2)
    vmax = np.percentile(np.abs(np.concatenate([b0.ravel(), bM.ravel()])), 99.5) + 1e-12            # ONE colour scale for both maps
    cw = (Rr - L) / 250
    for pi, (title, blocks) in enumerate((("B. Single-class IG for " + meta["class"] + " (logit f_c)", b0), ("C. Contrastive IG for " + meta["class"] + " vs " + meta["competitor"] + " (margin f_c - f_k)", bM))):
        h0 = yb + 60 + pi * 245
        s.text(L, h0 - 6, title + "  - red raises, blue lowers; same colour scale in B and C", 12, weight="bold")
        for l in range(12):
            s.text(L - 8, h0 + 18 * (l + 0.5) + 3, fg.LEADS[l], 9, "end")
            for i in range(250):
                s.rect(L + i * cw, h0 + l * 18, cw + 0.3, 18, fg.div_color(blocks[l, i] / vmax))
        s.rect(L, h0, Rr - L, 12 * 18, "none", stroke="#444")
    s.text(L, yb + 60 + 2 * 245 - 14, f"colour scale +/-{vmax:.3g} logit units per 20-sample block (99.5th percentile over both maps)", 10, c="#333")
    cx0 = 1090
    s.text(cx0, top - 8, "D. Window evidence for the margin (contrastive IG)", 12, weight="bold")
    pos = np.array([np.maximum(aM[:, w * 125:(w + 1) * 125], 0).sum() for w in range(nw)])
    neg = np.array([np.minimum(aM[:, w * 125:(w + 1) * 125], 0).sum() for w in range(nw)])
    mid, scale = top + 6 * rowh, 6 * rowh * 0.9 / (max(pos.max(), -neg.min()) + 1e-12)
    for w in range(nw):
        xx = cx0 + 10 + w * 7
        s.rect(xx, mid - pos[w] * scale, 5, pos[w] * scale, "#D55E00", 0.9)
        s.rect(xx, mid, 5, -neg[w] * scale, "#0072B2", 0.9)
    s.line(cx0 + 10, mid, cx0 + 10 + nw * 7, mid, "#444", 1)
    s.text(cx0 + 10, mid + 6 * rowh + 6, "250 ms windows (0-10 s). Red up: evidence for the target over the competitor; blue down: evidence for the competitor.", 9, c="#333")
    s.text(cx0 + 10, mid + 6 * rowh + 20, f"sum positive {pos.sum():.2f}, sum negative {neg.sum():.2f}, net {pos.sum() + neg.sum():.2f} (= M(x) - M(0) up to IG error)", 9, c="#333")
    s.text(cx0, yb + 60 + 245 - 6 + 10, "E. Lead sums (single vs contrastive)", 12, weight="bold")
    ls0, lsM = a0.sum(1), aM.sum(1)
    mxv = max(np.abs(ls0).max(), np.abs(lsM).max()) + 1e-12
    for l in range(12):
        y = yb + 60 + 245 + 14 + l * 16
        s.text(cx0 + 24, y + 11, fg.LEADS[l], 10, "end")
        for j, (v, col) in enumerate(((ls0[l], "#888"), (lsM[l], "#D55E00" if lsM[l] >= 0 else "#0072B2"))):
            wv = v / mxv * 110
            s.rect(min(cx0 + 150, cx0 + 150 + wv), y + j * 6, abs(wv), 5, col, 0.9)
    s.line(cx0 + 150, yb + 60 + 245 + 14, cx0 + 150, yb + 60 + 245 + 14 + 12 * 16, "#444", 1)
    s.text(cx0 + 40, yb + 60 + 245 + 14 + 12 * 16 + 14, "grey: single-class sum; colour: contrastive sum", 9, c="#333")
    s.text(L, H - 14, "Attribution describes the MODEL's sensitivity along the zero-baseline path; it is not evidence of a physiological cause. No region is labelled with a clinical finding.", 10, c="#555")
    s.save(path)


def main():
    cfg = json.load(open(os.path.join(PREV, "config.json")))
    assert sha_file(os.path.join(PREV, "results.json")).startswith(xa.PREV_RESULTS_PREFIX)
    run = os.path.join(ROOT, "results", "xai_contrastive", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run)
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    pairs = list(csv.DictReader(open(os.path.join(PREV, "pairs.csv"))))
    for p in pairs:
        for k in ("pair", "class_idx", "va_idx"):
            p[k] = int(p[k])
        for k in ("f_x", "f_base", "logit_cached", "comp_abs", "comp_rel"):
            p[k] = float(p[k])
        p["correct"] = p["correct"] == "True"
    z = np.load(os.path.join(PREV, "attributions.npz"))
    A0 = z["attr"]
    rec = np.array([p["record_id"] for p in pairs])
    cls = np.array([p["class"] for p in pairs])
    cidx = np.array([p["class_idx"] for p in pairs])
    corr = np.array([p["correct"] for p in pairs])
    cl = xa.Clust(rec)
    assert xf.sha_file(xf.CKPT) == xf.CKPT_SHA
    names = [str(c) for c in np.load(os.path.join(xf.CACHE, "merged.npz"))["class_names"]]
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(xf.CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    assert all(r in rel for r in rec)
    sc_ = {}
    sig = lambda r: sc_.setdefault(r, load_ecg(os.path.join(xf.DS, rel[r])))
    X = np.stack([sig(r) for r in rec])
    donor_ids = cfg["donor_record_ids"]
    D = np.stack([sig(i) for i in donor_ids])
    donor_of = [D[(k + 1) % len(D)] if donor_ids[(k + 1) % len(D)] != rec[k] else D[(k + 2) % len(D)] for k in range(len(pairs))]
    dev = "cuda"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = load_model(xf.CKPT, dev)
    h0 = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    thr = np.load(xf.THR, allow_pickle=False)
    with torch.no_grad():
        LG = np.stack([model(torch.from_numpy(X[k])[None].to(dev))[0].cpu().numpy().astype(np.float64) for k in range(len(pairs))])
    assert np.abs(LG[np.arange(len(pairs)), cidx] - np.array([p["f_x"] for p in pairs])).max() < 1e-3
    # ---- competitor selection (outputs only)
    comp, flags = [], []
    for k in range(len(pairs)):
        kk, fl = pick_competitor(LG[k], thr, cidx[k])
        comp.append(kk)
        flags.append(fl)
    comp, flags = np.array(comp), np.array(flags)
    target = [(int(cidx[k]), int(comp[k])) for k in range(len(pairs))]
    Mx_pre = LG[np.arange(len(pairs)), cidx] - LG[np.arange(len(pairs)), comp]
    say(f"{len(pairs)} pairs, {cl.R} unique records; competitors chosen from outputs only; flagged (no non-positive competitor): {int(flags.sum())}; margins M(x)>0 in {(Mx_pre > 0).mean():.3f} of pairs, min {Mx_pre.min():.2f}")
    comp_table = {}
    for k in range(len(pairs)):
        comp_table.setdefault(cls[k], {}).setdefault(names[comp[k]], 0)
        comp_table[cls[k]][names[comp[k]]] += 1
    # ---- contrastive IG
    t0 = time.time()
    AM = np.zeros_like(A0)
    Mx, M0, cabs, crel = np.zeros(len(pairs)), np.zeros(len(pairs)), np.zeros(len(pairs)), np.zeros(len(pairs))
    for k in range(len(pairs)):
        r = at.integrated_gradients(model, X[k], target[k], None, IG_STEPS)
        AM[k], Mx[k], M0[k], cabs[k], crel[k] = r["attr"], r["f_x"], r["f_baseline"], r["completeness_abs_err"], r["completeness_rel_err"]
    assert np.isfinite(AM).all() and np.abs(Mx - Mx_pre).max() < 1e-3
    say(f"contrastive IG in {time.time() - t0:.0f}s; completeness rel err median {np.median(crel):.4f}, abs median {np.median(cabs):.4f}")
    rng = np.random.RandomState(xf.SEL_SEED)
    sub = rng.choice(len(pairs), 24, replace=False)
    conv, lin = [], []
    for k in sub:
        r32 = at.integrated_gradients(model, X[k], target[k], None, 32)
        r256 = at.integrated_gradients(model, X[k], target[k], None, 256)
        conv.append({"rel32": r32["completeness_rel_err"], "rel64": crel[k], "rel256": r256["completeness_rel_err"], "pearson_64_256": float(np.corrcoef(AM[k].ravel(), r256["attr"].ravel())[0, 1])})
    for k in sub[:6]:
        rk = at.integrated_gradients(model, X[k], int(comp[k]), None, IG_STEPS)
        lin.append(float(np.abs(AM[k] - (A0[k] - rk["attr"])).max()))
    say(f"convergence (24 pairs): median rel err 32/64/256 = {np.median([c['rel32'] for c in conv]):.4f}/{np.median([c['rel64'] for c in conv]):.4f}/{np.median([c['rel256'] for c in conv]):.4f}; "
        f"IG(M) vs IG(c)-IG(k) max abs diff over 6 pairs {max(lin):.2e}")
    AK = A0 - AM                                                                            # competitor's own IG (exact by linearity; verified above)
    # ---- single vs contrastive comparison
    tr, cr = at.region_ids("time", 12, 5000, WIN), at.region_ids("cell", 12, 5000, WIN)
    sw0 = np.stack([at.region_scores(A0[k].astype(float), tr[0], 40, "signed") for k in range(len(pairs))])
    swM = np.stack([at.region_scores(AM[k].astype(float), tr[0], 40, "signed") for k in range(len(pairs))])
    swK = np.stack([at.region_scores(AK[k].astype(float), tr[0], 40, "signed") for k in range(len(pairs))])
    sc0 = np.stack([at.region_scores(A0[k].astype(float), cr[0], 480, "signed") for k in range(len(pairs))])
    scM = np.stack([at.region_scores(AM[k].astype(float), cr[0], 480, "signed") for k in range(len(pairs))])
    scK = np.stack([at.region_scores(AK[k].astype(float), cr[0], 480, "signed") for k in range(len(pairs))])
    cor = lambda a, b: np.array([np.corrcoef(a[k], b[k])[0, 1] for k in range(len(a))])
    top8 = lambda sc: [set(np.argsort(-sc[k])[:8]) for k in range(len(sc))]
    t0s, tMs, tKs = top8(sw0), top8(swM), top8(swK)
    absw = lambda A: np.stack([at.region_scores(np.abs(A[k]).astype(float), tr[0], 40, "abs") for k in range(len(pairs))])
    cmp_ = {"pearson_full_map_single_vs_contrastive": cl.mean_ci([np.corrcoef(A0[k].ravel(), AM[k].ravel())[0, 1] for k in range(len(pairs))]),
            "pearson_window_signed_single_vs_contrastive": cl.mean_ci(cor(sw0, swM)), "pearson_cell_signed_single_vs_contrastive": cl.mean_ci(cor(sc0, scM)),
            "top8_window_overlap_single_vs_contrastive_(chance 1.6)": cl.mean_ci([len(t0s[k] & tMs[k]) for k in range(len(pairs))]),
            "pearson_window_signed_target_vs_competitor_IG": cl.mean_ci(cor(sw0, swK)), "pearson_cell_signed_target_vs_competitor_IG": cl.mean_ci(cor(sc0, scK)),
            "top8_window_overlap_target_vs_competitor_IG_(chance 1.6)": cl.mean_ci([len(t0s[k] & tKs[k]) for k in range(len(pairs))]),
            "L1_ratio_contrastive_over_single": cl.mean_ci(np.abs(AM).sum((1, 2)) / np.abs(A0).sum((1, 2))), "L1_ratio_competitor_over_single": cl.mean_ci(np.abs(AK).sum((1, 2)) / np.abs(A0).sum((1, 2))),
            "positive_mass_share_single": cl.mean_ci(np.maximum(A0, 0).sum((1, 2)) / np.abs(A0).sum((1, 2))), "positive_mass_share_contrastive": cl.mean_ci(np.maximum(AM, 0).sum((1, 2)) / np.abs(AM).sum((1, 2))),
            "mean_sum_positive_single": float(np.maximum(A0, 0).sum((1, 2)).mean()), "mean_sum_negative_single": float(np.minimum(A0, 0).sum((1, 2)).mean()),
            "mean_sum_positive_contrastive": float(np.maximum(AM, 0).sum((1, 2)).mean()), "mean_sum_negative_contrastive": float(np.minimum(AM, 0).sum((1, 2)).mean()),
            "top_window_share_of_abs_IG": {"single": float((absw(A0) / absw(A0).sum(1, keepdims=True)).max(1).mean()), "contrastive": float((absw(AM) / absw(AM).sum(1, keepdims=True)).max(1).mean())},
            "per_class": {c: {"pearson_window_signed": float(cor(sw0, swM)[cls == c].mean()), "pearson_cell_signed": float(cor(sc0, scM)[cls == c].mean()),
                              "top8_overlap": float(np.mean([len(t0s[k] & tMs[k]) for k in np.where(cls == c)[0]])), "L1_ratio_contrastive_over_single": float((np.abs(AM).sum((1, 2)) / np.abs(A0).sum((1, 2)))[cls == c].mean())}
                          for c in sorted(set(cls))}}
    for nm, sel in (("correct", corr), ("incorrect", ~corr)):
        cmp_[f"pearson_window_signed_single_vs_contrastive_{nm}"] = cl.mean_ci(cor(sw0, swM), sel)
        cmp_[f"L1_ratio_contrastive_over_single_{nm}"] = cl.mean_ci(np.abs(AM).sum((1, 2)) / np.abs(A0).sum((1, 2)), sel)
    # ---- contrastive faithfulness
    curves, lead_single = suite(model, X, AM, A0, target, donor_of, say)
    assert h0 == hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest() and sha_file(xf.CKPT) == xf.CKPT_SHA
    say("model hash unchanged")
    G = raw_gaps(curves)
    faith, rows = {}, []
    for t in TESTS:
        key = "|".join(t)
        faith[key] = {}
        for o in ORDS[:-1]:
            v = G[t]["raw"][o]
            faith[key][o] = {"all": cl.mean_ci(v), "correct": cl.mean_ci(v, corr), "incorrect": cl.mean_ci(v, ~corr), "share_pairs_negative": float((v < 0).mean())}
            for c in sorted(set(cls)):
                m = cls == c
                rows.append({"test": key, "ordering": o, "class": c, "n_pairs": int(m.sum()), "n_records": int(len(set(rec[m]))), "raw_gap_vs_random": float(v[m].mean())})
        for a_, b_ in (("pos", "single_pos"), ("abs", "single_abs")):
            d = G[t]["raw"][a_] - G[t]["raw"][b_]
            faith[key][f"{a_}_minus_{b_}"] = {"all": cl.mean_ci(d), "correct": cl.mean_ci(d, corr), "incorrect": cl.mean_ci(d, ~corr), "share_pairs_contrastive_better": float(((d < 0) if t[2] == "delete" else (d > 0)).mean())}
    margin_stats = {"M_x_mean": float(Mx.mean()), "M_x_sd": float(Mx.std()), "share_M_x_positive": float((Mx > 0).mean()), "M_zero_baseline_mean": float(M0.mean()),
                    "M_x_minus_M0_mean": float((Mx - M0).mean()), "n_flagged_no_nonpositive_competitor": int(flags.sum()),
                    "completeness": {"rel_median": float(np.median(crel)), "abs_median": float(np.median(cabs)), "convergence": {k: float(np.median([c[k] for c in conv])) for k in conv[0]}, "linearity_check_max_abs_diff": float(max(lin))},
                    "competitor_counts_by_target_class": comp_table, "single_class_f_c": float(np.mean([p["f_x"] for p in pairs])),
                    "flip_note": "share of pairs whose margin after deleting 20% highest contrastive windows becomes <= 0 (zero baseline): " + str(float((curves[("time", "zero", "delete")]["pos"][:, 4] <= 0).mean())) +
                    "; random: " + str(float((curves[("time", "zero", "delete")]["random"][:, 4] <= 0).mean()))}
    res = {"n_pairs": len(pairs), "n_unique_records": int(cl.R), "n_correct": int(corr.sum()), "n_incorrect": int((~corr).sum()), "bootstrap": f"record-clustered, B={xa.BOOT_B}, seed={xa.BOOT_SEED}, percentile 95% CI",
           "competitor_rule": pick_competitor.__doc__, "margin": margin_stats, "single_vs_contrastive": cmp_, "faithfulness_margin_raw_gap_vs_random": faith,
           "expected_direction": "deletion of positively attributed regions: raw gap < 0 (margin falls faster than random); insertion: raw gap > 0; pro-competitor orderings (neg, neg_part) the opposite on deletion"}
    json.dump(res, open(os.path.join(run, "results.json"), "x"), indent=1, default=float)
    with open(os.path.join(run, "gap_by_class.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run, "pairs_competitors.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "record_id", "class", "competitor", "flagged_no_nonpositive_competitor", "correct", "f_c", "f_k", "M_x", "M_0", "comp_abs", "comp_rel"])
        for k in range(len(pairs)):
            w.writerow([k, rec[k], cls[k], names[comp[k]], bool(flags[k]), bool(corr[k]), LG[k, cidx[k]], LG[k, comp[k]], Mx[k], M0[k], cabs[k], crel[k]])
    np.savez_compressed(os.path.join(run, "contrastive_attributions.npz"), attr_margin=AM, margin_x=Mx, margin_zero=M0, competitor_idx=comp, flagged=flags, pair=np.arange(len(pairs)), record_id=rec, cls=cls,
                        **{f"curve_{g}_{b}_{d}_{o}": v for (g, b, d), cv in curves.items() for o, v in cv.items()})
    json.dump({"previous_run": os.path.relpath(PREV, ROOT), "checkpoint_sha256": xf.CKPT_SHA, "thresholds_sha256": sha_file(xf.THR), "ig": {"baseline": "zero in normalised space", "steps": IG_STEPS, "rule": "midpoint"},
               "pairs": "same 160 pairs as Experiment 1", "competitor_rule": pick_competitor.__doc__, "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("scripts/xai_contrastive_ig.py", "src/ecg_concept/attribution.py")}},
              open(os.path.join(run, "config.json"), "x"), indent=1)
    say(f"results saved; sha256 {sha_file(os.path.join(run, 'results.json'))[:16]}")
    figures(run, X, A0, AM, rec, cls, comp, names, pairs, Mx, M0, LG, cidx, crel, flags, corr, cor(sw0, swM), curves, G, cl, rows)
    print(run)


def figures(run, X, A0, AM, rec, cls, comp, names, pairs, Mx, M0, LG, cidx, crel, flags, corr, pw, curves, G, cl, rows):
    out = os.path.join(run, "figures")
    os.makedirs(out)
    idx_c = [i for i in np.argsort(pw) if corr[i]][:3]
    idx_w = [i for i in np.argsort(pw) if not corr[i]][:1]
    med = np.argsort(pw)[len(pw) // 2 - 1:len(pw) // 2 + 1]
    chosen = [(int(i), "among the 3 lowest single-vs-contrastive window correlation, correct prediction") for i in idx_c] + [(int(i), "lowest correlation among incorrect predictions") for i in idx_w] + \
             [(int(i), "median correlation (typical)") for i in med]
    for i, note in chosen:
        meta = {"record_id": rec[i], "class": cls[i], "competitor": names[comp[i]], "correct": bool(corr[i]), "f_c": LG[i, cidx[i]], "f_k": LG[i, comp[i]], "Mx": Mx[i], "M0": M0[i], "comp_rel": crel[i],
                "note": note + (" [no non-positive competitor]" if flags[i] else "") + f"; window corr {pw[i]:.2f}"}
        example_figure(os.path.join(out, f"example_{cls[i]}_vs_{names[comp[i]]}_{rec[i]}.svg"), X[i], A0[i].astype(float), AM[i].astype(float), meta)
    # curves
    s = fg.Svg(1500, 900)
    s.text(20, 28, "Class-margin perturbation tests: M(x) = f_c(x) - f_k(x) (mean change over pairs, band = 1.96 SE)", 15, weight="bold")
    s.text(20, 48, f"n = {len(rec)} pairs, {cl.R} unique records; deletion replaces regions by the baseline, insertion restores them. Expected: positively attributed regions pull the margin down when deleted and restore it when inserted.", 11, c="#333")
    col = {"pos": "#D55E00", "single_pos": "#7a2e00", "abs": "#E69F00", "low_abs": "#0072B2", "neg": "#CC79A7", "random": "#555555", "neg_part": "#009E73"}
    lab = {"pos": "contrastive IG, highest signed first", "single_pos": "single-class IG (class c), highest first", "abs": "contrastive, highest |attr| first", "low_abs": "contrastive, lowest |attr| first", "neg": "contrastive, most negative first (pro-competitor)",
           "random": "random order (mean of 10)", "neg_part": "contrastive, largest negative-part sum first"}
    panels = [("time|zero", "250 ms windows, zero baseline"), ("cell|zero", "lead x 250 ms cells, zero baseline"), ("time|donor", "250 ms windows, donor-ECG baseline")]
    for j, (k, t) in enumerate(panels):
        for r_, direction in enumerate(("delete", "insert")):
            g, b = k.split("|")
            cv = curves[(g, b, direction)]
            fr = KS[g] / KS[g][-1]
            start = cv["random"][:, 0]
            mats = {o: cv[o] - start[:, None] for o in col}
            allv = np.concatenate([m.mean(0) for m in mats.values()])
            lo, hi = float(min(allv.min(), 0)), float(max(allv.max(), 0))
            pad = (hi - lo) * 0.06 + 1e-9
            x0, y0 = 90 + j * 480, 100 + r_ * 400
            f = fg.axes(s, x0, y0, 380, 300, (0, 1), (lo - pad, hi + pad), [0, 0.25, 0.5, 0.75, 1], np.linspace(lo, hi, 5), "fraction " + ("deleted" if direction == "delete" else "restored"),
                        "change in margin M" + (" from M(x)" if direction == "delete" else " from baseline margin"), t if r_ == 0 else None)
            for o in ("random", "low_abs", "neg", "neg_part", "abs", "single_pos", "pos"):
                m = mats[o].mean(0)
                se = mats[o].std(0, ddof=1) / np.sqrt(len(rec))
                s.polygon([f(a, bb) for a, bb in zip(fr, m + 1.96 * se)] + [f(a, bb) for a, bb in zip(fr[::-1], (m - 1.96 * se)[::-1])], col[o], 0.13)
                s.poly([f(a, bb) for a, bb in zip(fr, m)], col[o], 2.2 if o in ("pos", "random") else 1.4)
            s.line(*f(0, 0), *f(1, 0), c="#888", sw=0.8, dash="3,3")
    for i, o in enumerate(("pos", "single_pos", "abs", "low_abs", "neg", "neg_part", "random")):
        x = 100 + (i % 4) * 350
        y = 850 + (i // 4) * 20
        s.line(x, y, x + 28, y, col[o], 3)
        s.text(x + 34, y + 4, lab[o], 10)
    s.save(os.path.join(out, "margin_deletion_insertion_curves.svg"))
    # gaps by class, correct vs incorrect
    s = fg.Svg(1500, 560)
    s.text(20, 28, "Margin deletion: contrastive vs single-class ordering, by class (time windows, zero baseline; raw AUC gap vs random, more negative = margin falls faster)", 13, weight="bold")
    cl_ = sorted(set(cls))
    def val(o, c):
        return [r["raw_gap_vs_random"] for r in rows if r["test"] == "time|zero|delete" and r["ordering"] == o and r["class"] == c][0]
    mn = min(min(val("pos", c), val("single_pos", c)) for c in cl_) * 1.15
    f = fg.axes(s, 90, 80, 1300, 320, (0, len(cl_)), (mn, 0.05), [], np.linspace(mn, 0, 5), "", "raw gap vs random (margin units)")
    for i, c in enumerate(cl_):
        for off, o, cc in ((0.1, "pos", "#D55E00"), (0.5, "single_pos", "#7a2e00")):
            x1, y1 = f(i + off, val(o, c))
            x2, y2 = f(i + off + 0.38, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), cc, 0.9)
        s.text(f(i + 0.5, 0)[0], 80 + 320 + 16, f"{c} (n={int((cls == c).sum())})", 10, "middle")
    s.rect(110, 450, 12, 12, "#D55E00")
    s.text(128, 461, "contrastive IG ordering", 11)
    s.rect(300, 450, 12, 12, "#7a2e00")
    s.text(318, 461, "single-class IG ordering (evaluated on the margin)", 11)
    s.save(os.path.join(out, "margin_deletion_gap_by_class.svg"))
    print(sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
