"""Experiment 4: robustness of class-contrastive Integrated Gradients to the choice of competitor class (frozen original LightECGNetV2, same 160 pairs, 157 records).
Competitor rules (all from model OUTPUTS only; ground truth never used):
  A  existing rule (Experiment 3): highest-probability non-target class that is NOT predicted positive under its saved threshold
  B  highest pre-sigmoid logit among non-target classes, regardless of threshold
  C  second-highest-probability non-target class that is NOT predicted positive (unavailable if fewer than two exist)
For each rule the margin M_r = f_c - f_kr is attributed with IG (zero baseline, 64 midpoint steps). IG(M_r) = IG(f_c) - IG(f_kr): IG(f_kr) is computed once per distinct competitor, the margin map is
formed by subtraction and checked against direct margin IG on a subset. Each contrastive ordering is evaluated on its own margin AND on f_c alone and f_kr alone, against ordinary IG(c)/IG(kr)
orderings, random and low-attribution controls. Record-clustered bootstrap. Validation role only; nothing is trained or modified.
Run: .venv\\Scripts\\python.exe -I scripts\\xai_competitor_robustness.py        -> results/xai_competitor/run_<stamp>/
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
from ecg_concept.preprocessing import load_ecg  # noqa: E402
import xai_attention_analysis as xa  # noqa: E402
import xai_contrastive_ig as xc  # noqa: E402
import xai_ig_faithfulness as xf  # noqa: E402

PREV, C3 = xa.PREV, os.path.join(ROOT, "results", "xai_contrastive", "run_20261120_033747")
KS, WIN, N_RAND, IG_STEPS = xf.KS, xf.WIN, xf.N_RAND, xf.IG_STEPS
RULES = ("A", "B", "C")
TESTS = [("time", "zero", "delete"), ("time", "zero", "insert"), ("cell", "zero", "delete"), ("cell", "zero", "insert"), ("time", "donor", "delete"), ("time", "donor", "insert")]
sha_file = xf.sha_file


# ------------------------------------------------------------------ competitor rules
def pick_highest_logit(logits, c):
    """Rule B: highest pre-sigmoid logit among non-target classes (ties -> lowest index), regardless of thresholds."""
    others = [i for i in range(len(logits)) if i != c]
    return int(max(others, key=lambda i: (float(logits[i]), -i)))


def pick_second_eligible(logits, thr, c):
    """Rule C: the SECOND-highest-probability non-target class that is not predicted positive; None if fewer than two such classes."""
    prob = 1 / (1 + np.exp(-np.asarray(logits, float)))
    cand = sorted([i for i in range(len(logits)) if i != c and prob[i] <= thr[i]], key=lambda i: (-prob[i], i))
    return int(cand[1]) if len(cand) >= 2 else None


def perturbed_multi(model, x, base, classes, masks, mode, bs=96):
    """logits of `classes` (list) for perturbed inputs; returns (M, len(classes))."""
    dev = next(model.parameters()).device
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
    base = torch.as_tensor(np.asarray(base), dtype=torch.float32, device=dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(masks), bs):
            m = torch.as_tensor(masks[i:i + bs], device=dev)
            xin = torch.where(m, base[None], x[None]) if mode == "delete" else torch.where(m, x[None], base[None])
            out.append(model(xin)[:, classes].float().cpu())
    return torch.cat(out).numpy()


def suite(model, X, A0, AM, AK, avail, comp, cidx, donor_of, say):
    """curves[(gran, base, direction)][ordering] -> (n_pairs, K, 4) logits of [f_c, f_kA, f_kB, f_kC] (random = mean over permutations)."""
    rid = {g: at.region_ids(g, 12, 5000, WIN) for g in ("time", "cell")}
    zero = np.zeros((12, 5000), np.float32)
    curves = {}
    t1 = time.time()
    for k in range(len(X)):
        prng = np.random.RandomState(1000 + k)
        classes = [int(cidx[k])] + [int(comp[r][k]) if avail[r][k] else int(cidx[k]) for r in RULES]
        for gran in ("time", "cell"):
            r_id, nreg = rid[gran]
            sc = lambda a, mode: at.region_scores(a.astype(np.float64), r_id, nreg, mode)
            od = {"c_pos": np.argsort(-sc(A0[k], "signed"), kind="stable"), "c_low": np.argsort(sc(A0[k], "abs"), kind="stable")}
            for r in RULES:
                if avail[r][k]:
                    od[f"{r}_cpos"] = np.argsort(-sc(AM[r][k], "signed"), kind="stable")
                    od[f"{r}_cneg"] = np.argsort(sc(AM[r][k], "signed"), kind="stable")
                    od[f"{r}_clow"] = np.argsort(sc(AM[r][k], "abs"), kind="stable")
                    od[f"{r}_kpos"] = np.argsort(-sc(AK[r][k], "signed"), kind="stable")
            rnd = np.stack([prng.permutation(nreg) for _ in range(N_RAND)])
            ks = KS[gran]
            plan = [(zero, "zero", d) for d in ("delete", "insert")]
            if gran == "time":
                plan += [(donor_of[k], "donor", d) for d in ("delete", "insert")]
            names = list(od) + ["random"]
            for base, bname, direction in plan:
                masks, tags = [], []
                for nme in names:
                    for perm in (rnd if nme == "random" else [od[nme]]):
                        masks.append(at.prefix_masks(perm, r_id, ks))
                        tags.append(nme)
                lg = perturbed_multi(model, X[k], base, classes, np.concatenate(masks), direction).reshape(len(tags), len(ks), 4)
                for nme in names:
                    arr = curves.setdefault((gran, bname, direction), {}).setdefault(nme, np.full((len(X), len(ks), 4), np.nan, np.float32))
                    arr[k] = lg[[i for i, t in enumerate(tags) if t == nme]].mean(0)
        if k % 40 == 0:
            say(f"  perturbation tests: pair {k}/{len(X)} ({time.time() - t1:.0f}s)")
    return curves


def aucs(curves, test, order, tcol):
    """per-pair AUC of the change of target-logit(s) from its starting value. tcol: function (n,K,4)->(n,K)."""
    cv = curves[test]
    fr = KS[test[0]] / KS[test[0]][-1]
    y = tcol(cv[order])
    return np.array([at.curve_auc(y[i] - y[i, 0], fr) if np.isfinite(y[i]).all() else np.nan for i in range(len(y))])


def main():
    cfg = json.load(open(os.path.join(PREV, "config.json")))
    assert sha_file(os.path.join(PREV, "results.json")).startswith(xa.PREV_RESULTS_PREFIX)
    c3res = os.path.join(C3, "results.json")
    run = os.path.join(ROOT, "results", "xai_competitor", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run)
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    pairs = list(csv.DictReader(open(os.path.join(PREV, "pairs.csv"))))
    for p in pairs:
        p["class_idx"] = int(p["class_idx"])
        p["f_x"], p["f_base"] = float(p["f_x"]), float(p["f_base"])
        p["correct"] = p["correct"] == "True"
    A0 = np.load(os.path.join(PREV, "attributions.npz"))["attr"]
    z3 = np.load(os.path.join(C3, "contrastive_attributions.npz"))
    rec, cls = np.array([p["record_id"] for p in pairs]), np.array([p["class"] for p in pairs])
    cidx = np.array([p["class_idx"] for p in pairs])
    corr = np.array([p["correct"] for p in pairs])
    n = len(pairs)
    cl = xa.Clust(rec)
    assert xf.sha_file(xf.CKPT) == xf.CKPT_SHA
    names = [str(c) for c in np.load(os.path.join(xf.CACHE, "merged.npz"))["class_names"]]
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(xf.CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    sc_ = {}
    sig = lambda r: sc_.setdefault(r, load_ecg(os.path.join(xf.DS, rel[r])))
    X = np.stack([sig(r) for r in rec])
    donor_ids = cfg["donor_record_ids"]
    D = np.stack([sig(i) for i in donor_ids])
    donor_of = [D[(k + 1) % len(D)] if donor_ids[(k + 1) % len(D)] != rec[k] else D[(k + 2) % len(D)] for k in range(n)]
    dev = "cuda"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = load_model(xf.CKPT, dev)
    h0 = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    thr = np.load(xf.THR, allow_pickle=False)
    with torch.no_grad():
        LG = np.stack([model(torch.from_numpy(X[k])[None].to(dev))[0].cpu().numpy().astype(np.float64) for k in range(n)])
    # ---- Step 1: competitors
    comp = {r: np.full(n, -1) for r in RULES}
    avail = {r: np.zeros(n, bool) for r in RULES}
    for k in range(n):
        comp["A"][k], _ = xc.pick_competitor(LG[k], thr, cidx[k])
        comp["B"][k] = pick_highest_logit(LG[k], cidx[k])
        cC = pick_second_eligible(LG[k], thr, cidx[k])
        if cC is not None:
            comp["C"][k] = cC
        avail["A"][k], avail["B"][k], avail["C"][k] = True, True, cC is not None
    assert (comp["A"] == z3["competitor_idx"]).all(), "rule A differs from Experiment 3"
    prob = 1 / (1 + np.exp(-LG))
    pos_cls = prob > thr[None]
    marg = {r: np.where(avail[r], LG[np.arange(n), cidx] - LG[np.arange(n), np.maximum(comp[r], 0)], np.nan) for r in RULES}
    sel_diffB = comp["B"] != comp["A"]
    summary1 = {"n_pairs": n, "n_unique_records": int(cl.R),
                "B_identical_to_A": int((~sel_diffB).sum()), "B_differs_from_A": int(sel_diffB.sum()), "B_is_predicted_positive_class": int(pos_cls[np.arange(n), comp["B"]].sum()),
                "C_available": int(avail["C"].sum()), "C_unavailable": int((~avail["C"]).sum()), "C_identical_to_A_or_B": int(((comp["C"] == comp["A"]) | (comp["C"] == comp["B"])).sum()),
                "nonpositive_margin_count": {r: int((marg[r][avail[r]] <= 0).sum()) for r in RULES}, "margin_mean": {r: float(np.nanmean(marg[r])) for r in RULES},
                "margin_min": {r: float(np.nanmin(marg[r])) for r in RULES}, "competitor_counts": {r: {names[i]: int((comp[r][avail[r]] == i).sum()) for i in sorted(set(comp[r][avail[r]]))} for r in RULES},
                "per_class_B_differs_from_A": {c: int((sel_diffB & (cls == c)).sum()) for c in sorted(set(cls))},
                "per_class_C_available": {c: int((avail["C"] & (cls == c)).sum()) for c in sorted(set(cls))}}
    say(f"Step 1: B differs from A in {summary1['B_differs_from_A']}/{n} pairs ({summary1['B_is_predicted_positive_class']} B-competitors are predicted-positive classes); C available {summary1['C_available']}; "
        f"non-positive margins {summary1['nonpositive_margin_count']}")
    # ---- Step 2: IG(f_k) per distinct competitor; margin map = IG(c) - IG(k)
    t0 = time.time()
    AK = {r: np.zeros_like(A0) for r in RULES}
    comp_rel = {r: np.full(n, np.nan) for r in RULES}
    cache = {}
    for k in range(n):
        for r in RULES:
            if not avail[r][k]:
                continue
            key = (k, int(comp[r][k]))
            if key not in cache:
                if r == "A":
                    cache[key] = (A0[k] - z3["attr_margin"][k], None)                         # reuse Experiment 3 (IG(k) = IG(c) - IG(M))
                else:
                    rr_ = at.integrated_gradients(model, X[k], int(comp[r][k]), None, IG_STEPS)
                    cache[key] = (rr_["attr"], rr_)
            AK[r][k] = cache[key][0]
    AM = {r: (A0 - AK[r]) * avail[r][:, None, None] for r in RULES}
    assert np.abs(AM["A"] - z3["attr_margin"]).max() < 1e-4, "rule A margin map differs from Experiment 3"
    say(f"Step 2: IG(f_k) for {sum(1 for v in cache.values() if v[1] is not None)} new (pair, competitor) combinations in {time.time() - t0:.0f}s")
    # direct margin IG check on a subset (linearity) + margin completeness
    rng = np.random.RandomState(xf.SEL_SEED)
    chk = []
    for k in rng.choice(n, 24, replace=False):
        for r in ("B", "C"):
            if avail[r][k] and len(chk) < 36:
                d = at.integrated_gradients(model, X[k], (int(cidx[k]), int(comp[r][k])), None, IG_STEPS)
                chk.append({"pair": int(k), "rule": r, "max_abs_diff": float(np.abs(d["attr"] - AM[r][k]).max()), "margin_completeness_rel": d["completeness_rel_err"]})
    lin_max = max(c["max_abs_diff"] for c in chk)
    say(f"linearity check: max |IG(M) - (IG(c) - IG(k))| over {len(chk)} (pair, rule) checks = {lin_max:.2e}; margin completeness rel err median {np.median([c['margin_completeness_rel'] for c in chk]):.4f}")
    # ---- Step 3: explanation stability
    tw, cw = at.region_ids("time", 12, 5000, WIN), at.region_ids("cell", 12, 5000, WIN)
    W = {r: np.stack([at.region_scores(AM[r][k].astype(float), tw[0], 40, "signed") for k in range(n)]) for r in RULES}
    Cc = {r: np.stack([at.region_scores(AM[r][k].astype(float), cw[0], 480, "signed") for k in range(n)]) for r in RULES}
    Wp = {r: np.stack([at.region_scores(np.maximum(AM[r][k], 0).astype(float), tw[0], 40, "signed") for k in range(n)]) for r in RULES}
    Wn = {r: np.stack([at.region_scores(np.minimum(AM[r][k], 0).astype(float), tw[0], 40, "signed") for k in range(n)]) for r in RULES}
    cor = lambda a, b: np.array([np.corrcoef(a[k], b[k])[0, 1] for k in range(len(a))])
    top = lambda s, sgn=1: [set(np.argsort(-sgn * s[k])[:8].tolist()) for k in range(len(s))]
    T8 = {r: top(W[r]) for r in RULES}
    N8 = {r: top(W[r], -1) for r in RULES}
    stab = {}
    for a_, b_, name in (("A", "B", "A_vs_B"), ("A", "C", "A_vs_C"), ("B", "C", "B_vs_C")):
        ok = avail[a_] & avail[b_]
        if name == "A_vs_B":
            ok = ok & sel_diffB                                                                # only pairs where the competitor really changes
        full = np.array([np.corrcoef(AM[a_][k].ravel(), AM[b_][k].ravel())[0, 1] if ok[k] else np.nan for k in range(n)])
        ov_pos = np.array([len(T8[a_][k] & T8[b_][k]) if ok[k] else np.nan for k in range(n)])
        ov_neg = np.array([len(N8[a_][k] & N8[b_][k]) if ok[k] else np.nan for k in range(n)])
        wc, cc_ = cor(W[a_], W[b_]), cor(Cc[a_], Cc[b_])
        wp, wn = cor(Wp[a_], Wp[b_]), cor(Wn[a_], Wn[b_])
        mass = lambda r: (np.maximum(AM[r], 0).sum((1, 2)), np.minimum(AM[r], 0).sum((1, 2)))
        stab[name] = {"n_pairs_compared": int(ok.sum()), "pearson_full_map": cl.mean_ci(full, ok), "pearson_window_signed": cl.mean_ci(wc, ok), "pearson_cell_signed": cl.mean_ci(cc_, ok),
                      "top8_positive_window_overlap_(chance_1.6)": cl.mean_ci(ov_pos, ok), "top8_negative_window_overlap_(chance_1.6)": cl.mean_ci(ov_neg, ok),
                      "pearson_positive_part_window_scores": cl.mean_ci(wp, ok), "pearson_negative_part_window_scores": cl.mean_ci(wn, ok),
                      "mean_positive_mass": {a_: float(mass(a_)[0][ok].mean()), b_: float(mass(b_)[0][ok].mean())}, "mean_negative_mass": {a_: float(mass(a_)[1][ok].mean()), b_: float(mass(b_)[1][ok].mean())},
                      "per_class_window_pearson": {c: float(np.nanmean(wc[ok & (cls == c)])) if (ok & (cls == c)).any() else None for c in sorted(set(cls))},
                      "per_class_top8_positive_overlap": {c: float(np.nanmean(ov_pos[ok & (cls == c)])) if (ok & (cls == c)).any() else None for c in sorted(set(cls))},
                      "per_class_n": {c: int((ok & (cls == c)).sum()) for c in sorted(set(cls))},
                      "correct_window_pearson": cl.mean_ci(wc, ok & corr), "incorrect_window_pearson": cl.mean_ci(wc, ok & ~corr)}
    # reference: how much of the contrastive map is the shared f_c component
    stab["shared_component_note"] = {"pearson_window_IGc_vs_IG_kA_vs_kB(target_IGs_of_different_competitors)": cl.mean_ci(cor(np.stack([at.region_scores(AK['A'][k].astype(float), tw[0], 40, 'signed') for k in range(n)]),
                                                                                                                 np.stack([at.region_scores(AK['B'][k].astype(float), tw[0], 40, 'signed') for k in range(n)])), sel_diffB)}
    # ---- Step 4: faithfulness against own margin, f_c alone, f_k alone
    curves = suite(model, X, A0, AM, AK, avail, comp, cidx, donor_of, say)
    assert h0 == hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest() and sha_file(xf.CKPT) == xf.CKPT_SHA
    say("model hash unchanged")
    ri = {"A": 1, "B": 2, "C": 3}
    faith, rows = {}, []
    for t in TESTS:
        tk = "|".join(t)
        faith[tk] = {}
        for r in RULES:
            sel = avail[r] & (sel_diffB if r == "B" else np.ones(n, bool))
            cols = {"own_margin": lambda c_, j=ri[r]: c_[..., 0] - c_[..., j], "f_c_alone": lambda c_: c_[..., 0], "f_k_alone": lambda c_, j=ri[r]: c_[..., j]}
            faith[tk][r] = {"n_pairs": int(sel.sum())}
            for tg, fcol in cols.items():
                rnd = aucs(curves, t, "random", fcol)
                res_t = {}
                for o in (f"{r}_cpos", f"{r}_cneg", f"{r}_clow", f"{r}_kpos", "c_pos", "c_low"):
                    v = aucs(curves, t, o, fcol) - rnd
                    res_t[o] = {"all": cl.mean_ci(v, sel), "correct": cl.mean_ci(v, sel & corr), "incorrect": cl.mean_ci(v, sel & ~corr)}
                    for c in sorted(set(cls)):
                        m = sel & (cls == c)
                        if m.any():
                            rows.append({"test": tk, "rule": r, "target": tg, "ordering": o, "class": c, "n_pairs": int(m.sum()), "n_records": int(len(set(rec[m]))), "raw_gap_vs_random": float(np.nanmean(v[m]))})
                faith[tk][r][tg] = res_t
    # paired comparisons that matter: contrastive vs ordinary IG on each target
    paired = {}
    for t in TESTS:
        tk = "|".join(t)
        paired[tk] = {}
        for r in RULES:
            sel = avail[r] & (sel_diffB if r == "B" else np.ones(n, bool))
            j = ri[r]
            fc, fk, mg = (lambda c_: c_[..., 0]), (lambda c_, j=j: c_[..., j]), (lambda c_, j=j: c_[..., 0] - c_[..., j])
            comp_ = {"margin: contrastive_pos minus IG(c)_pos": (mg, f"{r}_cpos", "c_pos"), "f_c: contrastive_pos minus IG(c)_pos": (fc, f"{r}_cpos", "c_pos"),
                     "f_k: contrastive_neg minus IG(k)_pos": (fk, f"{r}_cneg", f"{r}_kpos"), "f_k: contrastive_pos minus IG(k)_pos": (fk, f"{r}_cpos", f"{r}_kpos")}
            paired[tk][r] = {}
            for nm, (fcol, o1, o2) in comp_.items():
                d = aucs(curves, t, o1, fcol) - aucs(curves, t, o2, fcol)
                paired[tk][r][nm] = {"all": cl.mean_ci(d, sel), "correct": cl.mean_ci(d, sel & corr), "incorrect": cl.mean_ci(d, sel & ~corr), "share_pairs_contrastive_effect_stronger": float(((d < 0) if t[2] == "delete" else (d > 0))[sel].mean())}
    out = {"n_pairs": n, "n_unique_records": int(cl.R), "bootstrap": f"record-clustered, B={xa.BOOT_B}, seed={xa.BOOT_SEED}, percentile 95% CI", "step1_competitors": summary1,
           "step2": {"linearity_check": chk, "linearity_max_abs_diff": lin_max, "tolerance": 1e-4}, "step3_stability": stab, "step4_faithfulness_raw_gap_vs_random": faith, "step4_paired_contrastive_vs_ordinary_IG": paired,
           "convention": "raw gap = AUC(ordering) - AUC(random) of the change of the target from its starting value; deletion: negative = target falls faster than random; insertion: positive = recovers faster. "
                         "Evaluated targets: the rule's own margin, f_c alone, f_k alone. 'A_cneg' = most negative contrastive attribution first (evidence for the competitor); '*_kpos' = ordinary IG(k) highest first."}
    json.dump(out, open(os.path.join(run, "results.json"), "x"), indent=1, default=float)
    with open(os.path.join(run, "gap_by_class.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run, "pairs_competitors.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "record_id", "class", "correct"] + [f"{x}_{y}" for x in RULES for y in ("competitor", "margin")] + ["B_equals_A", "B_predicted_positive"])
        for k in range(n):
            w.writerow([k, rec[k], cls[k], corr[k]] + sum([[names[comp[r][k]] if avail[r][k] else "", marg[r][k]] for r in RULES], []) + [bool(not sel_diffB[k]), bool(pos_cls[k, comp["B"][k]])])
    np.savez_compressed(os.path.join(run, "attributions.npz"), pair=np.arange(n), record_id=rec, cls=cls, comp_A=comp["A"], comp_B=comp["B"], comp_C=comp["C"], avail_C=avail["C"],
                        attr_margin_B=AM["B"].astype(np.float32), attr_margin_C=AM["C"].astype(np.float32), attr_k_B=AK["B"].astype(np.float32), attr_k_C=AK["C"].astype(np.float32),
                        **{f"curve_{g}_{b}_{d}_{o}": v for (g, b, d), cv in curves.items() for o, v in cv.items()})
    json.dump({"previous_runs": [os.path.relpath(PREV, ROOT), os.path.relpath(C3, ROOT)], "checkpoint_sha256": xf.CKPT_SHA, "thresholds_sha256": sha_file(xf.THR), "ig": {"baseline": "zero", "steps": IG_STEPS, "rule": "midpoint"},
               "rules": {"A": xc.pick_competitor.__doc__, "B": pick_highest_logit.__doc__, "C": pick_second_eligible.__doc__}, "exp3_results_sha256": sha_file(c3res),
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("scripts/xai_competitor_robustness.py", "src/ecg_concept/attribution.py")}}, open(os.path.join(run, "config.json"), "x"), indent=1)
    say(f"saved; results sha256 {sha_file(os.path.join(run, 'results.json'))[:16]}")
    make_figures(run, X, A0, AM, AK, comp, avail, cidx, cls, rec, names, corr, W, stab, faith, LG, pairs, sel_diffB, cl)
    print(run)


def make_figures(run, X, A0, AM, AK, comp, avail, cidx, cls, rec, names, corr, W, stab, faith, LG, pairs, sel_diffB, cl):
    import xai_figures_v2 as v2                                                             # applies the figure-layer fixes found by rendering (clip, tick format)
    fg = v2.fg
    out = os.path.join(run, "figures")
    os.makedirs(out)
    cl_ = sorted(set(cls))
    # 1. stability by class
    s = fg.Svg(1500, 560)
    s.text(20, 28, "Stability of the contrastive explanation when the competitor changes (window-score Pearson between maps, signed)", 14, weight="bold")
    s.text(20, 46, "A vs B: pairs where B differs from A. A vs C: pairs where a second eligible competitor exists. Bars = class mean; overall values with record-clustered 95% CI in the text below.", 11, c="#333")
    f = fg.axes(s, 90, 80, 1300, 320, (0, len(cl_)), (-0.2, 1.0), [], [-0.2, 0, 0.25, 0.5, 0.75, 1.0], "", "Pearson of 250 ms window scores")
    for i, c in enumerate(cl_):
        for off, nm, col in ((0.1, "A_vs_B", "#D55E00"), (0.5, "A_vs_C", "#0072B2")):
            v = stab[nm]["per_class_window_pearson"][c]
            if v is None:
                continue
            x1, y1 = f(i + off, v)
            x2, y2 = f(i + off + 0.38, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), col, 0.9)
        s.text(f(i + 0.5, 0)[0], 80 + 320 + 16, f"{c} (B≠A: {stab['A_vs_B']['per_class_n'][c]}, C: {stab['A_vs_C']['per_class_n'][c]})", 9, "middle")
    s.rect(110, 440, 12, 12, "#D55E00")
    s.text(128, 451, "rule A vs rule B", 11)
    s.rect(260, 440, 12, 12, "#0072B2")
    s.text(278, 451, "rule A vs rule C", 11)
    fm = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    for i, nm in enumerate(("A_vs_B", "A_vs_C", "B_vs_C")):
        st = stab[nm]
        s.text(110, 478 + i * 18, f"{nm.replace('_', ' ')} (n={st['n_pairs_compared']}): window Pearson {fm(st['pearson_window_signed'])}; full map {fm(st['pearson_full_map'])}; top-8 positive overlap {fm(st['top8_positive_window_overlap_(chance_1.6)'])}; "
               f"top-8 negative overlap {fm(st['top8_negative_window_overlap_(chance_1.6)'])}", 11)
    s.save(os.path.join(out, "stability_by_class.svg"))
    # 2. faithfulness on three targets (time windows, zero baseline, deletion)
    s = fg.Svg(1500, 640)
    s.text(20, 28, "Contrastive vs ordinary IG orderings evaluated on three targets (250 ms windows, zero baseline, DELETION; raw AUC gap vs random, more negative = target falls faster)", 13, weight="bold")
    s.text(20, 46, "Each group: own margin | f_c alone | f_k alone. For f_k the ordinary comparator is IG(k); the contrastive ordering used is the most NEGATIVE first (evidence for the competitor). Whiskers: record-clustered 95% CI.", 11, c="#333")
    tk = "time|zero|delete"
    groups = []
    for r in RULES:
        for tg, o_c, o_s, lbl in (("own_margin", f"{r}_cpos", "c_pos", "margin"), ("f_c_alone", f"{r}_cpos", "c_pos", "f_c"), ("f_k_alone", f"{r}_cneg", f"{r}_kpos", "f_k")):
            g = faith[tk][r][tg]
            groups.append((f"rule {r}: {lbl}", g[o_c]["all"], g[o_s]["all"]))
    lo = min(min(g[1][1], g[2][1]) for g in groups) * 1.15
    hi = max(0.1, max(max(g[1][2], g[2][2]) for g in groups) * 1.15)
    f = fg.axes(s, 90, 90, 1350, 380, (0, len(groups)), (lo, hi), [], np.linspace(lo, hi, 6), "", "raw gap vs random (logit-AUC units)")
    for i, (lbl, a, b) in enumerate(groups):
        for off, (m, l_, h_), col in ((0.1, a, "#D55E00"), (0.5, b, "#7a2e00")):
            x1, y1 = f(i + off, m)
            x2, y2 = f(i + off + 0.38, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), col, 0.9)
            xm = (x1 + x2) / 2
            s.line(xm, f(0, l_)[1], xm, f(0, h_)[1], "#000", 1.2)
        s.text(f(i + 0.5, 0)[0], 90 + 380 + 16 + (0 if i % 2 == 0 else 14), lbl, 10, "middle")
    s.rect(110, 540, 12, 12, "#D55E00")
    s.text(128, 551, "contrastive ordering for that target", 11)
    s.rect(380, 540, 12, 12, "#7a2e00")
    s.text(398, 551, "ordinary IG(c) / IG(k) ordering", 11)
    s.save(os.path.join(out, "faithfulness_three_targets.svg"))
    # 3. example with three competitors
    ok = [k for k in range(len(cls)) if sel_diffB[k] and avail["C"][k]]
    wp = np.array([np.corrcoef(AM["A"][k].reshape(-1), AM["B"][k].reshape(-1))[0, 1] for k in ok])
    k = ok[int(np.argsort(wp)[len(wp) // 2])]
    W_, Rr, L = 1400, 1010, 70
    s = fg.Svg(W_, 1260)
    c = int(cidx[k])
    s.text(20, 26, f"{rec[k]}: target {cls[k]} - contrastive IG with three competitor rules (median A-vs-B similarity among eligible pairs; label for {cls[k]}: {'PRESENT' if corr[k] else 'ABSENT'})", 14, weight="bold")
    vmax = max(np.percentile(np.abs(AM[r][k].reshape(12, 250, 20).sum(2)), 99.5) for r in ("A", "B", "C")) + 1e-12
    top, rowh = 70, 26
    for l in range(12):
        yc = top + rowh * (l + 0.5)
        s.text(L - 8, yc + 4, fg.LEADS[l], 10, "end", weight="bold")
        seg = X[k][l].astype(float).reshape(1250, -1)
        pts = []
        for i in range(1250):
            xx = L + i / 1250 * (Rr - L)
            pts += [(xx, yc - np.clip(seg[i].min(), -4, 4) * rowh * 0.1), (xx, yc - np.clip(seg[i].max(), -4, 4) * rowh * 0.1)]
        s.poly(pts, "#222", 0.8)
    s.text(L, top - 8, "Input (filtered, z-scored)", 12, weight="bold")
    cw, hh = (Rr - L) / 250, 14
    for pi, r in enumerate(RULES):
        h0 = top + 12 * rowh + 50 + pi * (12 * hh + 38)
        kk = int(comp[r][k])
        s.text(L, h0 - 6, f"Rule {r}: competitor {names[kk]}  (margin M = {LG[k, c] - LG[k, kk]:.2f}) - red raises M, blue lowers; same colour scale in all three", 12, weight="bold")
        blocks = AM[r][k].reshape(12, 250, 20).sum(2)
        for l in range(12):
            s.text(L - 8, h0 + hh * (l + 0.5) + 3, fg.LEADS[l], 8, "end")
            for i in range(250):
                s.rect(L + i * cw, h0 + l * hh, cw + 0.3, hh, fg.div_color(blocks[l, i] / vmax))
        s.rect(L, h0, Rr - L, 12 * hh, "none", stroke="#444")
    s.text(L, 1244, f"colour scale +/-{vmax:.3g} logit units per 20-sample block. Maps describe the MODEL's sensitivity, not a physiological cause.", 10, c="#555")
    s.save(os.path.join(out, f"example_three_competitors_{cls[k]}_{rec[k]}.svg"))
    print(sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
