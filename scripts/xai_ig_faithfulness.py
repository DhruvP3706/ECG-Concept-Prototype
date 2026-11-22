"""Experiment 1: input-level explanation faithfulness of the ORIGINAL frozen LightECGNetV2 (ft_fold0, 45 PhysioNet classes).
Integrated Gradients on the pre-sigmoid logit of a class the MODEL predicts (probability > its saved per-class threshold), zero baseline in the normalised signal space;
perturbation tests (deletion / insertion with contiguous 250 ms windows, lead x window cells, whole leads; attribution-ranked vs random vs low-attribution vs negative-attribution
orderings; zero baseline and a donor-ECG baseline). Population: records of the frozen VALIDATION role only (original fold 0: epoch/threshold-selection fold of ft_fold0, weights never
trained on these signals; its saved thresholds WERE selected on it). test_reserved and train records are never read. Nothing is trained or modified.
Run: .venv\\Scripts\\python.exe -I scripts\\xai_ig_faithfulness.py        -> results/xai_ig/run_<stamp>/
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
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import attribution as at  # noqa: E402
from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.preprocessing import LEAD_NAMES, load_ecg  # noqa: E402

CACHE = os.path.join(ROOT, "results", "step4_4", "cache")
CKPT = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")
THR = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0_thresh.npy")
CKPT_SHA = "7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72"
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
SEL_SEED, N_CLASSES, PER_CLASS, MAX_WRONG, IG_STEPS, WIN, N_RAND, N_DONORS, CONV_PAIRS = 2026, 8, 20, 5, 64, 125, 10, 60, 24
KS = {"time": np.round(np.linspace(0, 40, 21)).astype(int), "cell": np.round(np.linspace(0, 480, 21)).astype(int), "lead": np.arange(13)}
ORD = ("pos", "abs", "low_abs", "neg", "random")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def boot_ci(v, seed=0, B=4000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.RandomState(seed)
    m = v[rng.randint(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(v))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    assert sha_file(CKPT) == CKPT_SHA, "checkpoint hash mismatch"
    run = os.path.join(ROOT, "results", "xai_ig", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run)
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    z = np.load(os.path.join(CACHE, "merged.npz"))
    ids_all, role_all = z["ids"], z["role"]
    names = [str(c) for c in z["class_names"]]
    va = np.where(role_all == "validation")[0]
    assert len(va) == 4437 and set(np.unique(role_all)) <= {"train", "validation"}
    thr = np.load(THR, allow_pickle=False)
    logits_va = z["orig_logits"][va].astype(np.float64)
    pred = (1 / (1 + np.exp(-logits_va))) > thr[None]
    lab = z["labels"][va].astype(bool)
    ids_va = ids_all[va]
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    # ---- selection (seeded)
    rng = np.random.RandomState(SEL_SEED)
    cls_sel = list(np.argsort(-pred.sum(0), kind="stable")[:N_CLASSES])
    pairs = []
    for c in cls_sel:
        pos = np.where(pred[:, c])[0]
        correct, wrong = pos[lab[pos, c]], pos[~lab[pos, c]]
        nw = min(MAX_WRONG, len(wrong))
        take = list(rng.choice(wrong, nw, replace=False)) + list(rng.choice(correct, min(PER_CLASS - nw, len(correct)), replace=False))
        rng.shuffle(take)
        for i in take:
            pairs.append({"va_idx": int(i), "record_id": str(ids_va[i]), "class_idx": int(c), "class": names[c], "correct": bool(lab[i, c]), "logit_cached": float(logits_va[i, c])})
    for k, p in enumerate(pairs):
        p["pair"] = k
    say(f"selected {len(pairs)} (record, class) pairs, classes {[names[c] for c in cls_sel]}; predicted positives in validation {[int(pred[:, c].sum()) for c in cls_sel]}; "
        f"incorrect (model positive, label absent): {sum(not p['correct'] for p in pairs)}; seed {SEL_SEED}")
    assert all(role_all[va[p["va_idx"]]] == "validation" for p in pairs)
    # ---- signals with the EXACT original preprocessing
    sig_cache = {}

    def sig(rid):
        if rid not in sig_cache:
            sig_cache[rid] = load_ecg(os.path.join(DS, rel[rid]))
        return sig_cache[rid]

    dev = "cuda"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = load_model(CKPT, dev)
    h0 = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    X = np.stack([sig(p["record_id"]) for p in pairs])
    assert X.shape[1:] == (12, 5000) and X.dtype == np.float32 and np.isfinite(X).all()
    with torch.no_grad():
        rec = np.array([float(model(torch.from_numpy(X[k])[None].to(dev))[0, p["class_idx"]]) for k, p in enumerate(pairs)])
    err = np.abs(rec - np.array([p["logit_cached"] for p in pairs]))
    say(f"preprocessing/model check: recomputed logit vs cached Step 4.4 logit: max abs diff {err.max():.2e}")
    assert err.max() < 1e-3, "preprocessing or checkpoint does not reproduce the cached logits"
    donor_ids = [str(i) for i in rng.choice([r for r in ids_va if True], N_DONORS, replace=False)]
    D = np.stack([sig(i) for i in donor_ids])
    donor_of = [D[(k + 1) % N_DONORS] if donor_ids[(k + 1) % N_DONORS] != p["record_id"] else D[(k + 2) % N_DONORS] for k, p in enumerate(pairs)]
    json.dump({"selection_seed": SEL_SEED, "classes": [names[c] for c in cls_sel], "pairs": pairs, "donor_record_ids": donor_ids, "n_random_orders": N_RAND, "ig_steps": IG_STEPS, "window_samples": WIN,
               "population": "frozen role 'validation' (original fold 0); test_reserved and train never read",
               "model": {"checkpoint": "models/pretrained/lightv2_ft_fold0.pt", "sha256": CKPT_SHA, "thresholds_sha256": sha_file(THR)},
               "caveat": "thresholds were selected on this same fold; its signals were not in the weights; explanations are of the model's predictions, not of ground truth",
               "fold_manifest_sha256": sha_file(os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv")), "cache_merged_sha256": sha_file(os.path.join(CACHE, "merged.npz")),
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("src/ecg_concept/attribution.py", "scripts/xai_ig_faithfulness.py", "src/ecg_concept/preprocessing.py")}},
              open(os.path.join(run, "config.json"), "x"), indent=1)

    # ---- Integrated Gradients
    t0 = time.time()
    A = np.zeros((len(pairs), 12, 5000), np.float32)
    for k, p in enumerate(pairs):
        r = at.integrated_gradients(model, X[k], p["class_idx"], None, IG_STEPS)
        A[k] = r["attr"]
        p.update(f_x=r["f_x"], f_base=r["f_baseline"], comp_abs=r["completeness_abs_err"], comp_rel=r["completeness_rel_err"])
        assert np.isfinite(r["attr"]).all()
    say(f"IG ({IG_STEPS} steps) for {len(pairs)} pairs in {time.time() - t0:.0f}s; completeness rel err median {np.median([p['comp_rel'] for p in pairs]):.4f}, "
        f"max {max(p['comp_rel'] for p in pairs):.4f}")
    conv = []
    for k in np.random.RandomState(SEL_SEED).choice(len(pairs), CONV_PAIRS, replace=False):
        r32 = at.integrated_gradients(model, X[k], pairs[k]["class_idx"], None, 32)
        r256 = at.integrated_gradients(model, X[k], pairs[k]["class_idx"], None, 256)
        rid_t, nt = at.region_ids("time", 12, 5000, WIN)
        top = lambda a: set(np.argsort(-at.region_scores(a, rid_t, nt))[:8].tolist())
        conv.append({"pair": int(k), "rel_err_32": r32["completeness_rel_err"], "rel_err_64": pairs[k]["comp_rel"], "rel_err_256": r256["completeness_rel_err"],
                     "pearson_64_vs_256": float(np.corrcoef(A[k].ravel(), r256["attr"].ravel())[0, 1]), "pearson_32_vs_256": float(np.corrcoef(r32["attr"].ravel(), r256["attr"].ravel())[0, 1]),
                     "top8_window_overlap_64_vs_256": len(top(A[k]) & top(r256["attr"]))})
    say(f"convergence on {CONV_PAIRS} pairs: median completeness rel err 32/64/256 = {np.median([c['rel_err_32'] for c in conv]):.4f}/{np.median([c['rel_err_64'] for c in conv]):.4f}/"
        f"{np.median([c['rel_err_256'] for c in conv]):.4f}; median Pearson 64 vs 256 = {np.median([c['pearson_64_vs_256'] for c in conv]):.4f}; median top-8-window overlap {np.median([c['top8_window_overlap_64_vs_256'] for c in conv])}/8")

    # ---- perturbation tests
    rid = {g: at.region_ids(g, 12, 5000, WIN) for g in ("time", "cell", "lead")}
    curves = {}                                                                             # (gran, base, direction) -> ordering -> (n_pairs, K)

    def store(key, o, k, v):
        curves.setdefault(key, {}).setdefault(o, np.zeros((len(pairs), len(v))))[k] = v

    thr_logit = np.log(thr / (1 - thr))
    lead_single = np.zeros((len(pairs), 12))
    t1 = time.time()
    zero = np.zeros((12, 5000), np.float32)
    for k, p in enumerate(pairs):
        c = p["class_idx"]
        a = A[k].astype(np.float64)
        prng = np.random.RandomState(1000 + k)
        for gran in ("time", "cell", "lead"):
            r_id, nreg = rid[gran]
            od = at.orderings(at.region_scores(a, r_id, nreg, "signed"), at.region_scores(a, r_id, nreg, "abs"), prng, N_RAND)
            ks = KS[gran]
            plan = [(zero, "zero", d) for d in ("delete", "insert")] if gran != "lead" else [(zero, "zero", "delete")]
            if gran == "time":
                plan += [(donor_of[k], "donor", d) for d in ("delete", "insert")]
            for base, bname, direction in plan:
                masks, tags = [], []
                for o in ORD:
                    for perm in (od[o] if o == "random" else [od[o]]):
                        masks.append(at.prefix_masks(perm, r_id, ks))
                        tags.append(o)
                lg = at.perturbed_logits(model, X[k], base, c, np.concatenate(masks), direction).reshape(len(tags), len(ks))
                for o in ORD:
                    store((gran, bname, direction), o, k, lg[[i for i, t in enumerate(tags) if t == o]].mean(0))
                    if o == "random":
                        curves.setdefault(("rand_all", gran, bname, direction), {}).setdefault(k, lg[[i for i, t in enumerate(tags) if t == "random"]])
        lm = (np.arange(12)[:, None, None] == np.arange(12)[None, :, None]) & np.ones((12, 12, 5000), bool)      # lm[l] = lead l alone replaced by the baseline
        lead_single[k] = at.perturbed_logits(model, X[k], zero, c, lm, "delete")
        if k % 20 == 0:
            say(f"perturbation tests: pair {k}/{len(pairs)} ({time.time() - t1:.0f}s)")
    assert h0 == hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest() and sha_file(CKPT) == CKPT_SHA, "model changed"
    say(f"perturbation tests done in {time.time() - t1:.0f}s; model hash unchanged")

    # ---- metrics
    fx = np.array([p["f_x"] for p in pairs])
    fb = np.array([p["f_base"] for p in pairs])
    cls = np.array([p["class"] for p in pairs])
    corr = np.array([p["correct"] for p in pairs])
    out = {"n_pairs": len(pairs), "completeness": {"rel_err_median": float(np.median([p["comp_rel"] for p in pairs])), "rel_err_max": float(max(p["comp_rel"] for p in pairs)),
                                                    "abs_err_median": float(np.median([p["comp_abs"] for p in pairs]))}, "convergence": conv}
    summary, rows = {}, []
    for (gran, bname, direction), cv in [(k, v) for k, v in curves.items() if k[0] in ("time", "cell", "lead")]:
        fr = KS[gran] / KS[gran][-1]
        pref = fr <= 0.3 + 1e-9
        base_val = fx if direction == "delete" else fb
        res = {}
        for o in ORD:
            d = cv[o] - base_val[:, None]                                                       # change from the starting point (deletion: from f(x); insertion: from f(baseline))
            auc = np.array([at.curve_auc(d[i], fr) for i in range(len(d))])
            auc30 = np.array([at.curve_auc(d[i][pref], fr[pref]) for i in range(len(d))])
            res[o] = {"mean_curve": d.mean(0).tolist(), "se_curve": (d.std(0, ddof=1) / np.sqrt(len(d))).tolist(), "auc": boot_ci(auc), "auc_first30pct": boot_ci(auc30), "auc_per_pair": auc,
                      "auc30_per_pair": auc30}
        sgn = -1.0 if direction == "delete" else 1.0                                              # "better" = more negative (deletion) / more positive (insertion)
        gaps = {}
        for o in ("pos", "abs", "low_abs", "neg"):
            g = sgn * (res[o]["auc_per_pair"] - res["random"]["auc_per_pair"])                       # > 0: ordering is MORE effective than random
            g30 = sgn * (res[o]["auc30_per_pair"] - res["random"]["auc30_per_pair"])
            gaps[o] = {"gap_vs_random_auc": boot_ci(g), "gap_vs_random_auc30": boot_ci(g30), "share_pairs_better_than_random": float((g > 0).mean())}
            for c_ in np.unique(cls):
                rows.append({"gran": gran, "baseline": bname, "direction": direction, "ordering": o, "class": c_, "n": int((cls == c_).sum()), "mean_gap_vs_random": float(g[cls == c_].mean())})
            rows.append({"gran": gran, "baseline": bname, "direction": direction, "ordering": o, "class": "ALL_correct", "n": int(corr.sum()), "mean_gap_vs_random": float(g[corr].mean())})
            rows.append({"gran": gran, "baseline": bname, "direction": direction, "ordering": o, "class": "ALL_incorrect", "n": int((~corr).sum()), "mean_gap_vs_random": float(g[~corr].mean()) if (~corr).any() else float("nan")})
        summary[f"{gran}|{bname}|{direction}"] = {"fractions": fr.tolist(), "orderings": {o: {k2: v2 for k2, v2 in res[o].items() if not k2.endswith("per_pair")} for o in ORD}, "gaps": gaps}
        # normalised recovery/drop for pairs with a clear margin
        margin = (fx - fb) >= 2.0
        if margin.any():
            nrm = {}
            for o in ORD:
                v = (cv[o][margin] - fb[margin, None]) / (fx[margin] - fb[margin])[:, None]       # 1 = original logit, 0 = baseline logit
                nrm[o] = v.mean(0).tolist()
            summary[f"{gran}|{bname}|{direction}"]["normalised_logit_fraction_of_f(x)-f(base)_pairs_with_margin>=2"] = {"n": int(margin.sum()), "curves": nrm}
    out["curves"] = summary
    # flip rates (prediction leaves 'positive' under the model's own threshold)
    tl = thr_logit[[p["class_idx"] for p in pairs]]
    flips = {}
    for gran, kk in (("time", [4, 8, 20]), ("lead", [1, 2, 3])):
        cv = curves[(gran, "zero", "delete")]
        ka = {int(v): i for i, v in enumerate(KS[gran])}
        for o in ("pos", "abs", "low_abs", "random"):
            for kv in kk:
                if gran == "time" and kv not in (4, 8, 20):
                    continue
                flips[f"{gran}|k={kv}|{o}"] = float((cv[o][:, ka[int(kv)]] < tl).mean())
    # random flip rate should use all permutations rather than the mean curve
    for gran, kk in (("time", [4, 8, 20]), ("lead", [1, 2, 3])):
        ra = curves[("rand_all", gran, "zero", "delete")]
        ka = {int(v): i for i, v in enumerate(KS[gran])}
        for kv in kk:
            flips[f"{gran}|k={kv}|random"] = float(np.mean([(ra[k][:, ka[kv]] < tl[k]).mean() for k in range(len(pairs))]))
    out["prediction_flip_rate_after_deletion_(zero_baseline)"] = flips
    # lead-level
    drop = fx[:, None] - lead_single                                                          # logit drop when the lead alone is zeroed
    lead_signed = np.stack([at.region_scores(A[k].astype(np.float64), rid["lead"][0], 12, "signed") for k in range(len(pairs))])
    lead_abs = np.stack([at.region_scores(A[k].astype(np.float64), rid["lead"][0], 12, "abs") for k in range(len(pairs))])
    rho = np.array([spearman(lead_signed[k], drop[k]) for k in range(len(pairs))])
    rho_abs = np.array([spearman(lead_abs[k], np.abs(drop[k])) for k in range(len(pairs))])
    out["lead_level"] = {"spearman_IGsigned_vs_single_lead_drop": boot_ci(rho), "spearman_IGabs_vs_abs_single_lead_drop": boot_ci(rho_abs),
                         "top1_lead_agreement_signed": float(np.mean(lead_signed.argmax(1) == drop.argmax(1))), "chance_top1": 1 / 12,
                         "mean_max_single_lead_drop": float(drop.max(1).mean()), "mean_fx_minus_fbase": float((fx - fb).mean()),
                         "share_pairs_where_best_IG_lead_dropped_flips_prediction": float((lead_single[np.arange(len(pairs)), lead_signed.argmax(1)] < tl).mean()),
                         "share_pairs_where_any_single_lead_flips": float(((lead_single < tl[:, None]).any(1)).mean())}
    share = lead_abs / lead_abs.sum(1, keepdims=True)
    out["lead_level"]["mean_top_lead_share_of_abs_IG"] = float(share.max(1).mean())
    out["lead_level"]["share_pairs_top_lead_over_30pct"] = float((share.max(1) > 0.30).mean())
    # concentration / edge checks / within-class consistency
    wsc = np.stack([at.region_scores(np.abs(A[k]).astype(np.float64), rid["time"][0], 40) for k in range(len(pairs))])
    wsh = wsc / wsc.sum(1, keepdims=True)
    out["concentration"] = {"mean_top1_window_share_of_40": float(wsh.max(1).mean()), "uniform_value": 1 / 40, "mean_top4_windows_share": float(np.sort(wsh, 1)[:, -4:].sum(1).mean()),
                            "mean_edge_share_first_plus_last_window": float((wsh[:, 0] + wsh[:, -1]).mean()), "uniform_edge_share": 2 / 40}
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    within, between = [], []
    r2 = np.random.RandomState(7)
    for _ in range(4000):
        i, j = r2.choice(len(pairs), 2, replace=False)
        (within if cls[i] == cls[j] else between).append(cos(share[i], share[j]))
    out["lead_share_cosine_similarity"] = {"within_class_mean": float(np.mean(within)), "between_class_mean": float(np.mean(between)), "n_within": len(within), "n_between": len(between)}
    out["class_lead_share_mean"] = {c_: share[cls == c_].mean(0).tolist() for c_ in np.unique(cls)}
    out["lead_names"] = list(LEAD_NAMES)
    out["accuracy_note"] = {"n_correct": int(corr.sum()), "n_incorrect": int((~corr).sum())}
    json.dump(out, open(os.path.join(run, "results.json"), "x"), indent=1)
    with open(os.path.join(run, "gap_by_class.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run, "pairs.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "record_id", "class", "class_idx", "correct", "va_idx", "logit_cached", "f_x", "f_base", "comp_abs", "comp_rel"])
        w.writeheader()
        w.writerows([{k: p[k] for k in w.fieldnames} for p in pairs])
    np.savez_compressed(os.path.join(run, "attributions.npz"), attr=A, pair=np.arange(len(pairs)), x=X.astype(np.float16), lead_single_logits=lead_single,
                        **{f"curve_{key[0]}_{key[1]}_{key[2]}_{o}": v for key, cv in curves.items() if key[0] in ("time", "cell", "lead") for o, v in cv.items()})
    say(f"saved. results sha256 {sha_file(os.path.join(run, 'results.json'))[:16]}")
    print(run)


if __name__ == "__main__":
    main()
