"""Experiment 2: does LightECGNetV2's lead attention explain the near class-independent IG lead profiles of Experiment 1?
Part A (unmodified model): extract the real lead-attention gates for the SAME 160 (record, class) pairs and relate them to IG lead shares and to single-lead removal effects.
Part B (inference-only intervention, in memory, checkpoint untouched): replace the gates by ONE uniform gate per record (the mean of that record's own gates, detached), keep the
ORIGINAL predicted class as the attribution target, recompute IG (same baseline/steps) and repeat the perturbation tests. All statistics use record-clustered bootstrap
(resample the unique ECG records with replacement, keep all their class pairs). Fold: frozen validation role only. test_reserved and train are never read.
Run: .venv\\Scripts\\python.exe -I scripts\\xai_attention_analysis.py        -> results/xai_attention/run_<stamp>/
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
import xai_ig_faithfulness as xf  # noqa: E402
import xai_ig_figures as fg  # noqa: E402

PREV = os.path.join(ROOT, "results", "xai_ig", "run_20261120_030747")
PREV_RESULTS_PREFIX = "4b5dd1a52f705528"
BOOT_B, BOOT_SEED, MODE = 5000, 20261120, "uniform_mean"
KS, ORD, WIN, N_RAND, IG_STEPS = xf.KS, xf.ORD, xf.WIN, xf.N_RAND, xf.IG_STEPS
TESTS = [("time", "zero", "delete"), ("time", "zero", "insert"), ("cell", "zero", "delete"), ("cell", "zero", "insert"), ("lead", "zero", "delete"),
         ("time", "donor", "delete"), ("time", "donor", "insert")]
sha_file, spearman = xf.sha_file, xf.spearman


# ------------------------------------------------------------------ statistics (record-clustered)
class Clust:
    def __init__(self, rec, B=BOOT_B, seed=BOOT_SEED):
        self.u, self.inv = np.unique(rec, return_inverse=True)
        self.R = len(self.u)
        rng = np.random.RandomState(seed)
        self.counts = np.stack([np.bincount(rng.randint(0, self.R, self.R), minlength=self.R) for _ in range(B)]).astype(float)       # (B, R) record multiplicities

    def mean_ci(self, v, sel=None):
        """cluster bootstrap CI of the mean of per-pair values v (optionally on a boolean subset)."""
        v = np.asarray(v, float)
        ok = np.isfinite(v) & (np.ones(len(v), bool) if sel is None else sel)
        S = np.bincount(self.inv[ok], weights=v[ok], minlength=self.R)
        N = np.bincount(self.inv[ok], minlength=self.R).astype(float)
        den = self.counts @ N
        m = (self.counts @ S) / np.where(den > 0, den, np.nan)
        return [float(v[ok].mean()), float(np.nanpercentile(m, 2.5)), float(np.nanpercentile(m, 97.5))]

    def similarity(self, X, cls):
        """within-class and between-class mean cosine over pairs of DIFFERENT records, cluster-bootstrapped."""
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        C = Xn @ Xn.T
        rec_i = self.inv
        valid = rec_i[:, None] != rec_i[None, :]
        same = cls[:, None] == cls[None, :]
        res = {}
        for name, mask in (("within", valid & same), ("between", valid & ~same)):
            vals = []
            for b in range(0, len(self.counts), 1):
                w = self.counts[b][rec_i]
                W = np.outer(w, w) * mask
                vals.append((W * C).sum() / W.sum())
            vals = np.array(vals)
            w0 = mask.astype(float)
            res[name] = [float((w0 * C).sum() / w0.sum()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        d = []
        for b in range(len(self.counts)):
            w = self.counts[b][rec_i]
            W = np.outer(w, w)
            ww, bb = W * (valid & same), W * (valid & ~same)
            d.append((ww * C).sum() / ww.sum() - (bb * C).sum() / bb.sum())
        res["difference_within_minus_between"] = [res["within"][0] - res["between"][0], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        return res


def centered(S):
    return S - S.mean(0, keepdims=True)


def loro_accuracy(S, cls, rec):
    """nearest-class-centroid (cosine) classification of centred lead-share vectors; centroids exclude the held-out ECG record. Returns per-pair correctness."""
    Z = centered(S)
    out = np.zeros(len(S))
    classes = np.unique(cls)
    for i in range(len(S)):
        keep = rec != rec[i]
        cents = np.stack([Z[keep & (cls == c)].mean(0) if (keep & (cls == c)).any() else np.zeros(Z.shape[1]) for c in classes])
        sim = (cents @ Z[i]) / (np.linalg.norm(cents, axis=1) * np.linalg.norm(Z[i]) + 1e-12)
        out[i] = float(classes[int(np.argmax(sim))] == cls[i])
    return out


# ------------------------------------------------------------------ perturbation suite (same procedure as Experiment 1)
def perturbation_suite(model, X, A, pairs, donor_of, cls_idx, say):
    rid = {g: at.region_ids(g, 12, 5000, WIN) for g in ("time", "cell", "lead")}
    curves, lead_single = {}, np.zeros((len(pairs), 12))
    zero = np.zeros((12, 5000), np.float32)
    t1 = time.time()
    for k in range(len(pairs)):
        c = cls_idx[k]
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
                    curves.setdefault((gran, bname, direction), {}).setdefault(o, np.zeros((len(pairs), len(ks))))[k] = lg[[i for i, t in enumerate(tags) if t == o]].mean(0)
        lm = (np.arange(12)[:, None, None] == np.arange(12)[None, :, None]) & np.ones((12, 12, 5000), bool)
        lead_single[k] = at.perturbed_logits(model, X[k], zero, c, lm, "delete")
        if k % 40 == 0:
            say(f"  perturbation tests (intervened model): pair {k}/{len(pairs)} ({time.time() - t1:.0f}s)")
    return curves, lead_single


def pair_gaps(curves, fx, fb):
    """per-pair gap vs random (positive = ordering is MORE effective than random) for every test and ordering."""
    out = {}
    for (gran, bname, direction), cv in curves.items():
        fr = KS[gran] / KS[gran][-1]
        start = fx if direction == "delete" else fb
        sgn = -1.0 if direction == "delete" else 1.0
        auc = {o: np.array([at.curve_auc(cv[o][i] - start[i], fr) for i in range(len(fx))]) for o in ORD}
        out[(gran, bname, direction)] = {o: sgn * (auc[o] - auc["random"]) for o in ("pos", "abs", "low_abs", "neg")}
        out[(gran, bname, direction)]["auc"] = auc
    return out


def lead_scores(A, lead_rid):
    sg = np.stack([at.region_scores(A[k].astype(np.float64), lead_rid, 12, "signed") for k in range(len(A))])
    ab = np.stack([at.region_scores(A[k].astype(np.float64), lead_rid, 12, "abs") for k in range(len(A))])
    return sg, ab


def main():
    cfg = json.load(open(os.path.join(PREV, "config.json")))
    assert sha_file(os.path.join(PREV, "results.json")).startswith(PREV_RESULTS_PREFIX), "Experiment-1 results changed"
    run = os.path.join(ROOT, "results", "xai_attention", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
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
    cfg_pairs = {p["pair"]: p for p in cfg["pairs"]}
    assert all(cfg_pairs[p["pair"]]["record_id"] == p["record_id"] for p in pairs)
    z = np.load(os.path.join(PREV, "attributions.npz"))
    A0 = z["attr"]
    lead_single0 = z["lead_single_logits"]
    curves0 = {(g, b, d): {o: z[f"curve_{g}_{b}_{d}_{o}"] for o in ORD} for (g, b, d) in TESTS}
    rec = np.array([p["record_id"] for p in pairs])
    cls = np.array([p["class"] for p in pairs])
    cidx = np.array([p["class_idx"] for p in pairs])
    fx0, fb0 = np.array([p["f_x"] for p in pairs]), np.array([p["f_base"] for p in pairs])
    corr = np.array([p["correct"] for p in pairs])
    cl = Clust(rec)
    say(f"{len(pairs)} (record, class) pairs, {cl.R} unique ECG records, classes {sorted(set(cls))}; cluster bootstrap B={BOOT_B}, seed {BOOT_SEED}")

    # ---- exact signals and model
    assert xf.sha_file(xf.CKPT) == xf.CKPT_SHA
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(xf.CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    assert all(r in rel for r in rec), "a selected record is not in the frozen validation role"
    sc_ = {}
    sig = lambda r: sc_.setdefault(r, load_ecg(os.path.join(xf.DS, rel[r])))
    X = np.stack([sig(r) for r in rec])
    donor_ids = cfg["donor_record_ids"]
    D = np.stack([sig(i) for i in donor_ids])
    N_D = len(donor_ids)
    donor_of = [D[(k + 1) % N_D] if donor_ids[(k + 1) % N_D] != rec[k] else D[(k + 2) % N_D] for k in range(len(pairs))]
    assert np.abs(X - z["x"].astype(np.float32)).max() < 0.05, "reloaded signals differ from Experiment 1"
    dev = "cuda"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = load_model(xf.CKPT, dev)
    h0 = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    thr = np.load(xf.THR, allow_pickle=False)
    with torch.no_grad():
        f_re = np.array([float(model(torch.from_numpy(X[k])[None].to(dev))[0, cidx[k]]) for k in range(len(pairs))])
    assert np.abs(f_re - fx0).max() < 1e-3
    say(f"original logits reproduced (max abs diff {np.abs(f_re - fx0).max():.2e})")

    # ============================== PART A: attention weights of the unmodified model
    Xt = torch.from_numpy(X).to(dev)
    s_t, mu_t = at.lead_attention_weights(model, Xt)
    S, MU = s_t.cpu().numpy().astype(np.float64), mu_t.cpu().numpy().astype(np.float64)
    la = model.lead_attn
    with torch.no_grad():
        s_zero = torch.sigmoid(la.fc2(torch.relu(la.fc1(torch.zeros(1, 12, device=dev))))).cpu().numpy()[0].astype(np.float64)
    urec = {r: i for i, r in reversed(list(enumerate(rec)))}
    Su = S[[urec[r] for r in cl.u]]                                                              # one row per unique record
    A_stat = {"gate_mean_by_lead": dict(zip(LEAD_NAMES, S.mean(0).round(5).tolist())), "gate_sd_across_records_by_lead": dict(zip(LEAD_NAMES, S.std(0).round(6).tolist())),
              "gate_mean_over_leads": float(S.mean()), "gate_min_lead_mean": float(S.mean(0).min()), "gate_max_lead_mean": float(S.mean(0).max()),
              "lead_to_lead_sd_of_mean_gate": float(S.mean(0).std()), "max_between_record_sd_any_lead": float(S.std(0).max()),
              "max_between_record_range_any_lead": float((S.max(0) - S.min(0)).max()), "max_abs_lead_mean_of_input (z-scored input)": float(np.abs(MU).max()),
              "gate_at_zero_input_by_lead": dict(zip(LEAD_NAMES, s_zero.round(5).tolist())), "max_abs_diff_gate_vs_gate_at_zero_input": float(np.abs(S - s_zero[None]).max()),
              "n_unique_records": int(cl.R), "ratio_between_lead_sd_to_between_record_sd": float(S.mean(0).std() / max(S.std(0).mean(), 1e-12))}
    centS = centered(S)
    A_stat["class_mean_gate_max_abs_deviation_from_overall"] = {c: float(np.abs(S[cls == c].mean(0) - S.mean(0)).max()) for c in sorted(set(cls))}
    A_stat["gate_cosine_similarity_raw"] = cl.similarity(S, cls)
    A_stat["gate_cosine_similarity_centred"] = cl.similarity(centS + 1e-9 * np.random.RandomState(0).randn(*centS.shape), cls)
    lead_rid = at.region_ids("lead", 12, 5000, WIN)[0]
    sg0, ab0 = lead_scores(A0, lead_rid)
    sh0 = ab0 / ab0.sum(1, keepdims=True)
    drop0 = fx0[:, None] - lead_single0
    A_stat["spearman_gate_vs_IG_abs_share_per_pair"] = cl.mean_ci([spearman(S[k], sh0[k]) for k in range(len(pairs))])
    A_stat["spearman_gate_vs_IG_signed_score_per_pair"] = cl.mean_ci([spearman(S[k], sg0[k]) for k in range(len(pairs))])
    A_stat["spearman_gate_vs_single_lead_drop_per_pair"] = cl.mean_ci([spearman(S[k], drop0[k]) for k in range(len(pairs))])
    mean_share = sh0.mean(0)
    A_stat["pooled_mean_IG_share_by_lead"] = dict(zip(LEAD_NAMES, mean_share.round(4).tolist()))
    A_stat["spearman_meangate_vs_meanIGshare_12_leads"] = spearman(S.mean(0), mean_share)
    A_stat["pearson_meangate_vs_meanIGshare_12_leads"] = float(np.corrcoef(S.mean(0), mean_share)[0, 1])
    A_stat["spearman_meangate_vs_mean_single_lead_drop_12_leads"] = spearman(S.mean(0), drop0.mean(0))
    A_stat["per_class_spearman_meangate_vs_class_mean_IGshare"] = {c: spearman(S[cls == c].mean(0), sh0[cls == c].mean(0)) for c in sorted(set(cls))}
    say(f"Part A: gate lead-means {S.mean(0).round(3).tolist()}; between-record sd max {S.std(0).max():.2e}; max |mean of input| {np.abs(MU).max():.2e}; "
        f"max |gate - gate(zero input)| {A_stat['max_abs_diff_gate_vs_gate_at_zero_input']:.2e}")

    # ============================== PART B: uniform-attention intervention
    A1 = np.zeros_like(A0)
    fx1, fb1 = np.zeros(len(pairs)), np.zeros(len(pairs))
    comp = []
    with at.lead_attention_intervention(model, "original"):                                       # the 'original' formula must equal the unmodified model exactly
        with torch.no_grad():
            f_orig_mode = np.array([float(model(torch.from_numpy(X[k])[None].to(dev))[0, cidx[k]]) for k in range(len(pairs))])
    assert np.abs(f_orig_mode - f_re).max() == 0.0, "'original' mode differs from the unmodified model"
    t0 = time.time()
    with at.lead_attention_intervention(model, MODE):
        for k in range(len(pairs)):
            r = at.integrated_gradients(model, X[k], cidx[k], None, IG_STEPS)
            A1[k] = r["attr"]
            fx1[k], fb1[k] = r["f_x"], r["f_baseline"]
            comp.append((r["completeness_abs_err"], r["completeness_rel_err"]))
        assert np.isfinite(A1).all()
        conv = []
        for k in np.random.RandomState(xf.SEL_SEED).choice(len(pairs), xf.CONV_PAIRS, replace=False):
            r32 = at.integrated_gradients(model, X[k], cidx[k], None, 32)
            r256 = at.integrated_gradients(model, X[k], cidx[k], None, 256)
            conv.append({"pair": int(k), "rel_err_32": r32["completeness_rel_err"], "rel_err_256": r256["completeness_rel_err"], "rel_err_64": comp[k][1],
                         "pearson_64_vs_256": float(np.corrcoef(A1[k].ravel(), r256["attr"].ravel())[0, 1])})
        say(f"Part B: IG on intervened model in {time.time() - t0:.0f}s; completeness rel err median {np.median([c[1] for c in comp]):.4f}; abs err median {np.median([c[0] for c in comp]):.4f}")
        curves1, lead_single1 = perturbation_suite(model, X, A1, pairs, donor_of, cidx, say)
    assert h0 == hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest() and sha_file(xf.CKPT) == xf.CKPT_SHA
    assert "forward" not in model.lead_attn.__dict__
    with torch.no_grad():
        f_after = np.array([float(model(torch.from_numpy(X[k])[None].to(dev))[0, cidx[k]]) for k in range(len(pairs))])
    assert np.abs(f_after - f_re).max() == 0.0, "original model outputs changed after the intervention was disabled"
    say("model hash unchanged; original logits identical after the intervention is disabled")

    # ---- comparison
    tlog = np.log(thr / (1 - thr))[cidx]
    res = {"n_pairs": len(pairs), "n_unique_records": int(cl.R), "bootstrap": f"record-clustered, B={BOOT_B}, seed={BOOT_SEED}, percentile 95% CI", "intervention": MODE,
           "partA_attention_weights": A_stat,
           "intervention_effect_on_logits": {"mean_abs_change_f_x": float(np.abs(fx1 - fx0).mean()), "mean_f_x_original": float(fx0.mean()), "mean_f_x_intervened": float(fx1.mean()),
                                             "pearson_f_x": float(np.corrcoef(fx0, fx1)[0, 1]), "share_still_above_original_threshold": float((fx1 > tlog).mean()),
                                             "mean_f_x_minus_f_base_original": float((fx0 - fb0).mean()), "mean_f_x_minus_f_base_intervened": float((fx1 - fb1).mean())},
           "IG_completeness": {"original": {"rel_median": float(np.median([p["comp_rel"] for p in pairs])), "abs_median": float(np.median([p["comp_abs"] for p in pairs]))},
                               "intervened": {"rel_median": float(np.median([c[1] for c in comp])), "abs_median": float(np.median([c[0] for c in comp])),
                                              "convergence": {"rel_err_32_median": float(np.median([c['rel_err_32'] for c in conv])), "rel_err_64_median": float(np.median([c['rel_err_64'] for c in conv])),
                                                              "rel_err_256_median": float(np.median([c['rel_err_256'] for c in conv])), "pearson_64_vs_256_median": float(np.median([c['pearson_64_vs_256'] for c in conv]))}}}}
    sg1, ab1 = lead_scores(A1, lead_rid)
    sh1 = ab1 / ab1.sum(1, keepdims=True)
    drop1 = fx1[:, None] - lead_single1
    lead = {}
    for name, sh, sg, drop in (("original", sh0, sg0, drop0), ("intervened", sh1, sg1, drop1)):
        acc = loro_accuracy(sh, cls, rec)
        lead[name] = {"mean_share_by_lead": dict(zip(LEAD_NAMES, sh.mean(0).round(4).tolist())), "mean_top_lead_share": float(sh.max(1).mean()),
                      "cosine_raw": cl.similarity(sh, cls), "cosine_centred": cl.similarity(centered(sh), cls),
                      "class_identification_accuracy_from_centred_lead_share_(chance_0.125)": cl.mean_ci(acc),
                      "spearman_IGsigned_vs_single_lead_drop": cl.mean_ci([spearman(sg[k], drop[k]) for k in range(len(pairs))]),
                      "top1_lead_agreement_IGsigned_vs_removal": cl.mean_ci((sg.argmax(1) == drop.argmax(1)).astype(float)),
                      "mean_max_single_lead_drop": float(drop.max(1).mean()), "per_class_mean_share": {c: sh[cls == c].mean(0).round(4).tolist() for c in sorted(set(cls))}}
    lead["share_profile_similarity_original_vs_intervened"] = {"pearson_of_mean_profiles": float(np.corrcoef(sh0.mean(0), sh1.mean(0))[0, 1]),
                                                              "per_pair_spearman": cl.mean_ci([spearman(sh0[k], sh1[k]) for k in range(len(pairs))])}
    lead["spearman_gate_vs_intervened_IG_abs_share_per_pair"] = cl.mean_ci([spearman(S[k], sh1[k]) for k in range(len(pairs))])
    res["lead_level"] = lead
    # temporal pattern comparison
    tr = at.region_ids("time", 12, 5000, WIN)[0]
    w0 = np.stack([at.region_scores(A0[k].astype(np.float64), tr, 40) for k in range(len(pairs))])
    w1 = np.stack([at.region_scores(A1[k].astype(np.float64), tr, 40) for k in range(len(pairs))])
    wa0 = np.stack([at.region_scores(np.abs(A0[k]).astype(np.float64), tr, 40) for k in range(len(pairs))])
    wa1 = np.stack([at.region_scores(np.abs(A1[k]).astype(np.float64), tr, 40) for k in range(len(pairs))])
    res["temporal_comparison"] = {"pearson_signed_window_scores_per_pair": cl.mean_ci([np.corrcoef(w0[k], w1[k])[0, 1] for k in range(len(pairs))]),
                                  "pearson_abs_window_scores_per_pair": cl.mean_ci([np.corrcoef(wa0[k], wa1[k])[0, 1] for k in range(len(pairs))]),
                                  "pearson_full_signed_map_per_pair": cl.mean_ci([np.corrcoef(A0[k].ravel(), A1[k].ravel())[0, 1] for k in range(len(pairs))]),
                                  "top8_window_overlap_mean_of_8": cl.mean_ci([len(set(np.argsort(-w0[k])[:8]) & set(np.argsort(-w1[k])[:8])) for k in range(len(pairs))]),
                                  "top8_window_overlap_expected_by_chance": 8 * 8 / 40,
                                  "top_window_share_of_abs_IG": {"original": float((wa0 / wa0.sum(1, keepdims=True)).max(1).mean()), "intervened": float((wa1 / wa1.sum(1, keepdims=True)).max(1).mean())}}
    # faithfulness
    g0, g1 = pair_gaps(curves0, fx0, fb0), pair_gaps(curves1, fx1, fb1)
    faith, rows = {}, []
    for t in TESTS:
        key = "|".join(t)
        faith[key] = {}
        for o in ("pos", "abs", "low_abs", "neg"):
            a, b = g0[t][o], g1[t][o]
            faith[key][o] = {"original_gap": cl.mean_ci(a), "intervened_gap": cl.mean_ci(b), "intervened_minus_original": cl.mean_ci(b - a),
                             "share_pairs_better_than_random_original": float((a > 0).mean()), "share_pairs_better_than_random_intervened": float((b > 0).mean())}
            for c in sorted(set(cls)):
                m = cls == c
                rows.append({"test": key, "ordering": o, "class": c, "n_pairs": int(m.sum()), "n_records": int(len(set(rec[m]))), "original_gap": float(a[m].mean()), "intervened_gap": float(b[m].mean())})
    res["faithfulness_gap_vs_random"] = faith
    res["single_lead_removal_logit_drop"] = {"mean_max_single_lead_drop_original": float(drop0.max(1).mean()), "mean_max_single_lead_drop_intervened": float(drop1.max(1).mean()),
                                              "share_pairs_any_single_lead_removal_flips_original": float((lead_single0 < tlog.reshape(-1, 1) - 0).any(1).mean()),
                                              "share_pairs_any_single_lead_removal_flips_intervened": float((lead_single1 < np.log(thr / (1 - thr))[cidx].reshape(-1, 1)).any(1).mean())}
    json.dump(res, open(os.path.join(run, "results.json"), "x"), indent=1, default=float)
    with open(os.path.join(run, "gap_by_class.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(run, "attention_and_ig.npz"), gates=S, input_lead_means=MU, logit_original=fx0, logit_intervened=fx1, f_base_original=fb0, f_base_intervened=fb1,
                        attr_intervened=A1, lead_single_logits_intervened=lead_single1, pair=np.arange(len(pairs)), record_id=rec, cls=cls,
                        **{f"curve_{g}_{b}_{d}_{o}": v for (g, b, d), cv in curves1.items() for o, v in cv.items()})
    json.dump({"previous_run": os.path.relpath(PREV, ROOT), "previous_results_sha256_prefix": PREV_RESULTS_PREFIX, "previous_attributions_sha256": sha_file(os.path.join(PREV, "attributions.npz")),
               "checkpoint_sha256": xf.CKPT_SHA, "intervention": "inference-only, in memory: LeadAttention gates replaced by one gate per record = mean over leads of the record's own gates (detached); checkpoint untouched",
               "target": "the ORIGINAL predicted class index of each pair (fixed)", "ig": {"baseline": "zero in normalised space", "steps": IG_STEPS, "rule": "midpoint"},
               "pairs_file": "xai_ig run pairs.csv (same 160 pairs)", "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("scripts/xai_attention_analysis.py", "src/ecg_concept/attribution.py")}},
              open(os.path.join(run, "config.json"), "x"), indent=1)
    say(f"results written; sha256 {sha_file(os.path.join(run, 'results.json'))[:16]}")
    print(run)
    make_figures(run, res, S, cls, sh0, sh1, g0, g1, rec, cl)


def make_figures(run, res, S, cls, sh0, sh1, g0, g1, rec, cl):
    out = os.path.join(run, "figures")
    os.makedirs(out)
    L = fg.LEADS
    s = fg.Svg(1500, 980)
    s.text(20, 28, "Does lead attention explain the near class-independent IG lead profile?", 16, weight="bold")
    s.text(20, 48, f"n = {res['n_pairs']} (record, class) pairs from {res['n_unique_records']} ECG records; intervention: lead-attention gates -> one uniform gate per record (in memory only)", 11, c="#333")
    # A: gates
    f = fg.axes(s, 90, 100, 560, 250, (-0.5, 11.5), (0, max(0.6, float(S.max()) * 1.15)), range(12), np.linspace(0, max(0.6, float(S.max()) * 1.15), 5), "lead", "attention gate s", "A. Actual lead-attention gate by lead (mean over records, bars = +/- 1 sd)")
    for j in range(12):
        m, sd = S[:, j].mean(), S[:, j].std()
        x1, y1 = f(j - 0.35, m)
        x2, y2 = f(j + 0.35, 0)
        s.rect(x1, y1, x2 - x1, y2 - y1, "#888", 0.85)
        s.line(*f(j, m - sd), *f(j, m + sd), c="#000", sw=1.4)
        s.text(f(j, 0)[0], 100 + 250 + 14, L[j], 10, "middle")
    s.text(90, 395, f"gate sd across records: max {res['partA_attention_weights']['max_between_record_sd_any_lead']:.1e}; lead-to-lead sd of the mean gate {res['partA_attention_weights']['lead_to_lead_sd_of_mean_gate']:.3f}; "
           f"max |mean of z-scored input| {res['partA_attention_weights']['max_abs_lead_mean_of_input (z-scored input)']:.1e}", 11)
    # B: IG share original vs intervened vs gate
    f = fg.axes(s, 800, 100, 640, 250, (-0.5, 11.5), (0, max(sh0.mean(0).max(), sh1.mean(0).max()) * 1.2), range(12), np.linspace(0, max(sh0.mean(0).max(), sh1.mean(0).max()) * 1.2, 5), "lead", "mean share of |IG|", "B. Mean IG lead share: original vs uniform attention")
    for j in range(12):
        for off, v, col in ((-0.2, sh0[:, j].mean(), "#D55E00"), (0.2, sh1[:, j].mean(), "#0072B2")):
            x1, y1 = f(j + off - 0.18, v)
            x2, y2 = f(j + off + 0.18, 0)
            s.rect(x1, y1, x2 - x1, y2 - y1, col, 0.85)
        s.text(f(j, 0)[0], 100 + 250 + 14, L[j], 10, "middle")
    s.rect(820, 112, 12, 12, "#D55E00")
    s.text(838, 123, "original model", 11)
    s.rect(940, 112, 12, 12, "#0072B2")
    s.text(958, 123, "uniform attention (IG recomputed)", 11)
    # C: per-class heatmaps
    cl_ = sorted(set(cls))
    for pi, (title, sh) in enumerate((("C1. Per-class mean IG lead share, original", sh0), ("C2. Per-class mean IG lead share, uniform attention", sh1))):
        x0 = 90 + pi * 700
        s.text(x0, 450, title, 12, weight="bold")
        M = np.array([sh[cls == c].mean(0) for c in cl_])
        for j in range(12):
            s.text(x0 + 40 + j * 40 + 20, 470, L[j], 10, "middle")
        for i, c in enumerate(cl_):
            s.text(x0 + 32, 470 + i * 28 + 22, c, 11, "end")
            for j in range(12):
                s.rect(x0 + 40 + j * 40, 478 + i * 28, 39, 27, fg.seq_color(M[i, j] / 0.2))
                s.text(x0 + 40 + j * 40 + 20, 478 + i * 28 + 18, f"{M[i, j]:.2f}", 9, "middle")
    # D: similarity summary text
    lv = res["lead_level"]
    y = 740
    s.text(90, y, "D. Class information in the lead profile (record-clustered 95% CI)", 13, weight="bold")
    for name in ("original", "intervened"):
        c = lv[name]["cosine_centred"]
        a = lv[name]["class_identification_accuracy_from_centred_lead_share_(chance_0.125)"]
        r_ = lv[name]["cosine_raw"]
        s.text(90, y + 24 + (0 if name == "original" else 44), f"{name}: raw cosine within {r_['within'][0]:.3f} vs between {r_['between'][0]:.3f}; centred cosine within {c['within'][0]:.3f} vs between {c['between'][0]:.3f} "
               f"(difference {c['difference_within_minus_between'][0]:.3f} [{c['difference_within_minus_between'][1]:.3f}, {c['difference_within_minus_between'][2]:.3f}])", 12)
        s.text(90, y + 42 + (0 if name == "original" else 44), f"      leave-one-record-out class identification from the centred lead share: {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] (chance 0.125)", 12)
    s.save(os.path.join(out, "attention_vs_ig_lead_profiles.svg"))
    # faithfulness comparison figure
    s = fg.Svg(1500, 640)
    s.text(20, 28, "Attribution-guided vs random perturbation: original vs uniform-attention model (gap = ordering minus random; >0 means the ranking is more effective than random)", 14, weight="bold")
    s.text(20, 48, "Highest signed attribution ordering. Bars: pooled mean over pairs, whiskers: record-clustered 95% CI. The intervention changes the model itself, so a change is not evidence that the original attention was unfaithful.", 11, c="#333")
    tests = [("|".join(t), t) for t in TESTS]
    vals = [(res["faithfulness_gap_vs_random"][k]["pos"]["original_gap"], res["faithfulness_gap_vs_random"][k]["pos"]["intervened_gap"]) for k, _ in tests]
    top = max(max(v[0][2], v[1][2]) for v in vals) * 1.1
    bot = min(0, min(min(v[0][1], v[1][1]) for v in vals) * 1.2)
    f = fg.axes(s, 90, 100, 1350, 380, (0, len(tests)), (bot, top), [], np.linspace(bot, top, 6), "", "gap vs random (logit-AUC units)")
    for i, (k, t) in enumerate(tests):
        for off, (m, lo, hi), col in ((0.1, vals[i][0], "#D55E00"), (0.5, vals[i][1], "#0072B2")):
            x1, y1 = f(i + off, m)
            x2, y2 = f(i + off + 0.38, 0)
            s.rect(x1, min(y1, y2), x2 - x1, abs(y2 - y1), col, 0.85)
            xm = (x1 + x2) / 2
            s.line(xm, f(0, lo)[1], xm, f(0, hi)[1], "#000", 1.3)
        s.text(f(i + 0.5, 0)[0], 100 + 380 + 16 + (0 if i % 2 == 0 else 14), f"{t[0]} / {t[1]} / {t[2]}", 10, "middle")
    s.rect(110, 540, 12, 12, "#D55E00")
    s.text(128, 551, "original model", 11)
    s.rect(260, 540, 12, 12, "#0072B2")
    s.text(278, 551, "uniform attention", 11)
    s.save(os.path.join(out, "faithfulness_original_vs_uniform_attention.svg"))
    print(sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
