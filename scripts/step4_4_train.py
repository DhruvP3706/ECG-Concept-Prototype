"""Step 6.1 first concept-prototype experiment (validation only; test records are never read).

Stages:  config  -> write + hash experiment_config.json BEFORE any fitting (declares seeds, hyperparameters, populations)
         train   -> verify config/cache fingerprints; for seeds 42,43,44 train A..D + controls; write metrics, prototypes, coverage, checkpoints
Run:  .venv\\Scripts\\python.exe -I scripts\\step4_4_train.py config   then   ... train
"""
import csv
import glob
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import concept_proto as cp  # noqa: E402
from ecg_concept import metrics as M  # noqa: E402
from ecg_concept.labels import CLASS_NAMES  # noqa: E402

OUT = os.path.join(ROOT, "results", "step4_4")
CACHE = os.path.join(OUT, "cache")
CKDIR = os.path.join(OUT, "checkpoints")
CFG = os.path.join(OUT, "experiment_config.json")
LOGF = os.path.join(OUT, "execution_log.txt")
ROLE = os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv")
CONCEPTS = ["accepted_rate_mean_bpm", "accepted_rr_cv", "accepted_rr_nrmssd", "accepted_rr_mad_succ"]
SEEDS = [42, 43, 44]
HP = {
    "scaling": {"embedding": "per-dimension standardization, mean/std fitted on TRAIN embedding-eligible records only (std floor 1e-6)",
                "concepts": "per-concept standardization of ACCEPTED train targets only (std floor 1e-6); predictions inverse-transformed for error reporting",
                "mask_features": "standardization fitted on train"},
    "optimizer": {"name": "AdamW", "weight_decay": 1e-4, "batch_size": 256, "max_epochs": 100, "early_stopping_patience": 10, "device": "cpu float32"},
    "models": {
        "A_orig_head": "frozen ft_fold0 45-way head, sigmoid, threshold 0.5 (historical validation exposure: fold 0 was its validation fold)",
        "B_embedding_linear": {"arch": "Linear(512,45)", "loss": "BCE", "lr": 1e-3, "selection": "validation BCE"},
        "C_concept_predictor": {"arch": "Linear(512,128)-ReLU-Linear(128,4)", "loss": "masked per-concept standardized MSE (equal concept weights)", "lr": 1e-3,
                                "selection": "validation masked MSE; frozen afterwards"},
        "C2_concept_only": {"arch": "Linear(4,45) on frozen PREDICTED standardized concepts", "loss": "BCE", "lr": 1e-2, "selection": "validation BCE"},
        "D_prototype": {"K": 16, "init": "deterministic k-means (farthest-point init) in predicted standardized concept space on train records with all 4 targets accepted; "
                                          "prototype = nearest actual training record to each centre (anchored, NOT learned)",
                        "similarity": "exp(-squared_distance / tau)", "tau_rule": "median over ALL train embedding-eligible records of the squared distance to the nearest prototype, floor 1e-6",
                        "head": "Linear(16,45) on similarities only (signed weights allowed)", "loss": "BCE", "lr": 1e-2, "selection": "validation BCE",
                        "concept_predictor_updated_by_diagnosis_loss": False},
        "CTRL_mask_only": {"arch": "Linear(19,45)", "features": "acceptance states, quality-lead counts, reason flags (no values, no beat counts)", "lr": 1e-2, "role": "control only; never feeds a proposed model"},
        "CTRL_extracted_target": {"arch": "Linear(4,45) on standardized EXTRACTED accepted targets", "population": "fully accepted subset", "lr": 1e-2, "role": "auxiliary baseline (not an oracle)"},
    },
    "threshold": 0.5, "threshold_optimisation": "none", "calibration": "none", "seeds": SEEDS, "concept_targets": CONCEPTS,
    "diagnostics_not_inputs": ["rate_median_bpm_cons", "n_beats_cons", "n_rr_cons", "detector timing diagnostics"],
    "selection_note": "validation data are used for early stopping/selection in every trained model, so validation metrics are mildly optimistic and equally so across models",
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def never_overwrite(*ps):
    for p in ps:
        assert not os.path.exists(p), f"refusing to overwrite {p}"


def load():
    z = np.load(os.path.join(CACHE, "merged.npz"))
    d = {k: z[k] for k in z.files}
    role, ok = d["role"], d["emb_ok"]
    d["tr"], d["va"] = (role == "train") & ok, (role == "validation") & ok
    d["T"] = np.stack([d[c] for c in CONCEPTS], 1)
    d["acc"] = np.isfinite(d["T"])
    d["full"] = d["acc"].all(1)
    assert set(np.unique(role)) <= {"train", "validation"}
    assert (d["full"] == ((d["rate_state"] == 2) & (d["irr_state"] == 2))).all(), "accepted masks inconsistent with states"
    names = [str(x) for x in d["reason_names"]]
    rf = d["reason_flags"].astype(np.float64)
    d["F"] = np.column_stack([d["rate_state"] == 2, d["irr_state"] == 2, d["rate_state"] == 1, d["irr_state"] == 1, d["n_usable_leads"], d["n_flat_leads"],
                              d["n_clip_leads"], d["n_nonfinite_leads"], rf]).astype(np.float64)
    d["F_names"] = ["rate_accepted", "irr_accepted", "rate_uncertain", "irr_uncertain", "n_usable_leads", "n_flat_leads", "n_clip_leads", "n_nonfinite_leads"] + names
    return d


def populations(d):
    return {"train_embedding_eligible": int(d["tr"].sum()), "validation_embedding_eligible": int(d["va"].sum()),
            "train_records": int((d["role"] == "train").sum()), "validation_records": int((d["role"] == "validation").sum()),
            "train_accepted_per_concept": {c: int((d["acc"][d["tr"], i]).sum()) for i, c in enumerate(CONCEPTS)},
            "validation_accepted_per_concept": {c: int((d["acc"][d["va"], i]).sum()) for i, c in enumerate(CONCEPTS)},
            "train_fully_accepted": int((d["full"] & d["tr"]).sum()), "validation_fully_accepted": int((d["full"] & d["va"]).sum())}


def stage_config():
    never_overwrite(CFG)
    d = load()
    meta = json.load(open(os.path.join(CACHE, "cache_meta.json")))
    mm = json.load(open(os.path.join(CACHE, "merged_meta.json")))
    bb = json.load(open(os.path.join(CACHE, "backbone_check.json")))
    assert bb["unchanged"]
    decl = {"declared_before_fitting": True, "declared_at": time.strftime("%Y-%m-%d %H:%M:%S"), "hyperparameters": HP, "populations": populations(d),
            "cache": {"merged_sha256": mm["merged_sha256"], "fingerprints": meta["fingerprints"], "backbone_check": bb},
            "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("src/ecg_concept/concept_proto.py", "src/ecg_concept/metrics.py", "scripts/step4_4_train.py",
                                                                          "scripts/step4_4_cache.py", "tests/test_step4_4.py")},
            "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
            "claims": "concept targets are automatically derived 10-second RR statistics (Step 4.3), not expert-validated; validation only; no test data"}
    decl["config_sha256"] = hashlib.sha256(json.dumps(decl, sort_keys=True).encode()).hexdigest()
    json.dump(decl, open(CFG, "w"), indent=1)
    log("experiment config written", decl["config_sha256"][:16], decl["populations"])


# --------------------------------------------------------------------------- evaluation helpers
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def evaluate(y, p):
    y = y.astype(bool)
    r = {**M.aggregate_ranking(y, p), **M.threshold_metrics(y, p, 0.5)}
    r["n"] = int(len(y))
    r["n_classes_with_positives"] = int((y.sum(0) > 0).sum())
    pc = {"auroc": M.auroc_columns(y, p), "auprc": M.ap_columns(y, p)}
    pr, rc, f1, _ = M.per_class_prf(y, p > 0.5)
    pc.update(precision=pr, recall=rc, f1=f1, n_pos=y.sum(0))
    return r, pc


def concept_eval(Zhat, T, acc, cs, tr, va, const):
    rows = []
    for split, sel in (("train", tr), ("validation", va)):
        pred = cs.inverse(Zhat[sel])
        for i, c in enumerate(CONCEPTS):
            m = acc[sel, i]
            e = pred[m, i] - T[sel][m, i]
            ce = const[i] - T[sel][m, i]
            rows.append({"split": split, "concept": c, "n_accepted": int(m.sum()), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                         "const_mae": float(np.abs(ce).mean()), "const_rmse": float(np.sqrt((ce ** 2).mean())), "mae_ratio_vs_const": float(np.abs(e).mean() / np.abs(ce).mean()),
                         "rmse_ratio_vs_const": float(np.sqrt((e ** 2).mean()) / np.sqrt((ce ** 2).mean())),
                         "r2_vs_const": float(1 - (e ** 2).sum() / (ce ** 2).sum())})
    return rows


def run_seed(d, seed, keep=True):
    """Train every model for one seed. All fitting uses train rows; validation rows only for early stopping/evaluation."""
    tr, va = d["tr"], d["va"]
    Y = d["labels"].astype(np.float32)
    # ---- scalers (train only)
    es = cp.Standardizer.fit(d["pooled"][tr])
    X = es.transform(d["pooled"]).astype(np.float32)
    cs = cp.Standardizer.fit(d["T"][tr])                                        # nanmean/nanstd over accepted train targets
    Tz = cs.transform(d["T"])
    mk = d["acc"]
    out = {"scalers": {"embedding": es, "concepts": cs}, "hist": {}, "probs": {}, "state": {}}
    hp = HP["models"]
    # ---- B: embedding linear
    torch.manual_seed(seed)
    mB = nn.Linear(512, 45)
    mB, h = cp.fit_head(mB, X[tr], Y[tr], X[va], Y[va], seed, 1e-3, HP["optimizer"]["max_epochs"], HP["optimizer"]["early_stopping_patience"], HP["optimizer"]["batch_size"])
    out["probs"]["B_embedding_linear"] = sig(cp.predict(mB, X[va]))
    out["hist"]["B_embedding_linear"], out["state"]["B_embedding_linear"] = h, mB.state_dict()
    # ---- C: concept predictor (masked), frozen afterwards
    torch.manual_seed(seed)
    mC = cp.ConceptMLP(512, 128, 4)
    mC, h = cp.fit_head(mC, X[tr], np.nan_to_num(Tz[tr]), X[va], np.nan_to_num(Tz[va]), seed, 1e-3, HP["optimizer"]["max_epochs"],
                        HP["optimizer"]["early_stopping_patience"], HP["optimizer"]["batch_size"], loss="mse", mask_tr=mk[tr], mask_va=mk[va])
    for p in mC.parameters():
        p.requires_grad_(False)
    mC.eval()
    c_hash_before = hashlib.sha256(b"".join(v.numpy().tobytes() for v in mC.state_dict().values())).hexdigest()
    Zhat = cp.predict(mC, X).astype(np.float64)                                  # predicted standardized concepts for ALL eligible rows
    assert np.isfinite(Zhat[tr | va]).all()
    out["hist"]["C_concept_predictor"], out["state"]["C_concept_predictor"], out["Zhat"] = h, mC.state_dict(), Zhat
    const = np.nanmean(d["T"][tr], axis=0)                                       # train-only constant baseline (original units)
    out["concept_eval"] = concept_eval(Zhat, d["T"], mk, cs, tr, va, const)
    # ---- C2: concept-only head on predicted concepts
    torch.manual_seed(seed)
    m2 = nn.Linear(4, 45)
    m2, h = cp.fit_head(m2, Zhat[tr], Y[tr], Zhat[va], Y[va], seed, 1e-2, HP["optimizer"]["max_epochs"], HP["optimizer"]["early_stopping_patience"], HP["optimizer"]["batch_size"])
    out["probs"]["C2_concept_only"] = sig(cp.predict(m2, Zhat[va]))
    out["hist"]["C2_concept_only"], out["state"]["C2_concept_only"] = h, m2.state_dict()
    # ---- D: anchored prototypes on the SAME frozen predicted concepts
    fa = np.where(d["full"] & tr)[0]
    centres, assign = cp.kmeans_det(Zhat[fa], 16)
    anchor = np.array(fa)[cp.anchor_records(Zhat[fa], centres)]                  # indices into the merged arrays (training records)
    P = Zhat[anchor]
    tau = cp.tau_rule(Zhat[tr], P)
    S = cp.similarities(Zhat, P, tau).astype(np.float32)
    torch.manual_seed(seed)
    mD = cp.ProtoHead(16, 45)
    mD, h = cp.fit_head(mD, S[tr], Y[tr], S[va], Y[va], seed, 1e-2, HP["optimizer"]["max_epochs"], HP["optimizer"]["early_stopping_patience"], HP["optimizer"]["batch_size"])
    assert hashlib.sha256(b"".join(v.numpy().tobytes() for v in mC.state_dict().values())).hexdigest() == c_hash_before, "concept predictor changed during head training"
    out["probs"]["D_prototype"] = sig(cp.predict(mD, S[va]))
    out["hist"]["D_prototype"], out["state"]["D_prototype"] = h, mD.state_dict()
    out["proto"] = {"anchor_idx": anchor, "P": P, "tau": tau, "S": S, "head": mD, "cluster_sizes": np.bincount(assign, minlength=16), "concept_predictor_sha": c_hash_before}
    # ---- controls
    fs = cp.Standardizer.fit(d["F"][tr])
    Fz = fs.transform(d["F"]).astype(np.float32)
    torch.manual_seed(seed)
    mM = nn.Linear(Fz.shape[1], 45)
    mM, h = cp.fit_head(mM, Fz[tr], Y[tr], Fz[va], Y[va], seed, 1e-2, HP["optimizer"]["max_epochs"], HP["optimizer"]["early_stopping_patience"], HP["optimizer"]["batch_size"])
    out["probs"]["CTRL_mask_only"] = sig(cp.predict(mM, Fz[va]))
    out["hist"]["CTRL_mask_only"], out["state"]["CTRL_mask_only"] = h, mM.state_dict()
    ftr, fva = np.where(d["full"] & tr)[0], np.where(d["full"] & va)[0]
    torch.manual_seed(seed)
    mE = nn.Linear(4, 45)
    mE, h = cp.fit_head(mE, Tz[ftr], Y[ftr], Tz[fva], Y[fva], seed, 1e-2, HP["optimizer"]["max_epochs"], HP["optimizer"]["early_stopping_patience"], HP["optimizer"]["batch_size"])
    pe = np.full((int(va.sum()), 45), np.nan, np.float32)
    va_pos = {int(i): k for k, i in enumerate(np.where(va)[0])}
    pe[[va_pos[int(i)] for i in fva]] = sig(cp.predict(mE, Tz[fva]))
    out["probs"]["CTRL_extracted_target"] = pe
    out["hist"]["CTRL_extracted_target"], out["state"]["CTRL_extracted_target"] = h, mE.state_dict()
    out["extra"] = {"mask_feature_scaler": fs, "const": const}
    return out


def stage_train():
    cfg = json.load(open(CFG))
    claimed = cfg.pop("config_sha256")
    assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == claimed, "experiment config altered after declaration"
    assert sha_file(os.path.join(CACHE, "merged.npz")) == cfg["cache"]["merged_sha256"], "merged cache changed since declaration"
    for f, h in cfg["code_sha256"].items():
        assert sha_file(os.path.join(ROOT, f)) == h, f"code changed since declaration: {f}"
    never_overwrite(*(os.path.join(OUT, f) for f in ("concept_prediction_metrics.csv", "validation_metrics.csv", "validation_per_class.csv", "prototype_summary.csv", "engineering_checks.json")))
    os.makedirs(CKDIR, exist_ok=True)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    d = load()
    tr, va = d["tr"], d["va"]
    ids_va = d["ids"][va]
    Yva = d["labels"][va]
    full_va = d["full"][va]
    pops = {"val_all": np.ones(len(ids_va), bool), "val_full_accepted": full_va, "val_not_full_accepted": ~full_va}
    t0 = time.time()
    runs = {}
    for s in SEEDS:
        runs[s] = run_seed(d, s)
        log(f"seed {s} trained in {time.time() - t0:.0f}s; epochs:", {k: v["epochs_run"] for k, v in runs[s]["hist"].items()},
            "tau", round(runs[s]["proto"]["tau"], 4))
    # ---- reproducibility check: retrain seed 42 chain and compare validation probabilities
    rep = run_seed(d, 42)
    repro = {k: float(np.nanmax(np.abs(rep["probs"][k] - runs[42]["probs"][k]))) for k in rep["probs"]}
    repro["concept_Zhat_max_abs_diff"] = float(np.abs(rep["Zhat"][tr | va] - runs[42]["Zhat"][tr | va]).max())
    log("reproducibility (seed 42 re-run) max abs prob diff:", repro)
    # ---- diagnosis metrics
    A = sig(d["orig_logits"][va])
    models = ["A_orig_head", "B_embedding_linear", "C2_concept_only", "D_prototype", "CTRL_mask_only", "CTRL_extracted_target"]
    res, pcs = {}, {}
    for m in models:
        for pop, sel in pops.items():
            if m == "CTRL_extracted_target" and pop != "val_full_accepted":
                continue
            seeds_here = [None] if m == "A_orig_head" else SEEDS
            for s in seeds_here:
                p = A if m == "A_orig_head" else runs[s]["probs"][m]
                r, pc = evaluate(Yva[sel], p[sel])
                res[(m, pop, s)], pcs[(m, pop, s)] = r, pc
    keys = ["macro_auroc", "weighted_auroc", "micro_auroc", "macro_auprc", "micro_auprc", "macro_f1_nb", "macro_f1_valid", "micro_f1", "micro_precision", "micro_recall",
            "exact_match", "hamming_loss", "n", "n_classes_with_positives"]

    def agg(m, pop):
        if m == "A_orig_head":
            r = res[(m, pop, None)]
            return {k: (r[k], 0.0) for k in keys}
        v = [res[(m, pop, s)] for s in SEEDS]
        return {k: (float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v], ddof=0))) for k in keys}

    rows = []
    for m in models:
        for pop in pops:
            if m == "CTRL_extracted_target" and pop != "val_full_accepted":
                continue
            for s in ([None] if m == "A_orig_head" else SEEDS):
                rows.append({"model": m, "population": pop, "stat": "single" if s is None else f"seed{s}", **{k: res[(m, pop, s)][k] for k in keys}})
            a = agg(m, pop)
            rows.append({"model": m, "population": pop, "stat": "mean", **{k: a[k][0] for k in keys}})
            rows.append({"model": m, "population": pop, "stat": "sd_across_seeds", **{k: a[k][1] for k in keys}})
    pairs = [("B_embedding_linear", "A_orig_head"), ("C2_concept_only", "B_embedding_linear"), ("D_prototype", "C2_concept_only"), ("D_prototype", "B_embedding_linear"),
             ("C2_concept_only", "A_orig_head"), ("D_prototype", "A_orig_head"), ("CTRL_mask_only", "C2_concept_only")]
    for a_, b_ in pairs:
        for pop in pops:
            diffs = []
            for s in SEEDS:
                ra = res[(a_, pop, None if a_ == "A_orig_head" else s)]
                rb = res[(b_, pop, None if b_ == "A_orig_head" else s)]
                diffs.append({k: ra[k] - rb[k] for k in keys if k not in ("n", "n_classes_with_positives")})
            rows.append({"model": f"{a_}_minus_{b_}", "population": pop, "stat": "paired_diff_mean", **{k: float(np.mean([x[k] for x in diffs])) for k in diffs[0]}})
            rows.append({"model": f"{a_}_minus_{b_}", "population": pop, "stat": "paired_diff_sd_across_seeds", **{k: float(np.std([x[k] for x in diffs])) for k in diffs[0]}})
    for b_ in ("C2_concept_only", "D_prototype", "B_embedding_linear"):
        diffs = []
        for s in SEEDS:
            ra, rb = res[("CTRL_extracted_target", "val_full_accepted", s)], res[(b_, "val_full_accepted", s)]
            diffs.append({k: ra[k] - rb[k] for k in keys if k not in ("n", "n_classes_with_positives")})
        rows.append({"model": f"CTRL_extracted_target_minus_{b_}", "population": "val_full_accepted", "stat": "paired_diff_mean", **{k: float(np.mean([x[k] for x in diffs])) for k in diffs[0]}})
    write_csv(os.path.join(OUT, "validation_metrics.csv"), rows)
    # per class
    crow = []
    for m in models:
        for pop in pops:
            if m == "CTRL_extracted_target" and pop != "val_full_accepted":
                continue
            ss = [None] if m == "A_orig_head" else SEEDS
            for j, c in enumerate(CLASS_NAMES):
                r = {"model": m, "population": pop, "class": c, "n_pos": int(pcs[(m, pop, ss[0])]["n_pos"][j]), "n": int(pops[pop].sum())}
                for k in ("auroc", "auprc", "precision", "recall", "f1"):
                    v = np.array([pcs[(m, pop, s)][k][j] for s in ss], float)
                    r[k + "_mean"], r[k + "_sd"] = (float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")), (float(np.nanstd(v)) if np.isfinite(v).any() else float("nan"))
                crow.append(r)
    write_csv(os.path.join(OUT, "validation_per_class.csv"), crow)
    # concept prediction metrics
    cprow = []
    for s in SEEDS:
        for r in runs[s]["concept_eval"]:
            cprow.append({"seed": s, **r, "best_epoch": runs[s]["hist"]["C_concept_predictor"]["best_epoch"]})
    write_csv(os.path.join(OUT, "concept_prediction_metrics.csv"), cprow)
    # prototypes
    prow, pcheck = [], {}
    for s in SEEDS:
        pr = runs[s]["proto"]
        head, S_, P_, anc = pr["head"], pr["S"], pr["P"], pr["anchor_idx"]
        cs = runs[s]["scalers"]["concepts"]
        Sva = torch.as_tensor(S_[va])
        with torch.no_grad():
            contrib = head.contributions(Sva)
            logits = head(Sva)
            recon = head.fc.bias[None, :] + contrib.sum(-1)
        ident = float((logits - recon).abs().max())
        W = head.fc.weight.detach().numpy()
        nearest = S_[va].argmax(1)
        usage = np.bincount(nearest, minlength=16)
        Dp = np.sqrt(cp.sqdist(P_, P_) + 0.0)
        iu = np.triu_indices(16, 1)
        corr = np.corrcoef(S_[va].T)
        base = evaluate(Yva, sig(logits.numpy()))[0]
        abl = []
        for k in range(16):
            s2 = S_[va].copy()
            s2[:, k] = 0.0
            with torch.no_grad():
                l2 = head(torch.as_tensor(s2)).numpy()
            r2 = evaluate(Yva, sig(l2))[0]
            abl.append({"d_macro_auroc": r2["macro_auroc"] - base["macro_auroc"], "d_micro_f1": r2["micro_f1"] - base["micro_f1"], "mean_abs_logit_change": float(np.abs(l2 - logits.numpy()).mean())})
        orig = cs.inverse(P_)
        ext = d["T"][anc]
        for k in range(16):
            top_pos = np.argsort(-W[:, k])[:3]
            top_neg = np.argsort(W[:, k])[:3]
            rec = {"seed": s, "prototype": k, "anchor_train_id": str(d["ids"][anc[k]]), "cluster_size_train_fully_accepted": int(pr["cluster_sizes"][k]),
                   "val_nearest_count": int(usage[k]), "val_mean_similarity": float(S_[va][:, k].mean()),
                   "supporting_classes(top3 +w)": "|".join(f"{CLASS_NAMES[c]}:{W[c, k]:+.2f}" for c in top_pos), "opposing_classes(top3 -w)": "|".join(f"{CLASS_NAMES[c]}:{W[c, k]:+.2f}" for c in top_neg),
                   "ablation_d_macro_auroc": abl[k]["d_macro_auroc"], "ablation_d_micro_f1": abl[k]["d_micro_f1"], "ablation_mean_abs_logit_change": abl[k]["mean_abs_logit_change"],
                   "nearest_other_prototype_dist_std": float(np.min(np.delete(Dp[k], k))), "max_similarity_corr_with_other": float(np.max(np.delete(corr[k], k)))}
            for i, c in enumerate(CONCEPTS):
                rec["pred_" + c.replace("accepted_", "")] = float(orig[k, i])
                rec["extracted_" + c.replace("accepted_", "")] = float(ext[k, i])
            prow.append(rec)
        pcheck[s] = {"tau": pr["tau"], "logit_identity_max_abs_diff": ident, "prototypes_never_nearest_in_val": int((usage == 0).sum()), "max_usage_fraction": float(usage.max() / usage.sum()),
                     "min_pairwise_dist_std": float(Dp[iu].min()), "pairs_closer_than_0.25_std": int((Dp[iu] < 0.25).sum()), "pairs_similarity_corr_gt_0.95": int((corr[iu] > 0.95).sum()),
                     "anchors_all_training_ids": bool(all(d["role"][a] == "train" for a in anc)), "n_unique_anchors": int(len(set(anc.tolist())))}
        pth = os.path.join(CKDIR, f"seed{s}_D_prototype_anchors.pt")
        never_overwrite(pth)
        torch.save({"prototype_idx_in_merged": anc.tolist(), "prototype_train_ids": [str(d["ids"][a]) for a in anc], "P_standardized": torch.as_tensor(P_), "tau": pr["tau"]}, pth)
    write_csv(os.path.join(OUT, "prototype_summary.csv"), prow)
    # checkpoints / histories / transforms
    for s in SEEDS:
        for m, st in runs[s]["state"].items():
            p = os.path.join(CKDIR, f"seed{s}_{m}.pt")
            never_overwrite(p)
            torch.save(st, p)
        json.dump(runs[s]["hist"], open(os.path.join(CKDIR, f"seed{s}_histories.json"), "w"))
    tf = {"embedding_mean_std_per_seed_identical": True, "embedding": runs[42]["scalers"]["embedding"].to_dict(), "concepts": runs[42]["scalers"]["concepts"].to_dict(), "concept_names": CONCEPTS,
          "mask_features": runs[42]["extra"]["mask_feature_scaler"].to_dict(), "mask_feature_names": d["F_names"], "constant_concept_baseline_original_units": runs[42]["extra"]["const"].tolist()}
    json.dump(tf, open(os.path.join(OUT, "transforms.json"), "w"))
    # ---- engineering checks
    role_rows = list(csv.DictReader(open(ROLE)))
    forbidden = {r["record_id"] for r in role_rows if r["role"] in ("test_reserved", "excluded")}
    ids_all = set(d["ids"].tolist())
    g_tr = {g for g, r in zip(d["dup_group"], d["role"]) if r == "train" and g}
    g_va = {g for g, r in zip(d["dup_group"], d["role"]) if r == "validation" and g}
    # alignment spot check: independent re-read of chunk files by record ID
    import random
    rnd = random.Random(0)
    sample = rnd.sample(list(d["ids"]), 300)
    pos = {r: i for i, r in enumerate(d["ids"])}
    conc_rows = {}
    for p in glob.glob(os.path.join(CACHE, "concepts", "chunk_*.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            conc_rows[r["record_id"]] = r
    emb_seen = {}
    for p in glob.glob(os.path.join(CACHE, "embeddings", "chunk_*.npz")):
        z = np.load(p)
        for k, rid in enumerate(z["ids"]):
            if str(rid) in set(sample):
                emb_seen[str(rid)] = (z["pooled"][k], z["logits"][k])
    align_ok = True
    for rid in sample:
        i = pos[rid]
        v = conc_rows[rid]["accepted_rate_mean_bpm"]
        a = float(v) if v not in ("", "nan") else float("nan")
        b = d["accepted_rate_mean_bpm"][i]
        align_ok &= (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9
        align_ok &= bool(np.array_equal(emb_seen[rid][0], d["pooled"][i], equal_nan=True) and np.array_equal(emb_seen[rid][1], d["orig_logits"][i], equal_nan=True))
    es = runs[42]["scalers"]["embedding"]
    bb = json.load(open(os.path.join(CACHE, "backbone_check.json")))
    chk = {"backbone_state_unchanged_during_embedding": bb["unchanged"], "backbone_ckpt_sha256_matches_declared": sha_file(os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")) == cfg["cache"]["fingerprints"]["backbone_ckpt"],
           "no_train_val_id_overlap": len(set(d["ids"][d["role"] == "train"]) & set(d["ids"][d["role"] == "validation"])) == 0, "no_train_val_dup_group_overlap": len(g_tr & g_va) == 0,
           "no_test_or_excluded_records_in_caches": len(ids_all & forbidden) == 0 and len(set(conc_rows) & forbidden) == 0, "roles_only_train_validation": set(np.unique(d["role"]).tolist()) == {"train", "validation"},
           "embedding_scaler_fit_on_train_only": bool(np.allclose(es.mean, d["pooled"][tr].astype(np.float64).mean(0))) and not np.allclose(es.mean, d["pooled"][tr | va].astype(np.float64).mean(0)),
           "concept_scaler_fit_on_accepted_train_only": bool(np.allclose(runs[42]["scalers"]["concepts"].mean, np.nanmean(d["T"][tr], 0))),
           "no_nan_in_predictions": all(np.isfinite(runs[s]["probs"][m]).all() for s in SEEDS for m in models if m not in ("A_orig_head", "CTRL_extracted_target")),
           "cache_alignment_spot_check_300_ids": bool(align_ok), "prototype_logits_identity_max_abs_diff": max(v["logit_identity_max_abs_diff"] for v in pcheck.values()),
           "prototype_head_inputs_similarities_only": all(runs[s]["proto"]["head"].fc.in_features == 16 for s in SEEDS),
           "concept_predictor_frozen_during_head_training": True,                       # asserted inside run_seed (parameter hash before/after head training)
           "reproducibility_seed42_rerun_max_abs_diff": repro, "extraction_matches_frozen_config": (cfg["cache"]["fingerprints"]["extractor_params"] == json.load(open(os.path.join(ROOT, "results", "step4_3", "extraction_config.json")))["parameters_sha256"]
                                                and cfg["cache"]["fingerprints"]["extractor_sources"] == json.load(open(os.path.join(ROOT, "results", "step4_3", "extraction_config.json")))["source_sha256"])}
    json.dump({"checks": chk, "prototype_checks": pcheck}, open(os.path.join(OUT, "engineering_checks.json"), "w"), indent=1)
    json.dump({"models": models, "val_ids_sha256": hashlib.sha256("\n".join(ids_va).encode()).hexdigest(), "n_val": int(len(ids_va))}, open(os.path.join(OUT, "validation_ids_fingerprint.json"), "w"))
    log("engineering checks", {k: v for k, v in chk.items() if k != "reproducibility_seed42_rerun_max_abs_diff"})
    log(f"train stage finished in {time.time() - t0:.0f}s")


def write_csv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r.get(k), float) and np.isnan(r.get(k)) else (f"{r[k]:.6f}" if isinstance(r.get(k), float) else r.get(k, ""))) for k in cols})


if __name__ == "__main__":
    {"config": stage_config, "train": stage_train}[sys.argv[1]]()
