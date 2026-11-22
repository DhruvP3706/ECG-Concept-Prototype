"""Expanded non-diagnostic SCP concepts with OUT-OF-FOLD concept predictions -> strict concept bottleneck (validation-stage; fold 10 never read).
Precondition: Stage-1 audit (scripts/ptbxl_scp_audit_v2.py) passed.
Concept predictor : ConceptMLP 512-128-C, masked weighted BCE (unknown entries masked, label policy P1), as in scripts/ptbxl_scp_bottleneck.py (fit_concepts reused).
OOF scheme        : leave-one-fold-out over the ORIGINAL PTB-XL folds 1-8 (patient-disjoint by construction; asserted). For held-out fold f the predictor is fitted ONLY on eligible
                    records of folds {1..8}\\{f} (embedding standardisation, concept pos_weight and weights all computed on that partition) and predicts fold f. The 8 held-out
                    blocks form the OOF predicted-concept matrix for every diagnosis-head training record. The FINAL predictor is fitted on all eligible folds 1-8 and predicts fold 9.
                    Early stopping of EVERY predictor (OOF and final) uses the masked concept loss on fold 9 (the existing protocol): same procedure everywhere, but fold-9 concept targets
                    influence the epoch chosen, so validation-stage numbers stay optimistic. Fold 9 never contributes training rows to any predictor.
Diagnosis heads   : only standardized predicted concept PROBABILITIES (train mean/std of the head's own training inputs); no embedding, no true concept target, no diagnostic label, no mask.
                    linear Linear(C,5); prototype K=16 real training records nearest deterministic k-means centres (their OOF vectors), exp(-d^2/tau), tau = median train nearest-prototype d^2.
                    BCE with train-only pos_weight, AdamW lr 1e-2, wd 1e-4, batch 256, patience 10 (validation weighted BCE).
Configurations    : vocabulary {expanded 23, previous 17} x training input {oof, insample}; insample = head trained on the FINAL predictor's predictions of its own training records (historical protocol).
Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_scp_oof.py <results/ptbxl_scp_oof/run_dir>
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ecg_concept import concept_proto as cp  # noqa: E402
from ecg_concept import metrics as M  # noqa: E402
from ecg_concept import ptbxl as px  # noqa: E402
from ecg_concept import scp_concepts as sc  # noqa: E402
import ptbxl_baseline_train as bt  # noqa: E402
import ptbxl_concept_proto as rr  # noqa: E402
import ptbxl_scp_bottleneck as sb  # noqa: E402

SEEDS, VOCAB5, K = sb.SEEDS, bt.VOCAB, 16
LR_H, B, BOOT_SEED = sb.LR_H, 5000, 20261120
HIST = os.path.join(ROOT, "results", "ptbxl_scp_concepts", "run_20261120_021055", "bottleneck")
sha_file, sig = rr.sha_file, rr.sig


def main():
    run = os.path.abspath(sys.argv[1])
    aud = os.path.join(run, "audit")
    ares = json.load(open(os.path.join(aud, "audit_results.json")))
    assert ares["all_pass"], "audit did not pass"
    vfile = os.path.join(aud, "concept_vocabulary.json")
    assert sha_file(vfile) == ares["vocabulary_file_sha256"]
    vdoc = json.load(open(vfile))
    vocabs = {"exp": vdoc["vocabulary"], "v17": vdoc["previous_17_vocabulary"]}
    cat_of = vdoc["category"]
    cat_of.update({c: ("form" if c in vdoc["category"] and vdoc["category"][c] == "form" else "rhythm") for c in vocabs["v17"] if c not in cat_of})
    out = os.path.join(run, "oof")
    ckd = os.path.join(ROOT, "models", "ptbxl_scp_oof", os.path.basename(run))
    os.makedirs(out)
    os.makedirs(ckd)
    log = open(os.path.join(out, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    ids, emb, man = rr.load()
    records = {r.ecg_id: r for r in px.read_metadata(os.path.join(sb.DATA, "ptbxl_database.csv"))}
    stm = px.read_statements(os.path.join(sb.DATA, "scp_statements.csv"))
    form, rhythm, _ = sc.split_statements(stm)
    members = {"form": form, "rhythm": rhythm}
    recs = [records[i] for i in ids]
    fold = np.array([r.strat_fold for r in recs])
    pid = np.array([r.patient_id for r in recs])
    role = np.array([man[i]["role"] for i in ids])
    tr, va = role == "train", role == "val"
    assert (tr.sum(), va.sum()) == (16289, 2034) and fold[tr].max() == 8 and fold[va].min() == fold[va].max() == 9
    # ---- patient separation in every internal split
    for f in range(1, 9):
        held, rest = pid[tr & (fold == f)], pid[tr & (fold != f)]
        assert len(set(held) & set(rest)) == 0 and len(set(held) & set(pid[va])) == 0, f"patient overlap at internal fold {f}"
    assert len(set(pid[tr]) & set(pid[va])) == 0
    say("patient separation verified: no patient shared between any held-out internal fold and its training partition, nor with fold 9")
    Y = np.array([[int(man[i][c]) for c in VOCAB5] for i in ids], np.float32)
    pos_weight = torch.from_numpy(((tr.sum() - Y[tr].sum(0)) / Y[tr].sum(0)).astype(np.float32))
    pat_va = pid[va]
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    Xraw = emb
    Targets = {v: sc.concept_targets(recs, vocabs[v], cat_of, members, "P1") for v in vocabs}

    def fit_predict(train_mask, vname, seed, pred_rows):
        """Fit a concept predictor on `train_mask` rows (stats from them only), early-stop on fold 9, return probabilities for `pred_rows` and the model."""
        T = Targets[vname]
        es = cp.Standardizer.fit(Xraw[train_mask])
        Xs = es.transform(Xraw).astype(np.float32)
        npos, nneg = (T[train_mask] == 1).sum(0), (T[train_mask] == 0).sum(0)
        cpw = torch.from_numpy((nneg / np.maximum(npos, 1)).astype(np.float32))
        torch.manual_seed(seed)
        m = cp.ConceptMLP(512, 128, T.shape[1])
        m, h = sb.fit_concepts(m, Xs[train_mask], T[train_mask], Xs[va], T[va], seed, cpw)
        P = sig(cp.predict(m, Xs[pred_rows])).astype(np.float64)
        assert np.isfinite(P).all()
        return m, h, P

    def run_seed(seed, vnames=("exp", "v17")):
        o = {}
        for v in vnames:
            T = Targets[v]
            Cn = T.shape[1]
            oof = np.zeros((len(ids), Cn))
            models, hist = {}, {}
            for f in range(1, 9):
                trm = tr & (fold != f)
                hold = tr & (fold == f)
                assert not (trm & hold).any() and not (trm & va).any()
                m, h, P = fit_predict(trm, v, seed, np.where(hold)[0])
                oof[hold] = P
                models[f"fold{f}"], hist[f"fold{f}"] = m.state_dict(), {k: h[k] for k in ("best_epoch", "epochs_run", "best_val")}
            mf, hf, Pf = fit_predict(tr, v, seed, np.arange(len(ids)))            # final predictor: all eligible folds 1-8; predicts everything (train rows in-sample)
            for p in mf.parameters():
                p.requires_grad_(False)
            hb = hashlib.sha256(b"".join(x.numpy().tobytes() for x in mf.state_dict().values())).hexdigest()
            models["final"], hist["final"] = mf.state_dict(), {k: hf[k] for k in ("best_epoch", "epochs_run", "best_val")}
            o[v] = {"T": T, "oof": oof, "final": Pf, "states": models, "hist": hist, "final_sha": hb, "probs": {}, "headhist": {}, "proto": {}, "headstate": {}}
            ce = []
            for nm, P, sel in (("oof_train", oof, tr), ("insample_train", Pf, tr), ("final_validation", Pf, va)):
                ce += [{"vocab": v, "seed": seed, **r} for r in sb.concept_metrics(P[sel], T[sel], vocabs[v], nm)]
            o[v]["concept_eval"] = ce
            # ---- diagnosis heads
            for mode, Pin in (("oof", None), ("insample", None)):
                Ptr = oof[tr] if mode == "oof" else Pf[tr]
                sc_ = cp.Standardizer.fit(Ptr)
                Ztr, Zva = sc_.transform(Ptr).astype(np.float64), sc_.transform(Pf[va]).astype(np.float64)
                torch.manual_seed(seed)
                ml, h = rr.fit_pw(nn.Linear(Cn, 5), Ztr, Y[tr], Zva, Y[va], seed, LR_H, pos_weight)
                o[v]["probs"][f"{mode}_linear"] = sig(cp.predict(ml, Zva)).astype(np.float64)
                o[v]["headhist"][f"{mode}_linear"], o[v]["headstate"][f"{mode}_linear"] = h, ml.state_dict()
                if mode == "oof":
                    centres, assign = cp.kmeans_det(Ztr, K)
                    tr_idx = np.where(tr)[0]
                    anchor = tr_idx[cp.anchor_records(Ztr, centres)]
                    assert tr[anchor].all() and not va[anchor].any()
                    P_ = Ztr[cp.anchor_records(Ztr, centres)]
                    tau = cp.tau_rule(Ztr, P_)
                    S_tr, S_va = cp.similarities(Ztr, P_, tau).astype(np.float32), cp.similarities(Zva, P_, tau).astype(np.float32)
                    torch.manual_seed(seed)
                    mp, h = rr.fit_pw(cp.ProtoHead(K, 5), S_tr, Y[tr], S_va, Y[va], seed, LR_H, pos_weight)
                    o[v]["probs"]["oof_proto"] = sig(cp.predict(mp, S_va)).astype(np.float64)
                    o[v]["headhist"]["oof_proto"], o[v]["headstate"]["oof_proto"] = h, mp.state_dict()
                    o[v]["proto"] = {"anchor_ecg_id": ids[anchor].tolist(), "tau": tau, "cluster_sizes": np.bincount(assign, minlength=K).tolist(),
                                     "head_weight": mp.fc.weight.detach().numpy().tolist(), "anchor_unknown_concept_entries": int((T[anchor] == -1).sum()),
                                     "anchor_true_concept_positives": {vocabs[v][j]: int((T[anchor, j] == 1).sum()) for j in range(Cn)}}
                    perm = np.random.RandomState(seed).permutation(len(Ztr))
                    torch.manual_seed(seed)
                    ms, h = rr.fit_pw(nn.Linear(Cn, 5), Ztr[perm], Y[tr], Zva, Y[va], seed, LR_H, pos_weight)
                    o[v]["probs"]["oof_shuffled"] = sig(cp.predict(ms, Zva)).astype(np.float64)
                    o[v]["headhist"]["oof_shuffled"], o[v]["headstate"]["oof_shuffled"] = h, ms.state_dict()
            assert hashlib.sha256(b"".join(x.numpy().tobytes() for x in mf.state_dict().values())).hexdigest() == hb, "final concept predictor changed"
        return o

    t0 = time.time()
    R = {}
    for s in SEEDS:
        R[s] = run_seed(s)
        say(f"seed {s} done ({time.time() - t0:.0f}s); final-predictor epochs exp {R[s]['exp']['hist']['final']['epochs_run']}, v17 {R[s]['v17']['hist']['final']['epochs_run']}; "
            f"OOF best epochs exp {[R[s]['exp']['hist'][f'fold{f}']['best_epoch'] for f in range(1, 9)]}")
    rep = run_seed(42, ("exp",))
    repro = {k: float(np.abs(rep["exp"]["probs"][k] - R[42]["exp"]["probs"][k]).max()) for k in rep["exp"]["probs"]}
    repro["oof_matrix"] = float(np.abs(rep["exp"]["oof"] - R[42]["exp"]["oof"]).max())
    say(f"reproducibility (expanded, seed 42 rerun) max abs diff: {repro}")

    # ---- evaluation tables
    Yva = Y[va].astype(bool)
    res = {f"{v}|{m}|{s}": bt.evaluate(Yva, R[s][v]["probs"][m]) for s in SEEDS for v in vocabs for m in R[s][v]["probs"]}
    for nm in ("concept_prediction_metrics.csv",):
        rows = [r for s in SEEDS for v in vocabs for r in R[s][v]["concept_eval"]]
        with open(os.path.join(out, nm), "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # ---- paired bootstrap (same saved draws)
    pred = np.load(os.path.join(rr.BOOT, "predictions.npz"))
    hist = np.load(os.path.join(HIST, "predictions_fold9.npz"))
    dz = np.load(os.path.join(rr.BOOT, "bootstrap_patient_draws.npz"))
    draws, upat = dz["draws"], dz["unique_patient_ids"]
    assert (pred["ecg_id"] == ids[va]).all() and (hist["ecg_id"] == ids[va]).all() and (pred["y"] == Y[va]).all() and (pred["patient_id"] == pat_va).all()
    assert int(dz["seed"]) == BOOT_SEED and draws.shape[0] == B and (np.unique(pat_va) == upat).all()
    allm = {m: {s: pred[f"{m}_seed{s}"] for s in SEEDS} for m in ("frozen_linear", "frozen_mlp")}
    allm["hist17_insample_linear"] = {s: hist[f"main_linear_seed{s}"] for s in SEEDS}
    allm["hist17_insample_proto"] = {s: hist[f"main_proto_seed{s}"] for s in SEEDS}
    for v in vocabs:
        for m in R[42][v]["probs"]:
            allm[f"{v}_{m}"] = {s: R[s][v]["probs"][m] for s in SEEDS}
    _, inv = np.unique(pat_va, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(upat)))
    lens = np.bincount(inv, minlength=len(upat))
    names = list(allm)
    keys = [(m, s) for m in names for s in SEEDS]
    HYP = VOCAB5.index("HYP")
    auc_b, ap_b = np.zeros((B, len(keys), 5)), np.zeros((B, len(keys), 5))
    t1 = time.time()
    for b in range(B):
        idx = order[np.concatenate([np.arange(starts[k], starts[k] + lens[k]) for k in draws[b]])]
        yb = Yva[idx]
        for j, (m, s) in enumerate(keys):
            auc_b[b, j] = M.auroc_columns(yb, allm[m][s][idx])
            ap_b[b, j] = M.ap_columns(yb, allm[m][s][idx])
    say(f"bootstrap {B} replicates in {time.time() - t1:.0f}s; undefined-class replicates AUROC {int(np.isnan(auc_b).any(axis=(1, 2)).sum())}, AP {int(np.isnan(ap_b).any(axis=(1, 2)).sum())}")
    stat, pt = {}, {}
    for m in names:
        js = [keys.index((m, s)) for s in SEEDS]
        stat[(m, "macro_auroc")] = np.nanmean(auc_b[:, js], axis=2).mean(1)
        stat[(m, "macro_ap")] = np.nanmean(ap_b[:, js], axis=2).mean(1)
        stat[(m, "hyp_auroc")] = auc_b[:, js, HYP].mean(1)
        pt[(m, "macro_auroc")] = float(np.mean([np.nanmean(M.auroc_columns(Yva, allm[m][s])) for s in SEEDS]))
        pt[(m, "macro_ap")] = float(np.mean([np.nanmean(M.ap_columns(Yva, allm[m][s])) for s in SEEDS]))
        pt[(m, "hyp_auroc")] = float(np.mean([M.auroc_columns(Yva, allm[m][s])[HYP] for s in SEEDS]))
    METR = ("macro_auroc", "macro_ap", "hyp_auroc")
    ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])] + [int(np.isnan(v).sum())]
    comps = [("exp_oof_linear", "frozen_mlp"), ("exp_oof_linear", "frozen_linear"), ("exp_oof_proto", "frozen_mlp"), ("exp_oof_linear", "v17_oof_linear"), ("exp_oof_proto", "v17_oof_proto"),
             ("exp_oof_proto", "exp_oof_linear"), ("v17_oof_proto", "v17_oof_linear"), ("v17_oof_linear", "hist17_insample_linear"), ("v17_insample_linear", "hist17_insample_linear"),
             ("exp_oof_linear", "exp_insample_linear"), ("v17_oof_linear", "v17_insample_linear"), ("exp_insample_linear", "v17_insample_linear"), ("exp_oof_linear", "hist17_insample_linear"),
             ("exp_oof_linear", "exp_oof_shuffled"), ("v17_oof_linear", "v17_oof_shuffled"), ("v17_oof_linear", "frozen_mlp")]
    brow = []
    for m in names:
        for k in METR:
            lo, hi, nb = ci(stat[(m, k)])
            brow.append(["model", m, k, pt[(m, k)], lo, hi, nb])
    for a_, b_ in comps:
        for k in METR:
            lo, hi, nb = ci(stat[(a_, k)] - stat[(b_, k)])
            brow.append(["difference", f"{a_} - {b_}", k, pt[(a_, k)] - pt[(b_, k)], lo, hi, nb])
    with open(os.path.join(out, "bootstrap_summary.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model_or_comparison", "metric", "fold9_point_estimate_seed_mean", "ci95_low", "ci95_high", "invalid_replicates"])
        w.writerows(brow)
    # ---- artefacts
    np.savez(os.path.join(out, "split_assignments.npz"), ecg_id=ids, patient_id=pid, strat_fold=fold, role=role, internal_heldout_fold=np.where(tr, fold, 0))
    np.savez(os.path.join(out, "predictions_fold9.npz"), ecg_id=ids[va], patient_id=pat_va, y=Y[va], **{f"{v}_{m}_seed{s}": R[s][v]["probs"][m] for s in SEEDS for v in vocabs for m in R[s][v]["probs"]})
    np.savez(os.path.join(out, "predicted_concepts.npz"), ecg_id=ids, **{f"{v}_{k}_seed{s}": R[s][v][k] for s in SEEDS for v in vocabs for k in ("oof", "final")},
             **{f"{v}_targets": Targets[v] for v in vocabs})
    for s in SEEDS:
        for v in vocabs:
            torch.save({"concept_predictors": R[s][v]["states"], "heads": R[s][v]["headstate"], "proto": R[s][v]["proto"], "vocab": vocabs[v], "seed": s, "final_sha": R[s][v]["final_sha"]},
                       os.path.join(ckd, f"{v}_seed{s}.pt"))
    json.dump({"results": res, "vocab": vocabs, "repro": repro, "seeds": SEEDS,
               "concept_hist": {f"{v}|{s}": R[s][v]["hist"] for s in SEEDS for v in vocabs},
               "head_hist": {f"{v}|{s}|{m}": {x: h[x] for x in ("best_epoch", "epochs_run", "best_val")} for s in SEEDS for v in vocabs for m, h in R[s][v]["headhist"].items()},
               "proto": {f"{v}|{s}": R[s][v]["proto"] for s in SEEDS for v in vocabs}, "final_predictor_sha": {f"{v}|{s}": R[s][v]["final_sha"] for s in SEEDS for v in vocabs},
               "pos_weight_diagnosis": pos_weight.tolist(), "hyper": {"lr_concept": sb.LR_C, "lr_heads": LR_H, "weight_decay": sb.WD, "batch": sb.BATCH, "max_epochs": sb.MAXEP, "patience": sb.PAT, "K": K,
                                                                   "oof": "leave-one-fold-out over original folds 1-8", "bootstrap_B": B, "bootstrap_seed": BOOT_SEED,
                                                                   "bootstrap_draws_sha256": sha_file(os.path.join(rr.BOOT, "bootstrap_patient_draws.npz"))},
               "inputs": {"audit_results_sha256": sha_file(os.path.join(aud, "audit_results.json")), "concept_vocabulary_sha256": ares["vocabulary_file_sha256"],
                          "manifest_sha256": ares["inputs"]["manifest_sha256"], "scp_statements_sha256": ares["inputs"]["scp_statements_sha256"],
                          "historical_predictions_sha256": sha_file(os.path.join(HIST, "predictions_fold9.npz"))},
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("scripts/ptbxl_scp_oof.py", "scripts/ptbxl_scp_bottleneck.py", "src/ecg_concept/scp_concepts.py")},
               "fold10_waveforms_read": 0}, open(os.path.join(out, "results.json"), "x"), indent=1)
    say(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
