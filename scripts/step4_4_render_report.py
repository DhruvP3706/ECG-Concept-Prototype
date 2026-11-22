"""Render results/step4_4/experiment_report.md from saved experiment outputs (no new fitting). Refuses to overwrite."""
import csv
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "step4_4")
dest = os.path.join(OUT, "experiment_report.md")
assert not os.path.exists(dest)
rd = lambda n: list(csv.DictReader(open(os.path.join(OUT, n), encoding="utf-8")))
cfg = json.load(open(os.path.join(OUT, "experiment_config.json")))
eng = json.load(open(os.path.join(OUT, "engineering_checks.json")))
fail = json.load(open(os.path.join(OUT, "failure_summary.json")))
meta = json.load(open(os.path.join(OUT, "cache", "cache_meta.json")))
bb = json.load(open(os.path.join(OUT, "cache", "backbone_check.json")))
V, PC, CP, PR, COV, LAB = rd("validation_metrics.csv"), rd("validation_per_class.csv"), rd("concept_prediction_metrics.csv"), rd("prototype_summary.csv"), rd("coverage_summary.csv"), rd("coverage_by_label.csv")
f4 = lambda x, n=4: "n/a" if x in ("", None) or (isinstance(x, float) and x != x) else f"{float(x):.{n}f}"


def vm(model, pop, stat, k):
    return float(next(x[k] for x in V if x["model"] == model and x["population"] == pop and x["stat"] == stat))


def ms(model, pop, k, n=4):
    if model == "A_orig_head":
        return f4(vm(model, pop, "single", k), n)
    return f"{vm(model, pop, 'mean', k):.{n}f} ± {vm(model, pop, 'sd_across_seeds', k):.{n}f}"


def pd_(a, b, pop, k, n=4):
    mod = f"{a}_minus_{b}"
    return f"{vm(mod, pop, 'paired_diff_mean', k):+.{n}f} (sd across seeds {vm(mod, pop, 'paired_diff_sd_across_seeds', k):.{n}f})"


MODELS = [("A_orig_head", "A: frozen original ft_fold0 head (threshold 0.5)"), ("B_embedding_linear", "B: linear head on frozen 512-d embedding"),
          ("C2_concept_only", "C2: linear head on 4 predicted concepts"), ("D_prototype", "D: anchored-prototype head (K=16) on the same predicted concepts"),
          ("CTRL_mask_only", "Control: mask-only linear head")]
KEYS = [("macro_auroc", "macro-AUROC"), ("micro_auroc", "micro-AUROC"), ("macro_auprc", "macro-AP"), ("micro_auprc", "micro-AP"), ("macro_f1_nb", "macro-F1 (all 45, undefined→0)"),
        ("macro_f1_valid", "macro-F1 (evaluable)"), ("micro_f1", "micro-F1")]
L = []
A = L.append
A("# Combined Step 4.4–6.1 — first concept → anchored-prototype experiment (validation only)")
A("")
A("**Read this first.** Concept targets are *automatically derived 10-second RR statistics* from the frozen Step 4.3 extractor, not expert-validated measurements; \"concept prediction error\" below is agreement with extractor targets. All results are on the **validation role only** (4,437 records); no test record was read, processed or scored. The validation set was also used for early stopping of every trained model, so validation numbers are mildly optimistic, equally so across models. Three seeds give a spread over initialisations, **not** a patient-level confidence interval, and no statistical-significance claim is made. This is an initial result for a minimal four-concept set, not evidence of publication readiness.")
A("")
A("## 1. Headline answers")
A("")
rate = {x["concept"]: x for x in CP if x["split"] == "validation" and x["seed"] == "42"}
cv = lambda c, k: np.mean([float(x[k]) for x in CP if x["split"] == "validation" and x["concept"] == c])
A(f"1. **Does the concept predictor beat a constant predictor? Yes, clearly (agreement with extractor targets).** Validation R² vs the train-only constant: rate {cv('accepted_rate_mean_bpm', 'r2_vs_const'):.3f} (MAE {cv('accepted_rate_mean_bpm', 'mae'):.2f} bpm vs {cv('accepted_rate_mean_bpm', 'const_mae'):.2f}), RR CV {cv('accepted_rr_cv', 'r2_vs_const'):.3f}, NRMSSD {cv('accepted_rr_nrmssd', 'r2_vs_const'):.3f}, median-abs-successive-difference {cv('accepted_rr_mad_succ', 'r2_vs_const'):.3f}. The irregularity concepts are predicted less well than rate, and all concepts generalise somewhat worse on validation than on training.")
A(f"2. **Cost of the four-concept bottleneck: large.** Versus the same-embedding linear head (B), concept-only (C2) loses macro-AUROC {pd_('C2_concept_only', 'B_embedding_linear', 'val_all', 'macro_auroc')}, micro-AUROC {pd_('C2_concept_only', 'B_embedding_linear', 'val_all', 'micro_auroc')}, micro-AP {pd_('C2_concept_only', 'B_embedding_linear', 'val_all', 'micro_auprc')}; versus the original head (A) macro-AUROC {pd_('C2_concept_only', 'A_orig_head', 'val_all', 'macro_auroc')}. Rhythm/rate-defined classes keep most of their discrimination; morphology/conduction/axis classes fall to near chance (§5).")
A(f"3. **Prototype layer vs concept-only head: a small, consistent gain, far short of the bottleneck gap.** D − C2: macro-AUROC {pd_('D_prototype', 'C2_concept_only', 'val_all', 'macro_auroc')}, micro-AP {pd_('D_prototype', 'C2_concept_only', 'val_all', 'micro_auprc')}, micro-F1 {pd_('D_prototype', 'C2_concept_only', 'val_all', 'micro_f1')}; positive for macro-AUROC in all three seeds. The largest single contribution is sinus rhythm (SR), a non-monotonic band of rate between bradycardia and tachycardia that a linear head cannot express. D is still far below the embedding head (D − B macro-AUROC {pd_('D_prototype', 'B_embedding_linear', 'val_all', 'macro_auroc')}).")
A(f"4. **Are masks predictive enough to confound? Not in aggregate.** The mask-only control reaches macro-AUROC {ms('CTRL_mask_only', 'val_all', 'macro_auroc')} (chance = 0.5; its micro-AUROC of {ms('CTRL_mask_only', 'val_all', 'micro_auroc')} reflects label prevalence) and every per-class AUROC is ≈ 0.5 (max 0.62 on 8 positives). Coverage does vary by label (e.g., ST and SA ≈ 92% accepted vs 96.6% overall), so coverage should keep being reported; but masks are not a usable shortcut here. Small subgroups (149 non-accepted validation records) are too few to conclude more.")
pc = fail  # noqa
A("5. **Prototypes distinct and explanations faithful? Computationally faithful, partly distinct, weights hard to read.** Logits equal bias + summed prototype contributions to ≤ 7.6e-6; all 16 anchors are distinct real training records in all seeds; no pairs are closer than 0.25 std in concept space and no similarity columns correlate > 0.95; every prototype is the nearest one for some validation records. But signed weights are very large and offsetting (the head had not converged at the 100-epoch cap), 14 of 48 prototype-seeds change macro-AUROC by < 0.001 when ablated, and each seed anchors a tiny outlier cluster of 3–6 training records (see §6). Computational faithfulness is not clinical validity, and four timing concepts cannot explain morphology diagnoses.")
A("6. **Single best-supported next improvement:** add concepts that carry what timing cannot — start with **QRS duration and frontal axis** (the deferred delineation-based concepts, validated first against the public/expert references in the Step 4.2 plan). The bottleneck loses most on conduction/axis/morphology classes (§5), while head or prototype changes recovered only ~0.01 macro-AUROC. Do not add an embedding bypass to close the gap; that would remove the interpretability claim.")
A("")
A("## 2. What was done")
A("")
A(f"- Fingerprints verified before use: extractor sources and parameters equal the Step 4.3 frozen config (parameters `{meta['fingerprints']['extractor_params'][:16]}…`), role and fold manifests unchanged, `lightv2_ft_fold0.pt` SHA-256 `{meta['fingerprints']['backbone_ckpt'][:16]}…`. **No mismatch.**")
A(f"- Records: only roles `train` ({meta['n_train']:,}) and `validation` ({meta['n_validation']:,}) were admitted; `test_reserved` ({meta['rejected_roles_not_read']['test_reserved']:,}) and `excluded` ({meta['rejected_roles_not_read']['excluded']}) were rejected explicitly and never read. All artifacts are joined by record ID.")
A(f"- Extraction: frozen Step 4.3 extractor over all {meta['n_train'] + meta['n_validation']:,} records (10 worker processes, ~8 min); 0 read failures. Embeddings: exact recovered backbone preprocessing, frozen `ft_fold0` in eval mode, pooled 512-d vector and original logits (~7 min on the GTX 1650); backbone state hash before/after identical ({bb['unchanged']}); 0 embedding failures. Sanity check: the cached original-head validation macro-AUROC is 0.9594, equal to the Step 3.5 audit value for the same fold.")
A(f"- Experiment configuration (seeds 42/43/44, four targets, hyperparameters, populations, thresholds) was written and hashed **before fitting** (`experiment_config.json`, SHA-256 `{cfg['config_sha256'][:16]}…`); the training stage verified it, the merged cache hash and the code hashes before running.")
A("- Common population: all 35,568 train and 4,437 validation records are embedding-eligible. Fully-accepted subsets (all four targets accepted): "
  f"{cfg['populations']['train_fully_accepted']:,} train, {cfg['populations']['validation_fully_accepted']:,} validation (validation records not fully accepted: {cfg['populations']['validation_embedding_eligible'] - cfg['populations']['validation_fully_accepted']}).")
A("")
A("**Correction of an earlier reporting claim.** The Step 4.3 report implied that a 0 ms *median* timing difference meant all matched detector beats had identical times. Actual differences between matched detector beats: "
  f"{fail['timing_between_matched_detector_beats']['train']['records_with_any_nonzero_dt']:,} of {fail['timing_between_matched_detector_beats']['train']['records']:,} training records ({fail['timing_between_matched_detector_beats']['train']['records_with_any_nonzero_dt'] / fail['timing_between_matched_detector_beats']['train']['records']:.1%}) and "
  f"{fail['timing_between_matched_detector_beats']['validation']['records_with_any_nonzero_dt']} of {fail['timing_between_matched_detector_beats']['validation']['records']:,} validation records have at least one matched pair with a nonzero difference; the maximum is {fail['timing_between_matched_detector_beats']['train']['max_abs_dt_ms']:.0f} ms (train) / {fail['timing_between_matched_detector_beats']['validation']['max_abs_dt_ms']:.0f} ms (validation); the 99th percentile of the per-record maximum is {fail['timing_between_matched_detector_beats']['train']['p99_of_record_max_dt_ms']:.0f} ms. Most matched times are identical, not all. No extractor bug was found; no detector parameter was changed.")
A("")
A("### Extraction coverage (states are detector-consensus acceptance, not validity)")
A("")
A("| Role | Group | ACCEPTED | UNCERTAIN | UNAVAILABLE |")
A("|---|---|---|---|---|")
for role in ("train", "validation"):
    for grp in ("rate", "irregularity"):
        g = {x["state"]: x for x in COV if x["role"] == role and x["concept_group"] == grp}
        A(f"| {role} | {grp} | {int(g['ACCEPTED']['n']):,} ({float(g['ACCEPTED']['fraction']):.1%}) | {int(g['UNCERTAIN']['n']):,} ({float(g['UNCERTAIN']['fraction']):.1%}) | {int(g['UNAVAILABLE']['n']):,} ({float(g['UNAVAILABLE']['fraction']):.1%}) |")
A("")
A("Reasons (record counts, train / validation): " + "; ".join(f"{k} {fail['reason_counts']['train'][k]:,}/{fail['reason_counts']['validation'][k]}" for k in fail["reason_counts"]["train"] if fail["reason_counts"]["train"][k]) + ". Per-label coverage in `coverage_by_label.csv`; summaries in `coverage_summary.csv` and `failure_summary.json`. Within-training duplicate records remaining: " + f"{fail['within_train_duplicate_records']} (43 exact pairs with 32 label-conflict flags; kept as-is, labels not merged or adjudicated; 1 validation record is in a duplicate group whose partner is outside validation/train roles; no train/validation duplicate-group overlap).")
A("")
A("## 3. Declared models (one initial architecture each; no search)")
A("")
A("| Model | Specification |")
A("|---|---|")
H = cfg["hyperparameters"]
A(f"| Common | {H['optimizer']['name']} (weight decay {H['optimizer']['weight_decay']}), batch {H['optimizer']['batch_size']}, ≤ {H['optimizer']['max_epochs']} epochs, early-stopping patience {H['optimizer']['early_stopping_patience']} on validation loss, CPU float32; threshold 0.5, no threshold optimisation or calibration |")
for k, v in H["models"].items():
    A(f"| {k} | {json.dumps(v) if isinstance(v, dict) else v} |")
A(f"| Scaling | {H['scaling']['embedding']}; {H['scaling']['concepts']} |")
A("")
A("## 4. Concept prediction (agreement with extractor targets; mean over 3 seeds)")
A("")
A("| Concept | Split | n accepted | MAE | const MAE | RMSE | const RMSE | R² vs const |")
A("|---|---|---|---|---|---|---|---|")
for c in cfg["hyperparameters"]["concept_targets"]:
    for sp in ("train", "validation"):
        xs = [x for x in CP if x["concept"] == c and x["split"] == sp]
        g = lambda k: np.mean([float(x[k]) for x in xs])
        A(f"| {c.replace('accepted_', '')} | {sp} | {int(xs[0]['n_accepted']):,} | {g('mae'):.4f} | {g('const_mae'):.4f} | {g('rmse'):.4f} | {g('const_rmse'):.4f} | {g('r2_vs_const'):.3f} |")
A("")
A("Units: bpm for rate; dimensionless for the RR statistics (inverse-transformed from standardized predictions). The constant predictor is the train-only mean of accepted targets.")
A("")
A("## 5. Diagnosis on validation (n = 4,437; mean ± SD over 3 seeds; threshold 0.5)")
A("")
A("| Model | " + " | ".join(n for _, n in KEYS) + " |")
A("|---|" + "---|" * len(KEYS))
for m, label in MODELS:
    A(f"| {label} | " + " | ".join(ms(m, "val_all", k) for k, _ in KEYS) + " |")
A("")
A("A is deterministic (no seed variance) and **its fold-0 validation exposure is historical**: fold 0 was the original validation fold for epoch, threshold and weak-class selection, so A is not a clean comparator; its head was trained with class weights and mixup, whereas B–D use plain BCE, which is why B has much higher micro-F1/micro-AP at a fixed 0.5 threshold but lower macro-AUROC. Undefined classes: AVRT has 0 validation positives and is excluded from macro-AUROC/AP; macro-F1 is shown under both conventions.")
A("")
A("### Paired changes on identical validation IDs (mean over seeds; sd is the spread across seeds, not a CI)")
A("")
A("| Comparison | macro-AUROC | micro-AUROC | micro-AP | micro-F1 |")
A("|---|---|---|---|---|")
for a, b in (("B_embedding_linear", "A_orig_head"), ("C2_concept_only", "B_embedding_linear"), ("D_prototype", "C2_concept_only"), ("D_prototype", "B_embedding_linear"), ("CTRL_mask_only", "C2_concept_only")):
    A(f"| {a} − {b} | " + " | ".join(pd_(a, b, "val_all", k, 3) for k in ("macro_auroc", "micro_auroc", "micro_auprc", "micro_f1")) + " |")
A("")
A("### Fully-accepted subset and the extracted-target baseline (n = 4,288 validation records; same IDs for all models)")
A("")
A("| Model | macro-AUROC | micro-AUROC | micro-AP | micro-F1 |")
A("|---|---|---|---|---|")
for m, label in MODELS + [("CTRL_extracted_target", "Auxiliary: linear head on EXTRACTED accepted concepts (extracted-target baseline, not an oracle)")]:
    A(f"| {label} | " + " | ".join(ms(m, "val_full_accepted", k) for k in ("macro_auroc", "micro_auroc", "micro_auprc", "micro_f1")) + " |")
A("")
A(f"The extracted-target baseline is not better than the head on *predicted* concepts (extracted − C2 macro-AUROC {pd_('CTRL_extracted_target', 'C2_concept_only', 'val_full_accepted', 'macro_auroc', 3) if False else vm('CTRL_extracted_target_minus_C2_concept_only', 'val_full_accepted', 'paired_diff_mean', 'macro_auroc'):+.4f}). Predicted concepts come from the embedding, so they may carry diagnosis-related information beyond the four extracted statistics (concept leakage), which weakens the claim that C2/D rely only on rate and RR irregularity.")
A("")
A("Records without all four accepted concepts (diagnostic only; n = 149): " + "; ".join(f"{lab.split(':')[0]} macro-AUROC {ms(m, 'val_not_full_accepted', 'macro_auroc')}" for m, lab in MODELS if m != "CTRL_extracted_target") + ". Concepts are predicted from the embedding for these records too; the sample is small.")
A("")
pcl = {}
for x in PC:
    if x["population"] == "val_all":
        pcl.setdefault(x["class"], {})[x["model"]] = x
rows = []
for c, m in pcl.items():
    if m["B_embedding_linear"]["auroc_mean"] == "":
        continue
    rows.append((c, int(m["B_embedding_linear"]["n_pos"]), float(m["B_embedding_linear"]["auroc_mean"]), float(m["C2_concept_only"]["auroc_mean"]), float(m["D_prototype"]["auroc_mean"]), float(m["CTRL_mask_only"]["auroc_mean"])))
rows = [r for r in rows if r[1] >= 30]
A("### Per-class AUROC (validation; classes with ≥ 30 positives; full table in `validation_per_class.csv`)")
A("")
A("| Class | n+ | B embedding | C2 concept-only | D prototype | mask-only | B − C2 |")
A("|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: -(r[2] - r[3])):
    A(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[2] - r[3]:+.3f} |")
A("")
A("Classes that keep discrimination from four timing concepts are rate/rhythm classes (SB, SVT, AFIB, AF, ST, SA, APB, VPB); the largest losses are conduction, axis, hypertrophy and morphology classes (ALS, 1AVB, RBBB, AQW, STE, TWO, TWC, LFBBB; SR is also lost by the linear concept head and largely recovered by the prototype layer). Support for several rare classes is ≤ 10, so their per-class values are noisy.")
A("")
A("## 6. Prototype analysis")
A("")
pcheck = eng["prototype_checks"]
A("| Seed | tau | logit identity max abs diff | prototypes never nearest in val | largest usage share | min pairwise distance (std) | pairs closer than 0.25 | pairs with similarity corr > 0.95 | anchors distinct training IDs |")
A("|---|---|---|---|---|---|---|---|---|")
for s, v in pcheck.items():
    A(f"| {s} | {v['tau']:.4f} | {v['logit_identity_max_abs_diff']:.2e} | {v['prototypes_never_nearest_in_val']} | {v['max_usage_fraction']:.1%} | {v['min_pairwise_dist_std']:.2f} | {v['pairs_closer_than_0.25_std']} | {v['pairs_similarity_corr_gt_0.95']} | {v['n_unique_anchors']} ({v['anchors_all_training_ids']}) |")
A("")
ab = np.array([float(x["ablation_d_macro_auroc"]) for x in PR])
sz = np.array([int(x["cluster_size_train_fully_accepted"]) for x in PR])
W = [torch.load(os.path.join(OUT, "checkpoints", f"seed{s}_D_prototype.pt"), weights_only=True) for s in (42, 43, 44)]
wmax = max(float(w["fc.weight"].abs().max()) for w in W)
bias = [float(w["fc.bias"].abs().max()) for w in W]
by_seed = {}
for x in PR:
    by_seed.setdefault(x["seed"], []).append(x)
top = []
for sd, xs in by_seed.items():
    t3 = sorted(xs, key=lambda x: float(x["ablation_d_macro_auroc"]))[:3]
    top.append(f"seed {sd}: " + ", ".join(f"≈{float(x['pred_rate_mean_bpm']):.0f} bpm / CV {float(x['pred_rr_cv']):.2f} ({float(x['ablation_d_macro_auroc']):+.3f})" for x in t3))
tiny = []
for sd, xs in by_seed.items():
    sm = [x for x in xs if int(x["cluster_size_train_fully_accepted"]) < 150]
    tiny.append(f"seed {sd}: " + ", ".join(f"{x['cluster_size_train_fully_accepted']} records ({float(x['nearest_other_prototype_dist_std']):.1f} std from nearest prototype, CV {float(x['pred_rr_cv']):.2f})" for x in sm))
A(f"- **Functional dependence (ablation: set one prototype's similarity to 0, validation):** macro-AUROC change per prototype ranges {ab.min():+.4f} to {ab.max():+.4f} (median {np.median(ab):+.4f}); {int((np.abs(ab) < 0.001).sum())} of {len(ab)} prototype-seeds change it by < 0.001. The three largest effects per seed come from large regular-rhythm prototypes (predicted rate / RR CV; macro-AUROC change): " + "; ".join(top) + ". This is a computational dependence on a rate profile; prototypes were defined from concepts only (no labels), and this is not clinical validity.")
A(f"- **Signed weights** are reported as supporting (+) or opposing (−) logit contributions in `prototype_summary.csv`. They are very large and offsetting (max |weight| {wmax:.0f}, max |bias| {max(bias):.0f}) because the similarity features are highly collinear and the head stopped at the 100-epoch cap (best epoch 100/100, 98/100, 100/100). Individual weights therefore should not be read as stand-alone evidence; the identity logit = bias + Σ contributions holds exactly, and the contribution decomposition is faithful to what the model computes.")
A(f"- **Redundancy / outliers:** no near-duplicate prototype pairs by the thresholds above, but deterministic farthest-point initialisation anchored a small high-irregularity outlier cluster in every seed — " + "; ".join(tiny) + f". {int((sz < 150).sum())} of {len(sz)} prototype-seeds have < 150 cluster members; these prototypes are distinct but rare and contribute almost nothing to validation performance. Prototypes are anchored to actual training records (IDs in the CSV); this is initial anchored selection, **not learned prototype optimisation**.")
A("- **Scope of explanation:** the four timing concepts cannot explain morphology-related diagnoses; the prototype layer explains only how rate/regularity profiles map to labels in this model.")
A("")
A("## 7. Engineering checks (all passed)")
A("")
A("| Check | Result |")
A("|---|---|")
for k, v in eng["checks"].items():
    A(f"| {k} | {v} |")
A("")
A("Unit tests: `tests/test_step4_4.py` (10 tests: masked-loss NaN safety, train-only scaling, deterministic k-means/anchors, tau safeguard, logit decomposition, similarity-only head, determinism) and the Step 4.3 tests (20) pass. The seed-42 chain was re-trained and reproduced bit-identically (max abs probability difference 0.0).")
A("")
A("## 8. Limitations and what was not checked")
A("")
for t in ("Validation is used for early stopping/selection of every trained model; validation numbers are mildly optimistic, equally for B–D and controls. The test fold was not touched.",
          "Concept heads (C2, D) are trained on predicted concepts of *training* records, which are in-sample for the concept predictor (lower error than on validation) — a train/validation shift in the head's inputs.",
          "The prototype head had not converged at the declared 100-epoch cap and the weights are ill-conditioned; no further tuning was done. Prototype selection is a single deterministic k-means with farthest-point initialisation (outlier-prone); K=16 and tau were not tuned.",
          "Concept targets are automatically derived and unvalidated against references; predicted concepts may encode information beyond them (see the extracted-target baseline).",
          "Original head A has historical validation exposure and a different training recipe (class weights, mixup), so A is a reference, not a fair competitor to B–D.",
          "Seed spread is not a confidence interval; no significance testing; within-training duplicates and conflicting labels (e.g., AF vs AFIB copies) are retained as-is.",
          "Not done: any test-set evaluation, threshold optimisation, calibration, hyperparameter search, bypass/residual variants, learned prototype optimisation, additional concepts, qualitative review of prototype records."):
    A("- " + t)
A("")
A("## 9. Files and commands")
A("")
A("Outputs in `results/step4_4/`: `experiment_report.md`, `experiment_config.json`, `concept_prediction_metrics.csv`, `validation_metrics.csv`, `validation_per_class.csv`, `prototype_summary.csv`, `coverage_summary.csv`, `coverage_by_label.csv`, `failure_summary.json`, `engineering_checks.json`, `transforms.json`, `validation_ids_fingerprint.json`, `execution_log.txt`, `checkpoints/` (per-seed state dicts, anchors, histories), `cache/` (record index, chunked concept CSVs, chunked embeddings, `merged.npz`, fingerprints; no waveforms). Code: `src/ecg_concept/concept_proto.py`, `scripts/step4_4_cache.py`, `scripts/step4_4_train.py`, `scripts/step4_4_coverage.py`, `scripts/step4_4_render_report.py`, `tests/test_step4_4.py`.")
A("")
A("```")
for c in (r".venv\Scripts\python.exe -I tests\test_step4_4.py", r".venv\Scripts\python.exe -I scripts\step4_4_cache.py index", r".venv\Scripts\python.exe -I scripts\step4_4_cache.py extract",
          r".venv\Scripts\python.exe -I scripts\step4_4_cache.py embed", r".venv\Scripts\python.exe -I scripts\step4_4_cache.py merge", r".venv\Scripts\python.exe -I scripts\step4_4_train.py config",
          r".venv\Scripts\python.exe -I scripts\step4_4_train.py train", r".venv\Scripts\python.exe -I scripts\step4_4_coverage.py", r".venv\Scripts\python.exe -I scripts\step4_4_render_report.py"):
    A(c)
A("```")
open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
with open(os.path.join(OUT, "execution_log.txt"), "a", encoding="utf-8") as lf:
    import time
    lf.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "coverage summaries and report rendered\n")
print("report written", dest)
