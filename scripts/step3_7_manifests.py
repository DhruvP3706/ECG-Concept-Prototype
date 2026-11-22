"""Step 3.7 stage A: verify fold reconstruction + duplicate evidence, write manifests and the PRE-DECLARED analysis config.

No model inference, no metrics. Refuses to overwrite existing Step 3.7 outputs. Historical folds are reproduced, never changed.
"""
import csv
import glob
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept.labels import CLASS_NAMES, build_index, reproduce_folds  # noqa: E402

OUT = os.path.join(ROOT, "results", "step3_7")
R5, R6 = os.path.join(ROOT, "results", "step3_5"), os.path.join(ROOT, "results", "step3_6")
NB = r"C:\Users\lenovo\HonsExAI\LightECGNet_v2_FIXED_faster.ipynb"
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
os.makedirs(OUT, exist_ok=True)
FILES = ["analysis_config.json", "original_fold_manifest.csv", "duplicate_group_manifest.csv", "prospective_role_manifest.csv"]
for f in FILES:
    assert not os.path.exists(os.path.join(OUT, f)), f"{f} exists; refusing to overwrite"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------- 1. reconstruct and verify the historical index / folds
classes, recs, labs = build_index(DS)
assert classes == CLASS_NAMES
ids = [os.path.basename(r) for r in recs]
folds = reproduce_folds(len(ids))
fold_of = {ids[i]: k for k, f in enumerate(folds) for i in f}
index_of = {r: i for i, r in enumerate(ids)}
cfg5 = json.load(open(os.path.join(R5, "evaluation_config.json")))
assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == cfg5["record_order_sha256"], "record order differs from Step 3.5"
test_ids = [ids[i] for i in folds[9]]
assert hashlib.sha256("\n".join(test_ids).encode()).hexdigest() == cfg5["test_record_ids_sha256"], "test IDs differ from Step 3.5"
pred = np.load(os.path.join(R5, "evaluation_predictions.npz"))
assert list(pred["record_ids"]) == test_ids and list(pred["class_names"]) == CLASS_NAMES
Ytest = np.zeros((len(test_ids), 45), np.uint8)
for j, i in enumerate(folds[9]):
    Ytest[j, labs[i]] = 1
assert np.array_equal(Ytest, pred["targets"]), "targets in predictions differ from reconstructed labels"
label_of = {ids[i]: sorted(CLASS_NAMES[j] for j in labs[i]) for i in range(len(ids))}

# ---------- 2. duplicate evidence (exact pairs only; partial-duplicate rows would have kind != exact_file)
rows = list(csv.DictReader(open(os.path.join(R6, "duplicate_pair_summary.csv"))))
assert all(r["kind"] == "exact_file" for r in rows), "unexpected non-exact rows"
summ = json.load(open(os.path.join(R6, "duplicate_summary.json")))["exact"]
pairs = [(r["id_a"], r["id_b"]) for r in rows]
partner = {}
for a, b in pairs:
    assert a not in partner and b not in partner
    partner[a], partner[b] = b, a
dup_ids = set(partner)


def cat(a, b):
    fa, fb = fold_of.get(a), fold_of.get(b)
    if fa is None or fb is None:
        return "4_not_in_experiment_set"
    if fa == 9 and fb == 9:
        return "1_both_test"
    return "2_cross_test_boundary" if (fa == 9) != (fb == 9) else "3_both_outside_test"


# recompute categories from my own fold map and compare with the Step 3.6 CSV/summary
assert all(cat(r["id_a"], r["id_b"]) == r["category"] for r in rows), "pair categories differ from Step 3.6 CSV"
cnt = Counter(cat(a, b) for a, b in pairs)
cross_test = sorted({(a if fold_of.get(a) == 9 else b) for a, b in pairs if cat(a, b) == "2_cross_test_boundary"})
any_test = sorted(r for r in dup_ids if fold_of.get(r) == 9)
both_out = [(a, b) for a, b in pairs if a not in fold_of and b not in fold_of]
one_out = [(a, b) for a, b in pairs if (a in fold_of) != (b in fold_of)]
verified = {"pairs": (len(pairs), summ["pairs"]), "unique_records": (len(dup_ids), summ["unique_records"]),
            "cross_test_pairs": (cnt["2_cross_test_boundary"], summ["2_cross_test_boundary"]),
            "test_records_in_cross_pairs": (len(cross_test), summ["test_records_in_cross_pairs"]),
            "test_records_in_any_pair": (len(any_test), summ["test_records_in_any_pair"]),
            "both_test": (cnt["1_both_test"], summ["1_both_test"]), "outside_test": (cnt["3_both_outside_test"], summ["3_both_outside_test"]),
            "category4": (cnt["4_not_in_experiment_set"], summ["4_not_in_experiment_set"])}
for k, (mine, theirs) in verified.items():
    assert mine == theirs, (k, mine, theirs)
membership_breakdown = {"category4_total": cnt["4_not_in_experiment_set"], "both_members_outside_experiment": [list(p) for p in both_out],
                        "one_member_outside_experiment": len(one_out), "one_outside_with_in_set_member_in_test": sum(1 for a, b in one_out if fold_of.get(a, fold_of.get(b)) == 9),
                        "unresolved_membership_pairs": 0}
group_id = {r: "DUP-" + min(r, partner[r]) for r in dup_ids}

# ---------- 3. PRE-DECLARED cohorts (rule-based on duplicate exposure only; no labels / predictions / performance used)
cohorts = {"A_all_test": {"exclude": [], "purpose": "historical comparability (Step 3.5 table)"},
           "B_exclude_cross_boundary": {"exclude": cross_test, "purpose": "main leakage-screened cohort for the existing-backbone study"},
           "C_exclude_any_duplicate": {"exclude": any_test, "purpose": "stricter sensitivity cohort"}}
sizes = {k: len(test_ids) - len(v["exclude"]) for k, v in cohorts.items()}
assert sizes == {"A_all_test": 4447, "B_exclude_cross_boundary": 4430, "C_exclude_any_duplicate": 4424}, sizes
assert set(cross_test) <= set(any_test)

decl = {
    "declared_before_computing_cohort_metrics": True, "declared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "cohorts": {k: {"n": sizes[k], "n_excluded": len(v["exclude"]), "excluded_ids": v["exclude"], "purpose": v["purpose"]} for k, v in cohorts.items()},
    "cohort_rule": "B: test records with a duplicate partner in folds 0-8; C: test records in ANY exact-duplicate pair (cross-boundary, within-test, or partner outside the experiment set). Same cohort for every model.",
    "models": {"ft_fold0": ["lightv2_ft_fold0", "original_file (VERIFIED)"], "p1_fold0": ["lightv2_p1_fold0", "original_file (VERIFIED)"],
               "p1_fold1": ["lightv2_p1_fold1", "regenerated_val (original file UNVERIFIED/stale, not used)"],
               "p1_fold2": ["lightv2_p1_fold2", "regenerated_val (no original file)"],
               "ensemble3": ["mean prob of ft_fold0, p1_fold1, p1_fold2", "ensemble_cv_regenerated"]},
    "predictions": "Step 3.5 single clean pass (evaluation_predictions.npz, probs__<model>__clean, thr__<model>__<name>); rows matched by record ID",
    "metrics": ["macro/weighted/micro AUROC", "macro/micro AUPRC", "macro-F1 (all 45, undefined->0)", "macro-F1 (evaluable classes)", "micro-F1", "common-evaluable-class macro AUROC/AUPRC/F1"],
    "paired_bootstrap": {"B": 1000, "seed": 20261013, "resamples_shared_across_models_within_cohort": True, "metrics": ["macro_auroc", "macro_f1_valid", "micro_f1"], "ci": "percentile 95%"},
    "reproduction_tolerance": 1e-8,
    "notes": "Subset-vs-full differences are descriptive (cohort composition changes); no CI by subtracting independent CIs; not patient-level CIs.",
}
decl["declaration_sha256"] = hashlib.sha256(json.dumps(decl, sort_keys=True).encode()).hexdigest()
inputs = {os.path.relpath(p, ROOT) if p.startswith(ROOT) else p: sha(p) for p in
          [os.path.join(R6, f) for f in ("duplicate_leakage_report.md", "duplicate_pair_summary.csv", "duplicate_summary.json", "exposure_counts.json")] +
          [os.path.join(R5, f) for f in ("evaluation_report.md", "evaluation_config.json", "threshold_audit.md", "aggregate_metrics.csv", "evaluation_predictions.npz", "paired_bootstrap.json", "regenerated_thresholds.npz")] +
          [os.path.join(ROOT, "scripts", f) for f in ("step3_6_duplicate_audit.py", "step3_6_exposure.py", "step3_5_evaluate.py", "step3_5_paired.py")] +
          [os.path.join(ROOT, "src", "ecg_concept", f) for f in ("metrics.py", "labels.py")] + [NB]}
json.dump({"declaration": decl, "input_fingerprints_sha256": inputs,
           "software": {"python": platform.python_version(), "numpy": np.__version__}, "verified_duplicate_totals_mine_vs_step3_6": verified,
           "category4_membership_breakdown": membership_breakdown,
           "record_order_sha256": cfg5["record_order_sha256"], "test_ids_sha256": cfg5["test_record_ids_sha256"]},
          open(os.path.join(OUT, "analysis_config.json"), "w"), indent=1)

# ---------- 4. manifests
rel = lambda r: os.path.relpath(recs[index_of[r]] + ".mat", DS).replace(os.sep, "/")
cross_set, any_set = set(cross_test), set(any_test)


def pf(r):
    p = partner.get(r)
    return fold_of.get(p) if p else None


with open(os.path.join(OUT, "original_fold_manifest.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["record_id", "dataset_rel_path", "original_index", "original_fold", "n_labels", "labels", "dup_group_id", "dup_partner_id",
                "dup_partner_in_experiment", "dup_partner_fold", "is_test_in_cross_pair", "is_nontest_partner_of_test", "is_test_in_any_pair",
                "ft_fold0_weights_trained_on_this_signal", "ft_fold0_weights_trained_on_duplicate_signal"])
    for i, r in enumerate(ids):
        p = partner.get(r)
        pfold = pf(r)
        w.writerow([r, rel(r), i, fold_of[r], len(label_of[r]), "|".join(label_of[r]), group_id.get(r, ""), p or "",
                    (p in fold_of) if p else "", "" if pfold is None else pfold, r in cross_set,
                    bool(p and pfold == 9 and fold_of[r] != 9), r in any_set,
                    1 <= fold_of[r] <= 8, bool(pfold is not None and 1 <= pfold <= 8)])

# prospective roles for the frozen ft_fold0 backbone
role, reason = {}, {}
for r in ids:
    fo, p, pfo = fold_of[r], partner.get(r), pf(r)
    if fo == 9:
        role[r], reason[r] = "test_reserved", ""
    elif p and pfo == 9:
        role[r], reason[r] = "excluded", "nontest_partner_of_test_record"
    elif fo == 0 and p and pfo is not None and 1 <= pfo <= 8:
        role[r], reason[r] = "excluded", "val_candidate_duplicate_in_backbone_training_folds"
    elif fo == 0:
        role[r], reason[r] = "validation", ""
    else:
        role[r], reason[r] = "train", ""
# invariants
active = {"train", "validation"}
for g in {group_id[r] for r in dup_ids}:
    mem = [r for r in dup_ids if group_id[r] == g]
    assert len({role.get(m) for m in mem if role.get(m) in active}) <= 1, ("group spans active roles", g)
    assert not ({role.get(m) for m in mem} & {"test_reserved"} and {role.get(m) for m in mem} & active), ("test group has active member", g)
rid_by_role = defaultdict(set)
for r, ro in role.items():
    rid_by_role[ro].add(r)
assert not (rid_by_role["train"] & rid_by_role["validation"]) and not (rid_by_role["test_reserved"] & (rid_by_role["train"] | rid_by_role["validation"]))
assert rid_by_role["test_reserved"] == set(test_ids)
assert len(role) == 44476 and sum(map(len, rid_by_role.values())) == 44476
assert all(role[p] == "excluded" for r in test_ids if (p := partner.get(r)) in role and fold_of[p] != 9)

with open(os.path.join(OUT, "prospective_role_manifest.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["record_id", "original_fold", "role", "exclusion_reason", "labels", "dup_group_id", "dup_partner_id", "partner_role",
                "within_train_dup_group", "dup_label_conflict", "partner_outside_experiment", "cohort_B_member", "cohort_C_member",
                "backbone_exposure_ft_fold0"])
    for r in ids:
        p = partner.get(r)
        inB = role[r] == "test_reserved" and r not in cross_set
        inC = role[r] == "test_reserved" and r not in any_set
        exposure = {"train": "weights_trained_on_this_signal", "validation": "epoch_threshold_selection_only",
                    "test_reserved": "none_in_weights; test already inspected"}.get(role[r], "weights" if 1 <= fold_of[r] <= 8 else "validation_selection" if fold_of[r] == 0 else "n/a")
        w.writerow([r, fold_of[r], role[r], reason[r], "|".join(label_of[r]), group_id.get(r, ""), p or "", role.get(p, "outside_experiment") if p else "",
                    bool(p and role[r] == "train" and role.get(p) == "train"), bool(p and p in label_of and label_of[p] != label_of[r]),
                    bool(p and p not in fold_of), inB, inC, exposure])

# duplicate group manifest (all 174 members, incl. out-of-experiment)
allhea = {os.path.basename(p)[:-4]: p[:-4] for p in glob.glob(os.path.join(DS, "WFDBRecords", "*", "*", "*.hea"))}
catof = {}
for a, b in pairs:
    catof[a] = catof[b] = cat(a, b)
with open(os.path.join(OUT, "duplicate_group_manifest.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dup_group_id", "record_id", "partner_id", "pair_category", "in_experiment", "original_fold", "prospective_role", "labels_45class",
                "raw_dx_codes", "age", "sex", "dataset_rel_path", "is_test_in_cross_pair", "is_test_in_any_pair"])
    for r in sorted(dup_ids, key=lambda x: (group_id[x], x)):
        hd = {"#Age": "", "#Sex": "", "#Dx": ""}
        for l in open(allhea[r] + ".hea", encoding="utf-8", errors="replace"):
            for k in hd:
                if l.startswith(k):
                    hd[k] = l.split(":", 1)[1].strip()
        w.writerow([group_id[r], r, partner[r], catof[r], r in fold_of, fold_of.get(r, ""), role.get(r, "excluded_out_of_experiment"),
                    "|".join(label_of.get(r, [])), hd["#Dx"], hd["#Age"], hd["#Sex"],
                    os.path.relpath(allhea[r] + ".mat", DS).replace(os.sep, "/"), r in cross_set, r in any_set])

# role counts + class support -> appended to config (explicit update of the file this script created)
lab_idx = {r: [CLASS_NAMES.index(c) for c in label_of[r]] for r in ids}
support = {}
for ro in ("train", "validation", "test_reserved", "excluded"):
    v = np.zeros(45, int)
    for r in rid_by_role[ro]:
        v[lab_idx[r]] += 1
    support[ro] = dict(zip(CLASS_NAMES, map(int, v)))
reasons = Counter(reason[r] for r in ids if role[r] == "excluded")
flags = {"within_train_dup_groups": sum(1 for r in rid_by_role["train"] if partner.get(r) and role.get(partner[r]) == "train") // 2,
         "label_conflict_pairs_in_experiment": sum(1 for a, b in pairs if a in label_of and b in label_of and label_of[a] != label_of[b])}
c = json.load(open(os.path.join(OUT, "analysis_config.json")))
c["prospective_roles"] = {"policy": "see evaluation_protocol.md", "counts": {k: len(v) for k, v in rid_by_role.items()}, "exclusions_by_reason": dict(reasons),
                          "flags": flags, "class_support": support,
                          "original_fold_sizes": [len(x) for x in folds]}
json.dump(c, open(os.path.join(OUT, "analysis_config.json"), "w"), indent=1)
print(json.dumps({"verified": verified, "category4": membership_breakdown, "cohort_sizes": sizes, "roles": c["prospective_roles"]["counts"],
                  "exclusions": dict(reasons), "flags": flags, "python": platform.python_version()}, indent=1))
