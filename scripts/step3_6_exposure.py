"""Step 3.6: descriptive exposure counts from duplicate_pair_summary.csv (no metrics, no model outputs)."""
import csv, json, os
from collections import Counter
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "step3_6")
rows = [r for r in csv.DictReader(open(os.path.join(R, "duplicate_pair_summary.csv"))) if r["kind"] == "exact_file"]
VAL = {"ft_fold0": 0, "p1_fold0": 0, "p1_fold1": 1, "p1_fold2": 2}
cross = [r for r in rows if r["category"] == "2_cross_test_boundary"]
def split(r):  # (test id, test labels, partner fold)
    a = r["fold_a"] == "9"
    return (r["id_a"], r["labels_a"], int(r["fold_b"])) if a else (r["id_b"], r["labels_b"], int(r["fold_a"]))
out = {}
for name, pred in (("any_cross_pair", lambda r: True), ("identical_labels", lambda r: r["labels_equal"] == "True"),
                   ("shares_>=1_label", lambda r: set(r["labels_a"].split("|")) & set(r["labels_b"].split("|")))):
    sel = [split(r) for r in cross if pred(r)]
    ex = {ck: sorted(t for t, _, f in sel if f in set(range(9)) - {v}) for ck, v in VAL.items()}
    ens = sorted({t for ck in ("ft_fold0", "p1_fold1", "p1_fold2") for t in ex[ck]})
    cls = Counter(c for t, l, _ in sel for c in l.split("|"))
    out[name] = {"test_records": len(sel), "exposed_per_checkpoint": {k: len(v) for k, v in ex.items()},
                 "exposed_to_any_ensemble_member": len(ens), "class_counts_of_test_records": dict(cls.most_common())}
# duplicates straddling the validation folds used for thresholds / epoch selection (pairs with both members in folds 0-8)
c3 = [r for r in rows if r["category"] == "3_both_outside_test"]
out["outside_test_pairs"] = {"pairs": len(c3), "same_fold": sum(r["fold_a"] == r["fold_b"] for r in c3),
                             "different_fold": sum(r["fold_a"] != r["fold_b"] for r in c3),
                             "straddling_validation_fold_of": {ck: sum((int(r["fold_a"]) == v) != (int(r["fold_b"]) == v) for r in c3) for ck, v in VAL.items()}}
out["age0_in_pairs"] = {"pairs_with_age_0_on_one_side": sum(("0" in (r["age_a"], r["age_b"])) for r in rows),
                        "of_cross_pairs": sum(("0" in (r["age_a"], r["age_b"])) for r in cross)}
json.dump(out, open(os.path.join(R, "exposure_counts.json"), "w"), indent=1); print(json.dumps(out, indent=1))
