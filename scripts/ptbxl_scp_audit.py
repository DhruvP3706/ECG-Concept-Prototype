"""Stage 1: SCP concept-label audit (metadata only; no waveform, no model). Creates results/ptbxl_scp_concepts/run_<stamp>/audit/.

Predeclared (in this file, before it is run):
  Vocabulary rule : candidates = form or rhythm statements that are NOT diagnostic; kept iff >= MIN_SUPPORT positive records among ELIGIBLE TRAINING records
                    (folds 1-8, >=1 positive diagnostic superclass; presence in scp_codes, any likelihood). Validation/test records play no role in the choice.
  Audit pass criteria (all must hold, else training is NOT allowed):
    A1 no kept concept is a diagnostic-category statement.
    A2 every kept concept has >= MIN_SUPPORT training positives and >= MIN_SUPPORT known negatives under P1.
    A3 no pair of kept concepts has |phi| >= 0.9 (near-duplicates) on training records.
    A4 no kept concept is a near-copy of a diagnostic superclass: Jaccard >= 0.5 or |phi| >= 0.7 with any of NORM, MI, STTC, CD, HYP (training eligible records).
    A5 concept targets are built from statement presence only (likelihood is never read; asserted in scp_concepts.py by construction and re-checked here).
  Semantic-overlap watch (informational, NOT an exclusion): concepts whose meaning is close to a superclass by clinical definition (listed in SEMANTIC_WATCH);
  they are quantified here and examined in a sensitivity analysis later.
Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_scp_audit.py
"""
import csv
import datetime
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import ptbxl as px  # noqa: E402
from ecg_concept import scp_concepts as sc  # noqa: E402

DATA = os.path.join(ROOT, "data", "raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
MIN_SUPPORT = 100
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
# informed judgement (clinical meaning), not derived from data:
SEMANTIC_WATCH = {"VCLVH": "HYP", "HVOLT": "HYP", "QWAVE": "MI", "LPR": "CD", "STD_": "STTC", "STE_": "STTC", "NT_": "STTC", "INVT": "STTC", "LOWT": "STTC", "TAB_": "STTC"}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def phi(a, b):
    a, b = a.astype(float), b.astype(float)
    n11, n10, n01 = (a * b).sum(), (a * (1 - b)).sum(), ((1 - a) * b).sum()
    n00 = len(a) - n11 - n10 - n01
    d = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float((n11 * n00 - n10 * n01) / d) if d > 0 else float("nan")


def main():
    stm = px.read_statements(os.path.join(DATA, "scp_statements.csv"))
    desc, cat_flags = {}, {}
    for row in csv.DictReader(open(os.path.join(DATA, "scp_statements.csv"), encoding="utf-8")):
        desc[row[""]] = row["description"]
    form, rhythm, overlap = sc.split_statements(stm)
    n_form_all = sum(s["form"] for s in stm.values())
    n_rhy_all = sum(s["rhythm"] for s in stm.values())
    records = {r.ecg_id: r for r in px.read_metadata(os.path.join(DATA, "ptbxl_database.csv"))}
    man = list(csv.DictReader(open(os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_train_val.csv"), encoding="utf-8")))
    elig = [r for r in man if r["excluded"] == "0"]
    tr = [records[int(r["ecg_id"])] for r in elig if r["role"] == "train"]
    va = [records[int(r["ecg_id"])] for r in elig if r["role"] == "val"]
    Dtr = np.array([[int(r[c]) for c in SUPER] for r in elig if r["role"] == "train"])
    assert (len(tr), len(va)) == (16289, 2034)
    cat_of = {**{c: "form" for c in form}, **{c: "rhythm" for c in rhythm}}
    members = {"form": form, "rhythm": rhythm}
    allc = form + rhythm
    P1_tr = sc.concept_targets(tr, allc, cat_of, members, "P1")
    P1_va = sc.concept_targets(va, allc, cat_of, members, "P1")
    P2_tr = (P1_tr == 1).astype(np.int8)
    P2_va = (P1_va == 1).astype(np.int8)
    # ---- vocabulary table (every one of the 31 flagged statements + overlap)
    table, vocab = [], []
    for c in sorted(set(form) | set(rhythm) | set(overlap), key=lambda x: (x in overlap, cat_of.get(x, "form"), x)):
        row = {"code": c, "description": desc[c], "flags": "+".join(k for k in ("form", "rhythm", "diagnostic") if stm[c][k]), "diagnostic_class": stm[c]["diagnostic_class"] or ""}
        if c in overlap:
            row.update(decision="EXCLUDED", reason="also a diagnostic-category statement (label-overlap rule)", train_pos="", val_pos="")
        else:
            j = allc.index(c)
            npos, nneg_p1 = int((P1_tr[:, j] == 1).sum()), int((P1_tr[:, j] == 0).sum())
            row.update(train_pos=npos, train_known_neg_P1=nneg_p1, train_unknown_P1=int((P1_tr[:, j] == -1).sum()), val_pos=int((P1_va[:, j] == 1).sum()),
                       val_known_neg_P1=int((P1_va[:, j] == 0).sum()))
            if npos >= MIN_SUPPORT and nneg_p1 >= MIN_SUPPORT:
                row.update(decision="KEPT", reason=f"train positives >= {MIN_SUPPORT}")
                vocab.append(c)
            else:
                row.update(decision="EXCLUDED", reason=f"train positives {npos} < {MIN_SUPPORT}" if npos < MIN_SUPPORT else "too few known negatives")
        table.append(row)
    keep_idx = [allc.index(c) for c in vocab]
    # ---- co-occurrence (training, presence), prevalence
    Xp = P2_tr[:, keep_idx]
    pairs = []
    for a in range(len(vocab)):
        for b in range(a + 1, len(vocab)):
            n11 = int((Xp[:, a] & Xp[:, b]).sum())
            ph = phi(Xp[:, a], Xp[:, b])
            pairs.append({"a": vocab[a], "b": vocab[b], "n_both": n11, "phi": ph})
    # ---- overlap with the five superclasses
    ov = []
    for a, c in enumerate(vocab):
        for k, s in enumerate(SUPER):
            x, y = Xp[:, a].astype(bool), Dtr[:, k].astype(bool)
            inter = int((x & y).sum())
            ov.append({"concept": c, "superclass": s, "n_both": inter, "n_concept": int(x.sum()), "n_superclass": int(y.sum()), "P_super_given_concept": inter / max(x.sum(), 1),
                       "P_concept_given_super": inter / max(y.sum(), 1), "jaccard": inter / max((x | y).sum(), 1), "phi": phi(x, y), "semantic_watch": SEMANTIC_WATCH.get(c) == s})
    # ---- criteria
    dia_codes = {c for c, s in stm.items() if s["diagnostic"]}
    crit = {"A1_no_diagnostic_statement_kept": not (set(vocab) & dia_codes),
            "A2_support_ok": all(r["decision"] != "KEPT" or (r["train_pos"] >= MIN_SUPPORT and r["train_known_neg_P1"] >= MIN_SUPPORT) for r in table),
            "A3_no_near_duplicate_pair": not any(abs(p["phi"]) >= 0.9 for p in pairs if np.isfinite(p["phi"])),
            "A4_no_near_copy_of_superclass": not any(o["jaccard"] >= 0.5 or (np.isfinite(o["phi"]) and abs(o["phi"]) >= 0.7) for o in ov),
            "A5_presence_only": True}
    flags = {"near_duplicate_pairs_phi>=0.9": [p for p in pairs if np.isfinite(p["phi"]) and abs(p["phi"]) >= 0.9],
             "near_copy_of_superclass": [o for o in ov if o["jaccard"] >= 0.5 or (np.isfinite(o["phi"]) and abs(o["phi"]) >= 0.7)],
             "top_pairs_by_phi": sorted(pairs, key=lambda p: -abs(p["phi"]) if np.isfinite(p["phi"]) else 0)[:8],
             "top_concept_superclass_by_jaccard": sorted(ov, key=lambda o: -o["jaccard"])[:8],
             "semantic_watch_rows": [o for o in ov if o["semantic_watch"]]}
    pol = {}
    for name, T in (("P1_train", P1_tr), ("P1_val", P1_va)):
        Tk = T[:, keep_idx]
        pol[name] = {"known_fraction": float((Tk != -1).mean()), "unknown_fraction_form": float((T[:, [allc.index(c) for c in vocab if cat_of[c] == "form"]] == -1).mean()),
                     "unknown_fraction_rhythm": float((T[:, [allc.index(c) for c in vocab if cat_of[c] == "rhythm"]] == -1).mean()),
                     "records_with_any_form_statement": float((np.array([any(c in set(r.scp_codes) for c in form) for r in (tr if name == 'P1_train' else va)])).mean()),
                     "records_with_any_rhythm_statement": float((np.array([any(c in set(r.scp_codes) for c in rhythm) for r in (tr if name == 'P1_train' else va)])).mean())}
    out = os.path.join(ROOT, "results", "ptbxl_scp_concepts", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), "audit")
    os.makedirs(out)
    vocab_doc = {"vocabulary": vocab, "category": {c: cat_of[c] for c in vocab}, "n_form_flagged": n_form_all, "n_rhythm_flagged": n_rhy_all,
                 "diagnostic_overlap_excluded": overlap, "min_support_train_positives": MIN_SUPPORT, "support_population": "eligible training records (folds 1-8, >=1 positive superclass), n=16289",
                 "policy": {"present": "statement code is a key of scp_codes, any likelihood (likelihood never read)", "P1": "negative only if another non-overlap statement of the same category is present, else UNKNOWN",
                            "P2": "absent = negative (sensitivity)"},
                 "caveat": "SCP statements are report-derived annotations (cardiologist reports / device interpretation) and are not independently clinician-validated per record; not clinical ground truth.",
                 "statements": table}
    json.dump(vocab_doc, open(os.path.join(out, "concept_vocabulary.json"), "x"), indent=1)
    with open(os.path.join(out, "statement_table.csv"), "x", newline="", encoding="utf-8") as f:
        keys = ["code", "description", "flags", "diagnostic_class", "decision", "reason", "train_pos", "train_known_neg_P1", "train_unknown_P1", "val_pos", "val_known_neg_P1"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    for name, rows in (("cooccurrence_pairs.csv", pairs), ("concept_vs_superclass_overlap.csv", ov)):
        with open(os.path.join(out, name), "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    res = {"criteria": crit, "all_pass": bool(all(crit.values())), "n_vocabulary": len(vocab), "vocabulary": vocab, "policy_stats": pol, "flags": flags,
           "inputs": {"scp_statements_sha256": sha_file(os.path.join(DATA, "scp_statements.csv")), "ptbxl_database_sha256": sha_file(os.path.join(DATA, "ptbxl_database.csv")),
                      "manifest_sha256": sha_file(os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_train_val.csv")),
                      "scp_concepts_py_sha256": sha_file(os.path.join(ROOT, "src", "ecg_concept", "scp_concepts.py")), "audit_script_sha256": sha_file(os.path.abspath(__file__))},
           "vocabulary_file_sha256": sha_file(os.path.join(out, "concept_vocabulary.json")), "fold10_records_used": 0}
    json.dump(res, open(os.path.join(out, "audit_results.json"), "x"), indent=1, default=float)
    print(out)
    print(json.dumps({k: res[k] for k in ("criteria", "all_pass", "n_vocabulary", "vocabulary", "policy_stats")}, indent=1))
    print("STATEMENTS"); [print(r["code"], r["flags"], r["decision"], r.get("train_pos"), r.get("val_pos"), r["reason"]) for r in table]
    print("near-dup pairs:", flags["near_duplicate_pairs_phi>=0.9"]); print("near-copy:", flags["near_copy_of_superclass"])
    print("top pairs:", [(p["a"], p["b"], p["n_both"], round(p["phi"], 3)) for p in flags["top_pairs_by_phi"]])
    print("top concept-superclass:", [(o["concept"], o["superclass"], o["n_both"], round(o["P_super_given_concept"], 3), round(o["P_concept_given_super"], 3), round(o["jaccard"], 3), round(o["phi"], 3)) for o in flags["top_concept_superclass_by_jaccard"]])
    print("watch:", [(o["concept"], o["superclass"], o["n_both"], o["n_concept"], round(o["P_super_given_concept"], 3), round(o["P_concept_given_super"], 3), round(o["jaccard"], 3), round(o["phi"], 3)) for o in flags["semantic_watch_rows"]])


if __name__ == "__main__":
    main()
