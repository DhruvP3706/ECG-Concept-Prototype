"""Step 3.5 diagnostic (validation folds only): does emulating the original bf16 autocast explain the small
threshold / F1 differences? The original evaluated under torch.autocast(cuda, bfloat16) on an RTX 4060; the
GTX 1650 has no native bf16, so this uses CPU bf16 autocast (an approximation of that numerics, not identical).
Primary results elsewhere stay fp32. Fold 9 is never touched here.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.labels import CLASS_NAMES, build_index  # noqa: E402
from ecg_concept.metrics import find_optimal_thresholds, notebook_phase1_thresholds, threshold_metrics  # noqa: E402
from ecg_concept.preprocessing import load_ecg  # noqa: E402
import glob  # noqa: E402

OUT = os.path.join(ROOT, "results", "step3_5")
MP = os.path.join(ROOT, "models", "pretrained")
meta = json.load(open(os.path.join(OUT, "cache", "meta.json")))
VAL = {"lightv2_ft_fold0": 0, "lightv2_p1_fold0": 0, "lightv2_p1_fold1": 1, "lightv2_p1_fold2": 2}
NB = {"lightv2_p1_fold0": 0.4541, "lightv2_ft_fold0": 0.4211, "lightv2_p1_fold1": 0.4321, "lightv2_p1_fold2": 0.4432}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# paths for validation records: re-derive from stored IDs (no tree walk needed)
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
lookup = {}
for p in glob.glob(os.path.join(DS, "WFDBRecords", "*", "*", "*.hea")):
    lookup[os.path.basename(p)[:-4]] = p[:-4]
log("path lookup", len(lookup))
models = {c: load_model(os.path.join(MP, c + ".pt"), "cpu") for c in VAL}
pool = ThreadPoolExecutor(6)
probs = {}
for fold in (0, 1, 2):
    fids = meta["fold_ids"][str(fold)]
    cache = np.load(os.path.join(OUT, "cache", f"fold{fold}.npz"))
    assert list(cache["ids"]) == fids
    mine = [c for c in VAL if VAL[c] == fold]
    acc = {c: np.zeros((len(fids), 45), np.float32) for c in mine}
    for b0 in range(0, len(fids), 32):
        x = np.stack(list(pool.map(lambda r: load_ecg(lookup[r] + ".mat"), fids[b0:b0 + 32])))
        xt = torch.from_numpy(x)
        for c in mine:
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                lo = models[c](xt)
            acc[c][b0:b0 + len(x)] = torch.sigmoid(lo.float()).numpy()
        if (b0 // 32) % 20 == 0:
            log("fold", fold, b0, "/", len(fids))
    for c in mine:
        probs[c] = acc[c]
    np.savez_compressed(os.path.join(OUT, "cache", f"bf16_val_fold{fold}.npz"), **{c: acc[c] for c in mine})

res = {}
for c in VAL:
    y = np.load(os.path.join(OUT, "cache", f"fold{VAL[c]}.npz"))["targets"].astype(bool)
    thr, f1 = notebook_phase1_thresholds(probs[c], y)
    fp = os.path.join(MP, c + "_thresh.npy")
    ent = {"bf16_regenerated_val_macro_f1": float(f1), "notebook_recorded": NB[c], "abs_diff": abs(f1 - NB[c])}
    if os.path.exists(fp):
        tf = np.load(fp)
        ent["file_val_macro_f1_bf16"] = threshold_metrics(y, probs[c], tf)["macro_f1_nb"]
        ent["file_equal_classes_vs_bf16_regenerated"] = int(np.isclose(tf, thr, atol=1e-9).sum())
    res[c] = ent
ens = ["lightv2_ft_fold0", "lightv2_p1_fold1", "lightv2_p1_fold2"]
cp = np.vstack([probs[c] for c in ens])
cy = np.vstack([np.load(os.path.join(OUT, "cache", f"fold{VAL[c]}.npz"))["targets"].astype(bool) for c in ens])
et = find_optimal_thresholds(cp, cy)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
src = open(os.path.join(ROOT, "scripts", "step3_5_threshold_audit.py")).read()
printed = eval(src[src.index("NB_ENS_PRINTED = ") + len("NB_ENS_PRINTED = "):src.index("}\n", src.index("NB_ENS_PRINTED = ")) + 1])
ok = sum(1 for i, cl in enumerate(CLASS_NAMES) if cl in printed and abs(et[i] - printed[cl]) < 1e-9)
unp = sum(1 for i, cl in enumerate(CLASS_NAMES) if cl not in printed and abs(et[i] - 0.5) <= 0.1 + 1e-9)
res["ensemble_cv_bf16"] = {"printed_exact_match": ok, "printed_total": len(printed), "unprinted_within_0.1": unp,
                          "unprinted_total": 45 - len(printed)}
json.dump(res, open(os.path.join(OUT, "bf16_diagnostic.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
