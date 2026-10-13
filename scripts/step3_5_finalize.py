"""Step 3.5: write evaluation_config.json and a dependency-free SVG of per-class test AUROC (matplotlib is not installed)."""
import csv, glob, hashlib, json, os, sys, platform
import numpy as np, scipy, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "step3_5")
MP = os.path.join(ROOT, "models", "pretrained")
meta = json.load(open(os.path.join(OUT, "cache", "meta.json")))
audit = json.load(open(os.path.join(OUT, "threshold_audit.json")))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
z = np.load(os.path.join(OUT, "evaluation_predictions.npz"))
ids = z["record_ids"].tolist()
cfg = {
    "step": "3.5 full held-out evaluation", "dataset_dir": meta["dataset_dir"], "dataset_root_relative": "data/raw/" + meta["dataset_dir"],
    "n_records_indexed": meta["n_records"], "record_order_sha256": meta["record_order_sha256"],
    "record_order_rule": "os.walk(WFDBRecords) with sorted filenames per directory; keep records with >=1 label among the 45 classes",
    "split": meta["split"], "fold_sizes": meta["fold_sizes"], "test_fold": 9, "n_test": len(ids),
    "test_record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "test_record_ids": ids,
    "class_order": meta["classes"],
    "preprocessing": {"read": "scipy.io.loadmat key 'val' (12,5000) int16", "pad_crop": "zero-pad / crop tail to 5000",
                      "filter": "butter(3,[0.5,40]/250,'band') + filtfilt per lead", "int16_truncation_after_filter": True,
                      "normalisation": "per-lead z-score, eps 1e-8", "dtype": "float32"},
    "inference": {"device": meta["device"], "gpu": meta["gpu"], "precision": "fp32 (no autocast)", "batch_size": meta["batch_size"],
                  "mode": "model.eval(), torch.inference_mode()", "output": "sigmoid(logits), float32",
                  "tta5": "clean + 4 passes of (+N(0,0.005) noise, x U(0.95,1.05)), seeded RandomState(1000*p+batch_start); probs averaged",
                  "ensemble3": "mean of probabilities of ft_fold0, p1_fold1, p1_fold2"},
    "bootstrap": {"type": "record-level resampling with replacement (multi-label records kept whole)", "B": 1000, "seed": 20261013,
                  "ci": "percentile 95%", "shared_resamples_across_models": True,
                  "undefined": "per-class metrics undefined in a resample (no positives/negatives) are skipped; counts in *_boot_valid"},
    "checkpoints_sha256": meta["checkpoints"], "threshold_files_sha256": meta["threshold_files"],
    "threshold_status": {c: audit["per_checkpoint"][c]["status"] for c in audit["per_checkpoint"]},
    "regenerated_thresholds_file": "results/step3_5/regenerated_thresholds.npz (fit on each checkpoint's own validation fold; NOT original artifacts)",
    "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
    "metrics_implementation": "src/ecg_concept/metrics.py (NumPy/SciPy; verified against brute-force AUROC/AP definitions and the scikit-learn doc examples)",
}
json.dump(cfg, open(os.path.join(OUT, "evaluation_config.json"), "w"), indent=1)

# ---- SVG: per-class AUROC with 95% CI, ensemble vs ft_fold0, classes ordered by test prevalence
rows = list(csv.DictReader(open(os.path.join(OUT, "per_class_metrics.csv"))))
def sel(m, t): return {r["class"]: r for r in rows if r["model"] == m and r["score_config"] == "clean" and r["threshold_config"] == t}
A, E = sel("lightv2_ft_fold0", "original_file"), sel("ensemble3", "ensemble_cv_regenerated")
cls = sorted([c for c in A if A[c]["auroc"] != ""], key=lambda c: -int(A[c]["n_pos"]))
W, H, L, T, rowh = 760, 40 + 14 * len(cls), 90, 30, 14
x = lambda v: L + (v - 0.70) / 0.30 * (W - L - 40)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="10">',
     '<rect width="100%" height="100%" fill="white"/>',
     f'<text x="{L}" y="16" font-size="12" font-weight="bold">Per-class test AUROC (95% bootstrap CI), classes ordered by #positives (n in brackets)</text>']
for g in (0.7, 0.8, 0.9, 1.0):
    s.append(f'<line x1="{x(g)}" y1="{T-6}" x2="{x(g)}" y2="{H-10}" stroke="#ddd"/><text x="{x(g)-8}" y="{H-0}" fill="#555">{g:.1f}</text>')
for i, c in enumerate(cls):
    y = T + i * rowh
    s.append(f'<text x="4" y="{y+9}" fill="#222">{c} [{A[c]["n_pos"]}]</text>')
    for r, col, dy in ((A[c], "#1f77b4", 3), (E[c], "#d95f02", 8)):
        lo, hi, v = float(r["auroc_lo"]), float(r["auroc_hi"]), float(r["auroc"])
        lo, hi = max(lo, 0.70), min(hi, 1.0)
        s.append(f'<line x1="{x(lo)}" y1="{y+dy}" x2="{x(hi)}" y2="{y+dy}" stroke="{col}" stroke-width="1.5"/><circle cx="{x(max(v,0.70))}" cy="{y+dy}" r="2.5" fill="{col}"/>')
s.append(f'<circle cx="{W-200}" cy="{H-18}" r="3" fill="#1f77b4"/><text x="{W-193}" y="{H-15}">ft_fold0</text><circle cx="{W-130}" cy="{H-18}" r="3" fill="#d95f02"/><text x="{W-123}" y="{H-15}">3-model ensemble</text>')
s.append("</svg>")
open(os.path.join(OUT, "per_class_auroc.svg"), "w").write("\n".join(s))
print("ok", len(cls), "classes plotted (axis clipped at 0.70)")
