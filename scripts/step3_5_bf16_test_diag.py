"""Step 3.5 diagnostic: ft_fold0 on the test fold under CPU bf16 autocast (clean pass), to separate the effect of the
original bf16 numerics from TTA randomness when comparing with notebook cell 24. Diagnostic only; no tuning.
Thresholds = the VERIFIED original ft_fold0 file."""
import glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept.inference import load_model
from ecg_concept.metrics import aggregate_ranking, threshold_metrics
from ecg_concept.preprocessing import load_ecg
OUT = os.path.join(ROOT, "results", "step3_5"); MP = os.path.join(ROOT, "models", "pretrained")
meta = json.load(open(os.path.join(OUT, "cache", "meta.json")))
ids = meta["fold_ids"]["9"]
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
lookup = {os.path.basename(p)[:-4]: p[:-4] for p in glob.glob(os.path.join(DS, "WFDBRecords", "*", "*", "*.hea"))}
f9 = np.load(os.path.join(OUT, "cache", "fold9.npz")); assert list(f9["ids"]) == ids
Y = f9["targets"].astype(bool)
m = load_model(os.path.join(MP, "lightv2_ft_fold0.pt"), "cpu"); pool = ThreadPoolExecutor(6)
P = np.zeros((len(ids), 45), np.float32); t0 = time.time()
for b in range(0, len(ids), 32):
    x = torch.from_numpy(np.stack(list(pool.map(lambda r: load_ecg(lookup[r] + ".mat"), ids[b:b + 32]))))
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        P[b:b + len(x)] = torch.sigmoid(m(x).float()).numpy()
    if (b // 32) % 20 == 0: print(time.strftime("%H:%M:%S"), b, flush=True)
thr = np.load(os.path.join(MP, "lightv2_ft_fold0_thresh.npy"))
fp32 = f9["lightv2_ft_fold0|probs"]
res = {"ranking_bf16": aggregate_ranking(Y, P), "thr_bf16": threshold_metrics(Y, P, thr),
       "ranking_fp32": aggregate_ranking(Y, fp32), "thr_fp32": threshold_metrics(Y, fp32, thr),
       "max_abs_prob_diff_bf16_vs_fp32": float(np.abs(P - fp32).max()), "mean_abs_prob_diff": float(np.abs(P - fp32).mean())}
json.dump(res, open(os.path.join(OUT, "bf16_test_diagnostic.json"), "w"), indent=1); print(json.dumps(res, indent=1))
