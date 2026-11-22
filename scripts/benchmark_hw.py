r"""Hardware benchmark on SYNTHETIC data only (no real ECG, no project weights: a randomly initialised throw-away model is stepped to measure memory/time).
Measures FP32 vs FP16(autocast+GradScaler) vs BF16 (emulated on Turing), feasible physical batch sizes, gradient accumulation cost, the BatchNorm-statistics difference between one batch of 256 and 4x64
micro-batches, and CPU augmentation throughput. Writes results/phase1_infrastructure/hw_benchmark.json (refuses to overwrite).
Usage: .venv\Scripts\python.exe -I scripts\benchmark_hw.py"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import data_v3 as dv  # noqa: E402
from ecg_concept.lightecgnet_paper import DualHeadLightECGNet, LightECGNet  # noqa: E402

OUT = os.path.join(ROOT, "results", "phase1_infrastructure")
DEV = "cuda"
MODES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def synth_batch(b, seed=0):
    """ECG-like synthetic batch (beats + baseline wander + noise), different per sample, roughly z-scored."""
    g = np.random.RandomState(seed)
    t = np.arange(5000) / 500.0
    x = np.zeros((b, 12, 5000), np.float32)
    for i in range(b):
        hr = g.uniform(0.8, 2.0)
        ph = (t * hr) % 1.0
        beat = np.exp(-((ph - 0.3) / 0.015) ** 2) - 0.2 * np.exp(-((ph - 0.5) / 0.06) ** 2) + 0.15 * np.exp(-((ph - 0.12) / 0.03) ** 2)
        for l in range(12):
            x[i, l] = (g.uniform(0.5, 1.5) * beat * (1 if l != 3 else -1) + 0.3 * np.sin(2 * np.pi * g.uniform(0.1, 0.4) * t + g.uniform(0, 6)) + 0.05 * g.randn(5000))
    x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)
    return torch.from_numpy(x)


def run_train(bs, mode, model_fn, steps=6, warm=2):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = model_fn().to(DEV).train()
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=mode == "fp16")
    x = synth_batch(bs).to(DEV)
    y = (torch.rand(bs, 45, device=DEV) < 0.05).float()
    ts = []
    try:
        for it in range(steps):
            torch.cuda.synchronize()
            t = time.time()
            with torch.autocast("cuda", dtype=MODES[mode], enabled=MODES[mode] is not None):
                out = m(x)
                loss = nn.functional.binary_cross_entropy_with_logits(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            ts.append(time.time() - t)
        dt = float(np.mean(ts[warm:]))
        r = {"ok": True, "sec_per_step": dt, "records_per_s": bs / dt, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
    except torch.cuda.OutOfMemoryError:
        r = {"ok": False, "error": "CUDA OOM"}
    del m, opt, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return r


def bn_stat_difference():
    """Train-mode forward of ONE batch of 256 vs the same samples as 4 micro-batches of 64 (dropout disabled): how much BatchNorm batch statistics change the logits."""
    torch.manual_seed(0)
    m = LightECGNet(45).to(DEV).train()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    x = synth_batch(256, seed=1).to(DEV)
    with torch.no_grad():
        full = m(x).float()
        micro = torch.cat([m(x[i:i + 64]) for i in range(0, 256, 64)]).float()
        micro32 = torch.cat([m(x[i:i + 32]) for i in range(0, 256, 32)]).float()
        # first BN layer activation statistics
        bn = m.stem_k3.bn
        h = m.stem_k3.pw(m.stem_k3.dw(m.lead_attn(x)))
        v_full = h.var(dim=(0, 2), unbiased=False)
        v_mic = torch.stack([h[i:i + 64].var(dim=(0, 2), unbiased=False) for i in range(0, 256, 64)]).mean(0)
    d64, d32 = (full - micro).abs(), (full - micro32).abs()
    return {"logit_abs_diff_256_vs_4x64": {"mean": float(d64.mean()), "max": float(d64.max())}, "logit_abs_diff_256_vs_8x32": {"mean": float(d32.mean()), "max": float(d32.max())},
            "logit_std_reference": float(full.std()), "first_bn_variance_relative_diff_4x64": float(((v_full - v_mic).abs() / v_full).mean()),
            "note": "synthetic ECG-like data; real data will differ, but the effect is real: micro-batch BN statistics are not the large-batch statistics"}


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "hw_benchmark.json")
    if os.path.exists(path):
        raise FileExistsError(path)
    p = torch.cuda.get_device_properties(0)
    res = {"device": p.name, "total_memory_gb": p.total_memory / 1e9, "torch": torch.__version__, "cuda": torch.version.cuda, "synthetic_data_only": True, "model": "LightECGNet default (== V2 architecture, random init, throw-away)", "train_step": {}}
    for mode in ("fp32", "fp16", "bf16"):
        res["train_step"][mode] = {}
        for bs in (16, 32, 48, 64, 96, 128, 192, 256):
            r = run_train(bs, mode, lambda: LightECGNet(45))
            res["train_step"][mode][str(bs)] = r
            print(mode, bs, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            if not r["ok"]:
                break
    res["dual_head_fp32_bs64"] = run_train(64, "fp32", lambda: _Dual())
    res["dual_head_fp16_bs64"] = run_train(64, "fp16", lambda: _Dual())
    # gradient accumulation: effective batch 256 as k micro-batches (time per EFFECTIVE step)
    res["accumulation_effective_256"] = {}
    for mode in ("fp32", "fp16"):
        for micro in (32, 64):
            r = res["train_step"][mode].get(str(micro))
            if r and r["ok"]:
                res["accumulation_effective_256"][f"{mode}_{micro}x{256 // micro}"] = {"sec_per_effective_step": r["sec_per_step"] * (256 // micro), "records_per_s": r["records_per_s"], "peak_allocated_gb": r["peak_allocated_gb"]}
    res["bn_statistics"] = bn_stat_difference()
    x = synth_batch(256)
    t = time.time()
    dv.paper_augment(x.numpy(), np.random.RandomState(0))
    res["cpu_augmentation_256_batch_sec"] = time.time() - t
    # inference throughput (eval, no grad)
    m = LightECGNet(45).to(DEV).eval()
    xb = synth_batch(128).to(DEV)
    inf = {}
    for mode in ("fp32", "fp16"):
        with torch.inference_mode(), torch.autocast("cuda", dtype=MODES[mode], enabled=MODES[mode] is not None):
            m(xb)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(5):
                m(xb)
            torch.cuda.synchronize()
        inf[mode] = 5 * 128 / (time.time() - t)
    res["inference_records_per_s_bs128"] = inf
    with open(path, "x") as f:
        json.dump(res, f, indent=1)
    print("saved", path)


class _Dual(nn.Module):
    """sum of both heads' outputs so one forward/backward touches shared backbone + both heads (45 outputs returned)"""

    def __init__(self):
        super().__init__()
        self.m = DualHeadLightECGNet()

    def forward(self, x):
        o = self.m.forward_all(x)
        return o["physionet"] + o["ptbxl"].mean(1, keepdim=True)


if __name__ == "__main__":
    main()
