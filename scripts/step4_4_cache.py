"""Step 4.4 cache pipeline: frozen Step 4.3 extractor + frozen ft_fold0 pooled embeddings for TRAIN and VALIDATION records only.

Stages (resumable, chunked, fingerprinted; never overwrites an existing file):
  index    permitted records from the Step 3.7 role manifest (train + validation; test_reserved/excluded rejected explicitly)
  extract  frozen extractor (src/ecg_concept/rr_concepts.py, frozen parameters) in parallel worker processes -> cache/concepts/chunk_*.csv
  embed    exact recovered backbone preprocessing + frozen ft_fold0 (eval, inference_mode) -> cache/embeddings/chunk_*.npz (pooled 512-d, original logits)
  merge    join everything BY RECORD ID into cache/merged.npz (+ merged_meta.json); completeness and alignment asserted
No raw waveform arrays are stored. Run:  .venv\\Scripts\\python.exe -I scripts\\step4_4_cache.py <stage>
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import csv
import glob
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import beat_detection as bd  # noqa: E402
from ecg_concept import rr_concepts as rc  # noqa: E402
from ecg_concept.labels import CLASS_NAMES  # noqa: E402
from ecg_concept.preprocessing import read_raw_mat  # noqa: E402

OUT = os.path.join(ROOT, "results", "step4_4")
CACHE = os.path.join(OUT, "cache")
CONC, EMB = os.path.join(CACHE, "concepts"), os.path.join(CACHE, "embeddings")
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
ROLE = os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv")
FOLD = os.path.join(ROOT, "results", "step3_7", "original_fold_manifest.csv")
CKPT = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")
FROZEN = os.path.join(ROOT, "results", "step4_3", "extraction_config.json")
CHUNK = 500
LOGF = os.path.join(OUT, "execution_log.txt")
ALLOWED = ("train", "validation")


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


def verify_fingerprints():
    """Extractor sources/parameters, manifests and backbone must equal what was frozen/recorded; mismatch aborts."""
    fz = json.load(open(FROZEN))
    for k, v in fz["source_sha256"].items():
        assert sha_file(os.path.join(ROOT, k)) == v, f"extractor source changed since freeze: {k}"
    assert hashlib.sha256(json.dumps(rc.CONFIG, sort_keys=True).encode()).hexdigest() == fz["parameters_sha256"], "extractor parameters changed"
    assert sha_file(ROLE) == fz["input_sha256"]["role_manifest"], "role manifest changed"
    assert sha_file(FOLD) == fz["input_sha256"]["original_fold_manifest"], "fold manifest changed"
    return {"extractor_sources": fz["source_sha256"], "extractor_params": fz["parameters_sha256"], "role_manifest": sha_file(ROLE),
            "original_fold_manifest": sha_file(FOLD), "backbone_ckpt": sha_file(CKPT)}


def read_index():
    return list(csv.DictReader(open(os.path.join(CACHE, "record_index.csv"), encoding="utf-8")))


def stage_index():
    dest = os.path.join(CACHE, "record_index.csv")
    assert not os.path.exists(dest)
    os.makedirs(CONC, exist_ok=True)
    os.makedirs(EMB, exist_ok=True)
    fp = verify_fingerprints()
    rel = {r["record_id"]: r["dataset_rel_path"] for r in csv.DictReader(open(FOLD))}
    rows = list(csv.DictReader(open(ROLE)))
    rejected = {r: sum(x["role"] == r for x in rows) for r in ("test_reserved", "excluded")}
    keep = sorted((x for x in rows if x["role"] in ALLOWED), key=lambda x: (x["role"] != "train", x["record_id"]))
    assert sum(x["role"] == "train" for x in keep) == 35568 and sum(x["role"] == "validation" for x in keep) == 4437
    assert not any(x["role"] in ("test_reserved", "excluded") for x in keep)
    for x in keep:
        assert all(c in CLASS_NAMES for c in x["labels"].split("|")), x["record_id"]
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "role", "dup_group_id", "dup_partner_id", "within_train_dup_group", "dup_label_conflict", "labels", "rel_path", "chunk"])
        for i, x in enumerate(keep):
            w.writerow([x["record_id"], x["role"], x["dup_group_id"], x["dup_partner_id"], x["within_train_dup_group"], x["dup_label_conflict"], x["labels"], rel[x["record_id"]], i // CHUNK])
    json.dump({"fingerprints": fp, "n_train": 35568, "n_validation": 4437, "rejected_roles_not_read": rejected, "chunk_size": CHUNK, "class_order": CLASS_NAMES,
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(CACHE, "cache_meta.json"), "w"), indent=1)
    log("index written", len(keep), "records; rejected (never read):", rejected)


# ------------------------------------------------------------------ extraction
COLS = ["record_id", "read_ok", "structural_ok", "n_usable_leads", "n_flat_leads", "n_clip_leads", "n_nonfinite_leads", "n_beats_A", "n_beats_B", "n_beats_cons",
        "n_rr_cons", "n_matched", "n_unmatched_A", "n_unmatched_B", "median_support_A", "median_support_B", "match_median_abs_dt_ms", "match_max_abs_dt_ms",
        "n_matched_nonzero_dt", "rate_mean_bpm_A", "rate_mean_bpm_B", "rate_mean_bpm_cons", "rate_median_bpm_cons", "rr_cv_cons", "rr_nrmssd_cons", "rr_mad_succ_cons",
        "rate_state", "irr_state", "reason_codes", "accepted_rate_mean_bpm", "accepted_rate_median_bpm", "accepted_rr_cv", "accepted_rr_nrmssd", "accepted_rr_mad_succ",
        "runtime_s"]


def _row(rid, path):
    try:
        res = rc.extract_record(read_raw_mat(path), rc.CONFIG, return_beats=True)
        res["read_ok"] = True
    except Exception as e:
        res = rc.extract_record(None, rc.CONFIG, return_beats=True)
        res["read_ok"] = False
        res["reason_codes"] = "READ_ERROR:" + type(e).__name__
    b = res.pop("beats", {}) or {}
    fs = rc.CONFIG["fs"]
    res["match_max_abs_dt_ms"], res["n_matched_nonzero_dt"] = float("nan"), 0
    if len(b.get("A", [])) and len(b.get("B", [])):                      # actual timing differences between matched detector beats
        a, bb = np.asarray(b["A"]) * fs, np.asarray(b["B"]) * fs
        pairs, _, _ = bd.match_one_to_one(a, bb, rc.CONFIG["match_tol_s"] * fs)
        if pairs:
            d = np.array([abs(a[i] - bb[j]) for i, j in pairs]) / fs * 1000
            res["match_max_abs_dt_ms"], res["n_matched_nonzero_dt"] = float(d.max()), int((d > 0).sum())
    res["record_id"] = rid
    return res


def run_chunk(args):
    ci, items = args
    dest = os.path.join(CONC, f"chunk_{ci:04d}.csv")
    if os.path.exists(dest):
        return ci, 0
    rows = [_row(rid, os.path.join(DS, rel)) for rid, rel in items]
    tmp = dest + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([repr(r[c]) if isinstance(r.get(c), float) else r.get(c, "") for c in COLS])
    os.replace(tmp, dest)
    return ci, len(rows)


def chunks_of(idx):
    out = {}
    for r in idx:
        out.setdefault(int(r["chunk"]), []).append((r["record_id"], r["rel_path"]))
    return out


def stage_extract():
    verify_fingerprints()
    idx = read_index()
    ch = chunks_of(idx)
    todo = [(ci, it) for ci, it in sorted(ch.items()) if not os.path.exists(os.path.join(CONC, f"chunk_{ci:04d}.csv"))]
    log(f"extract: {len(ch)} chunks, {len(todo)} to do")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(10) as ex:
        for ci, n in ex.map(run_chunk, todo):
            done += n
            if ci % 10 == 0:
                log(f"  chunk {ci} done; {done} records in {time.time() - t0:.0f}s")
    log(f"extract finished: {done} records in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ embeddings
def _state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def stage_embed():
    import torch
    from ecg_concept.inference import load_model
    from ecg_concept.preprocessing import load_ecg
    fp = verify_fingerprints()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT, dev)                                          # safe load, strict, eval mode
    assert not model.training
    before = _state_hash(model)
    idx = read_index()
    ch = chunks_of(idx)
    pool = ThreadPoolExecutor(8)
    t0 = time.time()
    n_done = 0
    for ci, items in sorted(ch.items()):
        dest = os.path.join(EMB, f"chunk_{ci:04d}.npz")
        if os.path.exists(dest):
            continue

        def prep(it):
            try:
                x = load_ecg(os.path.join(DS, it[1]), quantize_int16=True)        # exact recovered backbone preprocessing
                return x if x.shape == (12, 5000) and np.isfinite(x).all() else None
            except Exception:
                return None

        xs = list(pool.map(prep, items))
        pooled = np.full((len(items), 512), np.nan, np.float32)
        logits = np.full((len(items), 45), np.nan, np.float32)
        ok = np.array([x is not None for x in xs])
        good = [i for i, x in enumerate(xs) if x is not None]
        with torch.inference_mode():
            for b0 in range(0, len(good), 64):
                sel = good[b0:b0 + 64]
                out = model.forward_features(torch.from_numpy(np.stack([xs[i] for i in sel])).to(dev))
                pooled[sel] = out["pooled"].float().cpu().numpy()
                logits[sel] = out["logits"].float().cpu().numpy()
        tmp = dest + ".tmp.npz"
        np.savez_compressed(tmp, ids=np.array([i for i, _ in items]), pooled=pooled, logits=logits, ok=ok)
        os.replace(tmp, dest)
        n_done += len(items)
        if ci % 10 == 0:
            log(f"  embed chunk {ci}: {n_done} records in {time.time() - t0:.0f}s")
    after = _state_hash(model)
    assert before == after and not model.training, "backbone changed during embedding"
    json.dump({"backbone_state_sha256_before": before, "backbone_state_sha256_after": after, "unchanged": before == after, "ckpt_sha256": fp["backbone_ckpt"],
               "device": dev, "preprocessing": "ecg_concept.preprocessing.load_ecg(quantize_int16=True)"}, open(os.path.join(CACHE, "backbone_check.json"), "w"), indent=1)
    log(f"embed finished in {time.time() - t0:.0f}s; backbone state hash unchanged: {before == after}")


# ------------------------------------------------------------------ merge
REASONS = ["DETECTOR_DISAGREEMENT", "LOW_LEAD_SUPPORT", "LOW_USABLE_LEADS", "CLIPPING_SUSPECT", "FLAT_LEAD", "NONFINITE_LEAD", "INSUFFICIENT_USABLE_LEADS",
           "INSUFFICIENT_BEATS", "INSUFFICIENT_BEATS_FOR_IRREGULARITY", "STRUCTURAL_SHAPE", "READ_ERROR"]
STATE = {"ACCEPTED": 2, "UNCERTAIN": 1, "UNAVAILABLE": 0}


def stage_merge():
    dest = os.path.join(CACHE, "merged.npz")
    assert not os.path.exists(dest)
    meta = json.load(open(os.path.join(CACHE, "cache_meta.json")))
    idx = read_index()
    ids = [r["record_id"] for r in idx]
    pos = {r: i for i, r in enumerate(ids)}
    assert len(pos) == len(ids)
    n = len(ids)
    conc = {}
    for p in sorted(glob.glob(os.path.join(CONC, "chunk_*.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            assert r["record_id"] in pos and r["record_id"] not in conc
            conc[r["record_id"]] = r
    assert len(conc) == n, f"concept cache incomplete: {len(conc)}/{n}"
    pooled = np.full((n, 512), np.nan, np.float32)
    logits = np.full((n, 45), np.nan, np.float32)
    ok = np.zeros(n, bool)
    seen = set()
    for p in sorted(glob.glob(os.path.join(EMB, "chunk_*.npz"))):
        z = np.load(p)
        for k, rid in enumerate(z["ids"]):
            i = pos[str(rid)]                                                  # join by record ID, never by row order
            assert str(rid) not in seen
            seen.add(str(rid))
            pooled[i], logits[i], ok[i] = z["pooled"][k], z["logits"][k], z["ok"][k]
    assert len(seen) == n, f"embedding cache incomplete: {len(seen)}/{n}"
    f = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else float("nan")
    arrs = {}
    for k in ("accepted_rate_mean_bpm", "accepted_rate_median_bpm", "accepted_rr_cv", "accepted_rr_nrmssd", "accepted_rr_mad_succ", "rate_mean_bpm_cons", "rate_median_bpm_cons",
              "rr_cv_cons", "rr_nrmssd_cons", "rr_mad_succ_cons", "n_beats_cons", "n_rr_cons", "match_max_abs_dt_ms", "match_median_abs_dt_ms", "n_matched_nonzero_dt", "n_matched",
              "n_usable_leads", "n_flat_leads", "n_clip_leads", "n_nonfinite_leads", "n_unmatched_A", "n_unmatched_B", "median_support_A", "median_support_B", "runtime_s"):
        arrs[k] = np.array([f(conc[i], k) for i in ids], np.float64)
    arrs["rate_state"] = np.array([STATE[conc[i]["rate_state"]] for i in ids], np.int8)
    arrs["irr_state"] = np.array([STATE[conc[i]["irr_state"]] for i in ids], np.int8)
    arrs["read_ok"] = np.array([conc[i]["read_ok"] == "True" for i in ids])
    arrs["reason_flags"] = np.array([[c in conc[i]["reason_codes"].split(";") for c in REASONS] for i in ids], np.uint8)
    lab = np.zeros((n, 45), np.uint8)
    for i, r in enumerate(idx):
        for c in r["labels"].split("|"):
            lab[i, CLASS_NAMES.index(c)] = 1
    role = np.array([r["role"] for r in idx])
    np.savez_compressed(dest, ids=np.array(ids), role=role, labels=lab, dup_group=np.array([r["dup_group_id"] for r in idx]), within_train_dup=np.array([r["within_train_dup_group"] == "True" for r in idx]),
                        pooled=pooled, orig_logits=logits, emb_ok=ok, reason_names=np.array(REASONS), class_names=np.array(CLASS_NAMES), **arrs)
    json.dump({**meta, "merged_sha256": sha_file(dest), "n": n, "n_embedding_ok": int(ok.sum()), "n_read_ok": int(arrs["read_ok"].sum()),
               "concept_chunks": len(glob.glob(os.path.join(CONC, "chunk_*.csv"))), "embedding_chunks": len(glob.glob(os.path.join(EMB, "chunk_*.npz"))),
               "merged": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(CACHE, "merged_meta.json"), "w"), indent=1)
    log(f"merged {n} records; embeddings ok {int(ok.sum())}; reads ok {int(arrs['read_ok'].sum())}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"index": stage_index, "extract": stage_extract, "embed": stage_embed, "merge": stage_merge}[sys.argv[1]]()
