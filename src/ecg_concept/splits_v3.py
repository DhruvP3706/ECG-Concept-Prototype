"""Inner (development) splits for Phase 1. Metadata only; never opens waveforms.

PTB-XL: inner fold k groups the official strat_folds {1,2},{3,4},{5,6},{7,8}; the official folds are patient-wise, and `ptbxl_inner_folds` verifies patient disjointness between inner folds (raises otherwise).
Fold 9 and fold 10 are never assigned an inner fold: fold 9 = exposed reference (after a finalist is frozen), fold 10 = locked.
PhysioNet: records that are byte-identical duplicates (Step 3.6 `dup_group_id`) form one group; groups are assigned to inner folds greedily, balancing fold sizes and per-class positives.
"""
import numpy as np

PTBXL_INNER_GROUPS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3}


def ptbxl_inner_folds(rows):
    """rows: dicts with ecg_id, patient_id, strat_fold. Returns {ecg_id: inner_fold} for strat_fold 1..8 only."""
    out, pat = {}, {}
    for r in rows:
        f = int(r["strat_fold"])
        if f not in PTBXL_INNER_GROUPS:
            continue
        k = PTBXL_INNER_GROUPS[f]
        out[r["ecg_id"]] = k
        prev = pat.setdefault(r["patient_id"], k)
        if prev != k:
            raise ValueError(f"patient {r['patient_id']} appears in inner folds {prev} and {k}")
    return out


def duplicate_groups(rows):
    """rows: dicts with record_id and dup_group_id ('' for singletons). Returns {record_id: group id} (singletons get their own id)."""
    return {r["record_id"]: (r.get("dup_group_id") or f"SINGLE-{r['record_id']}") for r in rows}


def group_kfold(ids, groups, k, seed, labels=None, n_classes=None):
    """Assign whole groups to k folds. labels: list of class-index lists per id (optional) for greedy class balancing. Returns {id: fold}."""
    ids = list(ids)
    g2i = {}
    for i, g in zip(ids, groups):
        g2i.setdefault(g, []).append(i)
    pos = {i: l for i, l in zip(ids, labels)} if labels is not None else {}
    nc = n_classes or (1 + max((c for l in pos.values() for c in l), default=0))
    tot = np.zeros(nc)
    for l in pos.values():
        for c in l:
            tot[c] += 1
    rng = np.random.RandomState(seed)
    gl = sorted(g2i)
    rng.shuffle(gl)
    # rarest-first: groups containing the rarest classes are placed first
    def rarity(g):
        cs = [c for i in g2i[g] for c in pos.get(i, [])]
        return -max((1.0 / max(tot[c], 1) for c in cs), default=0.0)
    gl.sort(key=lambda g: (rarity(g), -len(g2i[g])))
    size = np.zeros(k)
    cls = np.zeros((k, nc))
    target = tot / k
    n_total = len(ids)
    fold_of = {}
    for g in gl:
        gc = np.zeros(nc)
        for i in g2i[g]:
            for c in pos.get(i, []):
                gc[c] += 1
        best, bs = 0, None
        for f in range(k):
            after = cls[f] + gc
            over = np.maximum(after - target, 0) / np.maximum(target, 1)
            score = over.sum() + 2.0 * ((size[f] + len(g2i[g])) / (n_total / k))
            if bs is None or score < bs - 1e-12:
                best, bs = f, score
        size[best] += len(g2i[g])
        cls[best] += gc
        for i in g2i[g]:
            fold_of[i] = best
    return fold_of
