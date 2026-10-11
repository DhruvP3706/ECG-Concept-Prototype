"""Output-class vocabulary and split, recovered from the original training notebook
(`ECGDataset` + cell 8 of HonsExAI/LightECGNet_v2_FIXED_faster.ipynb).

Class order = sorted acronyms. Acronyms come from ConditionNames_SNOMED-CT.csv
via a dict keyed on SNOMED code, so when several acronyms share a code the LAST
CSV row wins (IVB over IDC, LFBBB over LBBB/LBBBB, MISW over MI/MIBW/MIFW/MILW,
SAAWR over WAVN -- and SAAWR is then excluded). Excluded acronyms:
ABI, VET, FQRS, SAAWR, JPT, VB.
"""
import csv
import os
from collections import Counter

import numpy as np

EXCLUDE = {"ABI", "VET", "FQRS", "SAAWR", "JPT", "VB"}

# Verified: equals both re-derivation from the dataset (derive_classes) and
# HonsExAI/class_names.npy.
CLASS_NAMES = [
    "1AVB", "2AVB", "2AVB1", "3AVB", "AF", "AFIB", "ALS", "APB", "AQW", "ARS", "AT", "AVB", "AVRT",
    "CCR", "CR", "ERV", "IVB", "JEB", "LFBBB", "LVH", "LVQRSAL", "MISW", "PRIE", "PWC", "QTIE", "RAH",
    "RBBB", "RVH", "SA", "SB", "SR", "ST", "STDD", "STE", "STTC", "STTU", "SVT", "TWC", "TWO", "UW",
    "VEB", "VFW", "VPB", "VPE", "WPW",
]


def snomed_to_acronym(dataset_root):
    m = {}
    with open(os.path.join(dataset_root, "ConditionNames_SNOMED-CT.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            m[row["Snomed_CT"].strip()] = row["Acronym Name"].strip()
    return m


def _read_dx(hea_path):
    with open(hea_path, "r") as f:
        for line in f:
            if "#Dx" in line.replace(" ", ""):
                return line.split(":")[1].strip().split(",")
    return []


def build_index(dataset_root):
    """Replicates ECGDataset's record selection. Returns (classes, record_paths_without_ext, label_index_lists).

    Record order = os.walk order with sorted filenames per directory, as in the original.
    """
    m = snomed_to_acronym(dataset_root)
    recs, labs = [], []
    for root, _, files in os.walk(os.path.join(dataset_root, "WFDBRecords")):
        for fname in sorted(files):
            if fname.endswith(".hea"):
                base = os.path.join(root, fname[:-4])
                names = [m[c] for c in _read_dx(base + ".hea") if c in m]
                if names:
                    recs.append(base)
                    labs.append(names)
    counts = Counter(n for sub in labs for n in sub)
    classes = sorted(c for c in counts if c not in EXCLUDE)
    idx = {c: i for i, c in enumerate(classes)}
    kept = [(r, [idx[n] for n in l if n in idx]) for r, l in zip(recs, labs)]
    kept = [(r, l) for r, l in kept if l]
    return classes, [r for r, _ in kept], [l for _, l in kept]


def reproduce_folds(n_records, seed=42):
    """10 index folds: folds[0..8] = CV validation folds, folds[9] = held-out test (original cell 8)."""
    ind = np.random.RandomState(seed).permutation(n_records)
    return np.array_split(ind, 10)
