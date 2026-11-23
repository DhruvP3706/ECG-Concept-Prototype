# Step 3 — LightECGNetV2 recovery and inference contract

Legend: **[V]** verified by code/file/experiment in this step · **[H]** hypothesis / inference · **[U]** unresolved.

## 1. Provenance (Phase A)

The ECG-Concept-Prototype repo contains no model code (git history: 3 commits, only README/requirements/checkpoints). The original work was found in sibling project folders on this machine:

| Source | What it contains | Role |
|---|---|---|
| `C:\Users\lenovo\HonsExAI\LightECGNet_v2_FIXED_faster.ipynb` (saved 2026-04-18 23:29) | Model, preprocessing, label vocabulary, split, training loops, threshold search, checkpoint saving, printed outputs | **Authoritative source** (it is the code that trained the checkpoints) |
| `C:\Users\lenovo\HonsExAI\class_names.npy` | 45 class names | Independent confirmation of class order |
| `C:\Users\lenovo\HonsExAI\lightv2_{ft_fold0,p1_fold1,p1_fold2}.pt` | Same bytes as the repo copies (SHA-256 equal) [V] | Provenance of 3 of 4 checkpoints |
| `C:\Users\lenovo\Desktop\shap-project\ecg_shap_explainability_v2.py` | A *later* copy of the model + inference used for SHAP | **Not reliable — see §3** |

[V] `lightv2_p1_fold0.pt` is not in HonsExAI (the notebook saves it to `models/`), so its link to the notebook is by filename/hash of structure only; it loads and behaves identically in kind to the others.
[U] No repository URL, license, or citation for the architecture. The notebook compares against an unnamed "paper" (P 0.394 / R 0.533 / F1 0.413 / AUROC 0.962 / Acc 0.956). Not searched: other notebooks in HonsExAI (`sem5*.ipynb`, `Untitled*.ipynb`, `untitled2.*`), `ECGenius/*.pdf`, `nirvana/ECGenius_Pipeline.pdf`.

## 2. Architecture (Phase B) — recovered

Implemented in `src/ecg_concept/lightecgnetv2.py` (copied from notebook cells 10–11).

- [V] Notebook printed `Parameters : 376,108`; the rebuilt model has exactly **376,108 trainable parameters**. The Step-2 figure 380,332 was a state-dict count that includes BatchNorm running statistics.
- [V] All four checkpoints load with `load_state_dict(strict=True)` (no missing/unexpected keys) via `torch.load(weights_only=True)`.
- [V] All four share one architecture (identical keys/shapes) and have different weights.
- **Shapes alone do not prove the architecture**, and indeed two variants (with and without the ReLUs in the TCN blocks) both load strictly. Discriminating test [V]: BatchNorm running statistics record what the real network saw in training. On 5 test-fold records (eval mode), mean |input mean − running mean| / running std over all 18 BN layers:

  | Variant | p1_fold0 | ft_fold0 | median var ratio (input/running) |
  |---|---|---|---|
  | Training-notebook (ReLU in DSConv, TCN) | **0.057** | **0.060** | ≈ 1.0 (0.87–1.31) |
  | SHAP-script (activations = Identity) | 8.2 | 8.0 | up to ~33,000× |

  The notebook architecture is the one the weights were trained with.
- Forward path [V from code]: `LeadAttention(12→6→12, sigmoid gate on time-mean)` → three depthwise-separable stems (k=3/7/15 → 16+16+32 ch, BN+ReLU) → 1×1 conv+BN+PReLU → AvgPool2 → `tcn1a/b` (64 ch, k13, dil 1,2) → AvgPool2 → `tcn2a/b` (64→128, k13, dil 4,8) → AvgPool2 → `tcn3a/b` (128→256, k9, dil 1,2) → [global max | global mean] (512) → Linear 512→256 → ReLU → Dropout .4 → Linear 256→45.
- Input `(B,12,5000)` float32 channels-first → output `(B,45)` logits [V]. Eval mode and CUDA placement work (GTX 1650; CUDA vs CPU probs differ ≤ 1e-6).

## 3. Warning: the SHAP script's model is wrong for these checkpoints

`shap-project/ecg_shap_explainability_v2.py` says its model is "unchanged" but defines `DSConv1d.act`, `DilatedTCNBlock.act` as `Identity` and removes the ReLUs after bn1/bn2 (see BN test above). It also loads with `weights_only=False`, uses `sosfiltfilt` in float (no int16 truncation), and has no thresholding (averaged sigmoid of 3 folds). Attributions/ablations produced by that script (and `paper.tex` if it relies on them) were computed on a network that is not the trained one. Not investigated further here.

## 4. Inference contract (Phase C)

| Item | Finding | Status |
|---|---|---|
| Lead order | I, II, III, aVR, aVL, aVF, V1–V6 (file order; no reordering in code) | [V] |
| Fs / length | 500 Hz, 5000 samples; shorter → zero-pad tail, longer → crop tail | [V] code + data (all 45,152 records are 5000) |
| Raw scaling | none; ADC counts (1000/mV) enter the filter, then z-score removes scale | [V] |
| Filter | Butterworth order 3, band 0.5–40 Hz, `scipy.signal.filtfilt` per lead | [V] code |
| int16 quirk | `signal[i] = filtfilt(...)` writes into the int16 array returned by `loadmat`, truncating filtered values toward zero before z-scoring | [V] code + dtype check. Default in `preprocessing.py`; float variant changes smoke-test probs by ≤ 0.014 |
| Normalisation | per-lead `(x − mean)/(std + 1e-8)` over the whole 10 s | [V] |
| No resampling | n/a (native 500 Hz) | [V] |
| Output transform | **sigmoid** (BCEWithLogits training, multi-label); never softmax | [V] code |
| Class order | sorted acronyms (below) | [V] re-derived from dataset, equals `class_names.npy`; also supported by smoke test (SB/SR/ST/SVT top-1 on indices 29/30/31/36) |
| Thresholds | **per-class**, rule `prob > thr` (strict), 0.5 for classes with no positives | [V] code |
| Threshold selection | grid `np.arange(0.1, 0.9, 0.05)`, per-class F1 maximum on the *validation fold of that checkpoint*, no TTA; saved at the best-macro-F1 epoch (falls back to all-0.5 if that scores higher) | [V] code; matches file values (0.1–0.85, 0.05 steps) |
| Training-time eval precision | bf16 autocast on CUDA; we use fp32 | [V] code; effect ≤ 1e-6 vs CPU fp32 not measured against bf16 |

**Class list (index 0–44):** 1AVB, 2AVB, 2AVB1, 3AVB, AF, AFIB, ALS, APB, AQW, ARS, AT, AVB, AVRT, CCR, CR, ERV, IVB, JEB, LFBBB, LVH, LVQRSAL, MISW, PRIE, PWC, QTIE, RAH, RBBB, RVH, SA, SB, SR, ST, STDD, STE, STTC, STTU, SVT, TWC, TWO, UW, VEB, VFW, VPB, VPE, WPW.

- [V] **The 45 outputs are not the 45 most frequent SNOMED codes.** Mapping = `ConditionNames_SNOMED-CT.csv` → dict keyed by SNOMED code (**last row wins** for shared codes: IDC→**IVB**, LBBB/LBBBB→**LFBBB**, MI/MIBW/MIFW/MILW→**MISW**, WAVN→SAAWR), then 6 acronyms excluded (ABI, VET, FQRS, SAAWR, JPT, VB), then sorted. Codes absent from the CSV (43 of the 94 data codes, incl. several very frequent ones) are ignored. Note AF = atrial *flutter*, AFIB = fibrillation. Reproduced: 44,476 records with ≥1 valid class (= notebook output); 1 labelled record dropped.
- [V] Thresholds are **not interchangeable** across checkpoints; each was fit to its own model's validation probabilities. Never apply one file to another checkpoint.
- Rule used in the smoke test: each checkpoint with its own threshold file; fold-2 gets no thresholded prediction.

## 5. Threshold provenance and caveats

- `ft_fold0_thresh`: [V] saved with `ft_fold0.pt` in the fine-tune loop (validation fold 0).
- `p1_fold0_thresh`: [V] saved with `p1_fold0.pt` in the Phase-1 loop (validation fold 0).
- `p1_fold1_thresh`: **[H] possibly stale.** It was written by the Phase-1 loop (cell 19, 120 epochs). Cell 21 later retrained fold 1 (80 epochs) and overwrote `p1_fold1.pt` *without* saving thresholds (also why fold 2 has none). Executed cell order in the notebook (exec counts 12 → 18) supports this. Needs checking before the thresholds are trusted for fold 1.
- `p1_fold2`: [V] no thresholds ever saved by the code. The ensemble in cell 22 refits one set of thresholds on the 3 folds' combined validation data but **does not save it** [U]; the printed values are visible only in the notebook output (partial).

## 6. Original split (needed to avoid leakage)

- [V] Code: `RandomState(42).permutation(N)` → `np.array_split(…, 10)`; `folds[9]` = held-out test (4,447 records, matches notebook output); folds 0–8 = CV. Checkpoint `p1_foldK` / `ft_fold0` validate on `folds[K]` / `folds[0]` and train on the other CV folds. All three ensemble members never train on `folds[9]`.
- [V] `src/ecg_concept/labels.py::build_index/reproduce_folds` reproduces N=44,476 and the fold sizes. **[H]** exact membership additionally assumes the same `os.walk` order as the original run (sizes match; membership not independently verifiable).
- Implication: for future work, treat `folds[9]` as the held-out test; folds 3–8 were *training* data for every checkpoint, and folds 0–2 are each training data for two of the three ensemble members. Note the notebook tuned threshold *strategies* (offset, blend, oracle) on `folds[9]`, so it is not pristine for threshold-dependent claims.

## 7. Smoke test (Phase D) — see `inference_smoke_test.json`

5 test-fold records (JS42769 ST, JS37210 SR, JS10192 SVT, JS33756 SB, JS11191 ST+TWC), all 4 checkpoints, fp32, CUDA, `eval()` + `inference_mode()`, batch 5.
- [V] Logits/probs `(5,45)`, finite, probs in (1e-14, 0.9996); CUDA vs CPU ≤ 1e-6.
- [V] Top-1 is the true label for 5/5 records on all 4 checkpoints. Thresholded predictions equal truth plus extras on some records (e.g. ST→{ST,STTC}; ST+TWC→{ST,LVQRSAL} for ft_fold0/p1_fold1).
- This is an engineering check on 5 records, not evidence of accuracy or reproduction of the original metrics (the notebook reports ensemble macro-F1 ≈ 0.39 on the test fold with CV thresholds).

## 8. Remaining unresolved / risks

1. [U] Source repository/citation of the architecture; `p1_fold0.pt` original copy.
2. [H] `p1_fold1_thresh.npy` may not belong to the current `p1_fold1.pt`; fold-2 thresholds do not exist (not created).
3. [H] Fold membership relies on identical directory-walk order.
4. Original F1 not reproduced (needs a full pass over ~4.4k records; deliberately outside the 5-record limit).
5. SHAP-script model mismatch (§3).
6. Dataset data-quality issues from Step 2 (2 corrupt headers, constant/saturated leads, duplicates) are not handled by the original pipeline; the training run loaded those records unchanged.

## 9. Files

Created: `src/ecg_concept/{__init__,preprocessing,lightecgnetv2,labels,inference}.py`, `scripts/step3_smoke_test.py`, `results/step3/*`. Modified: `requirements.txt` (+`scipy==1.18.1`, approved). `results/` is git-ignored. Nothing committed.
