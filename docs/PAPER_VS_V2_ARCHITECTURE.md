# Paper (docs/ECG_paper.pdf) versus the repository LightECGNetV2 — second audit (figures inspected)

Source: `docs/ECG_paper.pdf`, SHA-256 `3d8937a625dc8658…ab7528c`. **It is byte-identical to `Downloads\ECG_paper.pdf`, the version used in the first audit** (same SHA-256), so the text conclusions of the first audit stand. The older Downloads PDF with the same title (accuracy 0.974 / AUROC 0.962) is a different version and is not used.
How the figures were inspected: the PDF has no PDF tooling in the environment; the figure bitmaps (Flate-compressed RGB + soft-mask objects 6, 12, 13, 18, 19) were decoded with a small stdlib script, composited on white and viewed. They are raster images, so no additional text/vector information exists beyond what is drawn.

Status codes: **P-text** confirmed by the paper text/tables · **P-fig** confirmed by a figure · **C** confirmed by repository code (V2 / notebook as documented in the handover) · **I** inferred, not confirmed · **U** unknown.

## What Figures 2–5 actually show
- **Fig. 1** system overview (digitisation → LightECGNet → explainability → outputs); not an architecture figure.
- **Fig. 2** (overall network): input (12 leads) feeds (a) a *Lead Attention Block* on a top branch and (b) the *Multi-Scale DS Stem* (k = {3,7,15}, "DS Conv 64 ch"); then AvgPool/2 → TCN stage 1 (Dilated TCN block ×2, d={1,2}, 64 ch, SE) → AvgPool/2 → stage 2 (d={4,8}, 128 ch, SE) → AvgPool/2 → stage 3 (d={1,2}, 256 ch, SE) → **an element-wise product (⊗) with the lead-attention output drawn AFTER stage 3** → GAP (green) and GMP (yellow) → **an element-wise sum (⊕)** → "FC Head 512 256 ReLU Dropout(0.5) 45 classes" → outputs (1AVB, AF, RBBB, TWC … WPW).
- **Fig. 3** (lead attention): Global AvgPool over time (B,12,1) → FC1 + ReLU (B,6,1) → FC2 + **ReLU** (B,12,1) → Sigmoid (B,12,1); "per-sample lead attention weights, values in [0,1]"; output labelled (B,12,1).
- **Fig. 4** (stem): three branches depthwise Conv1d (k=3/7/15) → pointwise Conv1d (B,16/16/32,5000) → BN + ReLU; concat along channels (B,64,5000); "Pointwise Conv1d + PReLU" (B,64,5000) (no BN drawn); AvgPool1d → (B,64,2500).
- **Fig. 5** (TCN block): Dilated Conv1d → BN + ReLU → Dilated Conv1d → BN + ReLU → SE → Dropout → ⊕ residual connection from the block input → output. SE: Global AvgPool → FC1 ReLU → FC2 Sigmoid → Scale.

## Figure/text contradictions and ambiguities (do not resolve by guessing)
1. **Lead-attention position.** The text says the sigmoid weights are applied "directly to the original ECG signal" before the stem (V2 does this). Fig. 2 draws the product after TCN stage 3, where the tensor is (B,256,625) and cannot be multiplied by a (B,12,1) weight. The figure is therefore a schematic with an impossible tensor shape; the text and V2 agree on input-level gating. Unresolved [U].
2. **GAP/GMP fusion.** Text: concatenate to 512-d (V2 does this); Fig. 2 draws ⊕ "element-wise sum" but also labels the head "512 256". A sum would give 256-d. Text + head dimension favour concatenation [I].
3. **FC2 ReLU before sigmoid (Fig. 3).** A ReLU before a sigmoid would restrict weights to [0.5,1). V2 has none. Likely a drawing artefact but not confirmed [U].
4. **Two dilated convolutions per TCN block (Fig. 5)** versus V2's dilated depthwise conv + non-dilated 1×1 pointwise. The figure names both convolutions "Dilated Conv1d (Temporal Convolution)"; it does not say depthwise/standard/kernel size. [U]
5. **Stem 1×1 + PReLU without BN (Fig. 4)** versus V2's BN between them [U].
6. **Post-residual ReLU:** Fig. 5 shows the sum as the block output (no ReLU after ⊕); V2 applies ReLU after the add [U].

## Component comparison
| Component | Paper | Repository V2 | Status |
|---|---|---|---|
| Input (B,12,5000), 500 Hz, 0.5–40 Hz Butterworth(3) zero-phase, per-lead z-score, pad/crop to 5000 | text | `preprocessing.py` (plus int16-truncation quirk not mentioned in the paper) | P-text, C; quirk C only |
| Lead attention: GAP → FC → bottleneck 6 → sigmoid, applied to the input | text; Fig. 3 (pool, FC 6, FC 12, sigmoid) | same; FC biases present (162 params) | P-text, P-fig (structure), C |
| Lead-attention parameter count 144 | text | 162 with biases | P-text vs C: **differs**; 144 = weights only [I] |
| Lead-attention applied after stage 3 | Fig. 2 only | no | contradicted by text [U] |
| Stem: 3 DS branches k=3,7,15 with 16/16/32 ch, concat 64, 1×1 conv + PReLU, AvgPool 2 → (B,64,2500) | text; Fig. 4 | same (+BN in branches and in the 1×1 projection) | P-text, P-fig, C; BN in projection [U] |
| Stem branch internals: depthwise → pointwise → BN → ReLU | Fig. 4 | same | P-fig, C |
| TCN: 3 stages, 64/128/256 ch, dilations (1,2),(4,8),(1,2), AvgPool between stages: 2500→1250→625 | text; Fig. 2 | same | P-text, P-fig, C |
| TCN block layers | "two dilated Conv1d, BN+ReLU, SE, dropout, residual"; Fig. 5 | dilated depthwise (k=13,13,13,13,9,9) + BN + ReLU + 1×1 pointwise + BN + ReLU, SE, dropout, residual | layer order P-fig+C; second layer dilated: P-fig only (**V2 differs**) |
| TCN kernel sizes | not given | 13,13,13,13,9,9 | C only; paper U |
| Block dropout rates | not given | 0.1/0.1/0.2/0.2/0.3/0.3 | C only; paper U |
| Skip projection when channels change | not given | 1×1 conv + BN | C only; paper U |
| Post-add ReLU | not shown | ReLU | C only; Fig. 5 differs [U] |
| SE: GAP → FC → ReLU → FC → sigmoid → scale, reduction 8 | text r=8; Fig. 5 | same (hidden ≥ 4) | P-text, P-fig, C |
| Global pooling | GAP+GMP, concat 512 (text) / ⊕ (Fig. 2) | concat [max\|mean] | P-text, C; Fig. 2 conflicts [U] |
| Head | 512 → 256, ReLU, dropout → 45; dropout 0.5 | 512 → 256, ReLU, **dropout 0.4**, → 45 | P-text, P-fig (0.5), C: **head dropout differs** |
| Total parameters | not stated; "≈10× fewer than reference", "≈32× fewer than ~22 M", "≈2.7 MB" ⇒ ≈0.68 M | 376,108 (1.5 MB fp32) | **unreconcilable**: do not invent details to match [U] |
| Loss | BCEWithLogits, ω_c = N_neg/N_pos capped at 50 | same (notebook clip 1–50) | P-text, C |
| Optimizer / LR | AdamW, lr 1e-3, wd 1e-4 | same | P-text (Table III), C |
| Schedule | 5-epoch linear warm-up, cosine to 1e-5, ≤120 epochs | same (120 for fold 0; 80 for later folds) | P-text, C |
| Batch size / precision | 256 / BFloat16 | 256 / bf16 on RTX 4060 | P-text, C |
| Early stopping | patience 15, **macro-AUROC every 5 epochs** | notebook: patience 15 (fold 0) / 10, best epoch by **validation macro-F1** | P-text vs C: **differs** |
| Augmentation | noise σ=0.02, shift ±250, scale 0.8–1.2, lead dropout (p=0.3; others 0.5), mixup α=0.2 | same list in the notebook | P-text, C (details I) |
| Gradient clipping | 1.0 | 1.0 | P-text, C |
| Rare-class fine-tuning | classes F1<0.4 ×3, <0.6 ×1.5 pos weight, WeightedRandomSampler 1/√freq, lr 2e-4, 20–30 epochs, no mixup | `ft_fold0` only (lr 2e-4, 30 epochs, boosted weak-class weights, sampler, no mixup) | P-text, C for fold 0; **no fine-tuned fold 1/2** |
| CV / ensemble | "stratified 9-fold CV", 3-fold ensemble, pooled 13,344 validation samples for thresholds | random `RandomState(42)` permutation, 10 index folds, **not stratified**; p1_fold0/1/2 | P-text vs C: **differs**; 13,344 = folds 0–2 [I] |
| Thresholds | CV-pooled F1-optimal; global offset −0.05 used for headline | per-class grid thresholds on each model's own validation fold | P-text vs C: differs |
| Test-time augmentation | 5 passes (type not specified) | none | P-text; implementation U |
| Headline metrics | macro-AUROC 0.963, accuracy 0.977 (macro-averaged per-class binary accuracy), recall 0.551 | measured on historically exposed fold 9: ft_fold0 0.9628, ensemble 0.9626 | P-text, C (Step 3.5) |

## Can the paper's architecture be reconstructed reliably?
**Partially.** Confirmed and implemented identically in V2: input shape, three-branch DS stem (k 3/7/15; 16/16/32 channels; 64 → 1×1 + PReLU → AvgPool 2), three stages with 64/128/256 channels, dilations, SE (r=8), pooling positions, GAP+GMP→512→256→45 head, training-recipe values (loss, optimizer, schedule, augmentation list).
**Not reconstructable from the paper:** convolution type and kernel size inside the TCN blocks, block dropout rates, skip form, BN in the stem projection, post-add activation, whether lead attention gates the input (text) or the final features (Fig. 2), concat vs sum fusion, and the ≈0.68 M parameter count. V2 remains the best-supported recovered implementation (it reproduces the reported macro-AUROC on the historically exposed test fold) and is **not** claimed to be the paper's exact network.
Consequence for the plan: new work uses `ArchConfig` (src/ecg_concept/lightecgnet_paper.py): the default is V2-equivalent (verified bit-identical); the figure-literal alternatives (`block_variant='two_ds'|'two_conv'`, `post_add_relu=False`, `stem_proj_bn=False`, `la_bias=False`, `head_dropout=0.5`, `pool_fusion='sum'`) are separate, clearly labelled ablations, never described as "the paper's model". No attempt is made to match 0.68 M parameters.

## Errata to the first audit
- The first audit stated "bf16 55 records/s on this GPU". That figure was measured at batch 256 where the process exceeded the 4 GB card and spilled to shared system memory. Properly measured (`results/phase1_infrastructure/hw_benchmark.json`): bf16 ≈ 160 records/s at batch ≤ 192, FP16 ≈ 205–210, **FP32 ≈ 318 at batch 32–64 (fastest)**.
- The two-DS-layer variant count quoted in the first audit (558,124) is reproduced by the implemented `two_ds` variant. Measured counts of the implemented figure-literal alternatives: V2/default 376,108; `two_ds` 558,124; `two_conv` with kernel 3 in every block 1,147,116; `two_conv` with V2's kernels 3,260,652. **None equals the paper-implied ≈0.68 M**, which is why no match is attempted.
