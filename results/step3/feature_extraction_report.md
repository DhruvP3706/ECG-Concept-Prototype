# Step 3 — Feature extraction interface (Phase E)

Verified on `lightv2_p1_fold0`, 5 test-fold records, eval mode, fp32 CUDA (data in `inference_smoke_test.json → feature_extraction`). Interface: `LightECGNetV2.forward_features(x)` in `src/ecg_concept/lightecgnetv2.py`.

## Candidate locations

| Name | Where | Shape | Per-time-step? | Notes |
|---|---|---|---|---|
| `feature_map` | output of `tcn3b` | `(B, 256, 625)` | **Yes** — 1 step = 8 input samples = 16 ms | Post-ReLU, range 0–11.7 on the 5 records. Hook on `tcn3b` is bit-identical to `forward_features` |
| `pooled` | `cat[max_t(fmap), mean_t(fmap)]` | `(B, 512)` | No (pooled over time) | `[0:256]` = global max, `[256:512]` = global mean. Bit-identical to the recomputation from `feature_map` |
| `hidden` | `head[0]` Linear then `head[1]` ReLU | `(B, 256)` | No | Post-ReLU; 77% exact zeros on these 5 records (sparse). This is the "256-d hidden representation" from the Step-2 audit; it is *post-ReLU*, not the raw `head[0]` output |
| `logits` | `head[3]` | `(B, 45)` | No | Sigmoid → probabilities |

## Verified

- [V] Extraction does not change predictions: `forward`, `forward_features['logits']`, and a model with forward hooks attached give bit-identical logits (`torch.equal`). Dropout is inactive in `eval()`.
- [V] Step-2 candidates confirmed: `tcn3b` has 256 channels; `head.0` is Linear(512→256) with 256 hidden units.
- [V] Not local: the gradient of one centre time step of `feature_map` is non-zero over **all 5000 input samples** (and all 12 leads). The convolutions alone would not cover that, but `LeadAttention` and every SE block gate with a **whole-signal time-mean**, so each feature step depends on the entire 10 s record. "Time step t" therefore means "features read out at position t", not "features of the beat at t".
- Time axis: 625 steps ↔ input sample 8·t … 8·t+7 (three AvgPool2 stages; the convolutions are length-preserving with symmetric padding).

## Suitability for the future concept / prototype layers (design judgment — [H])

- **`feature_map` (256×625)** is the only representation that keeps temporal localisation, so it is the natural input for concepts that live at specific waves/intervals and for localised prototype parts. Caveats: non-local dependence (above); border/padding effects at the edges; large (160k values per record).
- **`pooled` (512) / `hidden` (256)** are fixed-size, class-level vectors, cheap to store and well suited to record-level concepts (rhythm class, global morphology) and to case-based prototypes. `pooled` is the direct input of the pretrained classifier head, so a concept layer on `pooled` preserves a clean route back to the original 45 logits. `hidden` is sparse after ReLU, which makes distances less informative.
- Suggested starting point for Step 4: take `pooled` (512) as the primary record-level embedding, keep `feature_map` as an optional localisation channel, and keep the frozen head so original logits remain reproducible. Not tested for concept predictability — that would be new modelling and was out of scope.
- Layers before `tcn3b` (e.g. `tcn2b`, 128×625 after pooling) are further candidates but were not examined.

## Not done / limits

- Statistics are from 5 records of one checkpoint; the same code path loads the other three but taps were only compared on `p1_fold0`.
- No concept layer or prototype network built.
