# Phase 3 region-only LiteCrack — verified 26 October 2026

Archive `phase3_results.zip`, SHA256:
`d1d9c4fa7c42efb64e2f30f775fbbf8f5973f22676f591ba1537523d63119470`.
Evidence retained under `artifacts/imported/phase3_d1d9c4fa/`.

## Verified evidence

- ZIP integrity and best/latest snapshot hashes passed.
- Frozen pilot manifest matches exactly; all 60 validation IDs are present once.
- Per-image averages match reported metrics. No source holdout or target query
  evaluation occurs in this archive.
- All 24 tests passed without skips, including CPU/CUDA actual trainer recovery
  for all three architectures, compact-model shapes/gradients/parameter budget,
  and the grouped-convolution MAC convention.
- Training completed 100 epochs / 3,000 steps; validation-best epoch was 99.
- Latest and best checkpoints are preserved locally in the imported archive.

## Measured validation and cost

| Measurement | LiteCrack |
|---|---:|
| Validation Dice | 73.72% |
| Validation IoU | 62.24% |
| Validation precision | 71.32% |
| Validation recall | 78.20% |
| Parameters | 339,265 |
| Reduction versus stabilized R2AU | 83.92% |
| Conv2d/Linear MACs at 256-square, batch one | 148,741,376 |
| FP32 weight bytes | 1,357,060 |
| T4 model-pass p95 | 9.78 ms |
| Kaggle CPU model-pass p95, four threads | 45.87 ms |

The compactness gate passes. Validation Dice is 9.97 percentage points below
stabilized R2AU (83.70%) and 10.36 below U-Net (84.09%). The compact model therefore
does not yet preserve baseline validation accuracy. The formal holdout/query
accuracy tradeoff remains unmeasured for LiteCrack. Do not claim superiority or
that boundary supervision/few-shot adaptation has already recovered the gap.

GPU p95 is lower than the separately recorded R2AU result (18.15 ms), but higher
than U-Net (6.92 ms). CPU p95 is lower than both earlier records (184.16 and
90.85 ms). These are different Kaggle runs on nominally comparable hardware;
session load and generic CPU identification limit strict timing comparisons.
Conv2d/Linear MACs omit normalization, nonlinearities, pooling, resize and
elementwise operations. Small parameter count alone does not prove lower latency.

Foreground coverage was 2.988% versus labeled 2.846%; no persistent trivial-mask
collapse is evident from these aggregate metrics. Individual prediction masks
were not exported, so visual prediction quality remains unreviewed.

## Next

Preserve this region-only control and its fixed weights. Evaluate the validation-
best checkpoint on frozen source holdout and target query without threshold or
training changes, then run the boundary-aware ablation as a separate experiment.
The prepared notebook is `output/kaggle/Road_Crack_Phase3_Final_Evaluation.ipynb`;
it exports `phase3_final_evaluation.zip` with metrics, per-image rows and randomly
selected prediction previews. It replays validation before evaluating test data
and can restore the verified Phase 3 archive if the session has reset.
Any later accuracy claim must distinguish validation from evaluation and retain
the single-seed, dataset-grouping and previously inspected target-query caveats.
