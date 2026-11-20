# Phase 4 validation evidence

Archive SHA256: `0851dc03b170e41df2639432a2373579e594a7398f8e0c78183eb5f2a86d652c`.
Evidence retained under `artifacts/imported/phase4_0851dc03/`.

All 28 tests passed without skips, including CPU/CUDA recovery for the boundary
model. ZIP integrity, frozen manifest, best/latest checkpoint hashes, all 60
validation IDs and per-image metric means were verified locally. Training ran
100 epochs; the region-Dice-selected best snapshot is epoch 86.

| Validation metric | Region-only control | Boundary-aware model |
|---|---:|---:|
| Dice | 73.72% | 73.17% |
| IoU | 62.24% | 61.44% |
| Boundary F1 | Not yet measured | 86.73% |
| Parameters | 339,265 | 339,282 |

The boundary branch adds 17 parameters. Region Dice decreased by 0.56 percentage
point, within the proposed one-point tolerance, but the required boundary-F1 gain
cannot be assessed until the frozen region-only control is measured using the
same boundary extraction/matching implementation. An absolute 86.73% score is
not evidence of improvement over the control.

Next: evaluate both fixed checkpoints with the same evaluator on validation and
the frozen holdout/query splits, preserving per-image differences. No retraining
or threshold adjustment is needed. Single-seed and previously inspected query
limitations remain; no boundary benefit or few-shot improvement is claimed yet.
