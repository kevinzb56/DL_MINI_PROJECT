# Phase 4 fixed-checkpoint comparison

Archive SHA256:
`c145f89d6b7174ed409f6864b5e6d3d5744fc89a7872e5ee4dd8fce369fe91fe`.
Preserved under `artifacts/imported/phase4_comparison_c145f89d/`.

ZIP integrity, frozen manifest, both checkpoint identities, all split IDs,
aggregate means and every paired per-image difference verified. The control
uses epoch 99; the boundary model uses epoch 86. Threshold remains 0.5.

| Split | Region Dice | Boundary-aware Dice | Region boundary F1 | Boundary-aware boundary F1 |
|---|---:|---:|---:|---:|
| Validation (60) | 73.72% | 73.17% | 85.80% | 86.73% |
| Source holdout (237) | 76.36% | 75.40% | 86.54% | 86.01% |
| Target query (95) | 34.01% | 41.72% | 56.91% | 63.14% |

On target query, boundary supervision improves Dice by 7.71 percentage points
and boundary F1 by 6.22 points. Target recall rises from 24.53% to 32.04%, while
precision falls from 84.71% to 82.50%. The model still misses many crack pixels.
The proposed +2-point boundary-F1 / at-most-one-point Dice-loss gate passes on
target point estimates only. Validation boundary gain is +0.93 point; source
holdout boundary F1 decreases by 0.54 point and Dice by 0.97 point. It is therefore
incorrect to claim universal boundary improvement or overall superiority.

This is a positive target-specific single-seed pilot, with the previously viewed
query caveat and incomplete capture/site independence. Confidence intervals and
repeated-seed confirmation are still outstanding; no significance is established.
Keep both checkpoints and the negative source-domain result in final reporting.

Phase 4's first ablation is complete. Phase 5 is few-shot adaptation using only
the frozen target acquisition pool and nested 1/5/10-image support draws. Use
source validation or predetermined adaptation settings, never query labels, to
select settings. Preserve static scores as K=0 references, and evaluate both
region-only and boundary-aware models to distinguish adaptation from architecture.
