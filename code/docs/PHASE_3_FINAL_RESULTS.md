# Phase 3 pilot — final evaluation verified

Evaluation archive `phase3_final_evaluation.zip`, SHA256:
`5906263ae6885ebd412e3862297c31d1306af71aa4c794eb615885d7d6aa57e9`.
Evidence is preserved in `artifacts/imported/phase3_final_5906263a/`.
Fixed region-only control is registered in `data/manifests/litecrack_registry_v1.json`.

## Same frozen evaluation splits

| Model | Parameters | Source holdout Dice (237) | Target query Dice (95) |
|---|---:|---:|---:|
| U-Net | 1,965,569 | 81.32% | 24.60% |
| Paper-inspired R2AU | 2,110,029 | 81.42% | 34.52% |
| LiteCrack, region only | 339,265 | 76.36% | 34.01% |

LiteCrack source IoU is 63.70%; target IoU is 22.65%. On the target split it
has 84.71% precision and 24.53% recall. It remains conservative and misses many
crack pixels. The target Dice gap versus R2AU is 0.51 percentage point, with
83.92% fewer parameters; source holdout Dice is lower by 5.06 points. This supports
a narrow size/target-accuracy tradeoff in this pilot, not overall superiority or
statistical equivalence. The models use one seed, and query data was inspected in
prior phases. Independent confirmation and uncertainty intervals remain needed.

## Verification and visual inspection

- ZIP integrity and manifest SHA256 passed.
- Checkpoint identity and epoch 99 match the verified Phase 3 best snapshot.
- Validation replay matches the previous metrics.
- Both per-image files contain exactly the frozen split IDs, without omissions,
  extras or duplicates; aggregate Dice/IoU/precision/recall match their means.
- Six seeded source and six seeded target previews were visually inspected.
  Source examples generally follow major cracks, with missed fine branches and
  some fragmentation. Target examples show strong under-segmentation and broken
  continuity. This is sampled review, not exhaustive validation of every mask.
- No retraining, adaptation, boundary supervision or threshold changes occurred.
- Training recovery and model tests were previously verified: all 24 passed.

Phase 3's initial region-only pilot is complete. Preserve this checkpoint as the
control for Phase 4. Add boundary supervision as a separate run, keeping source
data, initialization policy, seed, region objective and training budget controlled.
Evaluate both region and boundary metrics before claiming a boundary benefit;
boundary learning alone does not guarantee crack connectivity or domain adaptation.
