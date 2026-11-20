# Phase 2 pilot baseline evaluation — 26 October 2026

**Status: initial single-seed baseline training and final evaluation verified.**

Final evaluation archive SHA256:
`e14e1e4f611cdb953a7d4132f37fd742c3c4c5bf6e5bc8b896f6c04dced5b65c`.
Evidence is preserved under `artifacts/imported/phase2_final_e14e1e4f/`.
Checkpoint identities, archive members, dataset hash and scores are frozen in
`data/manifests/baseline_registry_v1.json`.

## Measured results

All scores are per-image macro means, using validation-best weights and threshold
0.5. Source holdout contains 237 images; target query contains 95 images.

| Model | Source Dice | Source IoU | Target Dice | Target IoU | Target recall |
|---|---:|---:|---:|---:|---:|
| U-Net | 81.32% | 70.44% | 24.60% | 15.75% | 16.22% |
| Paper-inspired R2AU | 81.42% | 70.27% | 34.52% | 22.92% | 23.94% |

Both source results are close. The target-domain drop is substantial. Target
precision is 79.49% for U-Net and 86.29% for R2AU, while recall is low: both models
miss much of the annotated crack area at the frozen threshold. Predicted target
foreground is 0.566% and 0.885%, respectively, versus 3.952% annotated foreground.
These measurements demonstrate a generalization gap on this split; they do not
establish which camera, surface, annotation style or other factor caused it.

This provides a useful fixed reference for the planned lightweight architecture,
boundary supervision and few-shot adaptation. It does not establish that those
future changes improve the results.

## Verification completed

- Final ZIP integrity and dataset manifest SHA256 verified.
- Evaluated checkpoints match the stabilization archive's validation-best hashes:
  U-Net epoch 87 and R2AU epoch 63.
- Validation replay matches the prior recorded scores for both models.
- Every source/target output ID matches the frozen split, with no missing,
  extra or duplicate entries; aggregate metrics match the per-image means.
- Earlier stabilization evidence includes 19 passing tests and verified CPU/CUDA
  exact mid-epoch recovery for both trainers. Best/latest snapshots are retained.
- Final evaluation performed no training or threshold changes.

## Scope and limits

Phase 2's initial single-seed pilot is complete. Paper reproduction, three-seed
comparisons, confidence intervals, boundary metrics, MAC counts and full prediction
visual QA remain outside the verified evidence. The pilot source/target datasets
and modified architectures differ from the reference paper. Capture/site metadata
is incomplete. The target query was previously inspected in the failed v1 run,
so it remains an exploratory benchmark rather than pristine final confirmation.

Preserve these baselines. Future configurations must use new run directories;
do not tune them against target query labels. Phase 3 is the lightweight model,
followed by a controlled boundary-loss ablation and isolated support-set adaptation.
