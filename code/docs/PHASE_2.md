# Phase 2 — Baseline implementation

Implemented U-Net and a **paper-inspired** recurrent residual attention U-Net.
Both use base width 16 for an initial compute-conscious pilot. The recurrent
baseline uses two recurrent units per block with shared-weight refinement, plus
attention on decoder skips. It is not claimed to reproduce unspecified paper
architecture details or the paper's five-source dataset protocol.
Both baselines use nearest-neighbor decoder resizing followed by convolution,
avoiding dependence on CUDA bilinear-backward determinism for exact recovery.

Configuration: random initialization, seed 42, FP32, Dice loss, Adam at 0.001,
batch eight, 256-square inputs, up to 100 epochs, early stopping after ten epochs
without validation Dice improvement. A validation-loss scheduler reduces learning
rate after five non-improving epochs. Horizontal/vertical flips are paired with
the masks. These choices are frozen for the initial pilot, not tuned on query data.

Source training is the only update split. Source validation selects the best
checkpoint. Source holdout and target query are evaluated afterward using that
checkpoint and a fixed 0.5 threshold. Metrics are macro per-image Dice, IoU,
precision, and recall with per-image outputs. Boundary F1, confidence intervals,
three-seed comparisons, and MAC profiling remain subsequent evaluation work.

Run `output/kaggle/Road_Crack_Phase2_Baselines.ipynb` on the existing Kaggle session.
It embeds the source and frozen manifest, verifies data hashes, executes actual
trainer CPU/CUDA recovery tests for both architectures, then trains and profiles.
Setup now searches for complete existing dataset copies and otherwise restores
the pinned source commits and safely unpacks DeepCrack. Enable Kaggle Internet
for restoration. Every image and mask must match the frozen manifest checksums;
restoration never regenerates or changes the splits. A changed or incomplete
existing repository is preserved and reported instead of silently overwritten.
Only one T4 is used; avoiding multi-GPU introduces fewer recovery-state variables.

The CLI is independently available:

```text
python scripts/experiments/train_baseline.py --config configs/baseline_unet.json --manifest data/manifests/pilot_v1.json --data-root DATA_ROOT --run-dir artifacts/phase2/unet
```

Add `--resume` to continue the same run. `--stop-after-steps N` requests a safe
stop at an absolute optimizer step for diagnostics. Different configurations or
training code require a new run. Dataloader workers and AMP are deliberately not
used initially; batching and sampling state are explicit for deterministic replay.

Every epoch and 30 optimizer steps saves an immutable full snapshot; best/latest
JSON pointers are atomic and checksummed. History, early-stop count, sampler order,
cursor, partial epoch loss and RNG states are included. Signals request a safe
boundary stop. Forced termination can roll back to the latest complete snapshot.
Disk-space checks fail without silently deleting earlier checkpoints.

The final ZIP includes latest and best snapshots, configuration, frozen manifest,
tests, per-image metrics and profiling. Download it outside the live session.
Historical intermediate snapshots remain on Kaggle and need separate export if
their long-term preservation is required. Local working storage alone is not a
persistent backup.

Current evidence: Python syntax checks and notebook structural checks only until
Kaggle execution. Local dependencies remain unavailable. No baseline training,
CUDA recovery, accuracy or timing result is claimed by implementing this phase.
