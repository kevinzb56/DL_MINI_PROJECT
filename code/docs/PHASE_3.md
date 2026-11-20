# Phase 3 — Region-only lightweight model

**Status: initial region-only pilot training and final evaluation verified.**

Final source-holdout Dice is 76.36%; target-query Dice is 34.01%.
See `PHASE_3_FINAL_RESULTS.md` for evidence, baseline comparison and limitations.

See `PHASE_3_RESULTS.md` for measured results. LiteCrack meets the size target
but has a substantial validation-accuracy gap. The remaining text documents the
implementation protocol and its original pre-execution expectations.

LiteCrack uses torchvision MobileNetV3-Small features with width multiplier 0.5,
random initialization (`weights=None`), and no classifier. Encoder BatchNorm is
replaced with GroupNorm(1, channels) to match the stabilized normalization choice.
Skip features at indices 0/1/3/8/12 feed depthwise-separable decoder blocks with
128/64/32/16 output channels. Nearest resizing and a final separable refinement
block produce original-size single-channel logits. There is no boundary branch
yet: this is the required region-only control for the next phase's ablation.

The MobileNetV3 builder and no-pretraining contract follow the official API:
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
This is a modified segmentation architecture, not the stock classification model.

Use `output/kaggle/Road_Crack_Phase3_Lightweight.ipynb`. It embeds code, frozen
splits and the baseline registry. It restores pinned datasets, verifies hashes,
requires compatible preinstalled torch/torchvision and gates training on all
CPU/CUDA trainer recovery tests plus model shape, gradient and compactness tests.

Parameter gate: at most 1,055,014, half the measured stabilized R2AU baseline's
2,110,029 parameters. The exact LiteCrack count is not claimed until Kaggle runs.
Use the same seed 42, 256-square inputs, batch eight, BCE+Dice, Adam 0.0001,
gradient clipping one, 100-epoch maximum and 15-epoch validation patience as the
stabilized baselines. Input range stays 0–1; pretrained ImageNet normalization is
not introduced. Checkpoint and augmentation behavior remain unchanged.

Validation selects weights and diagnoses collapse. Source holdout and target
query are intentionally excluded from this first Phase 3 run. Later fixed-weight
evaluation and multi-seed evidence are needed to assess the accuracy tradeoff.
No boundary quality or few-shot improvement is established by this phase.

Profiling reports parameters, weight/checkpoint sizes, GPU/CPU p50/p95, GPU peak
allocated memory and Conv2d/Linear MACs. The MAC convention excludes normalization,
pooling, activation, interpolation and elementwise operations. Smaller models are
not assumed faster; compare measured runtimes on equivalent hardware. Full
end-to-end latency and confidence intervals remain later evaluation work.

New runs live in `road_crack_phase3_v1`; Phase 2 evidence remains untouched. Code
hashes include the model factory and compact encoder/decoder. Future architecture
changes require a new run and cannot resume old checkpoints. Local source changes
do not alter the original code snapshots preserved inside Phase 2 archives.

Download `phase3_results.zip` after execution. It retains latest/best checkpoints,
tests, configuration, frozen manifests, validation metrics, logs and cost reports.
All intermediate snapshots remain in the live session without automatic pruning.
Keep an external copy before ending Kaggle. Do not claim full snapshot history
was backed up merely because latest/best were exported.
