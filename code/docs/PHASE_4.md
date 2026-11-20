# Phase 4 boundary-supervision ablation

Implementation prepared; remote execution and scientific benefit remain unverified.
Run `output/kaggle/Road_Crack_Phase4_Boundary.ipynb`, then download
`phase4_results.zip` into the project root.

Uses the same half-width MobileNetV3 encoder, separable decoder, seed, source
split, normalization, optimizer, learning rate, augmentation, threshold and
training budget as the Phase 3 region-only control. A 1x1 boundary head shares
the final decoder features. Its initialization preserves the region model's
random initialization and the subsequent sampling RNG sequence.

Boundary targets use a 3x3 morphological gradient after resizing/augmentation.
Replicated border padding prevents artificial image-border boundaries. Training
adds 0.2 times boundary BCE with fixed foreground weight five. This fixed pilot
choice is declared before evaluation and is not tuned on target-query labels.
Validation checkpoint selection remains region Dice, matching the control.

Boundary F1 is computed from the region prediction, not the auxiliary head. It
uses symmetric many-to-one boundary matching within Chebyshev radius two pixels
at 256-square resolution. Both-empty boundary sets score one; one-empty scores
zero. Report this definition alongside results; it is not one-to-one matching.

The notebook gates training on CPU/CUDA recovery tests including the new branch,
then trains a separate run in `road_crack_phase4_v1`. Earlier checkpoints remain
preserved. Auxiliary-head compute is included in current profiling.

After validation, evaluate the fixed Phase 3 control with the same new boundary
metric before claiming improvement. Desired benefit is at least two boundary-F1
points with at most one region-Dice point lost. Compare per-image differences;
one seed cannot establish broad superiority. Holdout/query evaluation follows
only after validation checks. Thin-crack continuity and cross-domain adaptation
are distinct from boundary localization and require their own evidence.
