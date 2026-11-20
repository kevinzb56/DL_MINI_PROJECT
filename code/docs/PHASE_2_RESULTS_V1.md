# First baseline results — verified, but training collapsed

Archive: `phase2_results.zip`; SHA256:
`e216010ff251fb773c29a33cc59648d31976610622bf43055b6a92a24c1b2d73`.
Evidence retained in `artifacts/imported/phase2_e216010f/`.

ZIP integrity passed, frozen manifest matched exactly, and all four best/latest
checkpoint pointer hashes verified. Both evaluations have 237 unique source
holdout IDs and 95 unique target-query IDs. Dataset restoration verified 657
pairs against the frozen file hashes.

All 18 tests passed in 11.45 seconds, including actual trainer mid-epoch resume
on CPU and CUDA for both architectures. Both runs completed early stopping at
epoch 11, step 330; their validation-best checkpoints were from epoch 1.

| Model | Best validation Dice | Source holdout Dice | Target query Dice | Parameters | GPU p95 model-pass ms |
|---|---:|---:|---:|---:|---:|
| U-Net | 26.57% | 30.45% | 14.04% | 1,962,625 | 4.34 |
| Paper-inspired R2AU | 5.46% | 8.09% | 7.53% | 2,107,085 | 9.63 |

These are unsuccessful pilot runs, not useful accuracy baselines. U-Net validation
Dice became zero at epoch 2 and stayed zero; training soft Dice loss became one.
R2AU validation stayed at 5.46% with recall one and precision equal to IoU,
consistent with foreground collapse. Exact probability/gradient diagnostics were
not exported, so the numerical cause is not established by the archive alone.

The two model costs do not justify performance superiority. CPU p95 was 63.07 ms
for U-Net and 184.78 ms for R2AU on Kaggle's generic x86_64 host (four threads).
Timing excludes IO/transfer. Source-only training used one seed, one source
collection and modified widths; it does not reproduce the paper's experiment.

## Corrective experiment

Preserve v1 unchanged and start a new v2 run: GroupNorm, BCE+Dice objective,
learning rate 0.0001 and gradient clipping at norm one. These changes are motivated
by the training/validation collapse; they are hypotheses, not confirmed fixes.
Record validation foreground fractions to expose trivial masks. Repeat actual
CPU/CUDA trainer recovery tests for the new configuration before training.

The v2 notebook evaluates source validation only. Do not repeatedly evaluate query
or source holdout while repairing training. The previously inspected query results
make subsequent research exploratory; eventual stronger claims need independent
evaluation beyond repeated reuse of this pilot query set.

Run `output/kaggle/Road_Crack_Phase2_Stabilization.ipynb`; export
`phase2_stabilization_results.zip`. It uses `road_crack_phase2_v2` and never resumes
or overwrites v1. Original model definitions are retained inside the v1 evidence
archive; new normalized definitions must not be used to load old model weights.
