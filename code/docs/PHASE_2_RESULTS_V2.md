# Stabilized baseline results — verified 26 October 2026

Archive: `phase2_stabilization_results.zip`; SHA256:
`578ea01fb8389abd0904f63ed621192bc6708b6d76cac8ca9c2840e01d1a5040`.
Preserved locally under `artifacts/imported/phase2_stabilization_578ea01f/`.

Archive integrity, all four best/latest checkpoint hashes, unchanged frozen
manifest, 60 correct validation IDs per model and aggregate metric means verified.
All 19 tests passed without skips, including actual CPU/CUDA trainer recovery.

| Model | Validation Dice | Validation IoU | Best epoch | Epochs run | Parameters | T4 p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 84.09% | 73.74% | 87 | 100 | 1,965,569 | 6.92 |
| Paper-inspired R2AU | 83.70% | 73.13% | 63 | 78 | 2,110,029 | 18.15 |

U-Net predicted foreground fraction 3.078% versus 2.846% ground truth; R2AU 2.926%.
The histories show useful learning rather than persistent all-background or
all-foreground collapse. The stabilization package changed multiple factors;
their individual contributions are not established without controlled ablations.

These are source-validation scores used to select checkpoints, not final test or
target-domain results. The 0.39-point validation Dice difference does not establish
significance or overall superiority. Both models are single-seed pilot baselines.
MAC counts, complete visual prediction QA and independent-domain claims remain
unverified. Model-pass timings exclude decoding and host/device transfer.

The next no-training notebook evaluates the immutable validation-best checkpoints
on source holdout and target query, first replaying validation to verify the
checkpoint/runtime. It fixes threshold at 0.5 and preserves the prior evaluations.
The pilot query was already inspected in v1; later results remain exploratory.

Run `output/kaggle/Road_Crack_Phase2_Final_Evaluation.ipynb` and download
`phase2_final_evaluation.zip`. Training stability and checkpoint recovery are
verified; final stabilized baseline test evaluation is still pending.
