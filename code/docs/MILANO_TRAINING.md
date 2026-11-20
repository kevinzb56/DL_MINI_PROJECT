# Milano GPU handoff — 8 November 2026

The raw publisher extraction is unchanged. `data/external/milano_prepared_v1/`
contains a separate full-resolution copy, binary masks, checksummed manifest,
preparation audit and training-only review sheet. White foreground denotes cracks
or pavement distress, rather than a guaranteed crack-only taxonomy.
Publisher description: https://zenodo.org/records/20040434.

Normalization uses Pillow luminance >=128 as foreground, otherwise background.
Non-black pixels with partial transparency cause an error rather than silent loss.
There are 1,480 training, 498 validation and 231 test pairs. Normalization changed
8,871 intermediate pixels. Empty masks number 261/115/45 respectively; these are
publisher annotations, not independently certified hard-negative scenes.
Twelve evenly spaced training examples were visually reviewed; their overlays
appear aligned. No validation/test visuals were used for model decisions.

Decoded RGB SHA256 found no exact image duplicates across splits. The broad
64-bit average-hash screen flagged 3,674 cross-split candidate pairs at distance
<=3. These are unreviewed similarity flags, not confirmed duplicates. Publisher
splits are preserved; physical-site independence is not established. Do not claim
an independent-site benchmark or vegetation-failure resolution from this run.

## Run on Kaggle

1. Open `output/kaggle/Road_Crack_Milano_v1.ipynb` in a GPU Kaggle session.
2. Attach `output/kaggle/Milano_Prepared_v1.zip` as a private input. Its adjacent
   SHA256 file records the package identity; the notebook embeds that identity.
3. Run bootstrap and the optional recovery cell (leave RECOVERY=None for new work).
4. Run LiteCrack, download its recovery ZIP and SHA256, then run LiteCrack-BA.
5. Run comparison and download the final recovery ZIP and SHA256.

Both configurations use seed 42, random initialization, 256px, batch 8, BCE+Dice,
learning rate 1e-4, max 100 epochs, validation patience 15. BA adds boundary weight
0.2. Only training data update weights; best validation Dice selects the checkpoint.
Test evaluation uses threshold 0.5 and per-image macro metrics, with per-image outputs.
This is a matched single-seed experiment; three-seed confirmation remains separate.
Road masks are absent, so non-road false-positive area cannot be measured here.

Snapshots retain optimizer, scheduler, RNG and sampler progress every 30 steps and
at epoch boundaries. Recovery exports retain latest/best payloads and logs for both
models without deleting original snapshots. Interrupts are forwarded for graceful
checkpointing. Hard crashes can replay up to 29 steps. Restore a trusted ZIP and
adjacent SHA256 in a fresh project; the runtime identity must match. Installed local
torch is CPU-only. On 8 November, the supplied existing Jupyter server was verified
with torch 2.11.0+cu128, torchvision 0.26.0+cu128 and two Tesla T4 GPUs. All 11
trainer tests passed there, including CUDA checkpoint/resume comparisons; see
`output/milano_v1/gpu_preflight.txt`. Connectivity and fixture tests do not establish
completed Milano training.

## Existing Jupyter session

`scripts/experiments/kaggle_session.py` reads the connection URL through a hidden prompt and
does not save it. It supports code execution, chunked file upload, status capture,
and downloading/verifying the latest recovery export. The credential-bearing URL
must never be put in a project file or command argument.

`scripts/operations/build_milano_session_launch.py` builds credential-free launch, status,
and manual-export code under `output/kaggle/`. The launch starts a detached worker
that checks the complete dataset hash, restores the embedded source, runs CPU/CUDA
tests, trains both models sequentially and compares the final test results. A
background thread exports recovery ZIPs every five minutes once a checkpoint exists.
Both models use one GPU each sequentially; this is not distributed two-GPU training.
Session resets can still lose unexported work; download backups externally.

The actual session was launched on 8 November (worker PID 515). LiteCrack was
observed past epoch 1 at optimizer step 270. A step-180 recovery ZIP was downloaded
to `output/milano_v1/recovery/` and passed archive SHA256, ZIP CRC and checkpoint
pointer checks. `output/milano_v1/session_status.json` and `live_status.txt` are
saved observations, not continuously current training metrics.

A local `scripts/experiments/kaggle_session.py --watch-recovery --output output/milano_v1/recovery`
process polls every minute and downloads new exports with checksum/pointer checks.
`output/milano_v1/backup_sync_status.json` records the last poll. Keep this PC awake
and the local sync process running for continued downloads. The URL stays only in
the process memory. Remote training runs independently of the local connection,
subject to the Kaggle session remaining alive.

## Verified session recovery — 9 November (local time)

The replacement session was empty and matched the original runtime. Restored
the SHA256/CRC-verified local backup from 8 November 18:44:19 UTC: 28 completed
epochs, latest optimizer step 5,190 and validation-best step 5,180. No checkpoint
pointer or training configuration was edited. `Milano_Resume_v1.zip` transfers the
exact prepared masks, manifest and recovery artifacts (about 15.6 MB); publisher
images are downloaded directly by Kaggle and checked against both the original
archive MD5 and every image SHA256 in the frozen manifest.

Rebuild recovery launch code with
`python scripts/operations/build_milano_session_launch.py --resume-package output/kaggle/Milano_Resume_v1.zip`.
The generated worker validates conflicts before restoring and enforces the original
runtime lock before training. This requires the package at the Kaggle working root.
The worker reran all 11 CPU/CUDA recovery tests successfully (16.04 seconds).
LiteCrack subsequently advanced beyond the restored state to step 5,250, confirming
actual resumption. At that observation, LiteCrack-BA and final evaluation were still
pending. The later completed run and verified results are recorded below.

## Rebuilding the runtime after Kaggle image changes

The replacement GPU image on 10 November provides Python 3.13.16 and torch/vision
2.11.0/0.26.0 with CUDA 13.0. The saved study requires Python 3.13.15, torch
2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.1.3 and Pillow 13.0.0.
`scripts/operations/build_milano_environment_launcher.py` creates an isolated environment at
`/kaggle/working/milano_runtime_cu128` using managed CPython 3.13.15 and official
CUDA 12.8 PyTorch wheels. It verifies the versions, two Tesla T4 devices and a
GPU matrix operation before launching the existing recovery worker. The default
Kaggle environment is preserved. Package source: https://download.pytorch.org/whl/cu128/torch/.

The rebuilt Python uses a different compiler/build stamp. Generate launch code
with `--python-executable /kaggle/working/milano_runtime_cu128/bin/python` and
`--allow-python-build-metadata-change` only when the exact Python patch and all
other locked runtime fields match. This records both original and rebuilt identity
in `artifacts/runtime_rebuild.json`; it never edits the original runtime lock.
Cross-build bitwise equivalence is not claimed. Eight focused guard tests verify
that altered package versions, Python patch versions or GPU models still fail.

`output/kaggle/milano_training_monitor.py` executes a notebook kernel monitor of
the actual training worker until completion and writes
`/kaggle/working/milano_live_progress.json`. It reports failures rather than
keeping a stopped job marked as running. Session lifetime remains subject to
Kaggle limits; keep external checkpoint downloads active.

## Local verification

Focused preparation, trainer and recovery tests: 10 passed, 4 CUDA tests skipped.
Notebook code cells are syntax checked; package files and loader compatibility
are verified separately. These checks are not a completed GPU run.

Reproduce preparation with `scripts/data/prepare_milano.py`; an existing prepared
directory is never overwritten. Build/rebuild the notebook with
`scripts/authoring/build_milano_notebook.py`; an existing data ZIP must match its recorded
checksum and manifest.

## Final run and remaining validation

Both models completed training and publisher-split evaluation. The authoritative
single-seed result and archive identity are in
`output/milano_v1/FINAL_REPORT.md`. This supersedes the pending status above.
The refined image screen has 17 high-similarity candidates; their provenance is
not included in the publisher data. Use
`output/milano_v1/milano_similarity_adjudication.csv` to record source-backed
capture/site relationships and evidence; see
`output/milano_v1/SIMILARITY_ADJUDICATION_GUIDE.md`. Until the cross-split candidates are
resolved, keep the publisher test result exploratory; do not delete or relabel
images based on visual similarity alone.

Independent-site validation also requires real, permissioned scenes and complete
human crack and road masks. The intake is `data/followup_external/`; its collection
protocol is `data/followup_external/COLLECTION_GUIDE.md`. Once records are supplied,
run `scripts/operations/followup_study_inputs.py freeze-scenes` to audit and freeze a new
manifest. Site-separated evaluation and multi-seed confirmation remain outstanding.
