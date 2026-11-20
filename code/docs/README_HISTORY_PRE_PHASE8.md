# Lightweight Few-Shot Road Crack Segmentation

**Phase 7 implemented:** entropy/random support suggestions, source replay with
separate retention validation, overlapping tiled inference, and verified FP32
ONNX export/runtime are integrated into the React application. Eleven focused
tests, browser coverage and real-checkpoint parity checks passed. The short replay
pilot reduced forgetting but still failed the retention gate; candidates remain
rejected. See [docs/PHASE_7.md](docs/PHASE_7.md) for measured results and limitations.

**React studio and project showcase:** `web/` now uses React + Vite, with a dark
slate/orange design, verified-results dashboard, interactive few-shot comparison,
and a dedicated research gallery. See [web/README.md](web/README.md) for frontend
setup and development. The existing launcher serves the production build.

**Phase 6 implemented:** the local CrackLab application supports image upload,
real-checkpoint inference, mask correction, model comparison, recoverable CPU
adaptation, candidate validation/activation/rollback, and exports.

```powershell
.\.venv\Scripts\python.exe scripts/run_inspection.py
```

Open **http://127.0.0.1:8765**, or double-click `Start_CrackLab.cmd` to start it.
Setup, workflow, recovery, and verification: [docs/PHASE_6.md](docs/PHASE_6.md).
The 32 trained model references are imported from the verified local result ZIPs.
Functional browser QA uses labeled synthetic fixtures; it is not a new accuracy
benchmark. Phase 5's 30 runs are verified in `output/phase5_archive_verification.json`.

Earlier phase notes below record prior states.

**Phase 4 comparison verified:** on the 95-image target query, boundary supervision
increases Dice from 34.01% to 41.72% and boundary F1 from 56.91% to 63.14%.
Source-holdout Dice and boundary F1 decrease slightly. This is a single-seed,
target-specific pilot result, not a universal improvement. See
`docs/PHASE_4_COMPARISON_RESULTS.md`. Next is isolated few-shot adaptation.

**Latest:** Phase 3 final evaluation is verified. Region-only LiteCrack achieves
76.36% source holdout Dice and 34.01% target query Dice with 339,265 parameters.
The target result is 0.51 point below R2AU with 83.92% fewer parameters, while
source accuracy is lower. This is a single-seed pilot, not overall-superiority
evidence. See `docs/PHASE_3_FINAL_RESULTS.md`; next is the boundary-aware ablation.

**Phase 3 result verified:** LiteCrack has 339,265 parameters (83.92% fewer than
R2AU), validation Dice 73.72% and IoU 62.24%; all 24 tests passed. Compactness is
achieved, but validation accuracy is below both baselines. Holdout/query evaluation
for this model remains pending. See `docs/PHASE_3_RESULTS.md`. Earlier setup notes
below describe prior states.

**Phase 3 implementation ready:** run
`output/kaggle/Road_Crack_Phase3_Lightweight.ipynb` on Kaggle to test and train
the region-only LiteCrack model. It preserves the frozen splits and gates training
on compactness and CPU/GPU recovery checks. Download `phase3_results.zip` afterward.
No Phase 3 accuracy, parameter count or runtime has been verified yet; see
`docs/PHASE_3.md` for the design and evidence limits.

**Latest verified status (26 October 2026):** Phase 2's single-seed pilot baselines
are trained and evaluated. Source-holdout Dice is 81.32% for U-Net and 81.42% for
paper-inspired R2AU; target-query Dice is 24.60% and 34.52%. The domain gap remains
large. Checkpoints and results are preserved; see `docs/PHASE_2_FINAL_RESULTS.md`
and `data/manifests/baseline_registry_v1.json`. Earlier notes below are historical.

**Phase 2 implementation is ready for Kaggle execution.** Open
`output/kaggle/Road_Crack_Phase2_Baselines.ipynb` using the connected GPU kernel.
It runs recovery tests, trains the two pilot baselines, profiles best checkpoints,
and exports `phase2_results.zip`. See `docs/PHASE_2.md`. Training has not yet been
executed or verified from this agent session.

**Update:** The first Phase 2 ZIP has now been verified: 18 tests passed and both
trainers completed, but segmentation training collapsed. Preserve those results.
Use `output/kaggle/Road_Crack_Phase2_Stabilization.ipynb` for the separate,
validation-only corrective run. See `docs/PHASE_2_RESULTS_V1.md` for measured
results and limitations. No successful corrective run has been verified yet.

Phase 1 implementation: reproducible setup, dataset auditing, group-separated
manifest generation, and checkpoint recovery primitives. No model has trained yet.

## Current state

**Latest:** Kaggle execution verified two Tesla T4 GPUs and five passing foundation
tests. The corrected audit validated 657 pairs. The pilot manifest is frozen:
240 source training, 60 validation, 237 holdout, 25 target acquisition, and 95
target query images. Five nested 1/5/10-shot support draws are included.
No crack model training has started. See `docs/PHASE_1_STATUS.md` for full evidence
and limitations; the local-session notes below describe the initial setup only.

The three root `phase1*.zip` files preserve the received remote evidence. The
virtual environment, caches, raw datasets, generated working artifacts, and future
checkpoints are excluded from Git. Checkpoints require separate persistent backup.

- Python 3.12.4 and project-local `.venv` verified.
- Installation attempted but blocked by socket/network restrictions in this session.
- Dataset GitHub access through the shell also failed; no dataset downloaded.
- NVIDIA tools unavailable and hardware query denied; GPU status unverified.
- Atomic-write interruption test passed. Python modules compile.
- PyTorch resume and data audit tests are provided but cannot run until dependencies install.
- No frozen real-data manifest or dependency lock exists yet.

See [phase status](docs/PHASE_1_STATUS.md) and [checkpoint contract](docs/CHECKPOINTING.md).

## Setup (PowerShell from this directory)

```powershell
# The virtual environment already exists; recreate only if needed.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/experiments/environment.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt
```

Use the PyTorch installer appropriate for the actual training hardware before
freezing dependencies: https://pytorch.org/get-started/locally/ . Do not assume a
generic installation has CUDA support. No activation is required for these commands.

## Obtain data

The initial pilot uses DeepCrack as the source and CrackMap as the target. This
is smaller than the paper's five-source training collection and must be reported
as a pilot, not an exact reproduction.

```powershell
git clone https://github.com/yhlleo/DeepCrack.git data/raw/DeepCrack
git clone https://github.com/ikatsamenis/CrackMap.git data/raw/CrackMap
```

Run these only when network access is available. Preserve repository commits
(`git -C data/raw/DeepCrack rev-parse HEAD` and equivalent for CrackMap).
DeepCrack permits noncommercial research/education according to its authors'
README. CrackMap usage terms and actual annotations still need an audit.
Sources: https://github.com/yhlleo/DeepCrack and
https://github.com/ikatsamenis/CrackMap (reviewed 25 October 2026).

## Build the dataset index

Copy `data/manifests/index_template.csv` to `data/manifests/index_v1.csv`.
Create one row per verified image/mask pair. Paths are relative to the CSV:

```csv
image_id,image,mask,dataset,group_id,partition
example,../raw/example/images/example.jpg,../raw/example/masks/example.png,example,verified_parent_frame_01,source_train
```

This row is illustrative only. Use real filenames and verified parent-frame or
sequence IDs. Do not invent groups to force the script to pass. If grouping
metadata cannot be recovered, document that and revise the protocol explicitly.

- DeepCrack native training pairs → `source_train`; native test pairs → `source_test`.
- CrackMap pairs → `target`.
- Labels must be grayscale binary 0/255; inspect dataset polarity before any conversion.
- Preserve raw masks; any conversion needs a separate derived file and documented rule.

```powershell
.\.venv\Scripts\python.exe scripts/data/prepare_data.py data/manifests/index_v1.csv --output data/manifests/pilot_v1.json
```

The audit rejects unreadable files, mismatched dimensions, nonbinary masks,
duplicate IDs, identical decoded images, missing group IDs, and groups spanning
native partitions. Source validation is 20% of training groups; target acquisition
is 25% of target groups. Other target groups form the fixed query set. Native
source test data remains source holdout. Fractions are by group, not image count.
An existing manifest is never overwritten. Freeze only after visually reviewing
pairs, inspecting near duplicates, and confirming group provenance. The script's
output is a candidate allocation until that review is complete.

## Before long training

Pass the PyTorch recovery test, freeze actual dependencies and data manifests,
confirm the GPU environment, and configure a backup outside this project folder.
Phase 2 must integrate checkpoints into the actual training loop; the helper alone
does not implement periodic saving or guarantee recovery of a future trainer.
