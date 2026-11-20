# Multi-Source Dataset: handoff for the next model

**Last checked:** 17 November 2026, Asia/Kolkata  
**Project:** `C:\Users\Kevin Shah\OneDrive\Desktop\DL_MINI_PROJECT`  
**Dataset:** [Zenodo record 22697998](https://zenodo.org/records/22697998), version 1.0.0, CC BY 4.0.

## Current state

**Final official test completed and verified locally:** the validation-selected
512px LiteCrack checkpoint (seed 42, epoch 60, step 18,960, threshold 0.5) scored
the 496-image publisher source-holdout test at Dice **0.287325**, IoU **0.197227**.
The result ZIP and all per-image records are saved in `output/multisource_test_v1/`.
See [the final test report](../output/multisource_test_v1/FINAL_TEST_REPORT.md)
for provenance, archive-level scores, verification, and limitations. The ground
camera results are strong relative to UAV results; high-altitude UAV Dice is
0.142357. Do not retune this frozen candidate on the test split. Earlier
queue, launch, and test-disabled statements below are historical snapshots.

**CrackLab integration completed locally:** `scripts/prepare_inspection_models.py`
registers the frozen checkpoint as `LiteCrack · multi-source 512 px` after verifying
its SHA256 and freeze record. Existing project defaults remain unchanged; select
the new model in the inspection screen or activate it explicitly in model history.
Use whole-image resize and threshold 0.5 for the evaluated inference setting.
The UI notes the weak UAV test performance. The repeatable local smoke check is
`.venv\Scripts\python.exe scripts\evaluation\smoke_multisource_app.py`; it passed direct
CPU inference and the upload/predict/mask API flow on a synthetic image. This
checks integration, not real-image accuracy or visual quality. Start the local
app with `Start_CrackLab.cmd` and open `http://127.0.0.1:8765`.

**Resume from epoch 41 queued:** `resume_multisource_512.py` selects the latest
verified continuation recovery (step 12,956 at the last inspection), uploads it
and the retained 512px package to a fresh session, verifies restored manifest and
runtime identities, then invokes the existing resumable trainer without changing
the training config. Local recovery synchronization restarts after launch.
Inspect `output/multisource_512_continuation/launch_status.json`; `uploading`
means the GPU worker has not yet started. Do not launch a duplicate resume.

**16 November continuation queued:** restored downloads completed; rebuilding
`output/multisource_512_pilot/Training_512_Pilot.zip`. The new session passed T4
execution checks. `launch_multisource_512_continuation.py` waits for the rebuilt
package, uploads the preserved epoch-20 recovery and starts a separate remote
project. `continue_multisource_512_remote.py` checks exact parent manifest,
implementation and runtime identities before explicitly branching the checkpoint
with epochs 20→100; all other config and optimizer/scheduler/RNG/sampler state
are retained. Parent ZIP remains immutable; lineage is recorded remotely.
Local recovery watcher targets `output/multisource_512_continuation/recovery`.
Inspect that directory's `launch_status.json` before restarting. Training is
not started while state is `waiting_for_package`. Test scoring remains disabled.

**16 November restore required:** local cleanup removed the six original ZIPs
and prepared training packages; final recovery checkpoints and package metadata
remain. A fresh Kaggle session passed GPU execution checks, but no continuation
was launched. Original archives are being restored by the verified downloader;
inspect `output/multisource_restore/download.log` and `download.err` before any
restart. Rebuild 512px inputs and verify restored row/member identities before
domain diagnosis or continuation. Keep the epoch-20 pilot checkpoint immutable;
any extended epoch horizon must be an explicitly recorded continuation branch.

**512px pilot preparation active:** training-only original-mask audit completed
on 144 masks. High-UAV resize fidelity improves 0.754→0.875 at 512px. A distinct
20-epoch seed-42 pilot is being built from original members; automatic launch
and local backup synchronization are queued in `launch_multisource_512.py`.
See `MULTISOURCE_512_PILOT.md` and
`output/multisource_512_pilot/launch_status.json`. Keep local computer and supplied
Kaggle session running; do not duplicate the builder/launcher. Training has not
started while status is `waiting_for_package`.

**Domain diagnosis completed and saved locally:** restored best full-model
checkpoint; scored only train/validation. Validation Dice is GoPro 0.6401,
iPhone 0.6659, low UAV 0.1674, mid UAV 0.1099, high UAV 0.0716, other 0.0830.
All 3,068 per-image records and grouped summaries are verified locally under
`output/multisource_full_v1/domain_diagnosis/`. See `DOMAIN_DIAGNOSIS.md` in that
output directory. Next is an original-resolution preprocessing audit followed
by a controlled resolution/crop pilot; no new training was launched in diagnosis.

**Full training completed:** seed-42 LiteCrack stopped at epoch 68 / step 21,488.
Best validation Dice 0.238616, IoU 0.164175 at epoch 53 / step 16,748 across 542
validation images. Final local recovery and best/latest payload hashes verified.
See `output/multisource_full_v1/TRAINING_REPORT.md`. Test scoring remains disabled.
Prior launch/queue notes below are historical; do not restart those workers.
Next work is domain-level validation diagnosis before test evaluation.

**Full training launch authorized and queued:** see `MULTISOURCE_FULL_TRAINING.md`.
`build_multisource_full.py` is preparing verified 256px train/validation inputs.
`launch_multisource_when_ready.py` waits for the package, verifies its hash,
uploads it to the supplied active session, starts the trainer, and runs local
recovery synchronization. Inspect `output/multisource_full_v1/launch_status.json`
and `launch.txt` before launching anything else. Do not duplicate these workers.
The session credential is held only in process memory. Keep this computer and
the Kaggle session running. Test evaluation is disabled; the unavailable final
Milano baseline is explicitly omitted from this run.

**Full audit completed:** all six archives downloaded and verified; all 3,564
pairs passed preparation (2,526 train / 542 val / 496 test), with zero decoded
cross-split duplicates. Manifest SHA256:
`ba80bc2b4663e49717f62217aba487c55f3bbc0b5a393fd0a35affef132873f7`.
The previous queue/waiting notes below are historical. Before the full training
launch, recover the frozen final Milano checkpoints: the report references
`output/milano_v1/recovery/milano_recovery_litecrack_ba_20261110T184426004641Z.zip`,
but that folder/archive is currently absent. A workspace search found only older
`output/kaggle/Milano_Resume_step13410.zip` and `Milano_Resume_step8400.zip`;
these must not be substituted for the final checkpoints in the transfer baseline.
New full-dataset training has not started.

**Automatic next stage configured:** `prepare_multisource.py --full` now requires
all six pinned archives and audits complete official ID coverage, pairing,
dimensions, masks and cross-archive decoded duplicates. A background
`wait_multisource_audit.py` waits for the existing queue to finish, then writes
`output/multisource_full_v1/manifest.json`. Inspect
`output/multisource_downloads/audit.log` and `audit.err`; do not launch a second
waiter. The resulting status is `AUDITED_MULTISOURCE_V1`, which the trainer does
not accept until the separate main-study protocol is frozen. This stage does
not automatically launch GPU training. Two preparation guard checks passed;
the real full-data audit remains pending download completion.

**Latest continuation, 15 November:** The follow-up diagnostic finished at
20 total epochs / 980 steps. Best GoPro validation Dice is **0.530971** at
epoch 19; the original model recovered with longer training. The eight-image
memorization check reached Dice 0.830959, and the cached inputs reproduced the
pilot's validation metrics. See the
[diagnostic report](../output/multisource_diagnostic/DIAGNOSTIC_REPORT.md).
Its final recovery is verified locally. Test scoring remains disabled and no
six-archive model has been trained. A local background queue now downloads the
remaining five archives; inspect `output/multisource_downloads/queue_status.json`
and `queue.log` before starting another downloader. The Kaggle diagnostic and
its local backup watcher have finished; the download queue is independent.

**Archive-1 pilot update, 15 November 2026:** The authorized bounded pilot
completed three epochs / 147 steps, and its final recovery is verified locally.
See [MULTISOURCE_ARCHIVE1_PILOT.md](MULTISOURCE_ARCHIVE1_PILOT.md) and the
[pilot report](../output/multisource_archive1_pilot/PILOT_REPORT.md). The full
six-archive study has not started. The poor three-epoch validation result was
investigated in the completed diagnostic above; test evaluation was disabled.

- The earlier Milano LiteCrack and LiteCrack-BA runs are complete. Their report is
  `output/milano_v1/FINAL_REPORT.md`; these are seed-42 results on the Milano
  publisher split.
- Only archive 1 has been used for the short pipeline pilot and its separate
  20-epoch diagnostic continuation. Checkpoints and metrics are under
  `output/multisource_archive1_pilot/` and `output/multisource_diagnostic/`;
  no full-dataset run or new-dataset test evaluation exists.
- Download root: `data/external/multisource_cracks_v1/`.
- Complete and verified: `01_ground_gopro.zip`, 2,103,023,240 bytes. Zenodo MD5
  and the release SHA256 both passed. Metadata, license, SHA256 list and official
  `splits.zip` are also present and verified.
- The five archives pending when the queue started total 13,654,418,253 bytes
  (12.717 GiB). At queue launch C: had 17,181,872,128 bytes free, enough for the
  compressed archives and its 2 GiB reserve. Do not make a second full extracted
  copy. Recheck current free space and the ledger before further preparation.
- A local resumable queue is running for the remaining five archives. Inspect
  its status/log before using the restart commands below; `.part` files survive
  interruptions. The queue script is `scripts/data/queue_multisource_downloads.py`.
- This machine has no usable local CUDA setup (`nvidia-smi` unavailable and
  global Python has no PyTorch). GPU training therefore needs an active Kaggle
  GPU notebook/session or another GPU host.

## Exact remaining downloads

Run these sequentially in PowerShell from the project root. Keep the root the
same so checksummed files and partial transfers are reused. The script also
checks/retains the official metadata. It verifies Zenodo's file checksum and the
dataset's SHA256 entry before marking an archive complete. It now merges each
invocation into `DOWNLOAD_STATUS.json` instead of forgetting prior chunks.

```powershell
python scripts/data/download_multisource_cracks.py --root "data/external/multisource_cracks_v1" --files 02_ground_iphone.zip
python scripts/data/download_multisource_cracks.py --root "data/external/multisource_cracks_v1" --files 06_uav_other.zip
python scripts/data/download_multisource_cracks.py --root "data/external/multisource_cracks_v1" --files 03_uav_low_3_10m.zip
python scripts/data/download_multisource_cracks.py --root "data/external/multisource_cracks_v1" --files 04_uav_mid_15_30m.zip
python scripts/data/download_multisource_cracks.py --root "data/external/multisource_cracks_v1" --files 05_uav_high_35_100m.zip
```

Published sizes and SHA256 values:

| Archive | Bytes | SHA256 |
|---|---:|---|
| `02_ground_iphone.zip` | 1,637,391,230 | `99cb2fd7b22603cec14ec78843883cf911dc26fd230c5c6bd5c6114fbac77688` |
| `06_uav_other.zip` | 1,906,268,391 | `a60c0741132dbf1947ad234e5144f088d0b0c357a45dd0cd74ac838a9dde1ec1` |
| `03_uav_low_3_10m.zip` | 2,307,063,695 | `33b7b31acb6dcd5ea474ee55b77fec55d267816b3039ea71127c6a27f9b3fad7` |
| `04_uav_mid_15_30m.zip` | 3,445,893,965 | `269d423bed34aa81c11f31ad66fc1028e4f1a4bd58c6de175c445ae522ef81b6` |
| `05_uav_high_35_100m.zip` | 4,357,800,972 | `b438623402c156bd12fafa226a9b6b36221025731e0d9110ea8c143cd1b3832c` |

Use `DOWNLOAD_STATUS.json` as the progress ledger. A finished chunk must show
`sha256_verified: true`. If interrupted, rerun the same command; do not rename or
delete `.part` files. After all five finish, confirm all six archive names appear
in the ledger and each has `sha256_verified: true`.

## Dataset split facts

`SPLIT_AUDIT.json` was generated from the official split table and reports 3,564
images: 2,526 train, 542 validation, 496 test. The table assigns 340 contiguous
10-image blocks within 21 broad collection groups; no derived 10-image block
crosses train/validation/test. All 21 broad groups occur in every split. These
blocks reduce neighboring-frame leakage, but the data do **not** establish
physical-site independence. The release has crack masks but no road-region masks.
The `image_path` and `mask_path` CSV columns point to the authors' original
Windows paths; use `image_id` to pair files from the archives.

Regenerate/confirm the split audit after download:

```powershell
python scripts/data/audit_multisource_splits.py `
  --splits-zip data/external/multisource_cracks_v1/splits.zip `
  --output data/external/multisource_cracks_v1/SPLIT_AUDIT.json
```

## Work required before training

ZIP-backed loading and the explicit archive-1 pilot protocol are implemented.
The six-archive preparation/main-study protocol is **not ready yet**. The official
table contains author-machine paths. Extend the audited pilot preparation to:

1. Indexes members in all six ZIPs without unsafe path extraction.
2. Pairs every official `image_id` with exactly one image and one mask by member
   stem, checks dimensions and mask semantics, and records archive/member paths
   plus SHA256 identities.
3. Maps only official `train`, `val`, and `test` records into the run manifest;
   never derives splits from archive names or trains on validation/test.
4. Reads samples from the ZIPs or an explicitly sized external staging volume so
   a second full extracted copy is not required on C:.
5. Integrates with checkpointing, run configuration, progress logs and final
   per-image metrics. Do not begin training if any member/pair/split audit fails.

## Training and evaluation sequence

1. Obtain an active Kaggle GPU session. Keep the credential-bearing proxy URL out
   of scripts, logs and artifacts; `scripts/experiments/kaggle_session.py` prompts for it
   without saving it. Check actual available Kaggle disk and GPU before uploading
   or downloading large files. Prefer downloading the public archives directly
   inside Kaggle if internet is enabled, rather than uploading 15.76 GB from C:.
2. Preserve the verified original Milano archive and report. Before any new
   training or tuning, freeze model/config choices and measure the existing
   Milano LiteCrack and LiteCrack-BA checkpoints on this dataset's 496-image test
   split as a cross-dataset transfer baseline. Save predictions/metrics as a
   separate immutable run.
3. For new training, sample/shuffle across **all** archive members assigned to
   official `train` during each epoch. Do not train one archive to completion
   before moving to the next; that would confound training with camera/height
   domain changes and may cause forgetting.
4. Use official `val` only for checkpoint selection. Freeze architecture,
   preprocessing, threshold and reporting before evaluating official `test`.
   Use test once for the newly trained runs; do not tune from its outcomes.
5. For a multi-seed comparison, use three recorded seeds (42, 43, 44) for each
   selected architecture, preserve the same split and hyperparameters, and
   checkpoint/export progress regularly because Kaggle storage is ephemeral.
   If compute only permits one seed, label it a single-seed pilot, not confirmed
   model superiority.
6. Report macro per-image Dice/IoU/precision/recall/boundary F1, positive-only
   performance, empty-mask behavior and per-group results. Do not report
   road-only false-positive area because the release has no road masks. Do not
   claim unseen-site generalization from this split.

## What the user needs to provide

The next model can continue the five downloads without another dataset choice.
To launch/manage the GPU run, the user must provide a **fresh active Kaggle
notebook session** (or another GPU environment) when requested. The next model
should use the existing hidden URL prompt and must not copy the credential into
the handoff file. The run must export checkpoints and reports to durable storage
before the Kaggle session ends.
