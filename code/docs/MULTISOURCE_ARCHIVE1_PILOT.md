# Archive 1 pipeline pilot

Authorized on 14 November 2026: validate the pipeline with the first verified
archive before the six-archive main study. This is a partial-dataset engineering
pilot, not a final benchmark or an archive-by-archive curriculum.

**Completed 15 November 2026 (Asia/Kolkata):** three epochs / 147 steps, all
538 pairs audited, 19 tests passed, final recovery verified locally. Best
validation Dice was 0.127864 at epoch 1 and fell to 0.021863 by epoch 3; the
pipeline works but model quality remains unresolved. See the
[pilot report](../output/multisource_archive1_pilot/PILOT_REPORT.md).

The subsequent [diagnostic continuation](MULTISOURCE_DIAGNOSTIC.md) recovered to
best validation Dice 0.530971 at epoch 19. The original results above remain
unchanged and are not relabeled as the longer run.

## Fixed scope

- LiteCrack, random initialization, seed 42, 256-pixel input, batch 8.
- Three epochs, Adam at 0.0001, BCE plus Dice, gradient clipping at 1.0.
- Archive `01_ground_gopro.zip` only, using official train and validation rows.
- Official test rows are checked for data integrity but never scored or trained on.
- No Milano checkpoint is overwritten or used as pilot initialization.
- Full-dataset transfer evaluation and the six-archive mixed training study remain
  separate work. Pilot validation results are not evidence of six-domain accuracy.

## Implementation and gates

`scripts/data/prepare_multisource.py` verifies the pinned archive SHA256, official
split table, exact ID coverage for the two GoPro groups, unique image/mask pairs,
member paths, dimensions, binary masks, publisher foreground counts, and decoded
cross-split duplicates. It writes a ZIP-backed manifest only after all checks pass.

`src/archive_io.py` reads members without extracting the dataset. The existing
trainer verifies member SHA256 values and shuffles training rows each epoch.
The explicit pilot protocol rejects test evaluation and mismatched split mappings.

`scripts/experiments/run_multisource_pilot.py` runs archive regression tests and a GPU model
forward/backward check, downloads and audits archive 1, trains two steps, exports
a recovery ZIP, then resumes the exact optimizer/RNG/sampler state to finish
three epochs. It exports code, configuration, manifest, logs and checkpoints at
completion. ZIPs do not include the dataset; its pinned identity is in the manifest.

## Remote locations

- Project: `/kaggle/working/cracklab_multisource_archive1_pilot`
- Log: `/kaggle/working/multisource_pilot.log`
- State: `artifacts/pipeline_status.json` under the project.
- Recovery exports: `/kaggle/working/multisource_pilot_recovery_*.zip`

Local captures and downloaded recovery files live under
`output/multisource_archive1_pilot/`. Session credentials must never be saved here.

Fetch a finished export using the existing hidden connection prompt:

```powershell
python scripts/experiments/kaggle_session.py --fetch-latest-recovery --recovery-prefix multisource_pilot_recovery_ --output output/multisource_archive1_pilot/recovery
```

The helper verifies the export SHA256, ZIP CRC and checkpoint pointer hashes.
An export present only on Kaggle is not a durable backup.

## Continuation boundaries

The preparation script currently accepts only archive 1. Extend and audit all
six archives before creating the main-study protocol. Do not resume this pilot
checkpoint with a changed manifest/configuration; use a separate documented run.
The launcher refuses an existing remote project directory to prevent duplicate
workers. Inspect state before any retry after interruption.
