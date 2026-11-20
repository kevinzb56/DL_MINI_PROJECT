# Next GPU run: three training seeds

Use `output/kaggle/Road_Crack_Followup_Multiseed.ipynb` in VS Code with the existing
Kaggle GPU Jupyter kernel. This is a new notebook: it does not rerun the completed
90-adaptation retention matrix and has no disabled retraining flag.

## Run it

1. Enable GPU and Internet, select the Kaggle kernel and open the new notebook.
2. Run bootstrap and the CPU/CUDA recovery checks. Exact audited source/configs
   are embedded; old parent-weight ZIPs are not required for training from scratch.
3. Run seed **42**, then **73**, then **101**. Each enabled block trains LiteCrack,
   LiteCrack-BA and stabilized R2AU. Each prints a new recovery ZIP and SHA256;
   download both before leaving a session. Run All runs every block in order if
   GPU time/storage permit. Stop between blocks if needed.
4. After all nine source runs finish, run source evaluation and then the separate
   multi-seed adaptation cell. Neither stage silently selects the best seed.

All new results use `/kaggle/working/road_crack_multiseed_v1`. The first retention
project, downloaded archive, Phase 8 release and active app model stay unchanged.
Raw data restoration reuses and verifies all 657 original pairs when available.

## Resume after a reset or low storage

Attach the latest `multiseed_*.zip`, or its complete extracted contents, as a
private Kaggle input. The bootstrap selects a matching study backup, verifies every
file and restores into the new project directory. It checks conflicts before
copying anything. Existing newer results are never overwritten by an older backup.

If a project already exists, its local state is used. If restoring a newer backup
over an existing but different project is necessary, preserve the existing project
and use a fresh session. Latest/best states, configs, optimizer/scheduler/RNG/sampler
data and code/data identities are retained. Runtime versions and GPU model are
locked across the new study. Source checkpoints keep the frozen 30-step/epoch
cadence; adaptation saves every optimizer step. A hard source crash may replay up
to 29 steps. Older snapshots remain at their origin, without automatic pruning.

Recovery is exported after every seed block and evaluation/adaptation stage,
including a gracefully stopped worker. A still-running worker is not packaged.
Download backups promptly; Kaggle storage is ephemeral. When the storage guard
stops a stage, restore latest/best states in fresh storage and rerun that stage.

## What the output establishes

`source_multiseed/parents.json` must contain exactly nine complete, checksummed
models. Source training selects checkpoints on source validation only. The later
development/matrix stages preserve the original gates and support protocol;
no-admissible-candidate outcomes remain failures.

The adaptation summary separates support-draw variability from training-seed
variability. The old query has already been inspected, so these remain exploratory
results despite repeated training seeds. New physically separate scenes and actual
human timing sessions are still needed for the other two studies below.

## Independent road-scene intake

The local intake is initialized at `data/followup_external/`:

- `scene_index.csv`: one row per real scene, using the existing headers.
- `images/`, `cracks/`, `roads/`: original RGB scenes and complete aligned 0/255 masks.
- `usage_terms.txt`: permission or applicable dataset terms.
- `physical_sites.txt`: collection/site/capture provenance and split assignment.

Minimums are 24 training images from three sites, 12 validation images from two
other sites and 30 test images from five further sites. Sites/capture groups cannot
cross partitions. Annotated road masks are used for evaluation, never as an oracle
road-region filter at inference. The inspected wide photograph remains diagnostic.

Freeze only complete real data:

```powershell
.\.venv\Scripts\python.exe scripts/operations/followup_study_inputs.py freeze-scenes --root data/followup_external --index data/followup_external/scene_index.csv --usage-terms data/followup_external/usage_terms.txt --site-provenance data/followup_external/physical_sites.txt --output data/followup_external/scenes_v1.json
```

The tool calculates file hashes, validates masks/dimensions, rejects pilot-image
reuse, incomplete/prediction-origin labels and cross-split sites, then enforces
sample minimums. It refuses to replace an existing frozen manifest. Collection
metadata is supplied by the owner; the tool cannot independently certify real sites.
Use the existing scene develop/freeze/test commands described in
`docs/FOLLOWUP_IMPLEMENTATION.md` after intake succeeds.

## Human correction effort

Use the existing editor plus `scripts/operations/followup_annotation.py start/finish` to time
actual correction tasks. Keep annotator codes, task IDs, starting-mask conditions
and the corrected-mask exports. Dataset reference imports cannot count as measured
human effort. Avoid repeated-image practice effects in the experimental design.

```powershell
.\.venv\Scripts\python.exe scripts/operations/followup_study_inputs.py annotation-report --sessions data/followup_external/annotation_sessions --output output/followup_v1/annotation_report.json
```

The report separates blank/prediction times, identifies paired tasks and repeated
annotators, and preserves missing-data status. It reports descriptive self-reported
human times; it does not infer expert-quality labels or a causal effort advantage.
No independent scenes, human work or completed GPU retraining were fabricated by
this implementation.
