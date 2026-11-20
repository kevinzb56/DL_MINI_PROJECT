# Follow-up implementation

Implemented 1 November 2026 as a separate research workflow. The Phase 8 ZIP,
original checkpoints, completed pilot and active application are unchanged.
No full GPU retraining or independent-field success is claimed by implementation.

The first GPU retention output is now [audited](../output/followup_v1/downloaded_audit/RESULTS.md):
12 development adaptations, 90 locked matrix adaptations and all 18 CPU/CUDA checks
passed verification. The selected BA strategy passed all 15 joint holdout/query
gates with a target-accuracy tradeoff. This download does not contain the opt-in
three-seed source retraining, new scene-test data or real human annotation sessions.
See `output/followup_v1/downloaded_audit/audit.json` for checkpoint/metric verification.

## Run retention experiments on the GPU

Open `output/kaggle/Road_Crack_Followup_Retention.ipynb` in VS Code and select the
existing Kaggle Jupyter Server kernel. Enable GPU and Internet. Attach the original
Phase 3/4 ZIPs, their complete extracted contents, or the Phase 8 release ZIP.
The notebook verifies the original parents, restores pinned datasets, checks all
657 image/mask pairs, runs CPU/CUDA recovery tests and records its actual GPU
environment. It never installs the CPU release lock over Kaggle's GPU PyTorch.

Run cells in order. A separate `/kaggle/working/road_crack_followup_v1` directory
holds new code/results. Rerunning identical code/configuration resumes checkpoints;
a code/protocol mismatch fails rather than silently continuing. A session reset
requires restoration of this new recovery backup as well as the original parents.

The six predeclared conditions in `configs/followup_v1.json` are no replay,
supervised replay, and four replay+distillation variants. They vary distillation
weight, replay ratio and learning rate in a bounded grid. Distillation is
Bernoulli KL at temperature one, **only on source-training examples**. The parent
is frozen/eval; encoder weights remain frozen in the adapted student. No labels
or teacher inputs from source validation, holdout or target query enter updates.

Development uses fixed draw 0, K=5, five support passes, the remaining 20 acquisition
images for target validation and all 60 source-validation images for retention.
The eight-image application subset is not substituted for the full validation.
A distillation candidate must increase target-validation Dice and lose at most
two source Dice points. If none qualifies, `no_admissible_candidate` is recorded.
No gates are loosened. Highest target gain wins among admissible candidates, with
source retention/name as deterministic tie breaks.

After selection is frozen, each qualifying parent is compared against no-replay
and replay controls across five support draws and K=1/5/10. Only after all locked
adaptations finish are the full source holdout and old target query evaluated.
This is fixed support passes/optimizer steps across conditions; source/background
examples and compute increase. Every run reports example counts. Original-query
reuse remains exploratory and cannot be undone by a new file name.

CLI equivalent after restoring verified data:

```powershell
.\.venv\Scripts\python.exe scripts/experiments/run_followup.py --stage preflight --data-root data/raw --output output/followup_v1
.\.venv\Scripts\python.exe scripts/experiments/run_followup.py --stage develop --data-root data/raw --output artifacts/followup_v1/development
.\.venv\Scripts\python.exe scripts/experiments/run_followup.py --stage confirm --data-root data/raw --development artifacts/followup_v1/development --output artifacts/followup_v1/matrix
```

The production configuration requires CUDA. CPU fixture tests use their own small
configuration; they demonstrate recovery/isolation, not scientific performance.

## Multi-seed source retraining

The separate notebook cell is opt-in because it is a substantial training run.
Enable `RUN_SOURCE_RETRAINING` and run seeds **42, 73, 101**, one block at a time.
Each block trains matched LiteCrack, LiteCrack-BA and stabilized R2AU from scratch
using original configs, source training and validation-based checkpoint selection.
No test scores appear during source training. The nine-run parent index must be
complete before aggregate evaluation/adaptation through `--parents-index`.

```powershell
.\.venv\Scripts\python.exe scripts/experiments/run_followup.py --stage source-train --only-seed 42 --data-root data/raw --output artifacts/followup_v1/source
```

Repeat for 73 and 101 with prior blocks restored. The latest/best index uses
paths relative to the source output so a recovered tree can be relocated.
Source snapshots use the original 30-step and epoch cadence; at most 29 source
steps may need replay after a hard crash. Adaptation snapshots occur every step.
Full optimizer/scheduler/RNG/sampler state and configuration/manifest/code hashes
are retained. Training leaves prior immutable snapshots intact. Monitor free
space and export backups between blocks; no automatic pruning is performed.
New adaptation checkpoints reserve 2 GiB of free space for recovery export; new
source jobs require 8 GiB headroom. If storage is low, export and restore latest/best
states in fresh storage instead of deleting original snapshots during a run.

After all blocks finish, use `source-evaluate`, then `develop`/`confirm` with
`--parents-index artifacts/followup_v1/source/parents.json`. The notebook provides
these calls. `scripts/evaluation/summarize_followup.py` reports support-draw SD within each
training seed and SD across seed means separately. It does not pool repeated
images/draws into an inflated independent sample count or claim confidence intervals.

## New broad-scene data and reserved testing

Copy `configs/followup_scenes.template.json` outside the frozen experiment, replace
its placeholders, and provide real original-resolution RGB images, complete binary
0/255 crack masks and road-region masks. Road masks are **evaluation annotations**;
they are never supplied to inference as an oracle road filter. Explicit negative
scenes may have empty crack masks; unannotated images cannot be called negatives.

Assign `scene_train`, `scene_validation`, `scene_test` by physically distinct site
and capture group before tuning. The auditor rejects split/site/group overlap,
exact pilot-image reuse, duplicate images, mismatched checksums/dimensions,
incomplete/prediction-origin labels and crack labels outside the road mask.
Actual site independence relies on honest collection metadata; image hashes
alone do not establish it. The inspected wide photo is diagnostic, not independent.
The prospective minimum is 24 training images from three sites, 12 validation
images from two other sites, and 30 test images from five further sites. These
minimums support a bounded follow-up; they do not establish universal generalization.

Add `--scene-manifest PATH --scene-root DIRECTORY` to a **new** development run
and its subsequent matrix. One scene-training example per support example is
added to the distillation conditions; controls do not receive those extra examples.
This evaluates the combined intervention, without attributing its gain solely to
distillation. Scene validation adds gates: at least 25% relative non-road
false-positive-area reduction, at most two-point recall loss and at most two-point
Dice loss. Positive-image macro recall guards against predicting no cracks.
Zero baseline non-road false positives or no positives makes the gate unassessable,
not passed. These are prospective follow-up targets, not retroactive pilot gates.

Before opening scene test metrics, run `scene-freeze` with development directory,
`--parent-name litecrack_ba`, scene paths and `--scene-lock PATH`. It locks the
selected checkpoint, parent, sampling mode, code and scene manifest, requiring
the same scene-validation gates to pass. Then `scene-test` evaluates the reserved
sites with that lock and writes a paired report. `--mode tiles` uses the existing
audited CPU tile pipeline; use a separate CPU configuration/study for it. Resize
is the default GPU path. Do not switch modes after looking at test scores.

No automatic road-region classifier is introduced. Background training and the
reserved paired evaluation are executable; a road-region model would require its
own trained weights and validation study. No invented independent labels are bundled.

## Timed human corrections

Use the app's existing editor and record actual sessions:

```powershell
.\.venv\Scripts\python.exe scripts/operations/followup_annotation.py start --image scene.png --annotator A01 --task-id task01 --starting-mask prediction
.\.venv\Scripts\python.exe scripts/operations/followup_annotation.py finish --session PATH_PRINTED_ABOVE --mask corrected.png --origin human_corrected
```

Pause unrelated work during a session; restart a new session if the OS reboots.
Finishing checks original-image identity and mask shape/binary values and records
elapsed time plus hashes. Dataset imports are refused as human-effort records.
Human provenance is self-reported; the timer cannot certify who annotated a mask.
For comparisons, counterbalance blank/prediction starting masks across annotators
and matched task difficulty; avoid same-person repeat-image practice effects.
Actual annotator work is still required; this implementation does not fabricate it.

## Recovery and validation

After a completed/safely interrupted stage, use the notebook backup cell or:

```powershell
.\.venv\Scripts\python.exe scripts/operations/package_followup.py --root artifacts/followup_v1 --output followup_backup_001.zip
```

The exclusive ZIP includes code, protocol/results and checksummed latest/best
checkpoint targets. It verifies CRC and every member hash and writes an external
SHA256. Older snapshots remain at origin. Download the backup before ending Kaggle.
Keep original parents/data available; backing up a run does not back up remote data.
For local results-only runs, `_followup_source/` preserves the workflow code;
restore that source tree to a separate checkout before resuming with the run tree.

## Local evidence

`scripts/evaluation/smoke_followup.py` executes one CPU optimizer step with the actual BA
parent and known bundled release subsets, without touching the app. Its output
is explicitly an integration smoke check: eight source-validation images and one
separate acquisition-validation image are not the full retention study. The full
follow-up preflight still requires all 657 original pairs and a GPU. A smoke gate
pass cannot replace the frozen matrix or independent-scene evaluation.

Focused tests cover exact resume for both model variants, frozen parent/encoder,
split/hash isolation, source-only teacher loss, validation-only selection, rejection
of trivial/non-retaining models, scene provenance/site gates, original-aligned
resize/tile metrics and background-training recovery. CPU checks do not establish
CUDA recovery; the notebook requires the CUDA cases to pass before GPU experiments.
