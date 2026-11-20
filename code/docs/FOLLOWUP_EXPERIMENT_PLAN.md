# Follow-up experiments for unmet goals

Prepared 1 November 2026 after the completed Phase 8 pilot. Status: the isolated
workflow is implemented; see [implementation and execution](FOLLOWUP_IMPLEMENTATION.md).
The GPU retention follow-up is complete and [audited](../output/followup_v1/downloaded_audit/RESULTS.md).
Three-seed source retraining and new independent-scene/human data collection remain pending.
Keep the Phase 1-8 artifacts and release ZIP immutable. New configurations,
checkpoints, metrics and reports must use a separate follow-up namespace.

## Retention and target adaptation

Develop on source training/validation and target acquisition only. Compare the
static parent, existing adaptation and a proposed replay-plus-parent-distillation
candidate using the same supports and fixed compute/accounted example budget.
Use teacher probabilities only on permitted training/replay images, never holdout
or query images. Freeze encoder initially and retain the original loss control.
Learning rate, replay ratio and distillation strength require a bounded search
and a written configuration frozen before final evaluation.

Select on separate target-acquisition validation plus source validation. Enforce
the existing two-percentage-point source-loss limit without relaxing it after
failure. Compare target gain and source retention jointly. An unchanged parent
trivially retains accuracy and must not be called successful adaptation without a
target gain. Report the full source holdout separately from the eight-image app
validation subset. Previously inspected query/holdout results remain exploratory.

## Broad-scene false positives

Build a separately identified development set of varied road scenes, including
trees, poles, shadows, texture and non-road regions. Obtain complete reference
masks and road-region labels when testing a road-region filter. Do not label
unannotated pixels as negatives or derive truth from current predictions.
Keep the user's already inspected road image diagnostic only; improvement on
that one image cannot establish generalization.

Compare current resize/tiling baselines with training using explicit background
negatives; evaluate any road-region filtering as a separately labeled pipeline.
Use validation to choose settings and freeze before a new test set is opened.
Report per-image Dice/IoU, recall and foreground false-positive area on non-road
regions. Freeze numerical acceptance targets before the new evaluation, including
a recall safeguard so predicting no cracks cannot pass by reducing false positives.

## Scientific confirmation

Use at least three predeclared training seeds with matched data/protocol and
support draws. Report variation across training seeds separately from support
draws and image-level uncertainty. Restore and checksum all original data before
claiming full-data reruns. A verified independent target collection with physically
separate sites/captures must be reserved before tuning; hashing alone cannot prove
site independence. Keep old-query analyses explicitly exploratory.

Measured annotation effort requires actual timed human correction sessions with
documented instructions and comparable starting masks. Existing dataset reference
imports do not measure expert correction effort.

## Current readiness

The local environment reports PyTorch 2.15.0+cpu and no CUDA device. `data/`
currently contains manifests rather than a complete restored dataset tree.
A pinned DeepCrack archive exists under `artifacts/phase7/source/`, and the release
contains small replay/demo subsets; these are not a verified full-data rerun.
Use the existing Kaggle/VS Code notebook workflow for substantial training after
confirming the live GPU kernel. Full data restoration must pass original manifest
checksums. Independent labeled scenes/site metadata and measured human annotation
sessions are external inputs that cannot be fabricated from the completed pilot.

Completion means running the frozen follow-up, reporting all candidates/seeds and
judging the unchanged gates honestly. Engineering these studies is feasible;
whether the models satisfy each accuracy/retention gate remains an empirical result.
