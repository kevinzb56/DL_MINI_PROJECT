# Archive-1 follow-up diagnostic

Started 15 November 2026 after the user authorized continuation in a fresh Kaggle
GPU session. The completed three-epoch pilot remains immutable.

**Completed:** epoch 20 / step 980. Best validation Dice 0.530971 at epoch 19;
training-only memorization Dice 0.830959. Cache equivalence and metric replication
passed; final recovery is verified locally. See the
[results report](../output/multisource_diagnostic/DIAGNOSTIC_REPORT.md).

## Questions and fixed checks

1. Can cached 256-pixel inputs reproduce the pilot's epoch-3 validation Dice and
   soft Dice loss within 0.000001? Failure blocks the run.
2. Can the same model learn eight fixed, evenly spaced official training images
   in 200 updates at learning rate 0.001, without augmentation? This is a separate
   memorization diagnostic, not a validation result or a candidate final model.
3. Does continuing the original epoch-3 model/optimizer/scheduler/RNG to 20 total
   epochs resolve the early deterioration without changing the loss, architecture,
   learning rate policy, input size, augmentation or reporting threshold?

The continuation starts from the latest pilot checkpoint, step 147. It is an
explicit new diagnostic branch with its own config, checkpoint lineage and
cache hash, not an overwritten or relabeled pilot run. The best continuation
checkpoint is selected on validation Dice at threshold 0.5. The eight-image
diagnostic never initializes the continuation.

Both unaugmented train and validation evaluation report BCE+Dice and soft Dice
loss, hard Dice/IoU/precision/recall, foreground fraction and mean predicted
probability on foreground/background pixels. A fixed 0.1–0.9 threshold grid is
reported diagnostically; it does not change checkpoint selection or the 0.5
reporting threshold. There is no test scoring.

The fixed 20-epoch diagnostic horizon replaces the short pilot's termination
limit and patience-based stopping. The original ReduceLROnPlateau scheduler
state and policy are retained; this is a deliberate diagnostic extension.

## Data and runtime

`scripts/operations/build_multisource_diagnostic.py` verifies the original archive and
recovery SHA256s and every selected source member hash, then caches the same
Pillow RGB/bilinear and binary-mask/nearest resize used by the pilot. It packages
only 390 training and 86 validation image/mask pairs. Original test IDs remain in
the parent manifest for provenance, but no test pixels enter the diagnostic.
Three original train/validation pairs are included as exact decoding witnesses.
Remote startup checks cache hashes, row identity, split assignments, witnesses,
parent implementation/runtime identities, and the reproduced validation results.

The inspected fresh runtime has two idle T4 GPUs and matching torch 2.11.0+cu130
and torchvision 0.26.0+cu130. One GPU is used for this bounded diagnostic.

## Recovery

`scripts/experiments/run_multisource_diagnostic.py` checkpoints every 50 memorization updates
and every continuation epoch. It exports after memorization, every third total
epoch, and at completion. Exports include the exact cached inputs, source,
parent checkpoint lineage, metrics and current checkpoints. Restore into a new
directory and invoke the same runner to resume; it rejects changed configuration
or cache identity. Local backup synchronization verifies ZIP checksums and
checkpoint pointer hashes.

The main six-archive study remains gated on these results, complete archive
coverage, and its separate frozen protocol. This diagnostic does not establish
multi-domain or unseen-site accuracy.
