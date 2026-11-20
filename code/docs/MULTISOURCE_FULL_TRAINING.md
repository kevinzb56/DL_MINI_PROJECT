# Full Multi-Source training protocol

Frozen for the user-authorized run: LiteCrack, seed 42, random initialization,
256px, batch 8, Adam 0.0001, BCE+Dice, clipping 1.0, deterministic execution,
100 epochs maximum, validation patience 15, checkpoint every 30 steps.
Shuffle all 2,526 official training samples each epoch. Use all 542 official
validation samples for best-Dice checkpoint selection at threshold 0.5.
Official test scoring is deferred. This is a single-seed full-dataset run.

The fully audited original manifest is preserved in the package. Training uses
lossless PNGs of the exact RGB/bilinear and binary-mask/nearest 256px resize;
every selected source member hash is checked before cache construction. Test
pixels are excluded. The trainer verifies cached member hashes before updates.

The final Milano checkpoints could not be located. The planned frozen Milano
transfer baseline is unavailable and is omitted explicitly; earlier resume
checkpoints are not substituted. This does not change the new training split.

Recovery exports run every two minutes once checkpoints exist and at exit.
They include matching best/latest pointers and their verified payloads, source,
config and manifest. Dataset pixels are retained separately in Full_Training_v1.zip.
Local synchronization verifies export SHA256, ZIP CRC and pointer payload hashes.
