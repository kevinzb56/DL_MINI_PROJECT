# Phase 7 extension protocol, frozen before validation

Purpose: validate the extended application capabilities. This does not amend
Phases 1–5, rerank their checkpoints, or claim the tree false positives are fixed.

- Ranking: mean binary predictive entropy, computed from original-size bilinear
  probabilities; seed 7301 for the random comparator. Complete corrections,
  frozen evaluation images, and existing job validation images are excluded.
  Ranking only suggests corrections; it never chooses supports automatically.
- Tiling: native model-sized 256-pixel tiles, overlap 64, edge padding, Hann window
  with each 1-D weight floored at 0.05; blend probabilities then threshold 0.50.
  Maximum 256 tiles. This is a new inference mode, with no promised accuracy gain.
- Replay: source buffer of 16 source-training images sampled with seed 7301;
  eight separate source-validation images selected using the same RNG after
  training selection. Files must match the frozen manifest hashes. One replay
  image per support image in the same batch, decoder/head only, Adam 0.0001,
  clipping 1.0, no augmentation. The support-pass budget and optimizer-step count
  stay fixed; examples and compute increase. Replay order is deterministic from
  seed and optimizer step. Source validation never enters training.
- Replay activation: unchanged encoder, local target-validation Dice loss at most
  two points, and source-validation-subset Dice loss at most two points. Report
  this eight-image subset separately from the full source holdout. Do not relax
  gates after viewing a failure. Phase 6 no-replay recovery code stays unchanged.
- ONNX: FP32, opset 17, static batch one and model input size, RGB/255 input,
  region probability output. Check graph structure, finite output, native/ONNX
  maximum probability error at most 0.001 on audited source validation, at most
  three project images, and black/white/noise diagnostics. On audited labeled
  source-validation images require per-image Dice and IoU differences <=0.002.
  If audited data are absent, disclose the unassessed metric gate. Exports include
  weights' source hash, graph checksum, sample IDs, versions, and input contract.

Validation scope: tests for tiling coverage/alignment, ranking isolation and
determinism, source split isolation, exact replay interruption/resume, activation
gates and project ownership; real static reference ONNX parity; browser exercise
of ranking, tiled prediction, replay and portable export controls. A small
paired replay smoke experiment may use fixed draw 0 K=5, one support pass,
remaining acquisition images as local validation, and the source subset above.
This is a functional pilot, not a replacement for a preregistered full E6/E7
comparison or an independent target-query result.

Reference APIs: [PyTorch ONNX](https://docs.pytorch.org/docs/stable/onnx/) and
[ONNX Runtime Python](https://onnxruntime.ai/docs/api/python/api_summary.html).
