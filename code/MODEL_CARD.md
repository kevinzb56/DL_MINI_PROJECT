# CrackLab model card

Version 1.0.0-pilot, 1 November 2026. Evidence:
[consolidated results](output/phase8/evidence.json), [final report](docs/FINAL_REPORT.md).
The separate 17 November multi-source follow-up is documented below; the
versioned Phase 8 ZIP and its original model identities remain unchanged.

## Model and intended use

LiteCrack is a compact RGB binary crack segmenter with a half-width
MobileNetV3-Small encoder, GroupNorm and a separable multiscale decoder.
LiteCrack-BA adds a boundary-supervision head. Both were initialized without
pretrained encoder weights. Region-only/BA parameter counts are 339,265/339,282.

Intended use is supervised local research, educational demonstration and human
inspection with editable masks. Predictions require visual review. They do not
establish road safety, physical crack depth/width, maintenance urgency or
autonomous inspection suitability. Pixel measurements are not calibrated lengths.

## Data and protocol

DeepCrack is the source and CrackMap the target. Frozen partitions contain 240
source-training, 60 source-validation, 237 source-holdout, 25 target-acquisition
and 95 target-query pairs. Manifest SHA256:
`9c97554166af5494c746d46885425f748ea46c4035863c22c90215b3506358ae`.
The split field determines membership even where IDs preserve native
`source_train` names. Capture/similarity groups are proxies; physical site
independence is unverified. The query was inspected in prior project phases.

Static/Phase 5 metrics use RGB /255, 256-square bilinear resize, nearest-neighbor
mask resize, threshold 0.50, per-image macro averaging and two-pixel boundary
tolerance. Two empty masks score one for Dice/IoU. App tiling uses 256-pixel
tiles, 64-pixel overlap, edge padding and blended probabilities at original scale;
it is a separate inference pipeline. Stored masks align with original dimensions.
UI thresholds do not rewrite the benchmark protocol.

## Results

| Model | Source Dice | Target Dice | Target IoU |
| --- | ---: | ---: | ---: |
| U-Net reference | 81.32% | 24.60% | 15.75% |
| Paper-inspired R2AU reference | 81.42% | 34.52% | 22.92% |
| LiteCrack | 76.36% | 34.01% | 22.65% |
| LiteCrack-BA | 75.40% | 41.72% | 28.91% |

BA target Dice at K=1/5/10 is 48.00 +/- 6.90, 57.23 +/- 2.00 and
59.09 +/- 1.76 percent (mean/sample SD over five support draws). These draws share
the same query set. Each adaptation starts from its original parent, freezes the
encoder and updates decoder/heads for five support passes. Optimizer-step count
changes with K. No best-query candidate selection is claimed.

BA source-holdout change at K=5/10 is -2.81/-4.87 points and fails the two-point
retention target. The one-pass Phase 7 replay pilot reduced source-subset forgetting
but still failed its source gate. Phase 8's separate K=1 real-image functional
demo passed its small local gates; this does not reverse the Phase 7 findings.

FP32 native model-only CPU p95 is 44.88/52.59 ms for LiteCrack/BA on an i5-1335U,
two threads, batch one, 256-square input, 20 warmups and 200 timed passes.
Loading, preprocessing, tiling and UI are excluded. No R2AU speed comparison or
end-to-end latency claim follows. Static ONNX parity passed on eight audited
source-validation samples; max probability errors were 0.00037462/0.00012633.
Parity does not establish global accuracy or ONNX speed.

## Known limitations

### Multi-source follow-up, 17 November 2026

The follow-up is a **separate LiteCrack model**, trained on six audited archives
with 2,526 training, 542 validation and 496 publisher test image/mask pairs.
It uses 512 px RGB input, seed 42, and threshold 0.5. Validation selected epoch
60 / step 18,960 (validation Dice 0.316174). The checkpoint SHA256 is
`7f2f0bb391f1eea70c33256e0f17aead23c6d8e208a85adf9168a9a1745ff376`.
The selection record was frozen before the official test was opened.

| Official test source | Images | Dice | IoU |
| --- | ---: | ---: | ---: |
| All six archives | 496 | 0.287325 | 0.197227 |
| Ground GoPro | 62 | 0.693625 | 0.534673 |
| Ground iPhone | 50 | 0.719845 | 0.565234 |
| UAV 3–10 m | 65 | 0.216736 | 0.129471 |
| UAV 15–30 m | 58 | 0.208454 | 0.120334 |
| UAV 35–100 m | 210 | 0.142357 | 0.080253 |
| Other UAV | 51 | 0.145943 | 0.081666 |

Overall test precision is 0.356516, recall 0.270186 and boundary F1 0.376827.
These are per-image macro means at the fixed threshold, on this dataset only.
The high-altitude UAV group accounts for 210 test images and has low recall
(0.118514). The model is suitable as a research/inspection reference with human
review, not autonomous road assessment. No independent physical-site claim or
multi-seed uncertainty estimate follows from this single-seed publisher split.
It can still mark vegetation on a broad road image and miss visible cracks;
that image has no complete reference mask, so it supplies no Dice score.
Changing the app threshold or selecting tiled inference changes the inference
setting and does not inherit the reported test metrics.

The working-tree app registers this hash-verified checkpoint as a selectable
reference. It does not automatically replace a project's active model. The
[final test report](output/multisource_test_v1/FINAL_TEST_REPORT.md) links the
frozen selection and per-image results. This model is not in the Phase 8 ZIP.

### Phase 8 pilot limitations

All 32 imported base/adapted references produce substantial vegetation false
positives on the user's broad road photograph. Tiling changes the output without
solving road-region detection. No full reference mask exists for that photo, so
no numerical winner is assigned. Shadows, texture, lighting, camera scale, other
surfaces and domain shift remain concerns. Boundary supervision does not guarantee
connected cracks. Entropy ranking is not calibrated confidence.

Single training seed, reused query, proxy grouping, no independent second target,
no measured expert correction effort and forgetting limit generalization.
Retrospective bootstrap intervals are descriptive, unadjusted and do not measure
between-training-seed uncertainty. Full GPU training was not rerun from this
release. Demo reference masks are existing dataset annotations.

## Identity and adaptation controls

Original Phase 3/4/5 archives and all 32 imported references are immutable.
Archive identities are in `RELEASE_MANIFEST.json` and `scripts/release_integrity.py`;
individual model identities remain in model registries and exports. A checksum
confirms identity, not scientific validity. Load only trusted checkpoint archives.

Corrections are stored separately; drafts cannot be complete supports. Explicit
support IDs and separately selected local validation IDs are recorded. Replay
uses 16 source-training pairs and eight separate source-validation pairs; those
eight are a retention subset, not the full holdout. Candidates need validation
before explicit activation and support rollback. Preserve recovery ZIPs and
immutable latest/best states; do not revise frozen experiments after review.

The app serves loopback locally. The package omits the main user workspace,
credentials, venv, browser binaries and node_modules. It is not a public-service
security assessment. Dataset/license scope is in [NOTICE.md](NOTICE.md).
