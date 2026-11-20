# CrackLab: final technical report
Release 1.0.0-pilot | 1 November 2026

**Post-release research addendum (17 November 2026):** a separate six-archive
LiteCrack experiment completed after this Phase 8 report and release ZIP were
frozen. Its official 496-image publisher test Dice is 0.287325. See the
addendum at the end of this document for the fixed protocol, source breakdown,
and app integration status. The existing PDF remains the original Phase 8 report.

CRACKLAB / FINAL TECHNICAL REPORT / 1.0.0-PILOT

Lightweight road crack segmentation with boundary-aware few-shot adaptation

## Lightweight road crack segmentation with boundary-aware few-shot adaptation

A reproducible research pilot and a local inspection application.

Release date: 1 November 2026. Prepared from the preserved experiment archives, current application source, clean-environment tests and recorded demonstration. No institutional approval or authorship fields are inferred.

Abstract

This project extends a paper-inspired road-crack segmentation baseline with a compact MobileNetV3-Small encoder, a separable decoder, auxiliary boundary supervision and bounded decoder-only adaptation. A frozen DeepCrack-to-CrackMap pilot separates source training, source validation, source holdout, target acquisition and target query. Thirty fixed-budget adaptations assess 1/5/10-image support sets across five draws for two parents.

The region-only network has 339,265 parameters, 83.92% fewer than the implemented R2AU baseline. Static target Dice increases from 34.01% to 41.72% with boundary supervision. Boundary-aware 10-shot mean target Dice reaches 59.09%, while source-holdout Dice falls by 4.87 points relative to its static parent. These are single-seed exploratory results on a previously inspected query set, not an exact paper reproduction or independent confirmation.

The delivered React/FastAPI application supports inspection, pixel-aligned correction, paired comparison, recoverable adaptation, retention validation, entropy/random suggestions, blended tiled inference and verified ONNX inference. A fresh release virtual environment passed 49 CPU tests with six CUDA-only skips. A real-acquisition-image application workflow is recorded; its reference masks are existing annotations, not measured expert corrections.

| Release outcome | Scientific boundary |
| --- | --- |
| Functional pilot delivered | Multi-seed confirmation and full GPU retraining remain unperformed. |
| Integrity-checked application bundle | Failed retention gates and broad-scene false positives remain explicit. |

01 / EXPERIMENTAL FOUNDATION

Data, split controls and evaluation protocol

## Data, split controls and evaluation protocol

DeepCrack is the source dataset and CrackMap is the target. The local source is narrower than the reference paper's five-source protocol. The frozen manifest contains 657 pairs and is identified by SHA256 9c97554166af5494c746d46885425f748ea46c4035863c22c90215b3506358ae.

| Partition | Images | Permitted use |
| --- | --- | --- |
| source holdout | 237 | Fixed source evaluation only |
| source train | 240 | Optimizer updates and optional replay |
| source validation | 60 | Validation and fixed retention subset |
| target acquisition | 25 | Support labels and local demonstration |
| target query | 95 | Fixed query evaluation; never support selection |

Input preprocessing uses RGB divided by 255 and bilinear square resize to 256 x 256. Binary masks use nearest-neighbor resize during training and 0/255 storage. The fixed evaluation threshold is 0.50. UI thresholds are display/inference settings and do not rewrite the benchmark.

Results are per-image macro means. Dice and IoU equal one for two empty masks and zero for a one-sided empty mask; both-empty cases must be interpreted separately. Boundary F1 uses a fixed two-pixel tolerance at evaluation resolution. Acquisition/query separation, checksum checks and group proxies are preserved.

Grouping uses filename captures and image-similarity proxies; physical site/sequence independence is unverified. The target query was inspected in earlier phases. Some immutable source-validation IDs contain source_train because their names encode the native dataset partition; the frozen split field determines actual membership.

Only the 24 replay/source-validation pairs and fixed acquisition assets needed by later checks were restored locally. Archived evidence is not presented as a new full-image re-audit of all 657 pairs.

02 / SYSTEM DESIGN

Model and application implementation

## Model and application implementation

U-Net and a paper-inspired R2AU implementation establish fixed references. LiteCrack uses a half-width MobileNetV3-Small encoder initialized without pretrained weights, GroupNorm, multiscale skip connections and depthwise-separable decoder blocks. LiteCrack-BA adds a region-sharing boundary head; it does not guarantee topological connectivity.

| Model | Parameters | Role |
| --- | --- | --- |
| U-Net | 1,965,569 | Frozen baseline |
| Paper-inspired R2AU | 2,110,029 | Frozen baseline |
| LiteCrack | 339,265 | Region-only control |
| LiteCrack-BA | 339,282 | Boundary-aware treatment |

Training uses stabilized BCE+Dice, GroupNorm, clipping and validation-based checkpoint selection. Boundary targets are a 3 x 3 morphological gradient, with the fixed auxiliary loss added to the region objective. Adaptation freezes encoder parameters and updates decoder/output heads with Adam 0.0001, batch two and a predeclared support-pass budget.

The application separates React state and original-resolution canvas editing from FastAPI inference, SQLite metadata and filesystem snapshots. Workers run in isolated processes. Every complete optimizer step saves immutable model/optimizer/RNG/sampler state with checksummed pointers. Draft corrections cannot become fully labeled support; activation remains an explicit action after validation.

Phase 7 was added without changing the Phase 6 worker or engine, preserving existing recovery code identities. The package includes trusted trained archives; browser users cannot upload executable checkpoint files or arbitrary server paths.

03 / FROZEN CHECKPOINT RESULTS

Static comparison and the domain gap

## Static comparison and the domain gap

| Model | Source Dice | Target Dice | Target IoU |
| --- | --- | --- | --- |
| U-Net | 81.32% | 24.60% | 15.75% |
| Paper-inspired R2AU | 81.42% | 34.52% | 22.92% |
| LiteCrack | 76.36% | 34.01% | 22.65% |
| LiteCrack-BA | 75.40% | 41.72% | 28.91% |

![Figure 1. Same frozen source holdout (237) and target query (95), macro means, threshold 0.50. Single training seed; no universal superiority claim.](../output/phase8/figures/static_comparison.png)

Region-only LiteCrack is 0.51 target Dice point below R2AU with much lower parameter count, but source Dice is 5.06 points lower. Boundary supervision increases target Dice by 7.71 points and target boundary F1 by 6.22 points. Source Dice and source boundary F1 decrease slightly, so the boundary benefit is target-specific.

The initial Dice-only baselines collapsed and were retained as failure evidence. Stabilized runs are reported separately. Later training configurations were not selected from target-query labels. Earlier query inspection nevertheless limits the final experiment's confirmatory status.

04 / FIXED-BUDGET ADAPTATION

Few-shot adaptation across five support draws

## Few-shot adaptation across five support draws

| Labels | LiteCrack Dice (%) | LiteCrack-BA Dice (%) | BA source change |
| --- | --- | --- | --- |
| 0 | 34.01 | 41.72 | Reference |
| 1 | 41.46 +/- 8.93 | 48.00 +/- 6.90 | -3.84 pts |
| 5 | 54.85 +/- 1.07 | 57.23 +/- 2.00 | -2.81 pts |
| 10 | 56.91 +/- 1.33 | 59.09 +/- 1.76 | -4.87 pts |

![Figure 2. Means across five fixed support draws; error bars are sample SD across draws, not population confidence intervals. Every draw uses the same query images.](../output/phase8/figures/fewshot.png)

There are 30 independent adaptation initializations: two parent models x five draws x K=1/5/10. Each starts from its original parent; K=10 does not continue from a K=5 candidate. Five support passes give different step counts as K changes; this is a fixed-pass, not equal-compute, comparison.

Boundary-aware five-shot mean gain is 15.51 target Dice points over its own static model; ten-shot gain is 17.38 points. Source-holdout changes of -2.81 and -4.87 points fail the optional two-point retention target. Existing public masks were used; actual correction minutes were not measured.

05 / RETROSPECTIVE PAIRED ANALYSIS

Descriptive uncertainty and its limits

## Descriptive uncertainty and its limits

| Comparison | Change (pts) | 95% interval | Groups |
| --- | --- | --- | --- |
| BA 10-shot change, source | -4.87 | [-6.38, -3.37] | 237 |
| BA 10-shot gain, target | +17.38 | [+11.70, +22.80] | 40 |
| BA 5-shot change, source | -2.81 | [-4.06, -1.54] | 237 |
| BA 5-shot gain, target | +15.51 | [+10.83, +19.95] | 40 |
| BA - region, source | -0.97 | [-1.78, -0.17] | 237 |
| BA - region, target | +7.71 | [+5.27, +10.17] | 40 |

Method: 10,000 paired proxy-group bootstrap resamples, seed 881, percentile intervals and an image-weighted mean. Group labels come from the frozen manifest. For adaptation, scores are averaged across the five support draws for each image before resampling; repeated draws are not treated as independent new test images.

These intervals are a retrospective descriptive analysis of the same previously inspected images. Proxy capture/similarity groups are not verified physical road/site independence. Intervals are unadjusted across the listed comparisons. They are not formal confirmatory significance tests, between-training-seed intervals, or evidence of broad generalization.

The project still lacks repeated training seeds for the main static comparison and an independently reserved second target domain. The narrow conclusions are compactness, target-specific boundary/adaptation gains on this pilot, and a measurable source-domain tradeoff.

06 / PHASE 7

Extended release: replay, tiles and portability

## Extended release: replay, tiles and portability

Entropy/random ranking only sees eligible incomplete acquisition images; it excludes complete corrections, frozen evaluation matches and existing job validation images. Entropy is not calibrated confidence. Supports are still chosen manually. Tiling retains spatial scale with 256-pixel inputs, 64-pixel overlap, edge padding and probability blending.

| Parent / condition | Local target Dice | Source subset change | Replay gate |
| --- | --- | --- | --- |
| LiteCrack / no replay | 57.35% | -7.74 pts | Source gate not applied |
| LiteCrack / replay | 56.66% | -4.44 pts | Rejected |
| LiteCrack-BA / no replay | 62.47% | -7.44 pts | Source gate not applied |
| LiteCrack-BA / replay | 61.96% | -4.68 pts | Rejected |

This paired smoke pilot uses draw 0 K=5, one support pass, five separate acquisition-validation images and eight fixed source-validation images. Replay mixes one source-training image per support image. It reduced forgetting, but source-subset losses still exceeded two points. Both replay candidates remain rejected. This is not a full E7 or source-holdout confirmation.

| Model | Native CPU p95 | ONNX maximum error |
| --- | --- | --- |
| LiteCrack | 44.88 ms | 0.00037462 |
| LiteCrack-BA | 52.59 ms | 0.00012633 |

CPU timing: i5-1335U, PyTorch 2.15.0+cpu, FP32, 256 x 256, batch one, two threads, 20 warmups and 200 timed model passes. Loading/preprocessing/rendering are excluded. ONNX FP32 opset 17 passes max probability error <=0.001 and per-image Dice/IoU difference <=0.002 on eight audited source-validation images; unlabelled and synthetic diagnostics are checked too.

07 / FUNCTIONAL EVIDENCE

Recorded real-image application demonstration

## Recorded real-image application demonstration

![Figure 3. Real known acquisition image, actual frozen-checkpoint prediction and existing dataset reference mask. The reference was imported explicitly; no measured expert correction or independent evaluation is claimed.](../output/phase8/figures/real_demo.png)

The demonstration runs against a fresh extracted release and separate virtual environment. It covers upload, native prediction, actual brush/undo/redo and a draft save; reference-mask import and agreement metrics; tiled and ONNX inference; entropy ranking; a real source-replay subprocess; candidate validation; paired comparison; project/recovery ZIPs; and reload persistence.

| Observed step | Result |
| --- | --- |
| Real images | Two fixed target-acquisition records, with hashes and split provenance. |
| Adaptation job | completed; target/source validation reviewed without changing gates. |
| Candidate action | Activated then rolled back |
| Exports | Project ZIP, recovery checkpoint SHA256 and ONNX graph checksum verified. |
| Browser errors | 0 JavaScript errors; recorded walkthrough.webm retained. |

Full-resolution screenshots, the video, source annotation provenance, per-image records and validation details are under output/phase8/demo/. This demonstrates application functionality on real images, not a field deployment, user study or fresh accuracy benchmark.

08 / RELEASE VERIFICATION

Acceptance ledger and clean installation

## Acceptance ledger and clean installation

| Gate | Status | Finding |
| --- | --- | --- |
| G1 | partial | Frozen split/hash and exclusion checks preserved; physical capture/site independence is unverified. |
| G2 | passed | 339,265 region-only parameters; 83.92% fewer than the implemented R2AU baseline, below 5M. |
| G3 | pilot pass | Static target Dice meets the within-two-point tradeoff in this single-seed pilot; source accuracy is lower. |
| G4 | partial | Target boundary F1 +6.22 points and Dice +7.71; source/validation do not show a universal improvement. Multi-seed confirmation absent. |
| G5 | pilot pass | Five-shot boundary mean Dice 57.23% vs static 41.72%; five support draws share the same query set. |
| G6 | unmet | Boundary source-holdout Dice changes -2.81/-4.87 points at K=5/10; replay smoke still fails the source-subset gate. |
| G7 | passed on named cpu | Native FP32 p95 44.88/52.59 ms, 20 warmups and 200 timed model passes, i5-1335U, two threads. |
| G8 | passed on audited samples | Both static ONNX exports pass probability and labeled sample metric parity gates; no global accuracy or ONNX speed claim. |
| G9 | passed demonstration | Real acquisition-image workflow, reference annotation import, actual adaptation, comparison and exports recorded; no independent accuracy study. |
| G10 | partial | Fresh venv/app bootstrap and CPU tests verified. Full original GPU training has not been rerun from the release. |

Fresh Python 3.12 virtual environment: 49 passed, six CUDA-only skips. Installed distributions match the release lock. The extracted app imports all 32 references and bundled replay pairs without a dataset download or Node rebuild. Application code and compiled frontend fingerprints match the tested candidate.

09 / WHAT IS NOT SOLVED

Known broad-scene failure and limitations

## Known broad-scene failure and limitations

![Figure 4. The same unlabelled wide road photograph in resize and tiled modes. Green denotes predictions, not ground truth; trees, sky edges or lamp regions still generate false positives.](../output/phase8/figures/wide_scene_failure.png)

All 32 original base/adapted references misclassified substantial vegetation on this photograph. Tiling changes the pattern and recovers additional foreground cracks but does not become a road-region detector. No Dice/IoU or overall winner is assigned without a complete ground-truth mask.

Remaining limits include single-seed training, query reuse, incomplete physical grouping, the absence of a second independent target, no measured annotation effort, failed retention targets, and source/target distribution shifts. No output establishes road safety, crack depth, physical dimensions or maintenance priority.

DeepCrack source terms restrict the data to noncommercial research/education. CrackMap usage terms remain unconfirmed in the frozen protocol; this local package does not assert public redistribution permission. Third-party data and the reference paper are not relicensed by the repository code license.

10 / HANDOFF

Reproduction, provenance and final scope

## Reproduction, provenance and final scope

1. Extract the versioned bundle into a new folder. Run scripts/verify_release.py before using trusted checkpoint files. 2. Run Setup_CrackLab.ps1 to create a Python 3.12 virtual environment and install pinned runtime dependencies. 3. Start_CrackLab.cmd serves the compiled interface at 127.0.0.1:8765. 4. Use the included acquisition demo only with its recorded annotation provenance. 5. Keep original archives and recovery backups; never overwrite a newer workspace.

The release includes Phase 3/4/5 trained archives, the compiled React app, source, tests, model card, setup/demo guides, selected verified replay/demo assets and Phase 7 recovery evidence. It excludes the main user workspace, credentials, venv, node_modules and browser binaries. Network access is needed for the initial dependency installation; no offline wheelhouse is claimed.

Full original GPU training has not been rerun from this CPU release. Notebook/configuration commands and frozen manifests remain available, but require the original compatible GPU environment and full permitted datasets. Code, weights, application reproducibility and scientific generalization are separate evidence levels.

| Reference | Source |
| --- | --- |
| [1] Katsamenis et al. (2023) | A Few-Shot Attention Recurrent Residual U-Net for Crack Segmentation. arXiv:2303.01582. |
| [2] Liu et al. (2019) | DeepCrack. Neurocomputing 338:139-153. doi:10.1016/j.neucom.2019.01.036. |
| [3] Pinned datasets | github.com/yhlleo/DeepCrack; github.com/ikatsamenis/CrackMap. Commits are retained in configs/data_sources.json. |
| [4] Runtime APIs | docs.pytorch.org/docs/stable/onnx/; onnxruntime.ai/docs/api/python/api_summary.html. |
| [5] Local evidence | output/phase8/evidence.json; Phase 5 archive audit; Phase 7 parity/pilot reports; fresh tests and recorded demo. |

Final conclusion: the pilot application and versioned research package are delivered with reproducible model identities and explicit scope. Target-specific gains and compactness are supported by preserved evidence; universal generalization, full retention and independent scientific confirmation remain future work.

## Post-release addendum: six-archive multi-source model (17 November 2026)

This is a separate LiteCrack experiment, not an extension of the Phase 8
DeepCrack-to-CrackMap score table. Six audited publisher archives yielded 3,564
pairs: 2,526 training, 542 validation and 496 test. The single-seed (42) 512 px
model was selected on validation at epoch 60 / step 18,960, with validation Dice
0.316174. Its checkpoint SHA256 is
`7f2f0bb391f1eea70c33256e0f17aead23c6d8e208a85adf9168a9a1745ff376`.
The selection and threshold 0.5 were frozen in
[`FROZEN_SELECTION.json`](../output/multisource_test_v1/FROZEN_SELECTION.json)
before official test evaluation.

| Publisher test source | Images | Dice | IoU | Recall |
| --- | ---: | ---: | ---: | ---: |
| All six archives | 496 | 0.287325 | 0.197227 | 0.270186 |
| Ground GoPro | 62 | 0.693625 | 0.534673 | 0.674456 |
| Ground iPhone | 50 | 0.719845 | 0.565234 | 0.732012 |
| UAV 3–10 m | 65 | 0.216736 | 0.129471 | 0.200462 |
| UAV 15–30 m | 58 | 0.208454 | 0.120334 | 0.184258 |
| UAV 35–100 m | 210 | 0.142357 | 0.080253 | 0.118514 |
| Other UAV | 51 | 0.145943 | 0.081666 | 0.137065 |

The overall per-image macro precision is 0.356516 and boundary F1 is 0.376827.
The high-altitude UAV group has low recall and represents 210 of 496 test
images. The predicted foreground fraction (0.024576) is below the truth
fraction (0.035704), consistent with missed labeled crack pixels. The ground
and UAV scores show that performance is highly dependent on capture setting.
No multi-seed or independently verified physical-site generalization is claimed.
The result does not establish reliable autonomous UAV inspection.

The downloaded `results.zip` passed CRC verification. Its 496 per-image IDs are
unique, cover the six archive counts above and reproduce the summary Dice/IoU
means exactly. The result ZIP SHA256 is
`ca5ec6ceabdd13e5c4fb927ec3c1f12d889f1a49bd89207e7a8b827d18c01845`.
See the [full test report](../output/multisource_test_v1/FINAL_TEST_REPORT.md),
[`summary.json`](../output/multisource_test_v1/summary.json), and
[`per_image.json`](../output/multisource_test_v1/per_image.json).

The working-tree CrackLab app registers the hash-verified model as a selectable
reference, without changing existing project defaults. A local CPU smoke check
passed direct inference and the upload/predict/mask API flow on a synthetic
image; it is an integration check, not another accuracy evaluation. A separate
[real-image app walkthrough](MULTISOURCE_APP_DEMO.md) then passed upload,
prediction, draft correction, existing-reference import, side-by-side model
comparison, project export and reload persistence, with zero JavaScript page
errors and no adaptation jobs. It is a functional demo on known acquisition
images, not a new benchmark or measured expert correction study. The
versioned Phase 8 release ZIP and its PDF report predate this follow-up. A new
release package and clean extracted-folder check are separate delivery steps.
No more model retraining is planned for this project handoff.

