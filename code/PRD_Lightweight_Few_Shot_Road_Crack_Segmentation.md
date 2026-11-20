# Product Requirements Document

## Lightweight Few-Shot Road Crack Segmentation with Boundary-Aware Learning

**Version:** 1.0 | **Date:** 23 October 2026 | **Status:** Proposed implementation scope  
**Product:** Road crack inspection and model adaptation research prototype  
**Reference:** Katsamenis et al., *A Few-Shot Attention Recurrent Residual U-Net for Crack Segmentation*, arXiv:2303.01582v1, 2023.

All proposed metrics below are acceptance targets or hypotheses, not measured results. Dataset access, compute availability, and the initial baseline will determine the final feasible experiment budget.

## 1. Product summary

Build a lightweight road crack segmentation system that predicts pixel-level crack masks, preserves thin crack boundaries, and adapts to a new road surface using a small number of corrected images. Deliver both a reproducible research pipeline and a local inspection application where a user can upload images, inspect predictions, correct masks, initiate adaptation, compare model versions, and export results.

The central research question is whether a compact model with explicit boundary supervision can offer a better accuracy–compute tradeoff and more reliable small-sample adaptation than the paper-inspired baseline. A successful implementation does not require every hypothesis to win: negative results must remain visible and reproducible.

### 1.1 Problem

Road cracks occupy a small fraction of an image and can be narrow, fragmented, shadowed, or visually similar to road texture. Dense annotation is expensive. Models trained on one collection may degrade on a new camera, surface, or lighting condition. A useful prototype must address segmentation quality, annotation effort, adaptation stability, and inference cost together.

### 1.2 Intended users

| User | Need | Main workflow |
|---|---|---|
| Road inspection engineer | Inspect cracks and correct errors | Upload → predict → review → correct → export |
| Annotation reviewer | Produce valid correction masks efficiently | Select uncertain images → edit → approve mask |
| Student or researcher | Test scientific claims fairly | Freeze splits → train → adapt → evaluate → compare |
| Demonstration operator | Run a repeatable local demonstration | Load checkpoint → process images → export report |

### 1.3 Assumptions

- Binary segmentation: crack versus background in RGB road images.
- Initial deployment is single-user and local; GPU training can run separately from CPU inference.
- A training GPU with approximately 12–16 GB VRAM is a planning assumption, not an available resource confirmed by this PRD.
- Physical crack dimensions are unavailable without calibration; default measurements use pixels and area fractions.
- The title's “few-shot” means supervised adaptation using a small labeled support set. Episodic meta-learning and unseen-class segmentation are outside the required scope.

## 2. Reference paper and extension rationale

### 2.1 What the attached paper actually reports

The following facts come from the attached PDF, especially Sections 2–3 and Table 2:

| Aspect | Paper approach |
|---|---|
| Architecture | R2AU-Net combines recurrent residual convolutional blocks and attention gates in a U-Net structure |
| Training data | Five collections totaling 4,717 images: CFD, CRACK500, Cracktree200, DeepCrack, and GAPS384 |
| Preprocessing | Images resized to 256 × 256; 3,774 training images and 943 validation images |
| Target dataset | CrackMap: 120 annotated 256 × 256 patches |
| Adaptation | Rank target images using predicted-crack confidence, correct the lowest-ranked 5% (six images), and fine-tune |
| Final evaluation | Remaining 114 CrackMap images; adaptation images excluded |
| Training | Dice loss, Adam, up to 100 epochs, batch size eight, early stopping; five adaptation epochs at a reduced learning rate |
| Static R2AU-Net | Mean Dice 72.43%; mean IoU 57.28% |
| Refined R2AU-Net | Mean Dice 77.06%; mean IoU 63.11% |

The reported improvement is **4.63 percentage points Dice and 5.83 percentage points IoU** on that study's protocol. These numbers are literature results, not project results or promised benchmarks. The paper does not establish a parameter/latency budget or evaluate an explicit boundary-supervision objective. Its confidence score considers predicted foreground and requires a defined fallback if no crack pixels are predicted.

### 2.2 Proposed extensions beyond the paper

| Extension | Purpose | Required evidence | Priority |
|---|---|---|---|
| Compact encoder and decoder | Reduce inference cost | Parameter count, MACs, memory, checkpoint size, measured latency | P0 |
| Explicit boundary head and loss | Improve thin edges and local delineation | Boundary F1 and ablation against the same model without the loss | P0 |
| Fixed 1/5/10-image adaptation budgets | Quantify annotation efficiency | Paired before/after scores over repeated support draws | P0 |
| Strict support/validation/query isolation | Prevent optimistic adaptation results | Frozen manifests, duplicate checks, group separation | P0 |
| Cross-dataset testing | Measure domain shift | Source-only and adapted results on held-out datasets | P0 |
| Correction and comparison application | Make the research usable | Complete upload-to-export demonstration | P0 |
| Uncertainty-guided sample selection | Improve use of annotation effort | Random versus uncertainty selection at equal image budgets | P1 |
| Replay during adaptation | Limit forgetting on the source domain | Source holdout degradation before and after adaptation | P1 |
| High-resolution tiled inference | Avoid shrinking away thin cracks | Seam inspection, full-resolution mask alignment, latency | P1 |
| ONNX export | Support portable inference | Output agreement and measured runtime on named hardware | P1 |
| Quantization, video sampling, topology loss | Explore further efficiency and continuity | Separate controlled experiments | P2 |

P0 is required for the core project, P1 is the extended release after the core works, and P2 is optional stretch work. The intended contribution is an evaluated combination and workflow; novelty relative to the entire literature remains to be established through a broader literature review.

## 3. Goals, hypotheses, and exclusions

### 3.1 Goals

1. Segment thin road cracks with a compact, reproducible model.
2. Measure whether boundary supervision helps beyond ordinary region losses.
3. Adapt using limited correction budgets without using query labels or silently overwriting a good checkpoint.
4. Report segmentation accuracy, boundary quality, annotation cost, and compute together.
5. Provide an understandable inspection and correction interface.

### 3.2 Research hypotheses

- **H1 — Compactness:** The proposed model uses at most five million trainable parameters and at least 50% fewer parameters than the implemented R2AU-Net baseline.
- **H2 — Boundary quality:** Boundary supervision improves mean boundary F1 by at least two percentage points against the identical compact architecture without boundary supervision, while losing no more than one point of Dice.
- **H3 — Adaptation:** Five labeled support images improve target-domain mean Dice by at least two percentage points over the same model before adaptation, averaged across the preregistered support draws.
- **H4 — Retention:** Replay limits source-domain Dice degradation to at most two percentage points after adaptation.
- **H5 — Utility:** A reviewer can produce a corrected mask, run an adaptation job, compare versions, and export an auditable result through the interface.

H1–H4 are go/no-go targets for particular claims, not conditions for suppressing an otherwise valid experiment. If a hypothesis fails, document the failure and keep the best validated configuration.

### 3.3 Exclusions

Required scope excludes robotic repair, autonomous maintenance decisions, certified road safety ratings, physical depth estimation, crack-type classification without labels, cloud multi-tenancy, continuous live video, and full mobile application deployment. No system output should imply that an unreviewed model has established road safety.

## 4. User journeys and application requirements

### 4.1 Primary journey

1. Create a project and choose a checkpoint.
2. Upload one image or a small batch and verify file validation feedback.
3. Run inference and inspect the original, probability heatmap, binary mask, and overlay.
4. Adjust a display threshold; record it with exports. Benchmark thresholds remain fixed independently.
5. Add or erase crack pixels using an image-aligned mask editor; undo and redo changes.
6. Save a complete corrected mask or explicitly flag it as incomplete.
7. Select an eligible support set and start a bounded adaptation job.
8. Review validation results and before/after predictions; activate or reject the candidate version.
9. Export image results, model identity, settings, and summary measurements.

### 4.2 Functional requirements

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-01 | Accept PNG/JPEG images; validate decoded contents, dimensions, and size | Valid images load; unsupported, corrupted, and oversized files receive clear errors; configurable initial limit 20 MB/file |
| FR-02 | Run binary segmentation | Probability map and original-size binary mask are saved with checkpoint ID, threshold, and preprocessing configuration |
| FR-03 | Display synchronized original/mask/overlay views | Zoom, pan, and brush coordinates remain aligned; transparency is adjustable |
| FR-04 | Correct segmentation masks | Brush, erase, brush size, undo/redo, and explicit save work; saved corrections round-trip exactly at original dimensions |
| FR-05 | Track annotation completeness | Incomplete masks cannot be treated as fully labeled support images; partial-mask learning is outside P0 |
| FR-06 | Create bounded adaptation jobs | Job records support IDs, parent checkpoint, seed, optimizer, trainable layers, steps, and output checkpoint |
| FR-07 | Manage candidate versions | Base checkpoint is immutable; candidate passes validation and explicit activation before becoming the default; rollback is available |
| FR-08 | Compare versions | Same images, preprocessing, and display threshold show before/after predictions; metrics appear only when ground truth exists |
| FR-09 | Export results | ZIP includes binary PNG masks, overlays, and JSON/CSV metadata; image-level measurements clearly state units |
| FR-10 | Track job progress and failure | Queued/running/completed/failed/cancelled states are visible; cancellation never activates partial checkpoints |
| FR-11 | Summarize crack extent | Report crack pixel count and area fraction; optional skeleton length is labeled pixels and its algorithm is recorded |
| FR-12 | Suggest corrections (P1) | Rank only eligible unlabeled pool images; record ranking policy; allow reviewer override |
| FR-13 | Tile large images (P1) | Overlap blending yields original-size predictions; padding, tile size, stride, and blending are recorded |
| FR-14 | Export model (P1) | ONNX model has a documented input/output contract and passes parity checks |

### 4.3 Screens

| Screen | Content |
|---|---|
| Project and upload | Project name, checkpoint, upload queue, validation results |
| Inspection workspace | Image list, synchronized image views, threshold and overlay controls, measurements |
| Correction workspace | Brush/erase tools, zoom, undo/redo, save status, annotation completeness |
| Adaptation | Support selection, budget, immutable parent version, progress, errors, validation summary |
| Comparison and experiments | Before/after views, metric tables, configuration and export links |

Do not show an arbitrary image-level “confidence percentage” as calibrated reliability. Label uncertainty maps as diagnostic scores unless calibration is explicitly evaluated.

## 5. Model and learning requirements

### 5.1 Baselines and proposed architecture

Implement a standard U-Net, a paper-inspired R2AU-Net, and the proposed compact model. Mark any unavailable or ambiguous paper details as implementation assumptions; do not call a partial recreation an exact reproduction.

**Proposed working name: LiteCrack-BA.** Use a MobileNetV3-Small encoder candidate with multiscale skip features and a U-Net-style decoder using depthwise separable convolution, pointwise channel mixing, and bilinear upsampling. Start with decoder widths of 128/64/32/16 where compatible with the encoder feature scales. Produce two full-resolution single-channel logits: crack region and crack boundary. Boundary supervision must share features with the segmentation path so that it can influence the region prediction; the boundary head may be omitted from inference if experiments support doing so.

Architecture details and parameter counts must be confirmed in the implementation. MobileNetV3 is supported as an efficient architecture candidate by its original paper [R3]; road-crack performance is unproven here. Start with random initialization for controlled architecture comparisons. Run optional pretrained initialization as a separate experiment, recording the source and external training data.

### 5.2 Boundary supervision

Generate a binary boundary target from the ground-truth mask using a morphological gradient, `dilate(mask) - erode(mask)`, with a 3 × 3 kernel at the training resolution. Record the target width and any resizing rules. This is boundary supervision, not an assumption that attention gates alone enforce boundaries.

Initial training objective:

`L = L_BCE(region_logits, mask) + L_Dice(sigmoid(region_logits), mask) + 0.2 * L_balanced_BCE(boundary_logits, boundary_target)`

Use numerically stable logit-based BCE and define the Dice smoothing constant. Tune the boundary coefficient on source validation only, using a small declared grid such as `{0.1, 0.2, 0.5}`. Specify class weighting from training data, cap extreme weights, and handle all-background targets without division by zero. Empty masks must not generate artificial boundaries at image borders. Apply image and mask spatial augmentations together; compute boundary targets after spatial transforms.

Optional P2 work may examine topology-aware losses for continuity. Boundary accuracy and crack connectivity are different properties; do not claim one proves the other.

### 5.3 Training procedure

- Start at 256 × 256 for comparison with the paper; evaluate 512 × 512 only as a separate resolution experiment.
- Use synchronized flips, modest rotations, and photometric augmentation that preserves plausible road appearances. Use nearest-neighbor interpolation for discrete masks.
- Train with Adam or AdamW; freeze the choice, learning rate, batch size, schedule, early stopping, and maximum epochs after source validation.
- Initial planning defaults: batch size eight where memory allows, learning rate 0.001, maximum 100 epochs, early stopping patience ten. Record actual settings and effective batch size.
- Include genuine non-crack images where available. Avoid a training sampler that entirely hides hard negatives.
- Save the best source-validation checkpoint and a final run manifest. Log random seeds, dataset versions, code revision, library versions, hardware, training time, and memory usage.

### 5.4 Few-shot adaptation

For each target split, define support budgets `K ∈ {1, 5, 10}` images plus `K=0` as the source-only baseline. A shot is one fully annotated image at the declared image/patch resolution; record labeled pixel count and annotation time as well. Ten patches from one frame are not ten independent road scenes.

Default adaptation freezes the encoder and updates the decoder and prediction heads. Use a conservative initial learning rate of 0.0001 and a fixed budget, initially five support passes with a declared maximum optimizer-step cap. Resolve that cap using development data, then freeze it before testing. Compare full fine-tuning only as a separate P1 condition.

Use a source replay option in P1, initially mixing equal counts of support and sampled source images per adaptation batch where feasible. Replay data must come from source training only. Report additional compute and whether replay changes the effective labeled-data budget.

Select candidates using source validation and/or a separate target development set. If no target validation labels are available, use frozen adaptation settings and report that limitation. Never choose epoch, support set, or checkpoint from target query scores. A failed or worse candidate remains archived and must not automatically replace the current model.

### 5.5 Uncertainty selection (P1)

Compare random sampling with a declared uncertainty score based on predictive entropy, optionally restricted to a predicted-boundary band. Include an explicit fallback for empty foreground or empty boundary bands. Do not interpret confident all-background predictions as evidence of correct segmentation. Score eligible support-pool images only; query images remain outside acquisition. Report equal image budgets and observed annotation time, since harder images can take longer to label.

## 6. Data requirements and leakage controls

### 6.1 Candidate datasets

- Use the paper's source collections where accessible and appropriate: CFD, CRACK500, Cracktree200, DeepCrack, GAPS384.
- Use CrackMap as a target-domain benchmark, subject to verifying downloadable image/mask pairs and permitted use.
- Reserve a second independent target dataset for cross-domain evaluation if feasible. A collection reserved as a target must be removed from source training in that experimental track.
- Optional locally collected road images provide a demonstration set. They become evaluation evidence only after annotation, grouping, and split controls are established.

The authors' CrackMap repository was opened during PRD preparation [R2]; no dataset was downloaded, and actual file counts, annotation conventions, licenses, and sequence metadata have not been audited. The paper's reported counts are not verified local inventories.

### 6.2 Canonical dataset contract

Each record contains `image_id`, image path, mask path where labeled, dataset name/version, source frame or sequence ID where available, capture/site group where available, split, native dimensions, preprocessing version, and checksums. Store binary masks as lossless PNG with a declared mapping, for example 0 = background and 255 = crack. Preserve original files.

Validate paired dimensions, unreadable files, empty masks, unique IDs, duplicate images, mask values, and foreground ratios. Report exclusions rather than silently skipping samples. Trace patches back to their parent frame whenever metadata permits.

### 6.3 Splitting protocol

1. Audit and deduplicate before augmentation or tiling.
2. Split source training/validation/holdout by road, capture sequence, or parent frame where possible. All patches from a parent stay in one split.
3. Split the target into an acquisition/support pool and a fixed, hidden-label query set before sample selection. Use a separate target development pool only if its annotation cost is reported.
4. Create repeated nested support draws: the 1-shot set is contained in the 5-shot set, which is contained in the 10-shot set for a given draw. Keep the same query images across all K values and compared methods.
5. Store immutable manifests and seeds. No correction UI, adaptation loader, tuning script, or sample-selection routine may access query masks.

For a CrackMap-only pilot with 120 valid images, a provisional split is 30 acquisition images and 90 fixed query images. Final counts depend on valid grouping and the dataset audit. If group metadata is unavailable, use duplicate/perceptual-similarity checks and disclose that unseen-road independence cannot be guaranteed. Do not claim this custom split reproduces the paper's six/114 protocol.

A separate historical-comparison track may reproduce the paper's 5% correction procedure as closely as available information allows. Keep its result table separate from the stricter fixed-query experiments.

## 7. Evaluation and success criteria

### 7.1 Metrics and definitions

| Category | Required measurements |
|---|---|
| Region quality | Per-image Dice, IoU, precision, recall; dataset macro means; pooled confusion metrics as secondary values |
| Boundary quality | Boundary F1 with symmetric matching at a fixed two-pixel tolerance on 256 × 256 evaluation images; document boundary extraction and one-to-one/matching convention |
| Adaptation | Query-score change at K=1/5/10, score variance, source holdout forgetting, adaptation time |
| Annotation effort | Corrected image count, total labeled pixels, measured correction time when a reviewer participates |
| Efficiency | Trainable/total parameters, MACs with profiler convention, serialized size, inference peak memory, p50/p95 latency |
| Robustness | False positives on non-crack images, subgroup scores where metadata exists, declared synthetic corruption tests |

Use threshold 0.5 by default; a source-validation-tuned threshold is allowed if frozen before target evaluation. UI threshold changes must not alter benchmark metrics. Report metric averaging explicitly. For empty ground truth and empty prediction, set Dice/IoU to one; for exactly one empty mask, set them to zero. Report empty-mask cases separately to prevent inflated overall averages. Specify zero-denominator rules for precision, recall, and boundary metrics in the evaluator documentation.

For high-resolution P1 evaluation, declare a fixed boundary tolerance in the relevant coordinate system; do not quietly compare different tolerances across models or resolutions.

### 7.2 Experimental matrix

| ID | Model / condition | Question |
|---|---|---|
| E1 | Standard U-Net, source-only | Basic segmentation reference |
| E2 | Paper-inspired R2AU-Net, source-only and K=5 | Cost and adaptation reference |
| E3 | LiteCrack, region losses only | Effect of compact architecture |
| E4 | LiteCrack-BA, source-only | Isolate boundary supervision: E4 versus E3 |
| E5 | E4 at K=1/5/10 | Adaptation benefit and sensitivity |
| E6 (P1) | Random versus uncertainty-selected support | Annotation selection benefit |
| E7 (P1) | Adaptation with versus without source replay | Forgetting and compute tradeoff |
| E8 (P1) | Native runtime versus ONNX | Export parity and measured efficiency |

Run three training seeds for E1–E4 when feasible; at minimum run the main E3/E4 comparison with three seeds. For adaptation, use at least five support draws per evaluated checkpoint; reuse draws across compared methods. If compute forces a smaller matrix, record the reduction before viewing final query results and avoid broad significance claims.

Use paired image-level or capture-group bootstrap confidence intervals for model differences; prefer group resampling when group metadata exists. Report between-seed and between-support variation separately. Do not treat multiple seeds on the same image as independent new test images. Statistical tests, if used, require a declared comparison and correction for multiple comparisons.

### 7.3 Acceptance gates

| Gate | Target / requirement | Evidence |
|---|---|---|
| G1 Data integrity | No known group overlap or exact image duplicates across splits; all exclusions recorded | Audit report and frozen manifests |
| G2 Compactness | ≤5M trainable parameters and ≥50% fewer than implemented R2AU-Net | Parameter audit of complete models |
| G3 Accuracy tradeoff | Proposed static Dice within two percentage points of R2AU-Net on the fixed query protocol | Same training-data and initialization-policy comparison |
| G4 Boundary improvement | ≥2-point boundary F1 gain over E3; ≤1-point Dice loss | Paired E4/E3 ablation with uncertainty intervals |
| G5 Few-shot benefit | ≥2-point mean Dice gain at five shots over its own static model | Repeated support-draw results |
| G6 Retention (P1) | ≤2-point source holdout Dice decline | Source evaluation before/after adaptation |
| G7 CPU inference target | p95 ≤200 ms/model pass at 256 × 256, batch one | Named CPU/runtime/thread count; 20 warmups and ≥200 timed passes |
| G8 Portable export (P1) | FP32 ONNX/native max probability difference ≤0.001 on an audited sample set; metric difference ≤0.2 point | Parity report, runtime/version/opset |
| G9 Application | Full upload → correction → adaptation → comparison → export journey succeeds | Demonstration recording/screenshots and saved artifacts |
| G10 Reproducibility | A clean environment can execute documented data-audit, train, adapt, evaluate, and infer commands | README, locked environment, run manifests |

G7 measures model execution, excluding file IO and rendering; also report end-to-end image latency separately. Benchmark the original and compact models on identical hardware. The hardware is not yet specified, so latency remains provisional until the first profiling milestone. An unsuccessful research target must be marked unmet; functional delivery must not be relabeled as proof of superiority.

## 8. Proposed technical design

### 8.1 Components

`Local web UI → Python API → inference/adaptation service → versioned model and artifact store`

`Dataset audit → frozen manifests → training → adaptation → evaluation → experiment summaries`

Proposed stack: Python with PyTorch for learning, OpenCV/Pillow for image operations, FastAPI for the local API, React with an image-aligned canvas for correction, SQLite for metadata, and filesystem storage for images/checkpoints. ONNX Runtime is a P1 deployment option. These are design choices to validate in the first milestone; pin compatible versions during implementation.

Use one background worker for adaptation and a bounded inference queue. Keep CPU inference available without requiring the browser machine to train a model. A CLI is required independently of the interface so experiments remain scriptable and testable.

### 8.2 API contracts

| Endpoint | Purpose |
|---|---|
| `POST /projects` | Create a local project |
| `POST /projects/{id}/images` | Upload and validate images |
| `POST /predictions` | Run a chosen model on selected image IDs |
| `GET /predictions/{id}` | Retrieve artifact locations and metadata |
| `PUT /images/{id}/annotation` | Save a versioned correction and completeness status |
| `POST /adaptations` | Start a bounded job using a parent version and eligible support IDs |
| `GET /jobs/{id}` | Retrieve status, progress, and sanitized errors |
| `POST /jobs/{id}/cancel` | Cancel safely |
| `POST /models/{id}/activate` | Activate a validated candidate |
| `GET /exports/{project_id}` | Download a generated results bundle |

Validate project ownership and paths even in a single-user deployment. Never accept arbitrary server filesystem paths from the browser. Keep training configuration presets bounded rather than exposing executable commands.

### 8.3 Data entities

| Entity | Minimum fields |
|---|---|
| Project | ID, title, creation time, active model ID |
| Image | ID, project ID, checksum, dimensions, storage key, dataset/group/split |
| Annotation | ID, image ID, mask key, complete flag, revision, creation time |
| ModelVersion | ID, parent ID, architecture/config hash, checkpoint hash, validation summary, activation state |
| Prediction | ID, image ID, model ID, threshold, probability/mask/overlay keys, runtime, measurements |
| AdaptationJob | ID, parent version, support annotation IDs, seed, configuration, status, candidate ID, timestamps |
| ExperimentRun | ID, code/environment versions, data manifests, seed, artifacts, metric outputs |

### 8.4 Suggested repository layout

```text
configs/                # Source training, boundary, adaptation, profiling settings
data/manifests/         # Versioned split records; raw data kept outside Git
src/data/               # Audit, loaders, transforms, boundary targets
src/models/             # U-Net, paper-inspired R2AU-Net, LiteCrack-BA
src/training/           # Training and bounded adaptation
src/evaluation/         # Metrics, bootstrap, efficiency profiling
src/inference/          # Resizing, tiling, prediction, export
api/                    # Project, prediction, annotation, and job routes
web/                    # Inspection/correction application
scripts/                # CLI entry points
tests/                  # Meaningful correctness and integration tests
artifacts/              # Local runs and model outputs, excluded from Git
docs/                   # Protocol, model card, data audit, final results
```

## 9. Nonfunctional requirements

- **Reproducibility:** Each reported result resolves to a checkpoint, manifest, code revision, configuration, seed, and metric implementation.
- **Reliability:** Interrupted jobs do not corrupt completed models or annotations; saves use atomic replacement and explicit revision IDs.
- **Resource control:** Only one adaptation job runs at a time by default. Report memory errors with a recoverable message and retain the previous active model.
- **Usability:** Important states are explicit: unsaved correction, incomplete mask, job running, candidate available, and candidate rejected.
- **Local data handling:** Uploaded images remain local by default. Exports and logs must not leak arbitrary filesystem contents. Decode validation and upload limits are required.
- **Storage:** Keep raw images and original masks immutable. Define project cleanup behavior separately from model activation; never delete research evidence implicitly.
- **Accessibility:** Keyboard-accessible controls, readable labels, and overlays distinguishable without relying only on red/green.
- **Measurement honesty:** Pixel measurements cannot be exported as millimeters. Optional calibration requires a documented scale and camera geometry assumptions.

## 10. Validation plan

| Test | Expected result |
|---|---|
| Mismatched image/mask dimensions | Data audit rejects the pair with image ID and reason |
| Same source frame assigned to train and query | Manifest validation fails |
| Empty ground truth/prediction combinations | Evaluator follows documented conventions with finite outputs |
| Thin synthetic line and transformed mask | Boundary targets remain aligned and nonempty where expected |
| Annotation under zoom/pan | Saved pixels match the intended original-image coordinates |
| Incomplete correction selected as support | Adaptation request rejects it |
| Query annotation referenced by adaptation | Dataset/API guard rejects it |
| Cancelled or failed training job | Active model remains unchanged; partial version cannot activate |
| Export reopened | Mask dimensions, image IDs, hashes, threshold, and model IDs match the prediction record |
| Non-crack image | Prediction is handled normally; false positive area is logged |
| Tiled inference (P1) | Output size is correct; overlap and borders show no implementation-induced seams |
| ONNX export (P1) | Parity and metric-difference gates are checked on multiple representative inputs |

Complement tests with manually reviewed difficult cases: shadows, road markings, wet textures, fine cracks, intersections, and repaired seams where available. Synthetic corruption results must be labeled synthetic, not real-world deployment proof.

## 11. Delivery plan

Indicative schedule assumes one student developer with part-time review and accessible GPU time. Estimate: eight weeks for the required project; P1/P2 work depends on the core release.

| Milestone | Timing | Deliverables | Exit criterion |
|---|---|---|---|
| M1 Protocol and data | Week 1 | Dataset audit, source/target choice, frozen splits, environment, compute profile | Usable pairs and leakage checks pass; latency hardware named |
| M2 Baselines | Week 2 | U-Net and paper-inspired R2AU-Net, common evaluator | Source-only runs and cost measurements complete |
| M3 Compact model | Week 3 | LiteCrack and LiteCrack-BA, boundary targets/loss | Model shapes, parameter budget, training sanity verified |
| M4 Adaptation | Week 4 | K=1/5/10 pipeline, checkpoint versioning | Supports/query isolated; bounded runs complete |
| M5 Research evaluation | Week 5 | Core ablations, support draws, intervals, failure analysis | Main experiment matrix complete or declared reduced protocol frozen |
| M6 Inspection application | Week 6 | Upload, visualization, correction, job tracking | Full workflow works with a trained checkpoint |
| M7 Extended release | Week 7 | Prioritize uncertainty/replay, then tiling/export | Chosen P1 features validated individually |
| M8 Final package | Week 8 | Clean setup, demonstration, report, model card | Reproducibility and functional gates checked |

If time is constrained, retain E3/E4 boundary ablation, fixed-query adaptation, compactness measurements, and the correction workflow. Defer video, quantization, and topology loss first. Reduce optional baseline breadth before compromising leakage controls.

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Dataset download or license unclear | Training plan cannot proceed as proposed | Audit early; use confirmed permitted alternatives and record deviation |
| Patch/sequence leakage | Inflated accuracy | Parent-group split, duplicate audit, explicit residual limitations |
| Resizing removes fine cracks | Poor thin-crack recall | Inspect mask survival; use higher resolution or tiling as a separate experiment |
| Boundary loss amplifies noisy labels | Worse precision or Dice | Audit boundary targets, tune only on validation, retain region-only baseline |
| Few-shot overfitting | Target gain or source retention degrades | Freeze encoder, bound updates, retain parent checkpoint, evaluate replay |
| Small target dataset | Unstable conclusions | Paired comparisons, repeated draws, uncertainty intervals, narrower claims |
| Parameter reduction fails to improve CPU speed | Weak efficiency claim | Measure runtime early and profile operators on named hardware |
| Support selection favors expensive images | Image budget hides annotation burden | Track correction minutes and labeled pixels |
| Missing source grouping metadata | Cross-road generalization uncertain | Similarity audit and qualified claims; acquire better metadata if possible |
| Large experimental matrix | Incomplete final evidence | Run core ablations first; preregister any budget reduction |

## 13. Deliverables and definition of done

Required deliverables:

1. This PRD and a frozen experimental protocol.
2. Dataset audit, manifest schema, split manifests, and preprocessing documentation.
3. Training, adaptation, inference, evaluation, and profiling CLI commands.
4. Baseline and proposed-model implementations with run configurations and retained checkpoints.
5. Per-image results, aggregated metrics, uncertainty intervals, efficiency results, and ablation tables.
6. Local inspection application with correction, adaptation job tracking, model comparison, rollback, and exports.
7. Setup guide, model card, limitations, and a complete demonstration using actual trained outputs.
8. Final report separating paper-reported results, locally measured results, unmet targets, and future work.

The project is functionally complete when the reproducible pipeline and full application journey pass their acceptance tests. A claim of improved boundaries, successful few-shot adaptation, or better efficiency requires its corresponding experimental gate and supporting evidence. A polished interface or successful training run alone is insufficient.

## 14. Decisions to resolve during implementation

These do not block the PRD, but must be resolved before freezing experiments:

- Actual GPU/CPU availability, training-hour budget, and demonstration machine.
- Permitted accessible datasets, native annotation formats, and source grouping metadata.
- Final source/target allocations and whether a second target domain is feasible.
- Paper architecture details that require reconstruction and the exact limits of reproducibility.
- Whether pretrained initialization belongs in the principal track or only a separate comparison.
- Which P1 features fit the deadline after the core results are available.

## 15. References and provenance

- **[R1] Attached reference paper:** `data/references/2303.01582v1.pdf`, Katsamenis et al., *A Few-Shot Attention Recurrent Residual U-Net for Crack Segmentation*. Full PDF text reviewed for this PRD; Sections 2–3 and Table 2 support the paper summary. [Versioned paper](https://arxiv.org/abs/2303.01582v1).
- **[R2] Authors' dataset repository:** [ikatsamenis/CrackMap](https://github.com/ikatsamenis/CrackMap). Repository page opened on 23 October 2026; dataset contents and licenses still require implementation-stage audit.
- **[R3] Efficient architecture reference:** Howard et al., [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244). Supports the choice of an efficient encoder candidate, not a claim of demonstrated crack-segmentation performance.

This PRD contains a proposed system and evaluation plan. No training, dataset inventory, deployment benchmark, application implementation, or claimed improvement has been completed as part of preparing this document.
