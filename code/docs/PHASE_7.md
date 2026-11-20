# Phase 7: extended release

Implemented and validated on 1 November 2026. The frozen Phase 1–5 experiment
archives, original checkpoints, active user model, and Phase 6 worker/engine were
preserved. New capabilities are additive; they do not establish that broad road
scenes or the user's tree false positives are solved.

## Using the features

Start/restart `Start_CrackLab.cmd`, then open http://127.0.0.1:8765.

### Overlapping tiled inference

In **Inspection studio**, select **Image sampling → Overlapping tiles**.
The default **Whole-image resize** keeps the old preprocessing. Tiling uses
256-pixel model inputs at original spatial scale, 64-pixel overlap, edge padding,
and Hann blending (1-D weights floored at 0.05). Probabilities are blended before
thresholding and cropped back to exactly the original dimensions. Maximum 256
tiles per image bounds local work. Comparison uses the same mode for both models.
All tile/padding/runtime settings are recorded with each prediction and export.

Tiling preserves detail but is a separate inference pipeline. It does not segment
the road region or automatically suppress non-road false positives. On the user's
480×640 road photo it uses nine tiles; the archived whole-image and tiled overlays
are `output/phase7/wide_resize_overlay.png` and `wide_tiled_overlay.png`. Tree,
sky-edge and lamp false positives remain visible. No accuracy score is reported
because this photograph has no ground truth.

### Uncertainty suggestions

In **Adaptation lab**, use **Rank correction pool**. Predictive entropy computes
mean binary entropy in bits from original-size bilinear probabilities. The
seeded-random policy (seed 7301) is an explicit comparator. The eligible pool is
incomplete/unlabeled project images, excluding complete corrections, frozen
evaluation matches, and images already used as job validation. Up to 100 images
can be ranked in one project. Ties use stable image IDs. Policies, checkpoint
identity, seed, scores and pool membership are saved and included in exports.

Scores are diagnostic uncertainty, not calibrated confidence or guaranteed error.
A confident all-background model can still be wrong. Suggestions never select
supports automatically, and the ranking code does not load query or correction
masks. Real equal-budget annotation-efficiency research remains a separate E6
study; no human annotation time was measured here.

### Source replay

The user workspace is already prepared with 16 verified source-training pairs and
eight separate source-validation pairs. For another workspace, run:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_source_replay.py --workspace artifacts/phase6/workspace
```

The helper fetches the pinned DeepCrack archive when needed, retains it locally,
and checks each selected file against the frozen manifest. Only source-training
images enter replay. Source validation and source holdout never enter optimizer
updates. The selection is fixed by seed 7301; it is not chosen from experiment
results. An existing verified buffer is reused. A mismatched buffer is rejected.

Enable **Mix in source replay** when creating an adaptation job. Every support
image is mixed with one source image in the same batch: up to two support plus
two replay examples. The number of support passes and optimizer steps remains
fixed, while examples and compute increase. Encoder frozen, Adam 0.0001, clipping
1.0, BCE+Dice plus the existing boundary objective, no augmentation. Source order
is deterministic from seed and optimizer step. Recovery records replay IDs,
optimizer, sampler position, RNG, code/configuration and data identities at every
step. Existing Phase 6 jobs still use the unchanged Phase 6 worker.

Both the local target-validation gate and the source-validation-subset gate must
pass (at most a two-point Dice loss), alongside exact encoder preservation. The
source subset is **not the full source holdout**. Rejected candidates remain
available for inspection but cannot become the active model.

### ONNX export and inference

Choose a model in the studio and click **Export & verify ONNX**. The export is FP32,
opset 17, batch one, fixed model input dimensions. Input is RGB/255 NCHW; output
is region probability. The boundary auxiliary output is omitted from the graph.
No arbitrary uploaded checkpoint is accepted.

The exporter checks graph validity, finite output and maximum probability error
<=0.001 against native PyTorch on the eight audited source-validation images,
up to three project images, plus black/white/noise diagnostics. For labeled audited
images it also requires per-image Dice and IoU differences <=0.002. Missing audited
data are explicitly reported as an unassessed dataset-metric gate. A failed
probability/metric gate never registers an available export.

Download the ZIP containing `model.onnx`, `parity.json`, and the input/output
contract. Its checksum is checked again before download and runtime use. Switch
**Runtime → ONNX Runtime** to use the verified graph; tiled ONNX is supported too.
For ONNX comparison, first export both selected models. Exports are project-scoped.
They do not activate a model or update research metrics.

Install dependencies on a clean checkout with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase7.txt
cd web
npm.cmd ci
npm.cmd run build
```

The existing application requirements now include ONNX and ONNX Runtime. The
complete tested environment is recorded in `requirements-phase7.lock.txt` and
`output/phase7/environment.json`.

## Verified results

### Functional and browser evidence

- **11 tests passed**: tiled identity/edge coverage, size/budget checks, entropy,
  ranking eligibility and deterministic random ordering, source-selection split
  isolation, ONNX parity/runtime/download/project ownership/corruption/failure
  rejection, exact replay resume, source-validation exclusion from optimizer
  batches, frozen encoder, and all five existing Phase 6 API/worker tests.
- The React build passed. Chromium exercised entropy and random ranking, tiled
  native and ONNX inference, audited ONNX export/download, brush/undo/redo and
  correction saves, a real replay subprocess, validation rejection, comparison,
  project export, reload persistence and mobile overflow checks. Zero JavaScript
  errors. Those two browser image fixtures are synthetic and explicitly labeled.
- Real-data validation is isolated in `artifacts/phase7/validation-workspace`;
  browser fixtures use another workspace. The user workspace is not populated
  with smoke-test candidates or fixture images.

### Real checkpoint ONNX parity

| Static checkpoint | Maximum probability difference | Audited Dice/IoU difference gate |
|---|---:|---|
| LiteCrack | 0.00037462 | Passed on eight source-validation images |
| LiteCrack-BA | 0.00012633 | Passed on eight source-validation images |

These checks establish numerical export agreement on the audited sample set, not
road-scene accuracy. Reports include all sample IDs, model/graph hashes, and runtime
versions under `output/phase7/onnx_phase3_parity.json` and `onnx_phase4_parity.json`.

### Paired replay functional pilot

Protocol fixed before validation: draw 0, K=5, one support pass, seed 42, five
different remaining acquisition images as local target validation; the same eight
source-validation images assess retention for both conditions. This uses existing
public masks, not measured human corrections. No query/holdout labels were used.

| Parent | No replay: local target Dice | Replay: local target Dice | No replay: source Dice change | Replay: source Dice change |
|---|---:|---:|---:|---:|
| LiteCrack | 57.35% | 56.66% | −7.74 pts | −4.44 pts |
| LiteCrack-BA | 62.47% | 61.96% | −7.44 pts | −4.68 pts |

Replay reduced forgetting in this small pilot but still failed the two-point
source-retention gate. Both replay candidates were rejected and not activated.
This is not a full E7 result, a multi-draw conclusion, or a replacement for the
Phase 5 results. No claim of adequate retention follows from implementing replay.

### CPU model-pass benchmark

13th Gen Intel Core i5-1335U, Windows, PyTorch 2.15.0+cpu, FP32, batch one,
256×256, two threads, 20 warmups and 200 timed model passes per model:

| Model | p50 | p95 | p95 <=200 ms target |
|---|---:|---:|---|
| LiteCrack | 33.60 ms | 44.88 ms | Passed on this machine |
| LiteCrack-BA | 34.67 ms | 52.59 ms | Passed on this machine |

Model loading, preprocessing, tiling and rendering are excluded from this benchmark.
The UI reports a different per-request latency. These numbers are not a speed
comparison against U-Net/R2AU on the same CPU and are not ONNX speed claims.

## Artifacts and reproduction

- Fixed protocol: `docs/PHASE_7_PROTOCOL.md`, `configs/phase7.json`.
- Full real-data report and per-image checks: `output/phase7/real_validation.json`.
- Paired jobs: `output/phase7/real_replay_jobs.json` and the retained validation
  workspace's original snapshots. No-replay source-subset assessments are added
  to the report by `scripts/evaluation/evaluate_phase7_retention.py`, without changing gates.
- Tests: `output/phase7-tests.xml`; browser evidence: `output/phase7/browser/`.
- `phase7_results.zip` retains reports, source code/configuration, manifests,
  source/target job snapshots, parent identities, and latest recovery checkpoints.
  Preserve the original Phase 3–5 ZIPs too; imported model references can be
  reconstructed from them. The SHA256 sidecar identifies this local backup.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase7.py tests/test_inspection.py -q -p no:cacheprovider --basetemp=artifacts/phase7/fresh-test-temp
.\.venv\Scripts\python.exe scripts/evaluation/validate_phase7.py
```

The real-data validator creates a separate validation project, downloads only the
fixed acquisition files it needs, verifies hashes, runs four short paired jobs,
exports both static models, and measures CPU latency. Re-running creates another
isolated validation project; it does not overwrite completed job checkpoints.

Later work: a full equal-budget uncertainty/random study with annotation effort,
multi-draw replay evaluation, full source-holdout retention confirmation, and
proper road/background handling for broad-scene uploads. Phase 8 is the final
documentation, reproducibility and demonstration package.
