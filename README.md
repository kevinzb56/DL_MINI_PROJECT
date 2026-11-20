# CrackLab

CrackLab is a road-crack segmentation research project and local inspection application. It combines compact neural networks, boundary-aware training, limited-label adaptation, and a React/FastAPI studio for inspecting, correcting, comparing, and exporting predictions. This is a research and education pilot, not a field-ready road survey system.

## Run the application

On Windows, install Python 3.12, open PowerShell in this workspace, and run:

```powershell
cd code
powershell -ExecutionPolicy Bypass -File .\Setup_CrackLab.ps1
.\Start_CrackLab.cmd
```

Open <http://127.0.0.1:8765>. Setup installs pinned dependencies and prepares bundled model references; the first install requires internet. The compiled React interface does not need Node to run. For a fresh extracted submission ZIP, verify it first with `py -3.12 scripts/verify_release.py`. See the [submission instructions](code/SUBMISSION_README.md) and [setup guide](code/docs/SETUP.md). Frontend development requires Node 22.12+; see [web/README.md](code/web/README.md).

## Project components

| Component | Purpose |
| --- | --- |
| LiteCrack | Compact region segmentation with a MobileNetV3-Small encoder and separable decoder. |
| LiteCrack-BA | Adds an auxiliary boundary head and boundary loss. |
| U-Net and paper-inspired R2AU | Frozen research baselines. |
| Inspection studio | Local upload, prediction, original-resolution mask editing, reference comparison, and export. |
| Adaptation workflow | Bounded decoder-only few-shot jobs, distinct support and validation images, recovery snapshots, validation gates, activation and rollback. |
| Multi-source follow-up | Separately trained 512 px LiteCrack model tested on six ground/UAV sources. |

The frontend uses React/Vite; the backend uses FastAPI, SQLite metadata, filesystem snapshots, and local inference workers. The app includes tiled and ONNX inference, model comparison, entropy/random image suggestions, and optional Groq-assisted suggestions. The Groq action sends the selected image and current mask to Groq and requires a separately supplied API key. See the [model card](code/MODEL_CARD.md) and [technical report](code/docs/FINAL_REPORT.md).

## Data and evaluation

The original pilot trains on DeepCrack and evaluates transfer to CrackMap. Its frozen 657-pair manifest contains 240 source-training, 60 source-validation, 237 source-holdout, 25 target-acquisition, and 95 target-query images. Models use 256 × 256 RGB inputs and a fixed 0.5 evaluation threshold. Target-acquisition masks supply few-shot support; target-query masks supply evaluation. The main comparison uses one training seed, and the query set had been inspected in earlier phases.

The later multi-source study is a **separate experiment**. Six audited archives provide 2,526 training, 542 validation, and 496 publisher-test image/mask pairs. Validation selected a 512 px LiteCrack checkpoint (seed 42, epoch 60, step 18,960) before the test split was evaluated. Its scores must not be directly compared with the DeepCrack-to-CrackMap pilot. See the [split workflow](code/docs/MULTISOURCE_DATASET_WORKFLOW.md), [frozen selection](code/output/multisource_test_v1/FROZEN_SELECTION.json), and [final test report](code/output/multisource_test_v1/FINAL_TEST_REPORT.md).

## Results: DeepCrack to CrackMap

Per-image macro means at threshold 0.5 on the frozen source holdout (237 images) and target query (95 images):

| Model | Parameters | Source Dice | Target Dice | Target IoU |
| --- | ---: | ---: | ---: | ---: |
| U-Net | 1,965,569 | 81.32% | 24.60% | 15.75% |
| Paper-inspired R2AU | 2,110,029 | 81.42% | 34.52% | 22.92% |
| LiteCrack | 339,265 | 76.36% | 34.01% | 22.65% |
| LiteCrack-BA | 339,282 | 75.40% | 41.72% | 28.91% |

LiteCrack has 83.92% fewer parameters than the implemented R2AU baseline. Boundary supervision raises target-query Dice by 7.71 percentage points relative to LiteCrack, while source-holdout Dice falls by about 0.97 point. These are exploratory single-seed results, not an exact reproduction of the reference paper.

Thirty fixed-budget adaptations were run: two parent models × five support draws × 1, 5, or 10 labeled target images. Each candidate started from its unchanged parent. The boundary-aware model reached **48.00% ± 6.90**, **57.23% ± 2.00**, and **59.09% ± 1.76** target Dice for 1, 5, and 10 shots respectively (mean ± sample SD across draws). Its source Dice decreased by 2.81 points at 5 shots and 4.87 points at 10 shots, failing the optional two-point retention target. Existing dataset masks were used; human correction time was not measured. The [final report](code/docs/FINAL_REPORT.md) includes boundary metrics, descriptive bootstrap intervals, and limitations.

## Results: separate six-source test

The validation-frozen 512 px LiteCrack model returned **Dice 0.287325**, **IoU 0.197227**, **precision 0.356516**, **recall 0.270186**, and **boundary F1 0.376827** on 496 publisher-test images. Values are per-image macro means at threshold 0.5.

| Test source | Images | Dice | IoU |
| --- | ---: | ---: | ---: |
| Ground GoPro | 62 | 0.693625 | 0.534673 |
| Ground iPhone | 50 | 0.719845 | 0.565234 |
| UAV 3–10 m | 65 | 0.216736 | 0.129471 |
| UAV 15–30 m | 58 | 0.208454 | 0.120334 |
| UAV 35–100 m | 210 | 0.142357 | 0.080253 |
| Other UAV | 51 | 0.145943 | 0.081666 |

Ground-camera performance is much stronger than UAV performance; the 210-image high-altitude group has recall 0.118514. This checkpoint is registered in the current application as a selectable reference and does not replace an existing project's active model. The [test report](code/output/multisource_test_v1/FINAL_TEST_REPORT.md) records artifact verification and checkpoint identity. A [recorded app walkthrough](code/docs/MULTISOURCE_APP_DEMO.md) covers upload, prediction, draft correction, reference import, comparison, and verified export. That walkthrough did not retrain or activate a model and is not an accuracy benchmark.

## Other completed experiment

A separate Milano pavement dataset study used publisher splits of 1,480/498/231 images. On its 231-image test split, LiteCrack scored **49.99% Dice / 39.55% IoU**, and LiteCrack-BA scored **51.12% Dice / 40.86% IoU**. The boundary-aware model's aggregate Dice gain is concentrated in empty-mask images; on the 186 nonempty-mask images, its Dice is **55.42%** versus **57.25%** for LiteCrack. Seventeen high-similarity cross-split pairs need provenance review, so these single-seed results remain exploratory. See the [Milano report](code/output/milano_v1/FINAL_REPORT.md).

## Use the studio

1. Create or open a local project and upload road images.
2. Select a model and inference settings, run prediction, and inspect the overlay and mask.
3. Correct the mask with brush/eraser and undo/redo. Save an incomplete draft or mark a genuinely complete reference; identify imported existing masks as such.
4. Compare models on the same image and settings. For adaptation, select distinct support and local-validation images, run a bounded job, inspect target/source gates, and explicitly activate or roll back a candidate.
5. Export the project and recovery data. Keep exports when moving installations.

The [Phase 8 recorded demo](code/docs/DEMO_GUIDE.md) shows adaptation and rollback. The [multi-source demo](code/docs/MULTISOURCE_APP_DEMO.md) shows the current 512 px model without adaptation. Demo images use existing annotations and do not measure expert annotation effort.

## Repository layout and verification

| Folder | Contents |
| --- | --- |
| [code/](code/) | Maintained application and research source, scripts, configs, tests, reports, model card, and release instructions. |
| [data/](data/) | Downloaded datasets, manifests, and demo/replay inputs. |
| [results/](results/) | Experiment outputs, model/recovery archives, reports, and submission ZIPs. |

Within `code/`, `inspection/` contains the API and inference services, `web/` the React UI, `src/` the research models, `scripts/` the preparation/training/evaluation tools, and `tests/` the automated checks. `code/data`, `code/output`, `code/artifacts`, and `code/release_assets` are Windows junctions into `data/` and `results/`; they preserve historical relative paths without duplicating files. See the [submission layout](code/docs/SUBMISSION_LAYOUT.md) and [NOTICE](code/NOTICE.md).

From `code/`, run the local Python suite after setup with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

The fresh extracted Phase 8 release recorded **49 passed tests and six CUDA-only skips**, plus a real-acquisition-image browser workflow with zero JavaScript errors. That is release evidence, not a claim that the current workspace was freshly retested. See [release validation](code/output/phase8/release_validation.json). Full GPU training requires the original permitted datasets and compatible environment; the submission includes selected trained models and evidence rather than all source data. The [reproduction guide](code/docs/SETUP.md) and [project README](code/README.md) give the notebook and follow-up paths.

## Limits and next work

The main pilot has one training seed, a previously inspected target query, unverified physical-site independence, source retention failures after adaptation, and no measured human correction effort. A broad-road photograph still yields vegetation false positives; it has no complete reference mask for a numerical score. The six-source model performs poorly on distant UAV images. Milano has possible adjacent-frame split overlap. Stronger claims require independent site-separated images with complete masks and explicit hard negatives, repeated seeds, and measured correction effort. The [model card](code/MODEL_CARD.md) and [final report](code/docs/FINAL_REPORT.md) document these boundaries in detail.
