# CrackLab

For the current source and demo submission, start with [SUBMISSION_README.md](SUBMISSION_README.md).

Lightweight road-crack segmentation research with a React inspection studio,
pixel-aligned editing, recoverable few-shot adaptation and model comparison.
**Phase 8: versioned pilot release, 1 November 2026.** A later multi-source
experiment finished on 17 November 2026; it is available in this working tree
but is not part of the Phase 8 release ZIP.

## Start

Extract `CrackLab_Phase8_v1.0.0.zip` into a new folder. With Python 3.12 installed:

```powershell
py -3.12 scripts/verify_release.py
powershell -ExecutionPolicy Bypass -File .\Setup_CrackLab.ps1
.\Start_CrackLab.cmd
```

Open http://127.0.0.1:8765. Setup installs pinned dependencies over the network,
imports 32 verified checkpoint references and prepares the bundled replay buffer.
For the optional Groq vision suggestion demo, stop the running app and start
`powershell -ExecutionPolicy Bypass -File .\Start_CrackLab_Groq.ps1` in a new
terminal. The launcher prompts for the API key without saving it in the project.
Upload an image and select **AI find cracks**. You can also run a prediction
first, in which case the vision model looks for cracks missed by that mask.
The Groq vision model adds approximate centerline paths to an unsaved correction;
review them with the brush and eraser before marking the mask fully corrected.
The image and current prediction mask are sent to Groq for this action.
The release includes compiled React; Node is needed for frontend development.
Existing users can keep their current workspace and launcher. See
[START_HERE.md](START_HERE.md) and [setup instructions](docs/SETUP.md).

## Evidence and handoff

### Multi-source follow-up (17 November 2026)

Six audited archives supplied 3,564 image/mask pairs (2,526 training, 542
validation, 496 publisher test). Validation selected a 512 px LiteCrack model,
seed 42, epoch 60, at threshold 0.5 before the test split was evaluated.
The official publisher test returned **Dice 0.2873**, IoU **0.1972**, precision
**0.3565**, recall **0.2702**, and boundary F1 **0.3768** (per-image macro means).

| Publisher test source | Images | Dice |
| --- | ---: | ---: |
| Ground GoPro | 62 | 0.6936 |
| Ground iPhone | 50 | 0.7198 |
| UAV 3–10 m | 65 | 0.2167 |
| UAV 15–30 m | 58 | 0.2085 |
| UAV 35–100 m | 210 | 0.1424 |
| Other UAV | 51 | 0.1459 |

The ground and UAV results differ substantially. These scores belong to the
separate six-archive dataset and must not be compared as if they used the Phase 8
DeepCrack/CrackMap test split. The model is registered as a selectable CrackLab
reference in this working tree; existing project defaults are unchanged.
See the [final test report](output/multisource_test_v1/FINAL_TEST_REPORT.md),
[frozen selection](output/multisource_test_v1/FROZEN_SELECTION.json), and
[model card](MODEL_CARD.md). The [current app walkthrough](docs/MULTISOURCE_APP_DEMO.md)
records upload, prediction, draft correction, existing-reference import,
comparison and verified export without training. No further model training is
planned for this handoff.

- [Original Phase 8 PDF report](output/pdf/CrackLab_Final_Report.pdf),
  [updated Markdown report with the multi-source addendum](docs/FINAL_REPORT.md),
  and [updated model card](MODEL_CARD.md).
- [Phase 8 delivery](docs/PHASE_8.md), [demo guide](docs/DEMO_GUIDE.md), and [recorded walkthrough](output/phase8/demo/walkthrough.webm).
- [Consolidated evidence](output/phase8/evidence.json), [clean-install verification](output/phase8/release_validation.json), and [fresh tests](output/phase8-clean-tests.xml).
- [Phase 6 application](docs/PHASE_6.md) and [Phase 7 extensions](docs/PHASE_7.md).

The fresh extracted application passed **49 tests; six CUDA-only tests skipped**.
The recorded real-acquisition-image workflow passed with zero JavaScript errors.
Reference masks are existing dataset annotations; correction time was not measured.

## Verified research results

| Model | Parameters | Source-holdout Dice | Target-query Dice |
| --- | ---: | ---: | ---: |
| U-Net | 1,965,569 | 81.32% | 24.60% |
| Paper-inspired R2AU | 2,110,029 | 81.42% | 34.52% |
| LiteCrack | 339,265 | 76.36% | 34.01% |
| LiteCrack-BA | 339,282 | 75.40% | 41.72% |

Boundary-aware five-/ten-shot target Dice is 57.23%/59.09% averaged over five
support draws. Source-holdout Dice decreases 2.81/4.87 points, exceeding the
two-point retention target. Phase 7 replay reduced forgetting in a short pilot
but still failed its source-subset gate; those candidates remain rejected.

These are single-seed exploratory results on a previously inspected query set.
The wide road photo still produces substantial vegetation false positives.
Multi-seed confirmation, physical site independence, independent target testing,
measured correction effort and full GPU retraining from the release remain open.
The application release does not establish field readiness.

[Submission layout](docs/SUBMISSION_LAYOUT.md) maps the code, evidence, archives and generated files.

## Source and reproduction

Milano dataset preparation and its separate GPU experiment are complete:
[Milano training handoff](docs/MILANO_TRAINING.md) and
[final single-seed report](output/milano_v1/FINAL_REPORT.md). All 2,209 pairs have
normalized binary masks and verified hashes. The publisher-split test result is
exploratory: 17 high-similarity cross-split candidates need source-backed review,
and physical-site independence is unknown. Use the
[candidate adjudication register](output/milano_v1/milano_similarity_adjudication.csv)
and [adjudication instructions](output/milano_v1/SIMILARITY_ADJUDICATION_GUIDE.md),
plus the [independent-scene collection guide](data/followup_external/COLLECTION_GUIDE.md)
to continue validation. This experiment does not establish vegetation-failure resolution.
The [multi-source external dataset workflow](docs/MULTISOURCE_DATASET_WORKFLOW.md)
contains the verified split audit; the [handoff runbook](docs/MULTISOURCE_DATASET_HANDOFF.md)
records the completed training and test evaluation. Older queue and download
entries in that runbook are historical.

The [follow-up workflow](docs/FOLLOWUP_IMPLEMENTATION.md) implements retention
search, multi-seed retraining, independent-scene checks and real annotation timers.
The downloaded GPU retention matrix is [verified](output/followup_v1/downloaded_audit/RESULTS.md):
90 adaptations and all 18 CPU/CUDA checks completed. Three-seed retraining and
independent scene/human studies remain pending. Its notebook is
`output/kaggle/Road_Crack_Followup_Retention.ipynb`; completed Phase 8 results stay frozen.

The optional historical GPU follow-up is documented in
`output/kaggle/Road_Crack_Followup_Multiseed.ipynb` and
[next-run instructions](docs/NEXT_GPU_RUN.md). It is outside the current
no-retraining handoff scope.

`inspection/` holds the API, store and inference services; `web/` contains React/Vite;
`src/` contains research models/training. Frozen manifests and model registries
are under `data/manifests/`; GPU notebooks are under `output/kaggle/`.
Run `Setup_CrackLab.ps1 -RunTests` for local tests. Report authoring dependencies
are separate from `requirements-release.lock.txt`. See [setup](docs/SETUP.md).

Code and third-party data terms are separate: see [NOTICE.md](NOTICE.md).
This is a local research/education handoff; public data redistribution permission
is not asserted. Earlier states remain in [README history](docs/README_HISTORY_PRE_PHASE8.md).
