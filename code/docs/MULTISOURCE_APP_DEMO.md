# CrackLab multi-source app walkthrough

**Recorded:** 17 November 2026, in a separate local workspace on port 8771. This walkthrough exercises the working-tree app with the frozen 512 px LiteCrack checkpoint. It performs no training or adaptation and does not change the user's existing inspection project.

## Watch and inspect

- [Walkthrough video](../output/multisource_demo/walkthrough.webm)
- [Upload](../output/multisource_demo/01-uploaded.png), [frozen-model prediction](../output/multisource_demo/02-frozen-model-prediction.png), [draft correction](../output/multisource_demo/03-draft-correction.png), [reference agreement](../output/multisource_demo/04-reference-agreement.png), and [side-by-side comparison](../output/multisource_demo/05-side-by-side-comparison.png)
- [Machine-readable run record](../output/multisource_demo/demo.json) and [verified project export](../output/multisource_demo/project-export.zip)

## What passed

1. Uploaded two real CrackMap target-acquisition images from `release_assets/demo/`; each image and its existing reference mask matched the recorded SHA256 in `manifest.json`.
2. Selected `LiteCrack · multi-source 512 px`, confirmed whole-image resize and threshold 0.5, and ran a local CPU prediction.
3. Started a correction from the prediction, used brush/undo/redo, and saved it as an **incomplete draft**.
4. Imported the existing dataset mask as a separate complete reference, reloaded the page, and checked that the app displayed prediction agreement. This was not a timed human annotation.
5. Compared the frozen multi-source model with the older LiteCrack-BA base on the same image and UI settings. The comparison is a known acquisition-image illustration, not an independent benchmark or a reason to change the frozen official test score.
6. Exported the project and verified the ZIP, two images, prediction model identities and complete reference. Reloading retained both images. The run recorded zero JavaScript page errors.

The export contains two images, four predictions, two corrections and zero adaptation jobs. No model was activated or retrained. The official 496-image result and its limitations remain in the [final test report](../output/multisource_test_v1/FINAL_TEST_REPORT.md).

The repeatable automation is `scripts/evaluation/record_multisource_demo.py`. It uses an isolated server on port 8771, a disposable workspace under `artifacts/multisource_demo_workspace`, Playwright with installed Microsoft Edge, and the existing demo assets. Keep the demo workspace separate from a personal inspection project.
