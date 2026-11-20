# CrackLab submission

This archive contains the current CrackLab source, compiled local interface, selected multi-source model, and key research evidence. The older `CrackLab_Phase8_v1.0.0.zip` is a separate historical pilot release.

## Run on Windows

1. Extract the ZIP into a new directory.
2. Install Python 3.12 and open PowerShell in that directory.
3. Run `py -3.12 scripts/verify_release.py` to verify every packaged file.
4. Run `powershell -ExecutionPolicy Bypass -File .\Setup_CrackLab.ps1` to create the environment and import the bundled models. The first dependency install needs internet.
5. Run `.\Start_CrackLab.cmd` and open `http://127.0.0.1:8765`.

The setup imports 33 model references, including the validation-frozen 512 px multi-source model. It does not download the external training datasets. To run the included Python tests after setup, use `.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`.

## What to review

- [Current multi-source final test](output/multisource_test_v1/FINAL_TEST_REPORT.md) and [frozen selection](output/multisource_test_v1/FROZEN_SELECTION.json)
- [Updated research report](docs/FINAL_REPORT.md), [original Phase 8 PDF](output/pdf/CrackLab_Final_Report.pdf), and [model card](MODEL_CARD.md)
- [Source layout](docs/SUBMISSION_LAYOUT.md), [setup details](docs/SETUP.md), and [demo guide](docs/DEMO_GUIDE.md)

The multi-source publisher test Dice is 0.2873 across 496 images. Ground-camera and UAV performance differ substantially; see the source breakdown in the final test report. The package is a research and education demo, not a field-ready road survey system.

Downloaded datasets, local workspaces, recovery archives, virtual environments, `node_modules`, caches, and intermediate outputs are excluded. The ZIP has a SHA-256 sidecar and an internal `RELEASE_MANIFEST.json` with per-file hashes.
