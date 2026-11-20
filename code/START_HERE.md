# Start here

CrackLab 1.0.0-pilot is the Phase 8 local research/education application release.

1. Extract `CrackLab_Phase8_v1.0.0.zip` into a new folder. Preserve the ZIP and its
   `.sha256` sidecar. Do not extract over a newer user workspace.
2. Install Windows Python 3.12, including the `py` launcher. Open PowerShell in the
   extracted directory and run `py -3.12 scripts/verify_release.py`.
3. Run `powershell -ExecutionPolicy Bypass -File .\Setup_CrackLab.ps1`.
   Initial dependency installation needs internet. The compiled React interface
   and trained models are bundled; no Kaggle connection is needed.
4. Run `Start_CrackLab.cmd` and visit http://127.0.0.1:8765.
5. Follow [the demo guide](docs/DEMO_GUIDE.md). Preserve project/recovery exports
   before moving your work between machines.

Read the [final report](output/pdf/CrackLab_Final_Report.pdf),
[model card](MODEL_CARD.md) and [setup guide](docs/SETUP.md).
The known wide-road photo failure remains documented. A successful demo or local
candidate gate does not establish reliable segmentation of arbitrary road scenes.
