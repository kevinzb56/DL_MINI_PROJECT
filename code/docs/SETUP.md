# Installation and reproduction

## Versioned Windows release

Use a fresh directory and Windows Python 3.12 with the `py` launcher. Network is
needed for initial pip installation; the bundle is not an offline wheelhouse.

```powershell
py -3.12 scripts/verify_release.py
powershell -ExecutionPolicy Bypass -File .\Setup_CrackLab.ps1 -RunTests
.\Start_CrackLab.cmd
```

To verify an unextracted archive:

```powershell
py -3.12 scripts/verify_release.py --archive .\CrackLab_Phase8_v1.0.0.zip
```

The stdlib verifier rejects unsafe/duplicate archive paths, missing files and
size/hash mismatches. It also checks fixed checkpoint-archive identities. An
embedded manifest is an integrity record, not a digital publisher signature.
Preserve the supplied external `.sha256` sidecar as an independent identity record.

Setup creates `.venv`, installs `requirements-release.lock.txt`, imports original
Phase 3/4/5 checkpoints and copies the bundled 16 source-training/eight separate
source-validation replay pairs. Existing changed assets are refused. Compiled
React is bundled, so no Node rebuild is needed. The launcher uses source-content
fingerprints instead of ZIP-extraction timestamps. Open http://127.0.0.1:8765.

The main workspace lives in `artifacts/phase6/workspace`. Preserve project/recovery
exports before moving work. Do not extract releases over an existing installation.
For a separate test port:

```powershell
.\.venv\Scripts\python.exe scripts/run_inspection.py --port 8769
```

Another port alone does not isolate storage. Use a fresh extracted directory for
the automated demonstration.

## Source checkout and frontend development

The checkout needs the three original result ZIPs at its root. Bundled replay
assets can be prepared with `scripts/data/prepare_release_assets.py` when verified
Phase 7 inputs are present; otherwise `prepare_source_replay.py` needs the original
permitted data. Setup has a source-checkout preparation fallback.

Use a compatible Node version (22.12+ recommended) for frontend development:

```powershell
npm.cmd --prefix web ci
npm.cmd --prefix web run build
```

The production build records its source fingerprint in `web/dist/build_inputs.json`.
Changing source requires rebuilding. See `web/README.md` for Vite development.

## Validation and authoring

The RC archive was extracted and installed into a new venv: 49 tests passed, six
CUDA-only tests skipped, and all 46 pinned runtime distributions matched. See
`output/phase8/release_validation.json` and `output/phase8-clean-tests.xml`.
Final documentation/report files were added afterward; tested application core
and compiled frontend are unchanged.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Report authoring uses separate dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe scripts/authoring/build_final_evidence.py
.\.venv\Scripts\python.exe scripts/authoring/build_final_report.py
```

Builders consume archives, the recorded demo and clean-test XML; they do not
retrain. Use an authoring copy: regenerated files alter the frozen inventory.
Inspect every rendered PDF page after changes. `requirements-authoring.lock.txt`
records the authoring environment; the runtime lock stays separate.

## Original research reproduction

The CPU release includes trained LiteCrack/BA archives and pilot manifests, not
the complete datasets or original Phase 2 baseline weight archive. Baseline
registry and verified metrics remain. Full GPU training was not rerun in Phase 8.
It needs the original compatible GPU environment, permitted complete datasets and
the predecessor archives explicitly required by each notebook.

Use notebooks in `output/kaggle/` in their original order:

1. `Road_Crack_Phase1_Data_Audit.ipynb`: verify datasets and frozen manifest.
2. Phase 2 baselines, stabilization and final evaluation.
3. Phase 3 lightweight training and final evaluation.
4. Phase 4 boundary training and comparison.
5. `Road_Crack_Phase5_FewShot.ipynb`: fixed 30-run support-draw protocol.

Keep original archives immutable and write reruns separately. Use the existing
VS Code notebook + Existing Jupyter Server workflow for Kaggle. Authenticated
session URLs are not bundled. Do not change frozen splits, budgets or gates after
query review. Multi-seed/independent-domain work requires a new protocol.
