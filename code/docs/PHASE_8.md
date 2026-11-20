# Phase 8: final pilot release and evidence

Implemented 1 November 2026. This phase consolidates preserved experiments and
the local application into a reviewable release. Frozen checkpoints and observed
gates are unchanged; no additional research training is claimed.

## Delivered

- Eleven-page PDF, Markdown/HTML versions and four generated research figures.
- Model card, current README, setup/demo guides and third-party notices; historical
  README states retained separately.
- Consolidated evidence and G1-G10 ledger derived from original archives, fresh
  tests, Phase 7 reports and a recorded real-image application demonstration.
- Seeded retrospective proxy-group bootstrap intervals with shared support-draw
  image scores averaged before resampling; descriptive, unadjusted analysis.
- Versioned allowlist ZIP, SHA256 inventory, stdlib archive/directory verifier,
  pinned runtime, compiled React and verified replay/demo assets.
- Fresh extraction/bootstrap, package/core fingerprint checks and a video of
  correction, adaptation, validation, activation, rollback, comparison and exports.

## Verification

The verified RC was extracted separately and installed into a fresh Python 3.12
venv. Setup returned zero; all 46 pinned distributions matched. All 32 trained
references and 24 replay pairs imported without full-dataset download or Node
rebuild. **49 tests passed; six CUDA-only tests skipped.** The real-acquisition
workflow passed with zero JavaScript errors. Masks are existing annotations,
not a measured expert correction study. All 11 final PDF pages were rendered and
visually inspected.

Final reporting/docs surround the tested candidate; application core and compiled
frontend are compared by content hash. The final manifest is verified again.
Archive SHA256 and final extraction receipt remain external to avoid a
self-referential archive hash. See `output/phase8/FINAL_RELEASE_CHECK.json` beside
the authoring copy after packaging.

## Acceptance boundary

| Gate | Interpretation |
| --- | --- |
| G1 | Partial: frozen splits verified; physical site independence unverified. |
| G2 | Passed: 339,265 parameters; 83.92% below implemented R2AU. |
| G3 | Pilot pass: static target within two points; lower source accuracy. |
| G4 | Partial: target boundary gain; no multi-seed/universal confirmation. |
| G5 | Pilot pass: fixed-draw few-shot target gains. |
| G6 | Unmet: source-retention loss exceeds two points. |
| G7 | Passed on named CPU/model-only benchmark. |
| G8 | ONNX parity passed on audited samples. |
| G9 | Real-image functional demo passed; correction time not measured. |
| G10 | Partial: fresh app/tests pass; original GPU retraining not rerun. |

Wide-road vegetation false positives remain unresolved. Named authors/institution
and any academic submission template must come from the project owner. Public
dataset redistribution permission is not asserted; see `NOTICE.md`.

## Rebuild

In an authoring copy with original archives unchanged:

```powershell
.\.venv\Scripts\python.exe scripts/authoring/build_final_evidence.py
.\.venv\Scripts\python.exe scripts/authoring/build_final_report.py
.\.venv\Scripts\python.exe scripts/build_release.py --output CrackLab_Phase8_v1.0.0.zip
.\.venv\Scripts\python.exe scripts/verify_release.py --archive CrackLab_Phase8_v1.0.0.zip
```

The builder refuses to overwrite a ZIP. Re-running the recorded demo requires a
new disposable installation; do not run it against the main user workspace.
