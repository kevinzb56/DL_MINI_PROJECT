# Submission layout

The working tree has three top-level folders: `code/` for application and research source, `data/` for datasets and bundled inputs, and `results/` for generated outputs, model archives, and submission packages. Run project commands from `code/`.

| Path | Purpose |
| --- | --- |
| `inspection/`, `web/` | Local API and React inspection app |
| `src/`, `followup/` | Segmentation models and research workflows |
| `scripts/` | App and release entry points |
| `scripts/data/` | Dataset download, preparation, and audits |
| `scripts/experiments/` | Training, adaptation, and run launchers |
| `scripts/evaluation/` | Metrics, reports, validation, and smoke checks |
| `scripts/authoring/`, `scripts/presentations/` | Notebook, report, and presentation generation |
| `scripts/operations/` | Packaging, publication, and cleanup utilities |
| `configs/`, `../data/manifests/` | Configuration and frozen registries |
| `tests/`, `docs/` | Checks and documentation |
| `../results/output/` | Evidence, reports, and generated presentations |
| `../results/archives/` | Historical checkpoint and recovery ZIPs |
| `../results/releases/` | Historical Phase 8 release |
| `../results/submission/` | Clean submission packages |
| `../data/release_assets/` | Bundled demo and replay inputs |
| `../data/references/` | Reference paper PDF |

The `data/`, `output/`, `artifacts/`, and `release_assets/` entries inside `code/` are Windows directory junctions. They point to the physical folders under `../data/` and `../results/`. Historical phase archives are opened from `../results/archives/` in the working tree and are included at the root of portable release ZIPs.

Local virtual environments, caches, frontend dependencies, downloaded datasets, and recovery workspaces are not part of the submission ZIP. See [the current submission guide](../SUBMISSION_README.md) for the packaged demo and [the project README](../README.md) for research results.
