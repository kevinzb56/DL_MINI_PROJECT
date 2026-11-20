# CrackLab React interface

React + Vite + Lucide icons. FastAPI continues to own model inference, project
storage, annotation revisions, adaptation, validation, and exports.

Phase 7 adds `src/Extensions.jsx` and `src/phase7.css`: tiled/portable runtime
controls, ONNX export/download, source replay, and entropy/random correction-pool
ranking. See `docs/PHASE_7.md` for setup, gates, and validated scope.

## Run the application

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/run_inspection.py
```

Open http://127.0.0.1:8765. The launcher uses `web/dist/` and rebuilds when source
files are newer. Existing projects and trained models stay in the same workspace.

For the first source checkout, install and build the frontend once:

```powershell
cd web
npm.cmd ci
npm.cmd run build
```

Node 22.12+ is supported. The tested dependency graph is in `package-lock.json`.
Once built, the Python service can serve the compiled frontend without a running
Node process. `dist/` and `node_modules/` are ignored by Git.

## Frontend development

Keep the Python service on port 8765, then in another terminal:

```powershell
cd web
npm.cmd run dev
```

Use the Vite URL printed in the terminal (the `/static/` base). API calls are
proxied to the local Python service. No hosted account is needed.

## Source map

- `src/App.jsx`: application state, project selection, navigation, notifications.
- `src/Overview.jsx`: project showcase, interactive support-budget exploration,
  research results, archived qualitative previews.
- `src/Studio.jsx`: image library, prediction, comparison, original-resolution
  canvas editor, undo/redo, correction completeness, save/export integration.
- `src/Adaptation.jsx`: support/validation roles, bounded jobs, recovery controls,
  candidate validation, activation and rollback.
- `src/api.js`: the shared API client and image loader.
- `src/research.json`: copied from the verified Phase 5 archive audit.
- `public/research/`: original Phase 3 preview collages and verification JSON.
  Showcase crops use CSS; no model output or image pixels were synthesized.
- `src/styles.css`, `src/refinements.css`: responsive layout, visual design,
  readable typography, and reduced-motion support.

The original research pipeline and checkpoint format are unchanged. Showcase
metrics distinguish the region-only parameter count from boundary-model results.
Few-shot means and spread are descriptive pilot evidence; the UI retains source
forgetting and query-reuse limitations. Archived preview images are labeled Phase
3 and are separate from live uploaded-image predictions.

The React build, five focused API/worker tests, and a Chromium browser workflow
passed. Browser checks use two synthetic fixtures with real project checkpoints;
they do not constitute new real-road accuracy measurements. Evidence is under
`output/phase6/react-browser/` and `output/phase6-react-tests.xml`.
