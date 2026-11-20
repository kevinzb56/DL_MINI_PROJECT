# CrackLab demonstration

Watch `output/phase8/demo/walkthrough.webm`. Nine screenshots and `demo.json`
record the actual run against a fresh extracted installation on port 8769.

1. Upload `release_assets/demo/support.png` and `validation.png` in the studio.
   Both are fixed real CrackMap **target-acquisition** records, with IDs and hashes
   in the adjacent manifest. They are not a new query set.
2. Rank incomplete acquisition images with entropy. Inspect a native prediction,
   draw on the mask canvas, undo/redo and save a draft. Drafts are not complete
   training labels.
3. Fully annotate before marking complete, or explicitly state that an existing
   reference mask is imported. The recording imports the supplied dataset masks
   via the correction API after brush/draft checks; no expert effort/time is measured.
4. Review reference agreement, tiled inference and ONNX inference. The prediction
   and reference are labeled separately; these are actual trained-model outputs.
5. Select the first image as support and the second as separate local validation.
   Run K=1, one support pass, with source replay. Replay has 16 source-training and
   eight separate source-validation pairs. Review target/source gates.
6. The recorded candidate passed its local gates, was activated, then rolled back.
   Compare candidate and parent with identical image/settings. A two-image demo
   is not an independent accuracy benchmark.
7. Export the project, ONNX graph and recovery ZIP. Reload to verify persistence.
   Retain backups before moving the installation.

`scripts/evaluation/record_release_demo.py` is the automation client. It requires Playwright
and compatible Chromium binaries, which are not bundled. Run it only against a
disposable fresh installation: it creates images, a job and activation history.
The application itself does not need Playwright. The recorded run has zero
JavaScript errors and verifies export/checkpoint checksums.

For a viva, explain the compact encoder/decoder, auxiliary boundary loss,
decoder-only adaptation, separate support/validation roles and rejection gates.
Phase 7's separate K=5 replay pilot still failed its source gate; its candidates
remain rejected. Show the known wide-road vegetation failure too. Tiling does
not solve road-region detection, and reference masks are existing annotations.
