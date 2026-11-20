# Phase 1 status — 25 October 2026

**State: pilot data protocol frozen; ready to implement Phase 2 with documented limitations.**

## Corrected audit and frozen pilot

The second audit ZIP SHA256 is
`e4066021729dd3c7f8ff6c8a541003cfea00dc2b8046b7567b122aeea3630f9d`.
Integrity passed. Evidence is retained under `artifacts/imported/data_audit_e4066021/`.

- All 657 pairs audited with no reported errors; all masks use 0/255.
- Final allocation: 240 source training, 60 source validation, 237 source holdout,
  25 target acquisition, and 95 target query images.
- All 24 contact-sheet examples visually inspected; foreground alignment looks
  correct. This is sampled visual QA, not inspection of every image.
- No duplicate decoded image hashes or split-crossing groups found. All 54
  CrackMap GOPR filename groups stay together (14 acquisition / 40 query).
- Independent local checks found 10 repeated DeepCrack filename IDs across native
  partitions. Canonical IDs now include dataset and native partition; image hashes
  differ. No samples were silently removed.
- `data/manifests/pilot_v1.json` is frozen, portable relative to `data/raw`, and
  checksummed. Five nested 1/5/10-shot draws use distinct capture groups per draw.
  All support IDs belong only to acquisition; target query membership is fixed.
- Freeze checks verified all 657 exported derived-mask checksums. Images remain
  on Kaggle, so no new local raw-image decode was possible.

This closes the pilot data preparation, not the entire original multi-dataset
research scope. Capture/site independence remains unverified beyond filename and
similarity checks, CrackMap usage terms need clarification, and raw data backup
outside Kaggle remains pending. Actual baseline trainer/CUDA recovery must be
tested in Phase 2 before a long run. The imported Kaggle dependency snapshot is
preserved; local requirements still need reconciliation for Pillow 13.

Earlier sections below are historical evidence and are superseded by this result.

## First real-data audit received

`phase1_data_audit.zip` has SHA256
`159fb36db8de9bf3e32535e7ccae28ea27b1f4bf0ce82bc86a5a17d4a952a5be`.
ZIP integrity passed and evidence is preserved under
`artifacts/imported/data_audit_159fb36d/`.

- 120 valid CrackMap pairs, all raw masks using 0/255. Eight contact-sheet
  examples visually reviewed: white foreground follows cracks.
- DeepCrack inventory contains `dataset/DeepCrack.zip`, not unpacked pair folders.
  Initial audit therefore has no source training/validation/holdout records.
- Initial 30-acquisition/90-query split is rejected: 13 of 54 filename capture
  groups span both splits. Shared GOPR identifiers must stay together.
- Updated audit safely extracts the source ZIP and unions same-GOPR patches
  before candidate allocation. Five local extraction/pair/grouping tests pass;
  full corrected audit still needs Kaggle execution.
- The updated notebook uses a separate `road_crack_data_audit_v2` code directory
  and exports `phase1_data_audit_v2.zip`, preserving earlier evidence.
- No final split is frozen and no model training has started.

## Kaggle evidence received

The user-downloaded `phase1_results.zip` was verified locally against the supplied
SHA256 `cad6028aa08a3ef487816042b13c11da9d893d673579962caaa08680afbe3270`.
ZIP integrity passed. Contents were preserved under
`artifacts/imported/kaggle_phase1_cad6028a/` without replacing local source.

- Environment report: two Tesla T4 devices, CUDA available, PyTorch 2.11.0+cu128,
  Python 3.13.15, NumPy 2.1.3, Pillow 13.0.0, pytest 8.4.2.
- Test report: five tests passed in 5.28 seconds, including interrupted-write
  preservation, exact next-step checkpoint recovery, deterministic data allocation,
  duplicate-image rejection, and nonbinary-mask rejection.
- The checkpoint test uses a small CPU model on the GPU-equipped host. Actual
  CUDA training recovery and recovery in the future trainer are not yet tested.
- DeepCrack and CrackMap commit records are included, indicating the acquisition
  cell reached commit recording. Dataset payloads and a full pair inventory are
  not included; usable dataset counts and annotation quality remain unverified.
- A Kaggle dependency snapshot is retained. Its Pillow 13.0.0 lies outside the
  current local requirements range; reconcile environment policy before claiming
  clean-environment reproducibility. Do not blindly replace the working GPU stack.
- No actual data index or frozen manifest, model training, or training metrics
  appear in the archive. Phase 1 remains incomplete pending the real-data audit.

The table below records the original local-session checks; the Kaggle evidence
above supersedes its dependency, GPU, and fixture-test limitations for the remote
environment only. Direct agent-to-Kaggle connectivity remains blocked.

| Item | Evidence/status |
|---|---|
| Workspace inspection | Only PRD and paper existed initially; no AGENTS.md found |
| Local isolation | `.venv` created, Python 3.12.4 |
| Dependencies | Install attempted; WinError 10013 socket restriction; none installed |
| Dataset acquisition | GitHub `ls-remote` failed to connect; no local real pairs |
| Dataset choice | Provisional DeepCrack source / CrackMap target pilot |
| Data audit | Pair validation, checksums, duplicate and grouping checks implemented |
| Split freezing | Script refuses overwrites; no real-data manifest generated |
| Checkpoint recovery | Immutable snapshots, checksums, atomic pointers, full state helper implemented |
| Atomic failure test | Passed: interrupted write preserves previous file and removes temporary file |
| PyTorch recovery test | Implemented, not executed: torch/numpy/pytest unavailable |
| Data audit fixture tests | Implemented, not executed: Pillow/numpy/pytest unavailable |
| Syntax verification | All Python modules compile |
| GPU | Not verified: nvidia-smi missing and CIM hardware query denied |
| Training / model metrics | Not started; no performance results claimed |

Phase 1 exit criteria: dependencies installed and locked, hardware recorded,
datasets acquired with provenance, annotations visually audited, group metadata
reviewed, frozen manifests created, and resume plus audit tests passing. Complete
those before beginning long baseline runs. Setup commands are in the README.
