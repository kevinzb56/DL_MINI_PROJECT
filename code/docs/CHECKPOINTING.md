# Checkpoint and result preservation contract

Each independent training/adaptation run gets a unique directory. Never resume
into a different experiment or overwrite the parent model during adaptation.

`src/checkpoints.py` writes immutable `step-*.pt` snapshots with SHA256 verification
and atomic `latest.json`/`best.json` pointers. A failed snapshot write leaves prior
snapshots intact. The latest pointer can lag if interruption occurs after snapshot
creation; that snapshot remains available for manual inspection.

Snapshot content: model, optimizer, scheduler, AMP scaler, epoch, optimizer step,
best score, current metrics, exact configuration, manifest hash, sampler
order/cursor supplied by the trainer, and Python/NumPy/PyTorch/CUDA random states.
The loader rejects changed configuration or manifests. Only trusted local files
may be loaded, because complete recovery state uses Python pickle.

Phase 2 trainer requirements:

1. Save after every epoch, every 100 completed optimizer steps, on validation
   improvement, and at a safe step boundary after a graceful stop request.
2. Update model/optimizer/scheduler/scaler consistently before taking a snapshot.
   Do not save midway through gradient accumulation unless accumulated gradients
   are also saved. Initial trainer should avoid accumulation for exact recovery.
3. Save current sampler permutation and next-example cursor; use `num_workers=0`
   initially to avoid unrecorded prefetched augmentations.
4. Restore all components and RNG states before creating the next stochastic batch.
   Test recovery on the actual trainer, not only the small unit fixture.
5. Write configuration/environment/split hashes at run creation and retain metrics
   keyed by global step, reconciling log entries if recovery rolls back a step.
6. Keep validation-best checkpoint independent of newest checkpoint. Never use
   target query scores to choose the best checkpoint.
7. Adaptation uses a new run with parent checkpoint hash and support annotation IDs.

Exact replay is limited to a compatible hardware/software setup and deterministic
operations. CPU loading of a CUDA run may be useful for inference but is not exact
CUDA training recovery. The current helper requires matching recovery components.

No automatic pruning is implemented. Monitor storage; old snapshots must not be
silently removed. Before a long run, copy checkpoints, logs, manifests, and configs
to a separate disk or persistent remote storage and verify copied hashes. OneDrive
folder location alone is not evidence of a completed backup. Sudden power loss can
lose work since the latest completed save; a disk failure can lose all local copies.

Design reference: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
