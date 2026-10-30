# ChefNMR Stage 2 v3.3 — checkpoint writeback optimization

## Why
On NMRbox iodine with the staged 42.7 GB HDF5, normal model training took about 208 seconds per epoch. The main Python process repeatedly blocked in `balance_dirty_pages` with 0% GPU utilization and NFS-hosted ~2.15 GB best-model and hourly checkpoints. Best checkpoints were eligible for saving after **every improving validation epoch**. In the Oct 30 observation epoch 26 to 27 incurred a prolonged pause.

## Narrow differences from v3.2
- `best/`: ModelCheckpoint monitors `val/diffusion_loss` but saves the best 2 **weights-only** checkpoints only at epochs divisible by 10; `save_weights_only=True`, `every_n_epochs=10`. Validation is still run **every epoch**; model selection is only sampled every 10 epochs. The exact best among *all* epochs is not guaranteed to be saved.
- `segment_XX/hourly/last.ckpt`: unchanged hourly **full** training recovery (`save_weights_only=False`, optimizer/scheduler retained).
- `segment_XX/last.ckpt`: unchanged final **full** checkpoint (SHA256 checked for DAG continuity).
- `ResumeSafeEpochTimer()` is retained. No changes to model, underlying ChefNMR repository, data splits, optimizer hyperparameters, loss, batch sizes, GPU race, staging, job duration or HTCondor DAG management.
- Corrected **all** generated script paths to the packaged `disjoint_training_staged_v3_3` directory. The original v3.2 archive incorrectly referenced v3.1; v3.3 includes explicit path assertions in tests.

## Operational caution
Best weights-only files are **not valid** for full-state training continuation. Resume only from `hourly/last.ckpt` or finalized segment `last.ckpt`. Freeze the original v3.2 full hourly checkpoint before cancelling v3.2; check the output's `FROZEN_CHECKPOINT_OK` and SHA256. Never modify scripts for an active checksum-pinned campaign.

## Estimate (not measured)
Reduction in best-model checkpoint write bytes per epoch can approach ~80–95% versus full saves on every improving epoch (depends on improvement frequency and exact weights-only size). End-to-end training-time savings may be substantially less because hourly full recovery, NFS throughput variability and GPU input/validation overhead remain. Re-measure 1–2 hours after v3.3 starts.

See `RUNBOOK_v3_3.md` for safe migration and submission.
