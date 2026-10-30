# v3.4 — True timed-only hourly recovery (minimal fix)

This is a NEW launch package. Never overwrite or edit the running v3.3 campaign.

## Root cause (Lightning 2.5.0)

`ModelCheckpoint(save_top_k=0, save_last=True, train_time_interval=60min)`
performs extra last.ckpt saves from training/validation epoch-end hooks,
bypassing the time interval. That created frequent 2.15GB writes and NFS stalls.

## Surgical fix

Use `ModelCheckpoint(save_top_k=1, save_last=False, filename='last',
train_time_interval=60min, every_n_epochs=0, save_on_train_epoch_end=False,
save_weights_only=False, enable_version_counter=False)`.

Time-triggered training-batch callbacks still write `hourly/last.ckpt`, full
optimizer/scheduler/loop state. Epoch/validation hooks no longer write recovery
checkpoints. Final segment checkpoint (`segment_NN/last.ckpt`) unchanged.
Best-model checkpoint policy unchanged (top 2 weights-only at eligible epochs).

## Verify on the host before any cancellation

Run `test_checkpoint_policy.py`, `test_recovery_schedule_runtime.py`,
`test_auto_workflow.py`, `test_resume_epoch_timer.py`, `test_stage_hdf5.py`,
using `venv/bin/python`, plus HTCondor submission dry-runs.
`test_recovery_schedule_runtime.py` is a lightweight CPU-only Lightning integration
smoke. It rejects a recovery file written before the time interval, and
confirms the timed trigger produces exactly `hourly/last.ckpt` with optimizer
states. It does not simulate a production multi-GPU 43h handover.

## Important caution

This remains a direct NFS checkpoint write. On slower storage, the once-hourly
save can still stall. No async I/O or crash-safe NFS staging is introduced here.
Do not launch a new run without a verified frozen FULL checkpoint.
