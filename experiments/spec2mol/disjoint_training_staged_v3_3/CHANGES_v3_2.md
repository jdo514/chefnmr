# v3.2: resume-safe epoch timer

Fixes an error when a Lightning checkpoint is resumed halfway through an epoch.
The pinned original `NMRTo3DStructureElucidation.on_train_epoch_end` subtracts `self.start_epoch_time`,
but that value may remain `None` because `on_train_epoch_start` was skipped on mid-epoch resume.

`train_disjoint.py` adds a new stateless `ResumeSafeEpochTimer` callback that initializes
`pl_module.start_epoch_time` on `on_fit_start` if needed. The regular epoch-start hook
continues resetting the timer for later complete epochs. This change affects only
console epoch-duration reporting. No model architecture, data, optimizer or scheduler changes.

Retains local verified HDF5 staging, 120-minute timeout, 50 GiB disk request,
NFS fallback, 43-hour epochs-boundary segments, 12 segments, 1/2 GPU candidates.

**Never modify the pinned active campaign**; create a *new* campaign name using this package.
Keep original v2.1 running until v3.2 trains.

Before submitting, inspect the previous job's traceback and verify this is the
observed failure. If the exception is different, address it first.
