# v3.4 — timed-only recovery checkpoint QUICKSTART

**This is a fresh launcher. Do NOT unzip into or modify v3.3, which is SHA-pinned and running.**

All commands run on indium within `~/spec2mol-chefnmr` unless specified. Package ZIP should be in that directory.

## Install and test (does not affect v3.3)

```bash
unzip -o ChefNMR_Stage2_LocalHDF5_AutoDAG_v3_4_TrueHourlyFix.zip -d launch/
PY=~/spec2mol-chefnmr/venv/bin/python
LAUNCH=launch/disjoint_training_staged_v3_4
$PY "$LAUNCH/test_checkpoint_policy.py"
$PY "$LAUNCH/test_recovery_schedule_runtime.py"
$PY "$LAUNCH/test_auto_workflow.py"
$PY "$LAUNCH/test_resume_epoch_timer.py"
$PY "$LAUNCH/test_stage_hdf5.py"
$PY "$LAUNCH/test_gpu_priority_race.py"
```

The runtime test uses the NMRbox-installed Lightning version and must show
`TIMED_ONLY_NO_EPOCH_SAVES_PASS` and `TIMED_ONLY_FULL_RESUME_CHECKPOINT_PASS`.

## Freeze a FULL checkpoint from v3.3, while it continues running

```bash
SRC=logs/extended_augmented_disjoint_staged_v33_20261030/segment_01/hourly/last.ckpt
DST=logs/stage2_resume_snapshot_v34_20261030/last.ckpt
$PY "$LAUNCH/freeze_checkpoint.py" --source "$SRC" --output "$DST" --attempts 4 --retry-seconds 30
```

Require `FROZEN_CHECKPOINT_OK`, record its hash; the source may be actively
rewritten by v3.3, so retry if source changes during copying.

## Generate and dry-run a NEW campaign (does not submit it)

```bash
$PY "$LAUNCH/make_campaign.py" \
  --campaign extended_augmented_disjoint_staged_v34_20261030 \
  --segments 12 --hours 43 --request-disk-gib 50 \
  --resume-checkpoint "$DST"
ROOT=logs/extended_augmented_disjoint_staged_v34_20261030
condor_submit -dry-run /tmp/v34_1.ad "$ROOT/condor/S01_1.sub"
condor_submit -dry-run /tmp/v34_2.ad "$ROOT/condor/S01_2.sub"
```

If all verification passes, decide explicitly whether to switch. **Do not**
run the next commands unless choosing to stop v3.3 and start v3.4 now.

## Optional switch (only after all tests and frozen checkpoint PASS)

```bash
condor_rm 2225804  # current v3.3 DAG, not 2225805 only
condor_submit_dag "$ROOT/campaign.dag"
condor_q -dag
```

After GPU allocation, local HDF5 staging may take around 50–60 minutes.
Check logs for `CHECKPOINT_POLICY_V34`, successful full restore, progress of
training epochs, a best weights-only checkpoint at an eligible 10th epoch,
and the full time-triggered `segment_01/hourly/last.ckpt`. Verify no extra
recovery write between timer intervals. The full final segment result and
subsequent DAG handover still need live verification.
