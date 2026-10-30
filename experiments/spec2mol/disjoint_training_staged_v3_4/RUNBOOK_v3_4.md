# Safe v3.2 → v3.4 migration (NMRbox/indium)

From `indium`, with `venv` activated:

```bash
cd ~/spec2mol-chefnmr
# Install this bundle under launch/: unzip -o ZIP -d launch
PY=~/spec2mol-chefnmr/venv/bin/python
LAUNCH=launch/disjoint_training_staged_v3_4

$PY "$LAUNCH/test_checkpoint_policy.py"
$PY "$LAUNCH/test_resume_epoch_timer.py"
$PY "$LAUNCH/test_auto_workflow.py"
$PY "$LAUNCH/test_stage_hdf5.py"

# Snapshot the latest FULL hourly recovery checkpoint. v3.2 continues running.
SRC=logs/extended_augmented_disjoint_staged_v32_fresh_20261030/segment_01/hourly/last.ckpt
DST=logs/stage2_resume_snapshot_v33_20261030/last.ckpt
$PY "$LAUNCH/freeze_checkpoint.py" --source "$SRC" --output "$DST" --attempts 4 --retry-seconds 30
cat logs/stage2_resume_snapshot_v33_20261030/last.json

# Submit v3.4 while v3.2 is still active only if GPU concurrency is acceptable.
# Prefer cancel v3.2 after successful frozen recovery and before submitting v3.4,
# to eliminate concurrent NFS checkpoint writes and avoid wasted GPU hours.
condor_rm 2225743

$PY "$LAUNCH/make_campaign.py" \
  --campaign extended_augmented_disjoint_staged_v33_20261030 \
  --segments 12 --hours 43 --request-disk-gib 50 \
  --resume-checkpoint "$DST"

ROOT=logs/extended_augmented_disjoint_staged_v33_20261030
$PY - "$ROOT" "$LAUNCH" "$DST" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,launch,dst=map(Path,sys.argv[1:]);meta=json.loads((root/'campaign.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert meta['train_script_sha256']==sha(launch/'train_disjoint.py')
assert meta['launcher_script_sha256']==sha(launch/'run_disjoint.sh')
assert meta['original_checkpoint_sha256']==sha(dst)
assert 'disjoint_training_staged_v3_4' in (root/'condor/S01_1.sub').read_text()
assert 'disjoint_training_staged_v3_4' in (root/'condor/S01_2.sub').read_text()
assert meta['model_selection_checkpoint_policy']=='best_top2_weights_only_every10_epochs'
print('V33_GENERATED_CAMPAIGN_AUDIT_PASS')
PY
condor_submit -dry-run /tmp/v33_1.ad "$ROOT/condor/S01_1.sub"
condor_submit -dry-run /tmp/v33_2.ad "$ROOT/condor/S01_2.sub"
condor_submit_dag "$ROOT/campaign.dag"
condor_q -dag
```

After one candidate begins, staging will take ~52 minutes on iodine at measured throughput. Then inspect:

```bash
ROOT=logs/extended_augmented_disjoint_staged_v33_20261030
cat "$ROOT/segment_01/staging.json"
grep -E 'CHECKPOINT_POLICY_V34|RESUME_SAFE|Restored all states|Training Epoch|Traceback|Error' "$ROOT/segment_01"/try_batch*.log | tail -30
ls -lh "$ROOT/best/" "$ROOT/segment_01/hourly/"
```

**Before leaving it unattended:** Confirm the first full hourly recovery checkpoint is created, an epoch reaches the next multiple of ten, and best weights-only checkpoints are smaller than the previous 2.15 GB; if not, inspect errors. The DAG handover itself has not yet been tested on a real completed 43-hour segment.

**Reminder:** all 12 segments are 43 training hours each; the DAG will continue beyond 72h unless stopped. This is not a 72-hour total training cap. The source checkpoint is on persistent NFS. `condor_rm` cancels jobs, but does not remove saved NFS files.
