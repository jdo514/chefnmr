#!/usr/bin/env bash
# Called by HTCondor from a DAG node; never submit this directly.
set -euo pipefail
CAMPAIGN="${1:?campaign}"; SEGMENT="${2:?segment}"; EXPECTED_GPUS="${3:?1 or 2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$HERE/../.." && pwd)"
PY="$BASE/venv/bin/python"
ROOT="$BASE/logs/$CAMPAIGN"
# Stop if an accidentally edited launcher could change the pinned campaign behavior.
"$PY" - "$ROOT/campaign.json" "$HERE/run_disjoint.sh" <<'PIN'
import sys,json,hashlib
meta=json.load(open(sys.argv[1]))
h=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
if h!=meta['launcher_script_sha256']:
    raise SystemExit('Pinned launcher script changed after submission')
PIN
FOLDER="$(printf '%s/segment_%02d' "$ROOT" "$SEGMENT")"
mkdir -p "$FOLDER"
# Give a simultaneously allocated 2-GPU candidate a 60-second head start.
# IMPORTANT: the one-GPU allocation remains occupied during this grace period;
# this is claim priority, NOT an HTCondor matchmaking priority guarantee.
# Wait BEFORE acquiring either the flock or persistent claim.  A 2-GPU
# candidate entering within this window can claim immediately and win.
if [[ "$EXPECTED_GPUS" -eq 1 ]]; then
  echo 'ONE_GPU_PRIORITY_WINDOW: waiting 60 seconds for a 2-GPU candidate'
  sleep 60
fi
# Exactly one candidate may train; hold the lock for the whole segment.
exec 9>"$FOLDER/.winner.lock"
if ! flock -n 9; then
  echo 'Another GPU candidate already training this campaign; exiting without training.'
  exit 0
fi
if [[ -f "$FOLDER/started.json" ]]; then
  echo 'Another candidate previously started; refusing a second run.'
  exit 0
fi
# Atomic directory creation also guards against cross-node NFS lock differences.
# This remains on disk after failure; require manual review before a restart.
if ! mkdir "$FOLDER/.winner_claim" 2>/dev/null; then
  echo 'Another candidate already claimed this campaign; exiting without training.'
  exit 0
fi
# Publish claim owner immediately, before potentially slow GPU/PyTorch startup.
# This also lets DAGMan identify the winner if the losing job exits immediately.
printf '{"gpus":%s,"pid":%s}\n' "$EXPECTED_GPUS" "$$" > "$FOLDER/.winner_claim/owner.json"
if [[ -f "$FOLDER/result.json" ]]; then
  echo 'Refusing duplicate training: segment result already exists' >&2; exit 10
fi
export CHEFNMR_DIR="$BASE/chefnmr"
export PYTHONPATH="$CHEFNMR_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ACTUAL_GPU_COUNT="$($PY -c 'import torch; print(torch.cuda.device_count())')"
echo "EXPECTED_GPUS=$EXPECTED_GPUS VISIBLE_GPUS=$ACTUAL_GPU_COUNT"
[[ "$ACTUAL_GPU_COUNT" == "$EXPECTED_GPUS" ]] || { echo 'GPU mismatch: abort' >&2; exit 11; }
# This marker is written only *after the allocation actually starts*.
"$PY" - "$FOLDER/started.json" "$EXPECTED_GPUS" <<'PY'
import json,sys,os,time,socket,pathlib
pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(gpus=int(sys.argv[2]),pid=os.getpid(),
    host=socket.gethostname(),started_utc=time.strftime('%FT%TZ',time.gmtime())))+'\n')
PY
cd "$BASE/chefnmr"
# Same initial effective (global) batch on 1- and 2-GPU jobs.
# 1 GPU: 128 -> 64 -> 32; 2 GPUs: 64 -> 32 (global: 128 -> 64).
if [[ "$EXPECTED_GPUS" -eq 2 ]]; then
  BATCHES=(64 32)
else
  BATCHES=(128 64 32)
fi
for BATCH in "${BATCHES[@]}"; do
  echo "RUNNING segment=$SEGMENT gpus=$EXPECTED_GPUS batch=$BATCH"
  LOG="$FOLDER/try_batch${BATCH}.log"
  set +e
  "$PY" "$HERE/train_disjoint.py" --base "$BASE" --campaign "$CAMPAIGN" \
    --segment "$SEGMENT" --expected-gpus "$EXPECTED_GPUS" --batch "$BATCH" \
    --segment-hours 43 > "$LOG" 2>&1
  RC=$?
  set -e
  tail -n 24 "$LOG" || true
  if [[ "$RC" -eq 0 && -s "$FOLDER/result.json" ]]; then
    echo "TRAINING SEGMENT PASS"; exit 0
  fi
  if grep -Eiq 'CUDA out of memory|torch\.OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED|OutOfMemoryError' "$LOG"; then
    echo "GPU OOM at batch=$BATCH (rc=$RC); trying smaller batch" >&2
    continue
  fi
  echo "TRAINING ERROR NOT OOM, stop: rc=$RC log=$LOG" >&2
  exit "${RC:-1}"
done
echo 'All batches failed with OOM; require manual inspection' >&2
exit 12
