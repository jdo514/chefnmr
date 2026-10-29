#!/usr/bin/env bash
# Called by HTCondor from a DAG node; never submit this directly.
set -euo pipefail
CAMPAIGN="${1:?campaign}"; SEGMENT="${2:?segment}"; EXPECTED_GPUS="${3:?2 or 4}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$HERE/../../../.." && pwd)"
PY="$BASE/venv/bin/python"
ROOT="$BASE/logs/$CAMPAIGN"
FOLDER="$(printf '%s/segment_%02d' "$ROOT" "$SEGMENT")"
mkdir -p "$FOLDER"
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
for BATCH in 128 64 32; do
  echo "RUNNING segment=$SEGMENT gpus=$EXPECTED_GPUS batch=$BATCH"
  LOG="$FOLDER/try_batch${BATCH}.log"
  set +e
  "$PY" "$HERE/train_segment.py" --base "$BASE" --campaign "$CAMPAIGN" \
    --segment "$SEGMENT" --expected-gpus "$EXPECTED_GPUS" --batch "$BATCH" \
    --segment-hours 44 > "$LOG" 2>&1
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
