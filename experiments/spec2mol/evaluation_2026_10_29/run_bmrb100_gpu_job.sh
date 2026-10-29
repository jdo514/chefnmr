#!/usr/bin/env bash
# Execute on NMRbox HTCondor execute node, not on login node.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source venv/bin/activate
export CHEFNMR_DIR="$REPO/chefnmr"
export PYTHONPATH="$CHEFNMR_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
TAG="${1:-manual}"
PILOT="$REPO/data/external_evaluation/bmrb100_chefnmr_pilot_v1"
EVAL="$REPO/launch/chefnmr_eval/run_structure_identification_batch.py"
OUT="$REPO/logs/bmrb100_graph_eval_${TAG}"
mkdir -p "$OUT"
echo "START $(date -Is) HOST=$(hostname) JOB=$TAG"
echo "GPUS: ${CUDA_VISIBLE_DEVICES:-unknown}"
# Preflight performs no GPU computation and fails on checksum/structure errors.
python "$EVAL" --base "$REPO" --dataset "bmrb100=$PILOT" --preflight \
  2>&1 | tee "$OUT/preflight.log"
# Minimal smoke on GPU: only 2 cases, both models, 1 candidate, 10 diffusion steps.
python "$EVAL" --base "$REPO" --dataset "bmrb100=$PILOT" \
  --limit 2 --candidates 1 --steps 10 \
  --modes experimental,zero --atom-orders native \
  --output "$OUT/smoke" \
  2>&1 | tee "$OUT/smoke.log"
# MAIN controlled identification experiment, using a single allocation for all conditions.
python "$EVAL" --base "$REPO" --dataset "bmrb100=$PILOT" \
  --candidates 10 --steps 50 \
  --modes experimental,zero,h1_only,c13_only \
  --atom-orders native,shuffled \
  --save-coordinates --output "$OUT/full" \
  2>&1 | tee "$OUT/full.log"
echo "BMRB100 SINGLE-GPU BATCH PASS $(date -Is)"
