#!/usr/bin/env bash
set -euo pipefail
# Usage: run_eval.sh MODEL DATASET VARIANT LIMIT CANDIDATES STEPS RUN_ID [EXTERNAL_DATADIR] [EXTERNAL_INVENTORY]
# Modes: reference/augmented; frozen-val/specteach/bmrb100
BASE="${HOME}/spec2mol-chefnmr"
MODEL="${1:?model}"; DATASET="${2:?dataset}"; VARIANT="${3:?variant}"
LIMIT="${4:?limit}"; CANDS="${5:?candidates}"; STEPS="${6:?steps}"; RUN_ID="${7:?run_id}"
EXTERNAL_DIR="${8:-}"; INVENTORY="${9:-}"
case "$MODEL" in
 reference) CKPT="$BASE/logs/full_reference_2222733.0/checkpoints/spec2mol-reference-seed42/diffusion_loss/epoch0010-loss88.19.ckpt" ;;
 augmented) CKPT="$BASE/logs/full_augmented_2222972.0/checkpoints/spec2mol-augmented-seed42/diffusion_loss/epoch0010-loss88.26.ckpt" ;;
 *) echo "Unknown model $MODEL" >&2; exit 2;;
esac
[[ -f "$CKPT" ]] || { echo "Missing best checkpoint: $CKPT" >&2; exit 3; }
source "$BASE/venv/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$BASE/chefnmr"
export CHEFNMR_DIR="$BASE/chefnmr"
export PYTHONPATH="$CHEFNMR_DIR${PYTHONPATH:+:$PYTHONPATH}"
OUT="$BASE/logs/evaluation_${DATASET}_${MODEL}_variant${VARIANT}_n${LIMIT}_k${CANDS}_s${STEPS}_${RUN_ID}"
ARGS=(--checkpoint "$CKPT" --dataset "$DATASET" --variant "$VARIANT"
      --limit "$LIMIT" --candidates "$CANDS" --sampling-steps "$STEPS"
      --batch-size 2 --cpus 2 --output-dir "$OUT")
if [[ "$DATASET" == frozen-val ]]; then
  ARGS+=(--data-dir "$BASE/data" --selection "$BASE/selections/strict_clean_conformers_v1" --variants-per-molecule 11)
else
  [[ -n "$EXTERNAL_DIR" && -n "$INVENTORY" ]] || { echo "External data require EXTERNAL_DATADIR and EXTERNAL_INVENTORY" >&2; exit 4; }
  ARGS+=(--data-dir "$EXTERNAL_DIR" --inventory "$INVENTORY" --variants-per-molecule 1)
fi
exec python "$BASE/launch/chefnmr_eval/evaluate.py" "${ARGS[@]}"
