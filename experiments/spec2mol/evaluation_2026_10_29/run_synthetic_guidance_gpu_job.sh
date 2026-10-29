#!/usr/bin/env bash
# Single HTCondor GPU allocation: synthetic 10-molecule guidance sensitivity.
# Does not modify the frozen checkpoints, dataset, or completed BMRB100 run.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source venv/bin/activate
export CHEFNMR_DIR="$REPO/chefnmr"
export PYTHONPATH="$CHEFNMR_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
TAG="${1:-manual}"
EVAL="$REPO/launch/chefnmr_eval/run_structure_identification_batch_v2.py"
COMPARE="$REPO/launch/chefnmr_eval/compare_synthetic_guidance_pilot_v1.py"
DATASET="$REPO/data/external_evaluation/synthetic_val100_graph_v1"
OUT="$REPO/logs/synthetic_guidance_pilot_${TAG}"
mkdir -p "$OUT"
if [[ ! -f "$EVAL" ]]; then echo "Missing evaluator $EVAL" >&2; exit 1; fi
if [[ ! -f "$COMPARE" ]]; then echo "Missing comparison script $COMPARE" >&2; exit 1; fi
if [[ ! -d "$DATASET" ]]; then echo "Missing synthetic dataset $DATASET" >&2; exit 1; fi
if [[ -e "$OUT/guidance0/candidates.csv" || -e "$OUT/guidance2/candidates.csv" ]]; then
    echo "Refusing to overwrite existing inference outputs: $OUT" >&2
    exit 1
fi
echo "START $(date -Is) HOST=$(hostname) JOB=$TAG" | tee "$OUT/job_start.txt"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$OUT/job_start.txt"
python "$EVAL" \
    --base "$REPO" \
    --dataset "synthetic_val100=$DATASET" \
    --limit 10 \
    --modes experimental,zero \
    --atom-orders native \
    --candidates 10 \
    --steps 50 \
    --preflight \
    2>&1 | tee "$OUT/preflight.log"

# Keep the two guidance values in separate, immutable run directories.
# Model/condition/candidate seeds are reset identically by the v2 evaluator.
for guidance in 0 2; do
    echo "START GUIDANCE=$guidance $(date -Is)" | tee -a "$OUT/job_start.txt"
    python "$EVAL" \
        --base "$REPO" \
        --dataset "synthetic_val100=$DATASET" \
        --output "$OUT/guidance$guidance" \
        --limit 10 \
        --modes experimental,zero \
        --atom-orders native \
        --guidance "$guidance" \
        --candidates 10 \
        --steps 50 \
        --save-coordinates \
        2>&1 | tee "$OUT/guidance$guidance.log"
done
python "$COMPARE" --run "$OUT" 2>&1 | tee "$OUT/comparison.log"
echo "SYNTHETIC GUIDANCE PILOT PASS $(date -Is)" | tee -a "$OUT/job_start.txt"
