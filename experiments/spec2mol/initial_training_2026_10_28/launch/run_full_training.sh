#!/usr/bin/env bash
# Start one ChefNMR-S 11-epoch training job; only after gate_200.py has approved benchmark.
set -euo pipefail
BASE=/home/nmrbox/0015/jdonner/spec2mol-chefnmr
MODE="${1:?Usage: run_full_training.sh smoke|reference|augmented CLUSTER.PROC}"
RUN_ID="${2:?Provide HTCondor job ID}"
[[ "$MODE" == smoke || "$MODE" == reference || "$MODE" == augmented ]] || { echo "Bad experiment: $MODE" >&2; exit 2; }
[[ "$RUN_ID" =~ ^[0-9]+\.[0-9]+$ ]] || { echo "Bad run ID: $RUN_ID" >&2; exit 2; }
APPROVAL="$BASE/launch/conditional_full_training/approved.json"
[[ -f "$APPROVAL" ]] || { echo "200-step approval missing" >&2; exit 3; }
PY="$BASE/venv/bin/python"
read -r BATCH STEPS < <("$PY" - "$APPROVAL" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['benchmark_job']=='2222727.0'
assert x['epochs']==11 and x['train_molecules']==95202
assert x['batch_size'] in (32,16,8)
assert x['max_steps']==((95202+x['batch_size']-1)//x['batch_size'])*11
print(x['batch_size'],x['max_steps'])
PY
)
if [[ "$MODE" == reference ]]; then VARIANT_MODE=reference; else VARIANT_MODE=paired_cyclic; fi
EPOCHS=11
EXTRA_TRAINER=()
if [[ "$MODE" == smoke ]]; then
  # Exercises actual ChefNMR src.main with epoch-end validation + checkpoint callbacks.
  # Two training batches and two validation batches, rather than 11 complete epochs.
  EPOCHS=1
  EXTRA_TRAINER=(+trainer.limit_train_batches=2 +trainer.limit_val_batches=2)
fi
EXPERIMENT="spec2mol-${MODE}-seed42"
RUN_DIR="$BASE/logs/full_${MODE}_${RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite $RUN_DIR" >&2; exit 4; }
for REPO in "$BASE/pipeline" "$BASE/chefnmr"; do
    git -C "$REPO" diff --quiet HEAD -- || { echo "Tracked changes in $REPO" >&2; exit 4; }
done
# Ensure training uses exactly the Git commits audited in the 200-step benchmark.
"$PY" - "$APPROVAL" "$BASE" <<'CHECK_COMMITS'
import json, subprocess, sys
from pathlib import Path
approved = json.load(open(sys.argv[1]))
base = Path(sys.argv[2])
for directory, key in [('pipeline', 'pipeline_commit'), ('chefnmr', 'chef_commit')]:
    actual = subprocess.check_output(['git', '-C', str(base/directory), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == approved[key], f'{directory} code changed after benchmark: {actual} != {approved[key]}'
print('PINNED_CODE_CHECK_PASS', flush=True)
CHECK_COMMITS
export PYTHONPATH="$BASE/chefnmr"
export CHEFNMR_DIR="$BASE/chefnmr"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" "$BASE/pipeline/scripts/nmrbox/preflight.py" --base "$BASE"
mkdir -p "$RUN_DIR"
"$PY" -m pip freeze > "$RUN_DIR/pip_freeze.txt"
echo "CHEFNMR FULL TRAINING start: mode=$MODE variant=$VARIANT_MODE batch=$BATCH epochs=$EPOCHS max_steps=$STEPS run=$RUN_ID at $(date -Is)"
echo "Pipeline commit $(git -C "$BASE/pipeline" rev-parse HEAD)"
echo "ChefNMR commit $(git -C "$BASE/chefnmr" rev-parse HEAD)"
cd "$BASE/chefnmr"
"$PY" -m src.main \
  +condition=h1c13nmr-10k-80 \
  +data=spectrabase \
  +device=gpu1 \
  +embedder=hybrid-baseline \
  +lddt=threshold5124 \
  +model=dit-s \
  +training_transform=center_rot_trans \
  +diffusion=edm-train_af3-sample_edm_sde \
  +aug_conf=conf3 \
  +guidance=cfg2 \
  "dataset_args.datadir=$BASE/data" \
  "+dataset_args.selection_manifest_dir=$BASE/selections/strict_clean_conformers_v1" \
  "+dataset_args.spectral_variant_mode=$VARIANT_MODE" \
  +dataset_args.spectral_variant_count=11 \
  +dataset_args.spectral_variant_seed=42 \
  dataset_args.augmentation_args.max_n_h1spectra=11 \
  dataset_args.augmentation_args.max_n_c13spectra=11 \
  "dataset_args.batch_size=$BATCH" \
  dataset_args.num_workers=0 \
  trainer.devices=1 \
  "trainer.max_epochs=$EPOCHS" \
  "+trainer.max_steps=$STEPS" \
  trainer.check_val_every_n_epoch=1 \
  trainer.num_sanity_val_steps=2 \
  "${EXTRA_TRAINER[@]}" \
  trainer.log_every_n_steps=50 \
  general.wandb=disabled \
  general.save_checkpoint=true \
  general.save_top_n_ckpts=2 \
  "general.experiment_name=$EXPERIMENT" \
  general.seed=42 \
  validation_args.sample_every_n_val_epoch=50 \
  "hydra.run.dir=$RUN_DIR" \
  hydra.job.chdir=true
# ChefNMR's main.py writes checkpoint paths relative to the Hydra output directory.
CKPT_DIR="$RUN_DIR/checkpoints/$EXPERIMENT/diffusion_loss"
if [[ ! -d "$CKPT_DIR" ]] || ! find "$CKPT_DIR" -name '*.ckpt' -type f -print -quit | grep -q .; then
  echo "Training exited but expected validation checkpoints absent: $CKPT_DIR" >&2
  exit 5
fi
"$PY" - "$CKPT_DIR" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]); c=list(p.glob('*.ckpt'))
assert c and all(x.stat().st_size>100*1024**2 for x in c), c
print('FULL_TRAINING_CHECKPOINTS_PASS', [(x.name,round(x.stat().st_size/1024**3,2)) for x in c],flush=True)
PY
echo "FULL_TRAINING_PASS mode=$MODE run=$RUN_ID at $(date -Is)"
