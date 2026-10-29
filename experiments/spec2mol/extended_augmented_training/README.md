# ChefNMR-S — extended augmented continuation (v1, 2026-10-29)

**Purpose:** Continue the existing 11-epoch augmented model from its complete Lightning checkpoint. Train on the same strict 95,202/10,575 molecular split with a random single paired H1/C13 variant per visit and worker-local selective HDF5 loading. Do **not** train a new reference model.

## Install and preserve in Git

Copy this *directory* into `~/spec2mol-chefnmr/chefnmr/experiments/spec2mol/extended_augmented_training/` on indium. Run from indium:

```bash
cd ~/spec2mol-chefnmr/chefnmr
python -m compileall -q experiments/spec2mol/extended_augmented_training
python experiments/spec2mol/extended_augmented_training/test_workflow.py
# Commit and push ONLY these new training files. Historical launch files stay untouched.
git add experiments/spec2mol/extended_augmented_training
git diff --cached --check
git commit -m "feat(training): add segmented augmented ChefNMR continuation with GPU fallback"
git push origin feature/spec2mol-augmented-training
```

## Generate a campaign — no GPU allocation

```bash
cd ~/spec2mol-chefnmr
source venv/bin/activate
export CHEFNMR_DIR="$PWD/chefnmr"
export PYTHONPATH="$CHEFNMR_DIR${PYTHONPATH:+:$PYTHONPATH}"
python chefnmr/experiments/spec2mol/extended_augmented_training/workflow.py make \
    --campaign extended_augmented_v1 --segments 3 --fallback-hours 6 \
    --segment-hours 44 --target-epochs 10000

condor_submit -dry-run /tmp/extended_4gpu.ad logs/extended_augmented_v1/condor/S01_4.sub
condor_submit -dry-run /tmp/extended_2gpu.ad logs/extended_augmented_v1/condor/S01_2.sub
condor_submit_dag -no_submit logs/extended_augmented_v1/campaign.dag
```

**Review** generated submit ads, especially `RequestGPUs`, `gpus_minimum_memory`, `periodic_remove`, paths and resource requests. Confirm cluster policy accepts `+MaxRuntime=172800` / 48h; `+MaxRuntime` is a site policy ClassAd, not a universal HTCondor guarantee. If `-no_submit` creates a .condor.sub, it does not launch any jobs.

Submit after verification:

```bash
condor_submit_dag logs/extended_augmented_v1/campaign.dag
condor_q -dag
```

The DAG has six nodes for three segments: `S01_4 → S01_2 → S02_4 → S02_2 → S03_4 → S03_2`. The four-GPU job has a scheduler-side idle-only six-hour `periodic_remove` rule. Its POST hook authorizes the two-GPU job **only if the four-GPU executable did not start**. If it completed, the two-GPU job is skipped. If four-GPU training started then failed, the DAG stops; it does **not** submit a second conflicting job. This requires DAGMan's documented POST behavior for removed nodes and PRE_SKIP. An idle-only removal might occur slightly after six hours depending on the scheduler's periodic-evaluation interval.

## Training behavior

- Uses `trainer.fit(..., ckpt_path=...)` with the original complete `last.ckpt`, restoring optimizer + training counters (subject to Lightning compatibility).
- `max_steps=-1`, `max_epochs=10000`; epoch target is cumulative, not reset at each segment.
- Random 1 of 11 paired H1/C13 variants per molecule visit; validation always fixed reference spectra.
- `num_workers=2` per GPU rank, batch128/GPU first, then batch64, then batch32 **only if an OOM is detected**. GPU count is fixed within each job.
- One-hour recovery checkpoint under `segment_N/hourly/last.ckpt`; best-validation checkpoints under `campaign/best/`; final complete resumable state under `segment_N/last.ckpt`.
- Segment timer uses *monotonic wall time per new job*, not a restored Lightning timer state; requests stopping on epoch/validation boundaries at up to 44h. The 4h margin is intended to accommodate validation/checkpoint I/O. It is not a hard runtime guarantee if an epoch hangs or lasts exceptionally long.
- Internal test is intentionally not launched; external evaluation stays separate.
- Training and validation diffusion losses are logged on different scales in this ChefNMR implementation (`train` raw, `val` ×100); divide the validation number by 100 before direct comparison. The checkpoint monitor remains `val/diffusion_loss` to preserve its original meaning.
- `validation_args.sample_every_n_val_epoch=1000000` disables very costly structure-generation validation sampling during these segments; ordinary validation losses remain computed every epoch. Evaluate structure accuracy externally at selected milestones.
- `result.json` records final checkpoint SHA256, epoch, optimizer step, batch, GPU count, code SHA and timestamp. CSV training-step and validation-epoch metrics are stored in each `segment_N/attempt_batch*/metrics/v1/metrics.csv`.

## Inspect status and troubleshooting

```bash
python chefnmr/experiments/spec2mol/extended_augmented_training/workflow.py status extended_augmented_v1
condor_q -dag
ls -lh logs/extended_augmented_v1/segment_*/last.ckpt
```

No simultaneous independent jobs should write to this campaign. If a started segment fails, inspect its `try_batch*.log` and `hourly/last.ckpt` before manually recovering. Do not change a campaign's Git SHA mid-run. For more than three segments, generate a **new** continuation DAG or add nodes only after the previous DAG has finished; never regenerate over an active campaign.

## Explicit scope and reproducibility cautions

1. This package was created from your tracked `feature/spec2mol-augmented-training` source. Before the first GPU job, verify the installed Lightning version and inspect `trainer.fit` resume logs for epoch/global_step + optimizer restoration. CPU syntax/DAG tests do not establish training correctness.
2. The original checkpoint is 11 epochs/32,736 steps on 1 GPU with batch32. Altering per-GPU batch and DDP world size changes effective global batch and optimization dynamics. Track *steps and examples* as well as epochs.
3. 4→2 GPU fallback is achieved by a **scheduler-enforced idle-timeout plus DAGMan POST/PRE_SKIP**, not by an unverified `RequestGPUs=time()` expression.
4. Changes to `src/main.py` are not required; this standalone training entry point is separate from the archived historical launcher.
5. The 48h cap is site-specific. Confirm queue policy before submitting. For preemption, recovery depends on last successful on-disk hourly checkpoint; a mid-epoch interruption can repeat some data.
6. The best checkpoint from the initial 11 epochs remains in its historical directory. New campaign best tracking starts from the resumed stage and does not erase it.
7. Candidate checkpoints are ~2.15GB; ensure sufficient shared-filesystem quota for repeated hourly, best, and segment-boundary checkpoints. If needed, reduce retention after confirming resumability.
