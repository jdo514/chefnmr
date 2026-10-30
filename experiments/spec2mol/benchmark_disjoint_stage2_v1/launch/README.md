# ChefNMR-S stage-2 molecule-disjoint training (2026-10-30)

**Purpose.** Expand the original ChefNMR-compatible SpectraBase training population to *molecule-disjoint* training while permitting scaffolds shared with BMRB100 and SpecTeach. Resume the original augmented model (11 epochs), not a new random initialization. The experimental evaluation references remain unavailable as supervised train/validation targets.

## Important methodological constraints

1. **Use ALL 100 BMRB100 source identities and ALL 238 official SpecTeach identities.** The experimental *pilot* conversions contain only 20 BMRB100 and 213 compatible SpecTeach molecules; those subsets are insufficient to define the complete benchmark holdout. `export_full_benchmarks.py` rejects incomplete sources.
2. The old `strict_clean_conformers_v1/train.npy` (95,202 molecules) is **mandatory**. Its IDs and stereochemistry-insensitive identity equivalents are assigned to TRAIN, never new validation. Otherwise the resumed checkpoint would already have seen some validation molecules.
3. The HDF5 stays read-only and is NOT copied or changed. All eleven paired variants remain in the same molecule split. No Murcko-scaffold exclusion is performed.
4. Identity audit uses RDKit nonisomeric canonical SMILES. It does not claim to detect every tautomeric, salt, charge-state, or chemical-standardization equivalence; investigate those before making stronger independence claims.
5. Stage 2 is **continuation training**, not an independently trained full-data-from-scratch model. Comparisons to the published ChefNMR-S model and the initial runs must disclose this.
6. The BMRB100 data helped develop shift-error calibration in the pipeline; do not present it as an untouched hyperparameter-selection benchmark.

## STEP 1 — Export complete benchmark reference identities on laptop

After unzipping this package locally, run (adjust only the original source paths if necessary):

```bash
cd ~/spec2mol-dataset-pipeline
micromamba run -n spec2mol-dataset python \
  ~/chefnmr_disjoint_work/disjoint_training_v1/export_full_benchmarks.py \
  --bmrb-manifest data/processed/bmrb100_experimental/manifest.csv \
  --specteach-h5 data/external/specteach_chefnmr_official/extracted/real-specteach/molecules.h5 \
  --output data/processed/disjoint_full_benchmark_refs_v1
```

Expected: `FULL BENCHMARK REFERENCE EXPORT PASS: BMRB100 100; SpecTeach 238`.

Transfer just the compact reference identities to NMRbox:

```bash
ssh jdonner@indium.nmrbox.org 'mkdir -p ~/spec2mol-chefnmr/selections/benchmark_references'
scp data/processed/disjoint_full_benchmark_refs_v1/*.csv \
  jdonner@indium.nmrbox.org:~/spec2mol-chefnmr/selections/benchmark_references/
```

## STEP 2 — Install package and build new selection on indium

Copy ZIP file to NMRbox, then:

```bash
cd ~/spec2mol-chefnmr
unzip -o ChefNMR_Benchmark_Disjoint_Stage2_v1.zip -d launch/
source venv/bin/activate
python launch/disjoint_training_v1/test_local.py

python launch/disjoint_training_v1/build_molecule_disjoint.py \
  --master data/molecules.h5 \
  --bmrb selections/benchmark_references/bmrb100_full.csv \
  --specteach selections/benchmark_references/specteach_full.csv \
  --previous-train selections/strict_clean_conformers_v1/train.npy \
  --output selections/benchmark_molecule_disjoint_v1
```

Expected: `BENCHMARK-MOLECULE-DISJOINT SPLIT BUILD PASS`. The script refuses overwrite and insists on a 142,277-eligible-ID source archive and 95,202 original training IDs (unless explicitly changed for synthetic tests). It can take some minutes to scan 142k HDF5 metadata records.

Review results:

```bash
cat selections/benchmark_molecule_disjoint_v1/audit.json
python - <<'PY'
import json,numpy as np
p='selections/benchmark_molecule_disjoint_v1/'
m=json.load(open(p+'manifest.json'))
print('TRAIN',m['train_molecules'],'VALIDATION',m['validation_molecules'])
print('EXCLUDED',m['excluded_reasons'])
print('AUDIT',m['audit'])
print('FULL BENCHMARKS',m['benchmarks'])
a=np.load(p+'train.npy',allow_pickle=False)
b=np.load(p+'val.npy',allow_pickle=False)
old=np.load('selections/strict_clean_conformers_v1/train.npy',allow_pickle=False)
assert len(np.intersect1d(old,b))==0
assert len(np.setdiff1d(old,a))==0
print('STAGE1 TRAIN IDS IN NEW TRAIN AND OUT OF NEW VAL: PASS')
PY
```

The existing `src.data.external_split_loader` accepts the resulting `manifest.json` and `train.npy`, `val.npy`, `test.npy` without modifying ChefNMR or the original HDF5.

## STEP 3 — Generate a NEW isolated HTCondor campaign (no submission yet)

```bash
cd ~/spec2mol-chefnmr
python launch/disjoint_training_v1/prepare_jobs.py --gpus 1 2
condor_submit -dry-run /tmp/disjoint_1g.ad logs/extended_augmented_disjoint_v1/condor/candidate_1gpu.sub
condor_submit -dry-run /tmp/disjoint_2g.ad logs/extended_augmented_disjoint_v1/condor/candidate_2gpu.sub
```

This pins the current ChefNMR commit, the new manifest hash, the launcher and training script hashes, and the original augmented 11-epoch checkpoint. It refuses overwrite of an existing campaign directory.

## STEP 4 — Retire old-data requests, then submit new 1/2 GPU candidates

Only AFTER the new dataset audit and dry runs pass, inspect the old DAG and any independently submitted strict-clean candidate:

```bash
condor_q -dag
condor_q -nobatch
```

To remove the **known old strict-clean DAG 2224092** (and DAG-managed children), when you have verified that no old-data training needs to be retained:

```bash
condor_rm 2224092
```

The old separate one-GPU candidate, if it was submitted, is **not** part of that DAG. Remove its job ID separately with `condor_rm <OLD_CANDIDATE_JOB_ID>`, after confirming the ID. Verify old campaign jobs are gone using `condor_q -nobatch`.

Submit one or both new jobs:

```bash
condor_submit logs/extended_augmented_disjoint_v1/condor/candidate_1gpu.sub
condor_submit logs/extended_augmented_disjoint_v1/condor/candidate_2gpu.sub
```

**Race policy.** Both requests queue, but `flock` plus an atomic `mkdir` claim prevent both candidates from training. A later candidate exits without training. The shared output is `logs/extended_augmented_disjoint_v1/segment_01/`. If the first candidate crashes, there is *no automatic retry*; the marker blocks another candidate until manually audited. Avoid editing files after job submission.

**Training.** Full Lightning optimizer-state resume of `logs/full_augmented_2222972.0/checkpoints/spec2mol-augmented-seed42/diffusion_loss/last.ckpt`. Training uses `paired_random`, selective HDF5 reads, two loader workers/GPU, per-GPU batch 128→64→32 on CUDA OOM, model configuration from the archived augmented config, separate validation scoring, hourly recovery checkpoints, and an approximately 43-hour epoch-boundary stop (48-hour cluster ceiling; no absolute guarantee). Runs must be monitored after first GPU allocation.

## STEP 5 — Verify first allocation

```bash
condor_q -nobatch
find logs/extended_augmented_disjoint_v1/segment_01 -maxdepth 3 -type f | sort | head -50
cat logs/extended_augmented_disjoint_v1/segment_01/started.json 2>/dev/null
head -80 logs/extended_augmented_disjoint_v1/segment_01/try_batch128.log 2>/dev/null
```

Look for optimizer/checkpoint restore at epoch 11 and global step 32736, confirmed full new split counts, finite losses, first validation, and first hourly checkpoint. **No automatic crash restart** in this version; preserve all checkpoints for a reviewed recovery.

## Code and test status

- Python syntax/shell syntax verified and the split construction tested end-to-end on a synthetic HDF5 with duplicate structures and benchmark exclusions.
- Real 142,277-molecule metadata scan, full external-source export, Lightning checkpoint continuation, and actual HTCondor allocation remain **to be executed on NMRbox**.
- No NMRbox job is submitted from ChatGPT; `condor_submit` runs only in your NMRbox terminal.
