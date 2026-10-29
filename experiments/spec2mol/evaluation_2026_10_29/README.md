# ChefNMR-S evaluation package — 28 October 2026

Source code reviewed from `jdo514/chefnmr`, branch `feature/spec2mol-external-splits`, commit used at NMRbox `b7918a68896597e7faec336b0345fb74197afca1`, and pipeline `24f1f22c7c7cadfd8ec64b46116ba6035b91d395`.

This is a **staged evaluation package**, not proof that experimental test data have already been converted. It does not edit either Git repository, the 42.7 GB master HDF5, frozen training/validation manifest, or original trained checkpoints. It creates new output directories only.

## What does loss 88.19 mean?

ChefNMR `src/model/model.py` logs `val/diffusion_loss = diffusion_loss * 100`. `src/model/modules/diffusion.py` computes `diffusion_loss = denoising_MSE + smooth_lDDT_loss`. Therefore 88.19 means validation objective ~0.8819; 88.26 means ~0.8826. Neither value is a percent of correct/wrong identifications. Actual structure match accuracy is `test/matching_accuracy` (percent), calculated by reconstructing predicted SMILES and matching exact connectivity without stereochemistry. Report top-k (`diffusion_samples`) explicitly.

## Deployment

On your laptop: unzip the archive and transfer its **contents** to the NMRbox folder:

```bash
rsync -av chefnmr_eval_2026-10-28/ jdonner@indium.nmrbox.org:spec2mol-chefnmr/launch/chefnmr_eval/
```

Then SSH to NMRbox:

```bash
cd ~/spec2mol-chefnmr
source venv/bin/activate
python -m py_compile launch/chefnmr_eval/*.py
bash -n launch/chefnmr_eval/run_eval.sh
```

## Stage 1: Verify checkpoints reload / EMA state

On NMRbox execute in `~/spec2mol-chefnmr/chefnmr` (ensure Python can import `src`):

```bash
cd ~/spec2mol-chefnmr/chefnmr
python ../launch/chefnmr_eval/checkpoints.py --load \
  --checkpoint ../logs/full_reference_2222733.0/checkpoints/spec2mol-reference-seed42/diffusion_loss/epoch0010-loss88.19.ckpt \
  --checkpoint ../logs/full_augmented_2222972.0/checkpoints/spec2mol-augmented-seed42/diffusion_loss/epoch0010-loss88.26.ckpt \
  --output ../logs/eval_checkpoint_audit.json
```

This loads roughly 2.15GB checkpoint files and needs several GiB of RAM. Avoid doing it on a login node if local NMRbox policy prohibits substantial compute there; stage 1 is also implicitly exercised by stage 2's short GPU smoke jobs. `--hash` optionally records SHA256.

## Stages 2 and 3: First genuine molecular reconstruction test on frozen validation data

This is **an internal held-out validation control** (simulated data), *not independent SpecTeach/BMRB100 testing*. It is an essential pipeline smoke before attempting experimental data.

```bash
cd ~/spec2mol-chefnmr
condor_submit launch/chefnmr_eval/eval_smoke.sub
# Only after BOTH jobs return 0 and print CHEFNMR_EVALUATION_PASS:
condor_submit launch/chefnmr_eval/eval_heldout.sub
```

Smoke: 2 molecules, 1 structure candidate, 10 diffusion steps. Main validation control: same deterministic subset of 100 held-out molecules, 1 candidate, 50 diffusion steps for both checkpoints. Each job saves `metrics.json`, `manifest.json`, and ChefNMR-predicted structures under `logs/evaluation_*`.

For a scientifically useful top-k study, rerun after smoke with `candidates=5` or `10` for both models and the same 100 or more molecules. Top-k is *not* comparable to top-1.

Pair the results with `compare.py`, using real job directory paths (do not mix run settings):

```bash
cd ~/spec2mol-chefnmr/chefnmr
python ../launch/chefnmr_eval/compare.py \
  --reference ../logs/evaluation_frozen-val_reference_variant0_n100_k1_s50_JOBREF \
  --augmented ../logs/evaluation_frozen-val_augmented_variant0_n100_k1_s50_JOBAUG \
  --output ../logs/paired_frozen_val_variant0.json
```

`compare.py` compares molecular connectivity-matching accuracy (stereo ignored, as in ChefNMR), validity, Tanimoto similarity, and molecule-paired bootstrap uncertainty. It checks dataset, molecule IDs, perturbation variant, sampling steps, seed, and candidate count match exactly. It refuses mismatched pairs.

## Stage 4: Spectral perturbation robustness

Your converted production archive contains 1 reference spectrum (variant 0) plus 10 paired augmented H/C spectrum variants (variants 1–10). Use **the same unseen validation molecules** to compare both models under identical pre-existing perturbations, keeping all other settings unchanged:

```bash
condor_submit launch/chefnmr_eval/eval_variants.sub
```

This submits 20 GPU jobs (two models × 10 variants) on the 100 selected validation molecules. Run only after the two-model smoke and reference-spectrum evaluation pass. The original paired augmentation can involve multiple changing properties; **do not attribute a difference specifically to linewidth/field/shift without examining per-variant metadata**. A stronger generalization study would use new independent spectral perturbations rather than recycling the exact augmentation generator; that is a later extension.

## Experimental testing: SpecTeach and BMRB100

Current ChefNMR model requires **both spectra in the correct numeric representation** `h_10k: (1,10000)` and `c_80: (1,80)`, correct normalization and axis orientation, compatible 3D coordinates/atom masks, CHON atom mapping (4 channels), <=59 atoms and compatible per-molecule `smiles`. Raw NMR peak lists or FIDs cannot be used directly. **Do not feed raw experimental data to this runner and call the result test accuracy.**

Inspect original archives:

```bash
python ~/spec2mol-chefnmr/launch/chefnmr_eval/inventory.py /path/to/experimental.zip
```

Once a source-specific, scientifically defensible adapter has converted each experimental dataset to a *separate* ChefNMR-compatible `molecules.h5`, audit it:

```bash
python ~/spec2mol-chefnmr/launch/chefnmr_eval/inventory.py \
  /path/to/bmrb100_converted/molecules.h5 \
  --output ~/spec2mol-chefnmr/logs/bmrb100_inventory.json
```

Audit reports eligible, excluded and reasons. Verify chemical identity, formula, NMR axis/bins, scaling, and 3D conformer provenance. All validation must use the **training-model CHON/59 atom type setting**, not a silently enlarged architecture. Enantiomers cannot be distinguished by achiral NMR spectra, and ChefNMR currently scores structure connectivity with stereochemistry removed.

Once audit reports `READY` and the experimental source-specific conversion is independently validated, a minimal command is:

```bash
cd ~/spec2mol-chefnmr
bash launch/chefnmr_eval/run_eval.sh reference bmrb100 0 12 1 50 manualrun \
  /path/to/bmrb100_converted ~/spec2mol-chefnmr/logs/bmrb100_inventory.json
```

Run the matching augmented model with identical arguments and an untouched selection manifest. Repeat for SpecTeach. These are **not automatically scheduled in the current package**, because neither external converted HDF5 nor its validated audit path was provided; claiming otherwise would be misleading.

## Caveats

- `torch.load`/Lightning checkpoint loading unpickles checkpoint contents. Load only the two *trusted* checkpoint files produced by your own training jobs.
- The source code's `test_step` does bond reconstruction and all-conformer RMSD, which may be costly. Begin with 2- and 100-molecule evaluations before full tests. GPU jobs request 4 CPU cores, 12 GB RAM and 1 GPU.
- All evaluations use EMA weights as in the original ChefNMR `on_test_start` hook, and 50 diffusion steps (except smoke). External testing may require a validated normalization and simulator-to-experimental harmonization procedure; missing 3D conformers prevent the original RMSD metric.
- A **single 11-epoch model per condition** does not establish seed-to-seed variation. Replicate if possible after initial testing.
- Neither reference nor augmented checkpoint has an observed experimental accuracy yet. Do not interpret 88.19 vs 88.26 as accuracy.
