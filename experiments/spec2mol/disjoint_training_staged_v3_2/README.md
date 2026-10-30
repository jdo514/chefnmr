# ChefNMR-S molecule-disjoint automatic training — v2

## Training policy

Resume original `full_augmented_2222972.0/.../last.ckpt` with full Lightning state.
Master: 142,277; external-identity-disjoint selection: 127,481 train / 14,165 validation; no scaffold filter. Do **not** change `molecules.h5` or frozen selection.

- Each segment independently queues **one 1-GPU candidate** and **one 2-GPU candidate**.
- The **2-GPU candidate claims immediately**; a **1-GPU candidate waits 60 seconds after starting** before attempting the same per-segment claim. This favors 2 GPUs if both jobs start close together, but **does not guarantee 2-GPU priority in HTCondor scheduling**: an allocated 1-GPU job holds its GPU during the 60-second wait. The loser exits.
- 1 GPU tries per-device batch **128, 64, 32**; 2 GPUs try **64, 32**. Initial global batch = **128** on either allocation. If OOM fallback occurs, global batch can drop to 64 (or 32 for 1 GPU).
- On successful result, winner's DAGMan POST script cancels the losing candidate (narrowly matched by unique campaign key, segment and alternative GPU count). DAGMan queues the next pair **only when both prior nodes complete**.
- Segment timer: **43 hours** by default; 48-hour NMRbox GPU limit. Stops at epoch boundary, so hard-limit preemption is still possible. Hourly Lightning checkpoint at `segment_XX/hourly/last.ckpt`; complete final checkpoint at `segment_XX/last.ckpt`, validated by SHA256 before subsequent segment.
- If training crashes or there is no valid final result, DAG stops for manual intervention. No autonomous crash restart in v2.
- Default is **12 segments**, not unlimited GPU use. User may choose `--segments 24` or another fixed cap. Stops early if `target_epochs=10000` is reached. Epoch count across Stage 1 and 2 is cumulative, but different populations and GPU counts mean samples-per-epoch and iterations-per-epoch differ.
- Changing between 1 and 2 GPUs between completed segments is supported by Lightning full checkpoint restore in principle. Verify first actual 1→2 or 2→1 transition; data shuffle and random augmentation sequences will not be bitwise identical, especially if resuming an hourly mid-epoch checkpoint.

## 1. Install on NMRbox (keep old package intact)

Transfer `ChefNMR_Disjoint_Auto_1or2GPU_v2.zip` to `~/spec2mol-chefnmr/`, then:

```bash
cd ~/spec2mol-chefnmr
unzip -o ChefNMR_Disjoint_Auto_1or2GPU_v2.zip -d launch/
source venv/bin/activate
python launch/disjoint_training_auto_v2/test_auto_workflow.py
python -m compileall -q launch/disjoint_training_auto_v2
bash -n launch/disjoint_training_auto_v2/run_disjoint.sh
```

`launch/disjoint_training_v1` is not modified. `patch_batch_policy.py` is an idempotent helper to fix a *copy* of any original v1 launcher, including per-segment mutex/claim. The v2 package already includes the fixed launcher; no patch invocation is needed for v2.

## 2. Version files BEFORE generating a campaign

```bash
cd ~/spec2mol-chefnmr
mkdir -p chefnmr/experiments/spec2mol/benchmark_disjoint_auto_v2
cp -a launch/disjoint_training_auto_v2/. chefnmr/experiments/spec2mol/benchmark_disjoint_auto_v2/
cd chefnmr
git add experiments/spec2mol/benchmark_disjoint_auto_v2/
git diff --cached --stat
git diff --cached --check
git commit -m 'feat(training): add multi-segment 1-2gpu disjoint continuation DAG'
git push origin feature/spec2mol-augmented-training
```

The campaign generator pins this commit, launcher checksum, training script checksum, controller checksum, and frozen manifest checksum. Source must remain unchanged while campaign is running.

## 3. Generate campaign, inspect without submitting

**Check that there are no previously submitted candidate jobs for the same new selection.** The separate old `extended_augmented_disjoint_v1` campaign, if submitted, must not run concurrently with this new campaign; identify and remove its specific jobs intentionally, not by broad `condor_rm -all`.

```bash
cd ~/spec2mol-chefnmr
condor_q -nobatch
python launch/disjoint_training_auto_v2/make_campaign.py --segments 12 --hours 43
condor_submit -dry-run /tmp/stage2_auto_1.ad logs/extended_augmented_disjoint_auto_v2/condor/S01_1.sub
condor_submit -dry-run /tmp/stage2_auto_2.ad logs/extended_augmented_disjoint_auto_v2/condor/S01_2.sub
condor_submit_dag -no_submit logs/extended_augmented_disjoint_auto_v2/campaign.dag
```

Before submission verify `condor_q` and `condor_rm` are accessible on the NMRbox submit/access point (used by POST script) and inspect `campaign.dag` and `condor/*.sub`. Test DAGMan parsing on the actual installed version. This `-no_submit` command produces a DAGMan submit description without submitting the DAG.

## 4. Submit and monitor

```bash
condor_submit_dag logs/extended_augmented_disjoint_auto_v2/campaign.dag
condor_q -dag
condor_q -nobatch
```

After a segment starts:

```bash
cat logs/extended_augmented_disjoint_auto_v2/segment_01/started.json
cat logs/extended_augmented_disjoint_auto_v2/segment_01/.winner_claim/owner.json
tail -n 60 logs/extended_augmented_disjoint_auto_v2/segment_01/try_batch128.log 2>/dev/null || true
tail -n 60 logs/extended_augmented_disjoint_auto_v2/segment_01/try_batch64.log 2>/dev/null || true
find logs/extended_augmented_disjoint_auto_v2 -name result.json -o -name '*.ckpt' | sort
```

After a segment passes, its POST script removes the waiting or redundant candidate. The next segment will queue one and two GPU requests independently. `condor_q -dag` counts submitted nodes, not completed training segments: inspect `segment_XX/result.json` for scientific progress.

## Failure and safety cases

- A winner that exits without `result.json` fails the DAG. A loser may exit successfully only if the segment has another documented winner.
- A removed loser whose winner is still training has its POST deferred and retried once per minute; on winner completion the POST succeeds.
- A winning POST checks the full checkpoint SHA256 before canceling the sibling. The next PRE also validates it.
- A persistent claim prevents reusing a failed segment without manual auditing. Don't delete `started.json`, `.winner_claim` or checkpoints while training is active.
- The original training history includes 3,178 Stage 1 *validation* molecules reused in Stage 2 validation; they were not Stage 1 optimizer-training samples. Disclose this for model-selection independence.
- The split's chemical identity rule is stereo-insensitive canonical SMILES, not exhaustive standardization of tautomer/salt/protonation equivalents.
- The first real 2-GPU checkpoint restoration and NMRbox POST `condor_rm` operations are **not yet live-tested**. Inspect DAGMan logs after the first completed segment. Do not assume tested across cluster until observed.

## v2.1 — 60-second two-GPU preference

The only runtime behavior change is in `run_disjoint.sh`: the 1-GPU candidate waits 60 seconds **before** opening the winner lock or creating the persistent claim; the 2-GPU candidate does not wait. When both jobs start within 60 seconds and the two-GPU job reaches the claim in time, the two-GPU job wins. If 1 GPU is the only available allocation, it trains after 60 seconds. The losing candidate exits without touching checkpoint state. The campaign metadata records `two_gpu_claim_priority_seconds=60`. All code must be committed and the campaign generated after patching; do not replace a launcher already checksum-pinned by a submitted campaign.
