# Stage 2 autonomous GPU campaign with opportunistic local HDF5 staging (v3)

- The first allocated 1- or 2-GPU candidate wins; 1 GPU waits 60 seconds to favor a near-simultaneous 2-GPU allocation.
- The winner copies `molecules.h5` to its *own* HTCondor node-local scratch **only if** the job requests sufficient disk, the filesystem has enough free space, and the verified copy fits.
- The HDF5 copy is SHA-256 verified while copying against the production checksum. If local space/copying is unavailable, it logs why and uses NFS. No staging is attempted by the losing GPU request.
- Each successful ~43h segment saves a complete persistent checkpoint, then DAGMan automatically queues another 1- and 2-GPU pair (up to 12). No automatic recovery from a failed winner.
- Each new job normally copies the HDF5 again. Disk capacity and GPU eligibility vary by node; 50 GiB `request_disk` might substantially reduce slot matches. Keep the existing v2 campaign running until the new campaign actually trains successfully.
- `--resume-checkpoint` must name a frozen, valid, full Lightning checkpoint on persistent NFS, **not a moving hourly `last.ckpt` currently being rewritten**.
- Never edit source scripts after campaign generation. The launcher, trainer, controller and staging code are pinned by hashes and Git commit.

## Local verification
```
python launch/disjoint_training_staged_v3/test_auto_workflow.py
python launch/disjoint_training_staged_v3/test_gpu_priority_race.py
python launch/disjoint_training_staged_v3/test_stage_hdf5.py
bash -n launch/disjoint_training_staged_v3/run_disjoint.sh
```

## Generate (after commit, split validation, frozen snapshot)
```
python launch/disjoint_training_staged_v3/make_campaign.py \
  --campaign extended_augmented_disjoint_staged_v3 \
  --segments 12 --hours 43 --request-disk-gib 50 \
  --resume-checkpoint /home/nmrbox/0015/jdonner/spec2mol-chefnmr/logs/stage2_resume_snapshot/last.ckpt
condor_submit -dry-run /tmp/stage1.ad logs/extended_augmented_disjoint_staged_v3/condor/S01_1.sub
condor_submit -dry-run /tmp/stage2.ad logs/extended_augmented_disjoint_staged_v3/condor/S01_2.sub
condor_submit_dag logs/extended_augmented_disjoint_staged_v3/campaign.dag
```

## Diagnostics
```
condor_q -dag
condor_q -better-analyze <NEW_GPU_JOB_ID>
cat logs/extended_augmented_disjoint_staged_v3/segment_01/staging.json
cat logs/extended_augmented_disjoint_staged_v3/segment_01/selected_datadir.txt
```

***Do not remove old DAG 2225719 before proving a new allocation can run.***

## Safe transition from active v2 DAG 2225719

1. DO NOT change the existing `chefnmr` Git HEAD (`79ed45ff`), old trainer scripts, old campaign metadata, or old HDF5.
2. Create a frozen snapshot of the latest *complete* hourly checkpoint while the v2 job continues:
   ```
   python launch/disjoint_training_staged_v3/freeze_checkpoint.py \
     --source logs/extended_augmented_disjoint_auto_v21/segment_01/hourly/last.ckpt \
     --output logs/stage2_resume_snapshot_v3/last.ckpt
   ```
3. Run CPU tests; generate a *new* staged campaign with `--resume-checkpoint logs/stage2_resume_snapshot_v3/last.ckpt`.
4. Submit the staged DAG **in parallel with** the old campaign. This temporarily duplicates training from the frozen checkpoint. Do not cancel the old DAG until the staged job both obtains GPU allocation AND reports `STAGING_RESULT` plus actual epoch progress.
5. If `condor_q -better-analyze` shows zero matching slots for 50 GiB even if drained, do not cancel the old DAG. Treat the stage campaign as unsuitable for the cluster; retain working v2.
6. After successful staged training progress, remove the old DAG controller `condor_rm 2225719`; verify it also removes its GPU children, without modifying v3. The new campaign has its own results, log directory and checkpoint chain.
7. **If you leave before v3 receives a match**, keep the already-proven old DAG active. Staging has not been verified on NMRbox and scheduler availability cannot be guaranteed.

All generated metadata is checksum-pinned but the staging scripts in the outer `launch/` directory do not need to be committed before initial trial. A commit to the old `chefnmr/` checkout would change HEAD and cause its future segments to reject the environment. Archive a dedicated Git commit/worktree after the controlled transition if desired.
