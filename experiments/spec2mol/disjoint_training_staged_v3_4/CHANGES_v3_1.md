# v3.1: 120-minute HDF5 staging deadline

This is a separate, immutable launcher directory. It must NOT overwrite `launch/disjoint_training_staged_v3` while the v3 DAG is active.

Changes:
- HDF5 copy deadline 2400 s (40 min) -> 7200 s (120 min).
- `run_disjoint.sh` explicitly supplies `--max-seconds 7200`.
- Campaign generator pins files in `launch/disjoint_training_staged_v3_4`.

Unchanged:
- 50 GiB HTCondor disk request, safeguards and NFS fallback.
- 1/2-GPU competition, 60-second priority, full Lightning checkpoint resume.
- Batch selection, 43h training segment limits, 12 segment DAG.

IMPORTANT:
- A running `stage_hdf5.py` process cannot inherit a revised timeout. To test local staging, cancel the old v3 DAG (NOT the active v2.1 baseline) and submit a new v3.1 DAG under a unique campaign name.
- Verify that `logs/stage2_resume_snapshot_v3/last.ckpt` is a frozen, valid full checkpoint before generating the campaign.
- Do not edit pinned scripts during a running campaign.
- The 43h training timer starts at `fit`, after staging. 120min staging + 43h fits under the NMRbox GPU 48h limit with ~3h headroom, but this is not a guarantee under unusually slow startup or shutdown.
