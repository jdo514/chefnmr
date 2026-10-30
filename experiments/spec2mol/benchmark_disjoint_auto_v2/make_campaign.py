#!/usr/bin/env python3
"""Generate (but DO NOT submit) an automatic multi-segment stage-2 HTCondor DAG.

Each segment queues independent 1- and 2-GPU candidates. First to acquire
shared per-segment claim trains. Winner's POST cancels sibling, and DAGMan
queues both candidates for the next segment only after this segment succeeds.
"""
import argparse
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path.home() / 'spec2mol-chefnmr')
    p.add_argument('--campaign', default='extended_augmented_disjoint_auto_v2')
    p.add_argument('--segments', type=int, default=12)
    p.add_argument('--hours', type=float, default=43.0)
    a = p.parse_args()
    if not (1 <= a.segments <= 200):
        raise ValueError('--segments must be 1..200')
    if not (0 < a.hours <= 43):
        raise ValueError('--hours must be >0 and <=43')
    if not a.campaign.startswith('extended_augmented_disjoint_') or not a.campaign.replace('_', '').isalnum():
        raise ValueError('Choose an unambiguous alphanumeric campaign name prefixed extended_augmented_disjoint_')
    base = a.base.expanduser().resolve(strict=True)
    repo, work = base/'chefnmr', base/'launch'/'disjoint_training_auto_v2'
    selection = base/'selections'/'benchmark_molecule_disjoint_v1'
    manifest_path = selection/'manifest.json'
    m = json.loads(manifest_path.read_text())
    if m['audit'] != {'benchmark_exact_structure_overlap_train': 0,
                      'benchmark_exact_structure_overlap_val': 0,
                      'train_val_structure_identity_overlap': 0,
                      'previously_trained_validation_overlap': 0}:
        raise RuntimeError('Benchmark disjointness audit failed')
    if m['benchmarks']['BMRB100']['rows'] != 100 or m['benchmarks']['SpecTeach']['rows'] != 238:
        raise RuntimeError('Require all benchmark sources')
    if (m['train_molecules'], m['validation_molecules']) != (127481, 14165):
        raise RuntimeError('Split changed from confirmed 127481/14165; inspect before continuing')
    old_train = base/'selections/strict_clean_conformers_v1/train.npy'
    if sha(old_train) != m['previous_training_ids_sha256']:
        raise RuntimeError('Original Stage 1 train selection changed')
    for split in ('train', 'val', 'test'):
        if sha(selection/f'{split}.npy') != m['split_file_sha256'][split]:
            raise RuntimeError(f'Split {split} hash changed')
    for name in ('run_disjoint.sh', 'train_disjoint.py', 'control.py'):
        if not (work/name).is_file():
            raise RuntimeError(f'Missing training workflow source: {work/name}')
    if 'BATCHES=(64 32)' not in (work/'run_disjoint.sh').read_text():
        raise RuntimeError('Global batch policy patch absent')
    if '"$FOLDER/.winner_claim"' not in (work/'run_disjoint.sh').read_text():
        raise RuntimeError('Per-segment lock patch absent')
    original = (base/'logs/full_augmented_2222972.0/checkpoints/'
               'spec2mol-augmented-seed42/diffusion_loss/last.ckpt')
    if original.stat().st_size < 100 * 1024**2:
        raise RuntimeError('Original full Lightning checkpoint absent or undersized')
    current = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if dirty:
        raise RuntimeError('Refusing dirty ChefNMR checkout; commit changes first')
    root = base/'logs'/a.campaign
    if root.exists():
        raise FileExistsError(f'{root} exists: never overwrite campaign')
    root.mkdir(parents=True)
    (root/'condor').mkdir()
    meta = dict(campaign=a.campaign, campaign_key=uuid.uuid4().hex,
                segments=a.segments, target_epochs=10000, segment_hours=a.hours,
                original_checkpoint=str(original), chef_commit=current,
                base=str(base), repo=str(repo), dataset_name='benchmark_molecule_disjoint_v1',
                split_manifest_sha256=sha(manifest_path),
                train_script_sha256=sha(work/'train_disjoint.py'),
                launcher_script_sha256=sha(work/'run_disjoint.sh'),
                control_script_sha256=sha(work/'control.py'),
                train_molecules=m['train_molecules'], validation_molecules=m['validation_molecules'],
                requested_global_batch=128, first_wins=True,
                protocol='stage2_original_augmented_ckpt_full_optimizer_continuation_1or2GPU_43h_segments')
    (root/'campaign.json').write_text(json.dumps(meta,indent=2)+'\n')
    dag = []
    for seg in range(1,a.segments+1):
        nodes=[]
        for g in (1,2):
            node = f'S{seg:02d}_{g}'
            nodes.append(node)
            submit = root/'condor'/f'{node}.sub'
            lines = [
                'universe = vanilla',
                'executable = /bin/bash',
                f'arguments = {work/"run_disjoint.sh"} {a.campaign} {seg} {g}',
                f'initialdir = {base}',
                f'request_gpus = {g}',
                f'request_cpus = {2 if g==1 else 4}',
                f'request_memory = {8 if g==1 else 12}GB',
                'request_disk = 3GB',
                'gpus_minimum_memory = 38000',
                '+Production = True',
                '+MaxRuntime = 172800',
                f'+Spec2MolCampaignKey = "{meta["campaign_key"]}"',
                f'+Spec2MolSegment = {seg}',
                f'+Spec2MolCandidateGPUs = {g}',
                'should_transfer_files = NO',
                'transfer_executable = FALSE',
                f'log = {root/"condor"/f"{node}.log"}',
                f'output = {root/"condor"/f"{node}.out"}',
                f'error = {root/"condor"/f"{node}.err"}',
                'queue 1', '',
            ]
            submit.write_text('\n'.join(lines))
            dag += [f'JOB {node} {submit}',
                    f'SCRIPT PRE {node} /usr/bin/python3 {work/"control.py"} pre {root} {seg}',
                    f'PRE_SKIP {node} 42',
                    f'SCRIPT DEFER 75 60 POST {node} /usr/bin/python3 {work/"control.py"} post {root} {seg} {g} $RETURN $JOBID']
        if seg > 1:
            previous = [f'S{seg-1:02d}_{g}' for g in (1, 2)]
            dag += [f'PARENT {" ".join(previous)} CHILD {" ".join(nodes)}']
    dag += [f'NODE_STATUS_FILE {root/"campaign.node_status"} 60', '']
    (root/'campaign.dag').write_text('\n'.join(dag))
    print('AUTOMATIC DAG GENERATED (NOT SUBMITTED):', root/'campaign.dag')
    print('SEGMENTS',a.segments,'GPUS candidates 1 and 2, global batch initial 128')
    print('CHEFNMR COMMIT', current)
    print('GPU SEGMENT HOURS',a.hours)
    print('SPLIT train',m['train_molecules'],'val',m['validation_molecules'])
    print('DRY RUN: condor_submit -dry-run /tmp/auto1.ad', root/'condor/S01_1.sub')
    print('DRY RUN: condor_submit -dry-run /tmp/auto2.ad', root/'condor/S01_2.sub')
    print('SUBMIT: condor_submit_dag',root/'campaign.dag')


if __name__ == '__main__':
    main()
