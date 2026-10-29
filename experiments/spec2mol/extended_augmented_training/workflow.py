#!/usr/bin/env python3
"""GPU fallback/segment ordering for ChefNMR-S; stdlib only, DAGMan hooks."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

SKIP = 42


def project_base():
    # .../BASE/chefnmr/experiments/spec2mol/extended_augmented_training/workflow.py
    return Path(__file__).resolve().parents[4]


def campaign_dir(base, name):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_\-]*', name):
        raise ValueError('Campaign name must contain only letters, digits, _ and -')
    return base / 'logs' / name


def seg_dir(root, n):
    return root / f'segment_{int(n):02d}'


def result_info(root, n):
    path = seg_dir(root, n) / 'result.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    ckpt = Path(data.get('checkpoint', ''))
    if not ckpt.is_file() or ckpt.stat().st_size < 100 * 1024**2:
        raise RuntimeError(f'Invalid final segment checkpoint: {ckpt}')
    return data


def make(args):
    base = project_base()
    repo = base / 'chefnmr'
    archived = repo / 'experiments/spec2mol/initial_training_2026_10_28/configs/augmented/config.yaml'
    initial = base / 'logs/full_augmented_2222972.0/checkpoints/spec2mol-augmented-seed42/diffusion_loss/last.ckpt'
    if not archived.is_file():
        raise FileNotFoundError(archived)
    if not initial.is_file() or initial.stat().st_size < 100 * 1024**2:
        raise RuntimeError(f'Missing or undersized original resumable checkpoint: {initial}')
    if not (base / 'selections/strict_clean_conformers_v1/manifest.json').is_file():
        raise FileNotFoundError('Original strict split manifests missing on NMRbox')
    if not (base / 'data/molecules.h5').is_file():
        raise FileNotFoundError('Production HDF5 is missing')
    root = campaign_dir(base, args.campaign)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f'Campaign already exists: {root} (refusing overwrite)')
    root.mkdir(parents=True, exist_ok=True)
    condor_dir = root / 'condor'
    condor_dir.mkdir()
    workflow_file = root / 'campaign.dag'
    script = Path(__file__).resolve()
    python = base / 'venv/bin/python'
    launcher = script.with_name('run_segment.sh')
    if not python.is_file():
        raise FileNotFoundError(python)
    cfg = dict(campaign=args.campaign, segments=args.segments,
               fallback_hours=args.fallback_hours,
               segment_hours=args.segment_hours,
               target_epochs=args.target_epochs, original_checkpoint=str(initial),
               base=str(base), repo=str(repo))
    import subprocess
    cfg['chef_commit'] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    cfg['pipeline_commit'] = subprocess.check_output(['git', '-C', str(base/'pipeline'), 'rev-parse', 'HEAD'], text=True).strip()
    (root / 'campaign.json').write_text(json.dumps(cfg, indent=2) + '\n')
    dag = ['# Sequential extended ChefNMR-S training, 4 GPU then 2 GPU fallback.',
           '# Each four-GPU node is removed by HTCondor ONLY if idle > fallback_hours.',
           '# A failed started job stops the DAG. No two GPU jobs train the same segment.']
    fallback_seconds = int(args.fallback_hours * 3600)
    for n in range(1, args.segments + 1):
        for g in (4, 2):
            node = f'S{n:02d}_{g}'
            sub = condor_dir / f'{node}.sub'
            slot = seg_dir(root, n)
            sub.write_text('\n'.join([
                'universe = vanilla',
                'executable = /bin/bash',
                f'arguments = {launcher} {args.campaign} {n} {g}',
                f'initialdir = {base}',
                f'request_gpus = {g}',
                f'request_cpus = {8 if g==4 else 4}',
                f'request_memory = {24 if g==4 else 12}GB',
                'request_disk = 3GB',
                'gpus_minimum_memory = 38000',
                '+Production = True',
                '+MaxRuntime = 172800',
                'should_transfer_files = NO',
                'transfer_executable = FALSE',
                *([f'periodic_remove = (JobStatus == 1) && ((time() - QDate) > {fallback_seconds})'] if g == 4 else []),
                f'log = {condor_dir / node}.log',
                f'output = {condor_dir / node}.out',
                f'error = {condor_dir / node}.err',
                'queue 1', ''
            ]))
            dag.append(f'JOB {node} {sub}')
            if g == 4:
                dag.extend([
                    f'SCRIPT PRE {node} {python} {script} pre4 {args.campaign} {n}',
                    f'PRE_SKIP {node} {SKIP}',
                    f'SCRIPT POST {node} {python} {script} post4 {args.campaign} {n} $RETURN',
                ])
            else:
                dag.extend([
                    f'SCRIPT PRE {node} {python} {script} pre2 {args.campaign} {n}',
                    f'PRE_SKIP {node} {SKIP}',
                    f'SCRIPT POST {node} {python} {script} post2 {args.campaign} {n} $RETURN',
                ])
            dag.append('')
        dag.append(f'PARENT S{n:02d}_4 CHILD S{n:02d}_2')
        if n > 1:
            dag.append(f'PARENT S{n-1:02d}_2 CHILD S{n:02d}_4')
    workflow_file.write_text('\n'.join(dag) + '\n')
    print('CAMPAIGN CREATED:', root)
    print('CODE:', cfg['chef_commit'])
    print('SUBMIT: condor_submit_dag', workflow_file)
    print('DRY RUN: condor_submit -dry-run /tmp/chefnmr_4_gpu.ad', condor_dir/'S01_4.sub')
    print('DRY RUN: condor_submit -dry-run /tmp/chefnmr_2_gpu.ad', condor_dir/'S01_2.sub')


def previous_completed(root, n):
    if n == 1:
        return None
    data = result_info(root, n - 1)
    if data is None:
        raise RuntimeError(f'Segment {n-1} not finished; refusing to start segment {n}')
    return data


def pre4(args):
    root = campaign_dir(project_base(), args.campaign)
    prev = previous_completed(root, args.segment)
    if prev is not None and prev['target_reached']:
        print('SKIP: training target previously reached')
        return SKIP
    if result_info(root, args.segment):
        print('SKIP: segment already completed')
        return SKIP
    if (seg_dir(root, args.segment)/'started.json').exists():
        raise RuntimeError('Previous attempt started this segment but did not complete; inspect/recover before resubmission')
    print('PRE4 PASS: waiting for four GPUs')
    return 0


def post4(args):
    root = campaign_dir(project_base(), args.campaign)
    folder = seg_dir(root, args.segment)
    info = result_info(root, args.segment)
    if args.return_code == '0' and info:
        print('POST4 PASS: segment completed using four GPUs')
        return 0
    if (folder / 'started.json').is_file():
        raise RuntimeError('Four-GPU training started but did not complete; stop workflow for investigation')
    if args.return_code == '0':
        raise RuntimeError('Four-GPU job exited 0 but produced no result')
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'allow_2gpu_fallback').write_text('Four-GPU allocation did not start\n')
    print('POST4: four-GPU job ended without starting; allow two-GPU fallback')
    return 0


def pre2(args):
    root = campaign_dir(project_base(), args.campaign)
    prev = previous_completed(root, args.segment)
    if prev is not None and prev['target_reached']:
        print('SKIP: training target reached')
        return SKIP
    if result_info(root, args.segment):
        print('SKIP: segment completed using four GPUs')
        return SKIP
    folder = seg_dir(root, args.segment)
    if not (folder/'allow_2gpu_fallback').is_file():
        raise RuntimeError('Fallback not authorized by four-GPU POST script')
    print('PRE2 PASS: submit two-GPU fallback')
    return 0


def post2(args):
    root = campaign_dir(project_base(), args.campaign)
    if args.return_code != '0' or result_info(root, args.segment) is None:
        raise RuntimeError('Two-GPU segment failed or did not produce a verified checkpoint')
    print('POST2 PASS: segment completed using two GPUs')
    return 0


def status(args):
    root = campaign_dir(project_base(), args.campaign)
    cfg = json.loads((root/'campaign.json').read_text())
    print('CAMPAIGN', root)
    print('Git commit pinned:', cfg['chef_commit'])
    for n in range(1, cfg['segments'] + 1):
        s = seg_dir(root, n)
        item = result_info(root, n)
        if item:
            print(n, 'DONE', 'gpus=', item['gpus'], 'batch=', item['batch_size'],
                  'epoch=', item['epoch'], 'steps=', item['global_step'])
        else:
            print(n, 'started' if (s/'started.json').exists() else 'not completed')


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest='action', required=True)
    m = s.add_parser('make')
    m.add_argument('--campaign', default='extended_augmented_v1')
    m.add_argument('--segments', type=int, default=3)
    m.add_argument('--fallback-hours', type=float, default=6)
    m.add_argument('--segment-hours', type=float, default=44)
    m.add_argument('--target-epochs', type=int, default=10000)
    for name in ('pre4', 'post4', 'pre2', 'post2'):
        a = s.add_parser(name)
        a.add_argument('campaign')
        a.add_argument('segment', type=int)
        if name.startswith('post'):
            a.add_argument('return_code')
    t = s.add_parser('status')
    t.add_argument('campaign')
    args = p.parse_args()
    if args.action == 'make':
        if args.segments < 1 or args.segments > 100:
            p.error('segments must be 1..100')
        if args.fallback_hours <= 0 or args.segment_hours <= 0 or args.segment_hours > 44:
            p.error('invalid hours')
    try:
        return {'make': make, 'pre4': pre4, 'post4': post4,
                'pre2': pre2, 'post2': post2, 'status': status}[args.action](args) or 0
    except Exception as exc:
        print(f'WORKFLOW ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
