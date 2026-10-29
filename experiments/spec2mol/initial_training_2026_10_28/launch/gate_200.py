#!/usr/bin/env python3
"""DAGMan PRE gate: require successful, finite, sufficiently fast 200-step benchmark.

Exit 75: job pending (DAGMan defers). Exit 1: unsafe/failure (blocks training).
Exit 0: only after results verified and approved.json is written atomically.
Python 3.11 from the user's ChefNMR venv must run this script.
"""
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/home/nmrbox/0015/jdonner/spec2mol-chefnmr')
OUT = BASE / 'launch' / 'conditional_full_training'
WAIT = 75
MAX_STEP_SECONDS = 2.0
MAX_RESERVED_MIB = 32000.0
MIN_AVAILABLE_GIB = 35.0
MIN_CHECKPOINT_MIB = 100.0


def report(message, status):
    print(f'FULL_TRAINING_GATE: {message}', flush=True)
    return status


def condor_ad(job_id):
    for command in (
        ['condor_q', job_id, '-af', 'JobStatus', 'ExitCode', 'ExitBySignal'],
        ['condor_history', job_id, '-limit', '1', '-af', 'JobStatus', 'ExitCode', 'ExitBySignal'],
    ):
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=25)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f'FULL_TRAINING_GATE: scheduler unavailable: {exc}', flush=True)
            return None
        if p.returncode:
            print(f'FULL_TRAINING_GATE: query failed: {p.stderr.strip()}', flush=True)
            continue
        rows = [line.split() for line in p.stdout.splitlines() if line.strip()]
        if not rows:
            continue
        if len(rows) != 1 or len(rows[0]) < 1 or not rows[0][0].isdigit():
            raise ValueError(f'Unexpected HTCondor response: {p.stdout!r}')
        row = rows[0]
        return int(row[0]), row[1] if len(row) > 1 else 'undefined', row[2].lower() if len(row) > 2 else 'undefined'
    return None


def validate_benchmark(job_id):
    passing = []
    for batch in (32, 16, 8):
        run_dir = BASE / 'logs' / f'benchmark_{job_id}_batch{batch}'
        meta_path = run_dir / 'run.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get('status') != 'PASS':
            print(f'FULL_TRAINING_GATE: batch {batch} status={meta.get("status")}', flush=True)
            continue
        if meta.get('global_step') != 200 or meta.get('steps_requested') != 200:
            raise ValueError(f'Incorrect training steps in {meta_path}')
        if meta.get('selection') != 'strict_clean_conformers_v1' or meta.get('batch_size') != batch:
            raise ValueError(f'Incorrect selection or batch size in {meta_path}')
        if meta.get('pipeline_commit') != '24f1f22c7c7cadfd8ec64b46116ba6035b91d395':
            raise ValueError(f'Unexpected pipeline commit {meta.get("pipeline_commit")}')
        timing_path = run_dir / 'timings.json'
        t = json.loads(timing_path.read_text())
        records = t['records']
        if [r['step'] for r in records] != [20, 100, 200]:
            raise ValueError(f'Missing 20,100,200 milestones in {timing_path}')
        if t.get('batch_size') != batch:
            raise ValueError(f'Incorrect batch size in timing file: {batch}')
        for record in records:
            if not all(math.isfinite(float(record[k])) for k in ('training_loss', 'avg_seconds_per_step', 'segment_seconds_per_step', 'peak_cuda_reserved_mib')):
                raise ValueError(f'Nonfinite milestone: {record}')
            if float(record['training_loss']) < 0 or float(record['segment_seconds_per_step']) <= 0:
                raise ValueError(f'Invalid loss or timing milestone: {record}')
        last = records[-1]
        seconds_per_step = float(last['segment_seconds_per_step'])
        reserved_mib = float(last['peak_cuda_reserved_mib'])
        if seconds_per_step > MAX_STEP_SECONDS:
            raise ValueError(f'Last 100 steps too slow: {seconds_per_step:.3f}s/step > {MAX_STEP_SECONDS}s/step')
        if reserved_mib > MAX_RESERVED_MIB:
            raise ValueError(f'GPU memory {reserved_mib:.1f}MiB > {MAX_RESERVED_MIB:.1f}MiB')
        checkpoints = [Path(p) for p in meta.get('checkpoint', [])]
        if not checkpoints or not all(p.is_file() for p in checkpoints):
            raise ValueError(f'Full Lightning checkpoint missing for batch {batch}')
        sizes = [p.stat().st_size / (1024 ** 2) for p in checkpoints]
        if not all(size >= MIN_CHECKPOINT_MIB for size in sizes):
            raise ValueError(f'Unexpectedly small checkpoint(s): {sizes}')
        passing.append((batch, meta, seconds_per_step, reserved_mib, sizes))

    if len(passing) != 1:
        raise ValueError(f'Expected exactly one successful batch setting, got {len(passing)}')
    batch, meta, seconds_per_step, reserved_mib, sizes = passing[0]
    free_gib = shutil.disk_usage(BASE / 'logs').free / 2**30
    if free_gib < MIN_AVAILABLE_GIB:
        raise ValueError(f'Free filesystem space {free_gib:.1f}GiB < {MIN_AVAILABLE_GIB:.1f}GiB; check quota separately')
    steps_per_epoch = (95202 + batch - 1) // batch
    projected_training_hours = steps_per_epoch * 11 * seconds_per_step / 3600
    # A conservative estimate: add 25% for validation, checkpoints and system overhead.
    projected_run_hours = projected_training_hours * 1.25
    if projected_run_hours > 36:
        raise ValueError(f'Projected one-model runtime {projected_run_hours:.1f}h > 36h safety threshold')

    selection_manifest = json.loads((BASE / 'selections/strict_clean_conformers_v1/manifest.json').read_text())
    if any(int(selection_manifest[k]) != v for k, v in [('train_molecules',95202),('validation_molecules',10575),('test_molecules',0)]):
        raise ValueError('Frozen selection metadata changed')

    return dict(approved_utc=datetime.now(timezone.utc).isoformat(), benchmark_job=job_id,
                batch_size=batch, epochs=11, train_molecules=95202,
                steps_per_epoch=steps_per_epoch, max_steps=steps_per_epoch*11,
                segment_seconds_per_step=seconds_per_step,
                projected_hours_per_model_including_25pct_overhead=round(projected_run_hours,2),
                peak_cuda_reserved_mib=reserved_mib, checkpoints_mib=sizes,
                filesystem_free_gib=round(free_gib,1),
                pipeline_commit=meta.get('pipeline_commit'), chef_commit=meta.get('chef_commit'),
                selection='strict_clean_conformers_v1',
                note='Filesystem free space does not account for user/project storage quotas; verify quota before DAG submission.')


def main():
    if len(sys.argv) != 2 or not re.fullmatch(r'\d+\.\d+', sys.argv[1]):
        return report('Usage: gate_200.py CLUSTER.PROC', 2)
    job_id = sys.argv[1]
    try:
        ad = condor_ad(job_id)
        if ad is None:
            return report(f'{job_id} not yet visible; retry', WAIT)
        state, code, signaled = ad
        if state in (1, 2, 6, 7) or (state == 4 and code == 'undefined'):
            return report(f'{job_id} still pending (state={state})', WAIT)
        if state != 4 or code != '0' or signaled == 'true':
            return report(f'{job_id} failed/stopped: state={state}, exit={code}, signaled={signaled}; full training blocked', 1)
        approval = validate_benchmark(job_id)
        OUT.mkdir(parents=True, exist_ok=True)
        destination = OUT / 'approved.json'
        if destination.exists():
            prior = json.loads(destination.read_text())
            if prior.get('benchmark_job') != job_id:
                raise ValueError('Existing approved.json belongs to a different benchmark')
        tmp = OUT / 'approved.json.partial'
        tmp.write_text(json.dumps(approval, indent=2, sort_keys=True) + '\n')
        tmp.replace(destination)
        return report(f'PASS: batch={approval["batch_size"]} step={approval["segment_seconds_per_step"]:.3f}s; '
                      f'projected={approval["projected_hours_per_model_including_25pct_overhead"]}h/model; '
                      f'free={approval["filesystem_free_gib"]}GiB', 0)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        return report(f'FAIL (nothing launched): {exc}', 1)


if __name__ == '__main__':
    sys.exit(main())
