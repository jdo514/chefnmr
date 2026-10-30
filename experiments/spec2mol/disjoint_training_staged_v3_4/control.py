#!/usr/bin/env python3
"""DAGMan PRE/POST hooks for parallel 1/2-GPU candidates of each segment.

stdlib-only; runs on submit/access point, never consumes a GPU.
PRE skips completed or target-reached segments (PRE_SKIP 42).
POST validates the winner, cancels any queued sibling through HTCondor,
and treats only an authenticated losing candidate as successful.
POST returns 75 to request DAGMan's DEFER retry after 60 seconds when
an intentionally removed losing job is waiting for the winner's result.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SKIP = 42
DEFER = 75


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validated_result(campaign, segment):
    folder = Path(campaign) / f'segment_{segment:02d}'
    result = folder / 'result.json'
    if not result.is_file():
        return None
    info = read_json(result)
    checkpoint = folder / 'last.ckpt'
    if (info.get('segment') != segment or info.get('gpus') not in (1, 2)
        or Path(info.get('checkpoint', '')).resolve() != checkpoint.resolve()
        or not checkpoint.is_file() or checkpoint.stat().st_size < 100 * 1024**2
        or checkpoint.stat().st_size != info.get('checkpoint_bytes')):
        raise RuntimeError(f'Segment {segment}: malformed result or missing checkpoint')
    if sha(checkpoint) != info.get('checkpoint_sha256'):
        raise RuntimeError(f'Segment {segment}: checkpoint SHA256 mismatch')
    return info


def pre(campaign, segment):
    meta = read_json(Path(campaign) / 'campaign.json')
    if segment < 1 or segment > meta['segments']:
        raise ValueError('Invalid segment')
    if segment > 1:
        # If an earlier segment reached the training target, later PRE nodes
        # must keep skipping even though all intermediate segments were skipped.
        for previous in range(segment - 1, 0, -1):
            previous_file = Path(campaign) / f'segment_{previous:02d}' / 'result.json'
            if previous_file.is_file():
                prev = validated_result(campaign, previous)
                if prev.get('epoch', 0) >= meta['target_epochs']:
                    print(f'PRE_SKIP: target reached after segment {previous}', flush=True)
                    return SKIP
                if previous != segment - 1:
                    raise RuntimeError(f'Segment {segment-1} not complete; cannot submit next GPU pair')
                break
        else:
            raise RuntimeError(f'No completed previous segment for segment {segment}')
    current = validated_result(campaign, segment)
    if current is not None:
        print(f'PRE_SKIP: segment {segment} already completed', flush=True)
        return SKIP
    print(f'PRE_PASS segment={segment}', flush=True)
    return 0


def cancel_sibling(meta, segment, winner_gpus):
    # Do not cancel by owner or guessed JobId alone. Require unique campaign key
    # and explicitly the OTHER GPU count for this specific segment.
    key = meta['campaign_key']
    if not (len(key) == 32 and all(c in '0123456789abcdef' for c in key)):
        raise RuntimeError('Invalid campaign identity; refusing cancellation')
    loser_gpus = 1 if winner_gpus == 2 else 2
    expr = (f'Spec2MolCampaignKey == "{key}" && '
            f'Spec2MolSegment == {segment} && Spec2MolCandidateGPUs == {loser_gpus}')
    query = subprocess.run(['condor_q', '-constraint', expr, '-af', 'ClusterId', 'ProcId'],
                           text=True, capture_output=True, timeout=60)
    if query.returncode:
        raise RuntimeError(f'condor_q sibling lookup failed: {query.stderr.strip()}')
    for row in query.stdout.splitlines():
        fields = row.split()
        if len(fields) != 2 or not all(x.isdigit() for x in fields):
            raise RuntimeError(f'Unexpected condor_q sibling: {row!r}')
        jid = f'{fields[0]}.{fields[1]}'
        rm = subprocess.run(['condor_rm', jid], text=True, capture_output=True, timeout=60)
        if rm.returncode and 'not found' not in rm.stderr.lower():
            raise RuntimeError(f'condor_rm sibling {jid} failed: {rm.stderr.strip()}')
        print(f'CANCEL_LOSER segment={segment} gpus={loser_gpus} job={jid}', flush=True)


def post(campaign, segment, gpus, raw_return, jobid):
    meta = read_json(Path(campaign) / 'campaign.json')
    folder = Path(campaign) / f'segment_{segment:02d}'
    started = folder / 'started.json'
    start = read_json(started) if started.exists() else None
    owner_path = folder / '.winner_claim' / 'owner.json'
    owner = read_json(owner_path) if owner_path.exists() else None
    if start and start.get('gpus') not in (1, 2):
        raise RuntimeError('Unrecognized started.json; cannot classify candidate')
    if owner and owner.get('gpus') not in (1, 2):
        raise RuntimeError('Unrecognized winner claim owner; cannot classify candidate')
    if start and owner and start['gpus'] != owner['gpus']:
        raise RuntimeError('Claim owner and started marker disagree')
    winner_gpus = start['gpus'] if start else (owner['gpus'] if owner else None)
    failed_marker = folder / '.winner_failed'
    if failed_marker.exists():
        return 24
    if winner_gpus == gpus and raw_return != 0:
        failed_marker.write_text(f'winner_gpus={gpus} return_code={raw_return}\n')
        print(f'WINNER_FAILED segment={segment} gpus={gpus} exit={raw_return}', file=sys.stderr)
        return 20
    result = validated_result(campaign, segment)
    if winner_gpus == gpus:
        # This candidate acquired the actual training claim.
        if result is None or result['gpus'] != gpus:
            failed_marker.write_text(f'winner_gpus={gpus} missing_or_mismatched_result\n')
            print(f'WINNER_INCOMPLETE segment={segment}: no valid result.json', file=sys.stderr)
            return 21
        cancel_sibling(meta, segment, gpus)
        print(f'WINNER_PASS segment={segment} gpus={gpus} job={jobid}', flush=True)
        return 0
    if winner_gpus is not None and winner_gpus != gpus:
        if raw_return == 0:
            # Loser ran, saw the winner's persistent claim, and exited 0.
            print(f'LOSER_EXITED segment={segment} gpus={gpus}', flush=True)
            return 0
        if result is None:
            # This job was removed as a losing candidate while the winner is
            # still training; delay POST until winner writes final result.
            print(f'LOSER_AWAIT_WINNER segment={segment} gpus={gpus}', flush=True)
            return DEFER
        if result['gpus'] != winner_gpus:
            raise RuntimeError('Winner/result mismatch')
        print(f'LOSER_REMOVED_AFTER_WIN segment={segment} gpus={gpus}', flush=True)
        return 0
    if (folder / '.winner_claim').is_dir():
        # Extremely narrow race: another process claimed the directory but has
        # not yet written owner.json. Retry this POST rather than failing DAG.
        return DEFER
    print(f'UNCLAIMED_CANDIDATE_FAIL segment={segment} gpus={gpus} exit={raw_return}', file=sys.stderr)
    return 22


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest='action', required=True)
    for name in ('pre', 'post'):
        q = sp.add_parser(name)
        q.add_argument('campaign', type=Path)
        q.add_argument('segment', type=int)
        if name == 'post':
            q.add_argument('gpus', type=int, choices=(1, 2))
            q.add_argument('return_code', type=int)
            q.add_argument('jobid')
    a = p.parse_args()
    try:
        meta = read_json(a.campaign / 'campaign.json')
        if sha(Path(__file__)) != meta['control_script_sha256']:
            raise RuntimeError('DAG control script changed after campaign generation')
        if a.action == 'pre':
            return pre(a.campaign, a.segment)
        return post(a.campaign, a.segment, a.gpus, a.return_code, a.jobid)
    except Exception as exc:
        print(f'AUTO_DAG_CONTROL_ERROR: {exc}', file=sys.stderr, flush=True)
        if a.action == 'post':
            folder = a.campaign / f'segment_{a.segment:02d}'
            try:
                owner = read_json(folder / '.winner_claim' / 'owner.json')
                if owner.get('gpus') == a.gpus:
                    (folder / '.winner_failed').write_text(f'WINNER_POST_ERROR: {exc}\n')
                    return 24
                if (folder / '.winner_failed').exists():
                    return 24
            except (OSError, ValueError, KeyError):
                pass
            return DEFER
        return 23


if __name__ == '__main__':
    raise SystemExit(main())
