#!/usr/bin/env python3
"""Stage the verified ChefNMR HDF5 into an allocated job's local scratch.

Fail open to shared NFS: always emit a selected data directory and a JSON report.
Never writes to the production HDF5, changes the shared dataset, or touches
another job's scratch. Uses a streaming SHA-256 to authenticate the local copy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024 ** 3
EXPECTED_SHA256 = 'd71614feb17fd0e5d9e8c4f856a7c01e7141e063a0bb37cff99562c78af420e9'


def stage(source: Path, scratch: Path, request_disk_kib: int, expected_sha: str,
          max_seconds: float = 7200, min_spare_gib: float = 10.,
          min_alloc_spare_gib: float = 4.) -> dict:
    src = source.expanduser().resolve(strict=True)
    report = dict(source=str(src), scratch=str(scratch), mode='nfs',
                  datadir=str(src.parent), reason='', copied_bytes=0,
                  elapsed_seconds=0)
    target_file = None
    temp_file = None
    start = time.monotonic()
    try:
        assert src.is_file() and src.name == 'molecules.h5', 'source is not molecules.h5'
        if len(expected_sha) != 64 or not all(x in '0123456789abcdef' for x in expected_sha):
            raise RuntimeError('invalid expected checksum')
        scratch = scratch.expanduser().resolve(strict=True)
        if not scratch.is_dir():
            raise RuntimeError('scratch directory does not exist')
        if scratch == src.parent or scratch in src.parents or scratch == Path('/'):
            raise RuntimeError('unsafe scratch path')
        fstype = subprocess.run(['findmnt', '-n', '-T', str(scratch), '-o', 'FSTYPE'],
                                capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()[0]
        report['scratch_fstype'] = fstype
        if fstype in {'nfs', 'nfs4', 'autofs', 'cifs', 'fuse.sshfs'}:
            raise RuntimeError('scratch filesystem is not node-local: ' + fstype)
        size = src.stat().st_size
        available = shutil.disk_usage(scratch).free
        requested = int(request_disk_kib) * 1024
        report.update(source_bytes=size, scratch_available_bytes=available,
                      requested_disk_bytes=requested)
        if requested < size + int(min_alloc_spare_gib * GIB):
            raise RuntimeError('HTCondor requested disk smaller than HDF5 plus allocation margin')
        if available < size + int(min_spare_gib * GIB):
            raise RuntimeError('local disk free space smaller than HDF5 plus node spare margin')
        destination = scratch / 'spec2mol_chefnmr_h5_stage'
        destination.mkdir(exist_ok=True)
        target_file = destination / 'molecules.h5'
        temp_file = destination / 'molecules.h5.incomplete'
        if target_file.exists() or temp_file.exists():
            raise RuntimeError('refusing to overwrite an existing staged file')
        digest = hashlib.sha256()
        with src.open('rb') as inp, temp_file.open('xb') as out:
            while True:
                if time.monotonic() - start > max_seconds:
                    raise TimeoutError(f'copy exceeded {max_seconds}s limit')
                block = inp.read(8 * 1024 * 1024)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                report['copied_bytes'] += len(block)
            out.flush()
            os.fsync(out.fileno())
        if report['copied_bytes'] != size or digest.hexdigest() != expected_sha:
            raise RuntimeError('copy size or SHA256 differs from original production HDF5')
        os.replace(temp_file, target_file)
        report['mode'] = 'local'
        report['datadir'] = str(destination)
        report['reason'] = 'verified streaming copy completed'
        report['sha256'] = digest.hexdigest()
        temp_file = None
    except Exception as exc:
        report['reason'] = str(exc)
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - start, 2)
        if temp_file is not None and temp_file.exists():
            try: temp_file.unlink()
            except OSError: pass
        # For invalid local copy, revert to NFS and do not leave a corrupted HDF5.
        if report['mode'] != 'local' and target_file is not None and target_file.exists():
            try: target_file.unlink()
            except OSError: pass
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--scratch', required=True, type=Path)
    p.add_argument('--request-disk-kib', required=True, type=int)
    p.add_argument('--expected-sha256', default=EXPECTED_SHA256)
    p.add_argument('--max-seconds', default=7200, type=float)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--selected-datadir-file', required=True, type=Path)
    a = p.parse_args()
    # Always publish fallback as early as possible, so a failed copy never
    # leaves the training launcher without a valid source-data directory.
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.selected_datadir_file.parent.mkdir(parents=True, exist_ok=True)
    default_dir = str(a.source.expanduser().resolve(strict=True).parent)
    a.selected_datadir_file.write_text(default_dir + '\n')
    outcome = stage(a.source, a.scratch, a.request_disk_kib, a.expected_sha256,
                    max_seconds=a.max_seconds)
    a.report.write_text(json.dumps(outcome, indent=2) + '\n')
    a.selected_datadir_file.write_text(outcome['datadir'] + '\n')
    print('STAGING_RESULT:', json.dumps(outcome), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
