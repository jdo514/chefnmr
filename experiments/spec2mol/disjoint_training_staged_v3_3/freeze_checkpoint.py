#!/usr/bin/env python3
"""Create a frozen, validated full-Lightning checkpoint from a live hourly checkpoint.

Designed for an already running workflow that may update hourly/last.ckpt.
This never modifies the source checkpoint. Uses stat-before/after and the
checkpoint ZIP CRC to avoid archiving a partial or concurrently rewritten file.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--retry-seconds', type=float, default=30)
    a = p.parse_args()
    src = a.source.expanduser().resolve(strict=True)
    out = a.output.expanduser().absolute()
    if not src.is_file() or src.stat().st_size < 100 * 1024**2:
        raise SystemExit('ERROR: source checkpoint missing/undersized')
    if out.exists() or out.with_suffix(out.suffix+'.incomplete').exists():
        raise SystemExit('ERROR: output/incomplete already exists; choose unique output')
    out.parent.mkdir(parents=True, exist_ok=True)
    if src == out:
        raise SystemExit('ERROR: cannot snapshot onto source')
    tmp = out.with_suffix(out.suffix+'.incomplete')
    for attempt in range(1, a.attempts + 1):
        digest = hashlib.sha256()
        start = src.stat()
        try:
            with src.open('rb') as inf, tmp.open('xb') as outf:
                while True:
                    block = inf.read(8 * 1024 * 1024)
                    if not block: break
                    outf.write(block)
                    digest.update(block)
                outf.flush()
                os.fsync(outf.fileno())
            end = src.stat()
            if (start.st_size, start.st_mtime_ns, start.st_ino) != (end.st_size, end.st_mtime_ns, end.st_ino):
                raise RuntimeError('source changed while copying')
            if tmp.stat().st_size != start.st_size:
                raise RuntimeError('snapshot size differs from source')
            with zipfile.ZipFile(tmp) as z:
                names = z.namelist()
                if not names or not any(n.endswith('data.pkl') for n in names):
                    raise RuntimeError('not a recognizable PyTorch checkpoint archive')
                bad = z.testzip()
                if bad:
                    raise RuntimeError('checkpoint zip CRC error: '+bad)
            os.replace(tmp, out)
            result = dict(source=str(src),checkpoint=str(out),bytes=out.stat().st_size,
                          sha256=digest.hexdigest(),attempt=attempt,status='FROZEN_CHECKPOINT_OK')
            out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2),flush=True)
            return 0
        except Exception as exc:
            if tmp.exists(): tmp.unlink()
            print(f'SNAPSHOT_ATTEMPT_{attempt}_FAILED: {exc}',file=sys.stderr,flush=True)
            if attempt < a.attempts:time.sleep(a.retry_seconds)
    raise SystemExit('ERROR: unable to create stable valid snapshot; original training unaffected')

if __name__ == '__main__':
    raise SystemExit(main())
