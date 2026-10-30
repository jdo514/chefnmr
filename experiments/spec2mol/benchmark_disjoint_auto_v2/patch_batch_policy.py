#!/usr/bin/env python3
"""Idempotent, fail-closed patch of stage-2 launcher. Default: produce separate copy.

Preserves the original launcher; also scopes mutex and winner claim per segment,
which is essential when multiple segments are run in a single campaign.
"""
import argparse
from pathlib import Path

OLD_BATCH = 'for BATCH in 128 64 32; do'
NEW_BATCH = '''# Same initial effective (global) batch on 1- and 2-GPU jobs.
# 1 GPU: 128 -> 64 -> 32; 2 GPUs: 64 -> 32 (global: 128 -> 64).
if [[ "$EXPECTED_GPUS" -eq 2 ]]; then
  BATCHES=(64 32)
else
  BATCHES=(128 64 32)
fi
for BATCH in "${BATCHES[@]}"; do'''
PATCHES = [
    (OLD_BATCH, NEW_BATCH),
    ('exec 9>"$ROOT/.winner.lock"', 'exec 9>"$FOLDER/.winner.lock"'),
    ('mkdir "$ROOT/.winner_claim"', 'mkdir "$FOLDER/.winner_claim"'),
]


def transform(source):
    text = source
    for before, after in PATCHES:
        if after in text:
            if before in text:
                raise RuntimeError(f'Both old and new form present: {before}')
            continue
        if text.count(before) != 1:
            raise RuntimeError(f'Refusing unexpected script: expected one occurrence of {before!r}, found {text.count(before)}')
        text = text.replace(before, after, 1)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true', help='Explicitly allow replacing output, never input')
    a = p.parse_args()
    src = a.input.resolve(strict=True)
    dst = a.output.resolve()
    if src == dst:
        raise SystemExit('Refusing in-place edit: use a separate --output to preserve original scripts')
    if dst.exists() and not a.overwrite:
        raise FileExistsError(f'{dst} exists; use --overwrite only after checking the contents')
    original = src.read_text()
    fixed = transform(original)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(fixed)
    dst.chmod(src.stat().st_mode & 0o777)
    print('BATCH_AND_SEGMENT_LOCK_PATCH_PASS', dst)
    if original == fixed: print('Script was already patched; no changes needed')


if __name__ == '__main__':
    main()
