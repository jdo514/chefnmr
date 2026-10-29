#!/usr/bin/env python3
"""Non-destructive ChefNMR checkpoint/benchmark freeze with verified copies.

Run on indium (NMRbox). Does not change original checkpoints or the master HDF5.
Usage:
    python freeze_chefnmr_20261028.py --dry-run
    python freeze_chefnmr_20261028.py
    python freeze_chefnmr_20261028.py --verify
    python freeze_chefnmr_20261028.py --resume   # if copy was interrupted
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATE = "20261028"
REQUIRED = [
    "logs/fixed_noise_benchmark_full_20261028.csv",
    "logs/fixed_noise_benchmark_full_summary_20261028.csv",
    "logs/fixed_noise_benchmark_full_metadata_20261028.json",
    "logs/fixed_noise_benchmark_full_20261028.log",
    "logs/full_reference_2222733.0/.hydra/config.yaml",
    "logs/full_augmented_2222972.0/.hydra/config.yaml",
    "selections/strict_clean_conformers_v1/manifest.json",
    "selections/strict_clean_conformers_v1/train.npy",
    "selections/strict_clean_conformers_v1/val.npy",
    "selections/strict_clean_conformers_v1/test.npy",
]
OPTIONAL = [
    "logs/fixed_noise_denoising_20261028.csv",
    "logs/embedding_probe_20261028.txt",
    "logs/position_sensitivity_20261028.txt",
    "data/molecules.h5.sha256",
    "chefnmr/src/model/model.py",
    "chefnmr/src/model/modules/diffusion.py",
    "chefnmr/src/model/modules/score_models.py",
    "chefnmr/src/model/modules/embedders.py",
    "chefnmr/src/data/datamodule.py",
    "chefnmr/src/data/external_split_loader.py",
    "chefnmr/src/data/paired_spectra.py",
    "chefnmr/src/data/input_generator.py",
]
CKPT_GLOBS = {
    "reference": "logs/full_reference_*/checkpoints/*/diffusion_loss/epoch0010-loss88.19.ckpt",
    "augmented": "logs/full_augmented_*/checkpoints/*/diffusion_loss/epoch0010-loss88.26.ckpt",
}

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()

def git_info(base: Path) -> dict:
    out = {}
    for key, cmd in {
        "spec2mol_repo_head": ["git", "rev-parse", "HEAD"],
        "spec2mol_repo_status": ["git", "status", "--porcelain", "--untracked-files=no"],
    }.items():
        p = subprocess.run(cmd, cwd=base, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out[key] = p.stdout.strip() if p.returncode == 0 else None
    if (base / "chefnmr").is_dir():
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=base / "chefnmr", text=True, capture_output=True)
        out["chefnmr_repo_head"] = p.stdout.strip() if p.returncode == 0 else None
    else:
        out["chefnmr_repo_head"] = None
    return out

def source_files(base: Path):
    selected = {}
    for name, pattern in CKPT_GLOBS.items():
        matches = sorted(base.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"{name}: expected exactly 1 checkpoint for {pattern}, found {len(matches)}")
        selected[f"checkpoints/{name}/{matches[0].name}"] = matches[0]
    for rel in REQUIRED:
        p = base / rel
        if not p.is_file():
            raise FileNotFoundError(f"Required evidence missing: {p}")
        selected[f"evidence/{rel}"] = p
    for rel in OPTIONAL:
        p = base / rel
        if p.is_file():
            selected[f"evidence/{rel}"] = p
    for p in sorted((base / "launch/chefnmr_eval").glob("*.py")):
        selected[f"evidence/{p.relative_to(base).as_posix()}"] = p
    for p in sorted((base / "logs").glob("fixed_noise_benchmark_smoke*20261028*")):
        if p.is_file():
            selected[f"evidence/{p.relative_to(base).as_posix()}"] = p
    return selected

def verify_archive(dest: Path):
    manifest = dest / "freeze_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    data = json.loads(manifest.read_text())
    count = 0
    for item in data["files"]:
        p = dest / item["archived_relative_path"]
        if not p.is_file() or p.stat().st_size != item["size_bytes"] or hash_file(p) != item["sha256"]:
            raise RuntimeError(f"ARCHIVE VERIFY FAIL: {p}")
        count += 1
    print(f"FREEZE VERIFY PASS: {count} archived files; checksums match: {dest}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path.home() / "spec2mol-chefnmr")
    ap.add_argument("--dest", type=Path, default=None)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    base = args.base.expanduser().resolve()
    dest = (args.dest or base / "archives" / "chefnmr_epoch10_freeze_20261028").expanduser().resolve()
    if args.verify:
        verify_archive(dest)
        return
    if not base.is_dir():
        raise NotADirectoryError(base)
    selected = source_files(base)
    total = sum(p.stat().st_size for p in selected.values())
    free = shutil.disk_usage(dest.parent if dest.parent.exists() else base).free
    print(f"Base: {base}\nArchive: {dest}\nFiles: {len(selected)}\nTo copy: {total / 2**30:.3f} GiB\nAvailable: {free / 2**30:.3f} GiB")
    for rel, src in selected.items():
        if rel.startswith("checkpoints/"):
            print(f"  {rel}: {src.stat().st_size / 2**30:.3f} GiB")
    print("Master molecules.h5 is NOT copied (source remains unchanged).")
    if args.dry_run:
        print("DRY RUN: no changes made")
        return
    if dest.exists() and not args.resume:
        raise FileExistsError(f"Archive destination already exists: {dest}. Use --verify or --resume.")
    if not dest.exists() and free < total + 100 * 1024**2:
        raise OSError("Not enough free disk for an independent freeze copy plus 100 MiB safety margin")
    dest.mkdir(parents=True, exist_ok=True)
    items = []
    for rel, src in selected.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Freezing {rel}", flush=True)
        src_hash = hash_file(src)
        if target.exists():
            if not args.resume:
                raise FileExistsError(target)
            if target.stat().st_size != src.stat().st_size or hash_file(target) != src_hash:
                raise RuntimeError(f"Existing archive file does not match source: {target}")
        else:
            tmp = target.with_name(target.name + ".copying")
            with src.open("rb") as fsrc, tmp.open("wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, length=8*1024*1024)
            if hash_file(tmp) != src_hash:
                raise RuntimeError(f"Copy checksum mismatch: {tmp}")
            os.replace(tmp, target)
        target.chmod(0o444) # archived COPY only, never source
        items.append({
            "source": str(src), "archived_relative_path": rel,
            "sha256": src_hash, "size_bytes": src.stat().st_size,
        })
    selection = json.loads((base / "selections/strict_clean_conformers_v1/manifest.json").read_text())
    record = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "Frozen, verified epoch-10 ChefNMR-S reference vs paired-cyclic training and denoising benchmark evidence",
        "master_hdf5_path": str(base / "data/molecules.h5"),
        "master_hdf5_copied": False,
        "master_sha256_from_selection_manifest_not_recomputed": selection.get("master_sha256"),
        "training_count": 95202, "validation_count": 10575,
        "warning": "Archive is a checksum-verified read-only copy, not an enforced filesystem/WORM lock. Verify again before reuse.",
        "repositories": git_info(base), "files": items,
    }
    (dest / "freeze_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (dest / "SHA256SUMS").write_text("".join(f"{i['sha256']}  {i['archived_relative_path']}\n" for i in items))
    (dest / "README.txt").write_text(
        "ChefNMR-S FROZEN SNAPSHOT, 2026-10-28\n"
        "Contains two epoch-10 checkpoints and reproducibility evidence.\n"
        "Original training and data files were not changed.\n"
        "No 42.7 GB master HDF5 duplication; original location and manifest hash recorded.\n"
        "Verify: python freeze_chefnmr_20261028.py --verify\n"
        "Do not resume training in this snapshot; use original checkpoint paths for future read-only inference.\n"
    )
    verify_archive(dest)
    print("FREEZE COMPLETE. Originals remain untouched.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FREEZE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
