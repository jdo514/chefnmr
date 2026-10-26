"""Validated external split loading for ChefNMR."""

import json
from pathlib import Path

import h5py
import numpy as np


def load_external_split_indices(selection_dir, dataset_h5):
    root = Path(selection_dir).expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())

    fields = {
        "train": "train_molecules",
        "val": "validation_molecules",
        "test": "test_molecules",
    }

    splits = {}

    for name, count_field in fields.items():
        arr = np.load(root / f"{name}.npy", allow_pickle=False)

        if arr.ndim != 1 or arr.dtype.kind not in "iu":
            raise ValueError(f"{name}.npy must be a 1-D integer array")

        if arr.size and (
            np.any(arr < 0)
            or np.any(arr > np.iinfo(np.int32).max)
        ):
            raise ValueError(f"{name}.npy contains invalid indices")

        if len(np.unique(arr)) != len(arr):
            raise ValueError(f"Duplicate molecule IDs in {name}.npy")

        if len(arr) != int(manifest[count_field]):
            raise ValueError(f"{name}.npy count differs from manifest")

        splits[name] = arr.astype(np.int32, copy=False)

    if not len(splits["train"]) or not len(splits["val"]):
        raise ValueError("Training and validation must be nonempty")

    all_ids = np.concatenate(list(splits.values()))

    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("Train/validation/test indices overlap")

    if len(all_ids) != int(manifest["total_molecules"]):
        raise ValueError("Selection size differs from manifest")

    with h5py.File(dataset_h5, "r") as h5:
        valid = np.intersect1d(
            h5["valid_indices_h"][()],
            h5["valid_indices_c"][()],
        )

    unknown = np.setdiff1d(all_ids, valid)

    if len(unknown):
        raise ValueError(
            f"{len(unknown)} unavailable molecule IDs; "
            f"first={unknown[0]}"
        )

    # Verify checksum metadata without rehashing 42.7 GB
    recorded_hash = manifest.get("master_sha256")
    checksum_file = Path(dataset_h5).with_name(
        "molecules.h5.sha256"
    )

    if recorded_hash and checksum_file.is_file():
        stored_hash = checksum_file.read_text().split()[0]

        if stored_hash != recorded_hash:
            raise ValueError("Master checksum metadata mismatch")

    print(
        f"[EXTERNAL SPLIT] {root.name}: "
        f"train={len(splits['train']):,}; "
        f"val={len(splits['val']):,}; "
        f"test={len(splits['test']):,}",
        flush=True,
    )

    return splits
