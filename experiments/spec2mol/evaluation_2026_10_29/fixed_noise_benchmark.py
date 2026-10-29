#!/usr/bin/env python3
"""Paired fixed-noise ChefNMR-S NMR-conditioning benchmark.

Uses original strict_clean_conformers_v1 train/val split, reference spectral
variant index 0, first stored conformer, original two epoch-10 checkpoints,
and identical noise realizations for both models/conditions.

Run first: python fixed_noise_benchmark.py --smoke
Then:      python fixed_noise_benchmark.py

IMPORTANT: This is a diagnostic of denoising a known noisy conformer, not
an end-to-end structure-elucidation evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

PATTERNS = {
    "reference": "full_reference_*/checkpoints/*/diffusion_loss/epoch0010-loss88.19.ckpt",
    "augmented": "full_augmented_*/checkpoints/*/diffusion_loss/epoch0010-loss88.26.ckpt",
}
SIGMAS = (0.20320279896259308, 0.6035448312759399, 1.8553335666656494)
MAX_ATOMS = 59
SEED = 20261028
EXCLUDE_VAL = {292058, 295115}  # Two previously explored validation molecules.
FIELDS = [
    "model", "split", "molecule", "matched_donor", "formula_signature", "sigma",
    "replicate", "condition", "n_atoms", "noisy_rmse", "correct_rmse",
    "condition_rmse", "delta_rmse_alt_minus_correct",
    "coordinate_delta_rms_per_atom", "score_relative_l2",
]


def key(mid: int) -> str:
    return f"{int(mid):07d}"


def formula_signature(group: h5py.Group) -> tuple[int, ...]:
    """Counts of encoded atom types. This is what the model knows, not a SMILES check."""
    feat = group["atom_features"]
    one_hot = np.asarray(feat["atom_one_hot"], dtype=np.float32)
    mask = np.asarray(feat["atom_mask"], dtype=np.float32).astype(bool)
    if one_hot.ndim != 2 or one_hot.shape[1] != 4 or mask.shape != (len(one_hot),):
        raise ValueError(f"Unexpected atom-feature shapes: {group.name}")
    valid = one_hot[mask]
    if valid.size == 0 or not np.allclose(valid.sum(1), 1, atol=1e-5):
        raise ValueError(f"Invalid atom types: {group.name}")
    counts = valid.sum(0)
    if not np.allclose(counts, np.rint(counts), atol=1e-5):
        raise ValueError(f"Non-integer atom counts: {group.name}")
    return tuple(int(x) for x in np.rint(counts))


def load_item(h5: h5py.File, mid: int) -> dict:
    group = h5[key(mid)]
    atom = group["atom_features"]
    coord_all = atom["atom_coords"]
    if coord_all.ndim != 3 or coord_all.shape[0] < 1:
        raise ValueError(f"No conformers for molecule {mid}")
    coords_raw = np.asarray(coord_all[0], dtype=np.float32)
    feat_raw = np.asarray(atom["atom_one_hot"], dtype=np.float32)
    mask_raw = np.asarray(atom["atom_mask"], dtype=np.float32)
    h_raw = np.asarray(group["spectra/h_10k"][0], dtype=np.float32)
    c_raw = np.asarray(group["spectra/c_80"][0], dtype=np.float32)
    n = mask_raw.shape[0]
    if not (0 < n <= MAX_ATOMS and coords_raw.shape == (n, 3)
            and feat_raw.shape == (n, 4)
            and h_raw.shape == (10000,) and c_raw.shape == (80,)):
        raise ValueError(f"Unexpected shapes for {mid}")
    if not all(np.isfinite(a).all() for a in (coords_raw, feat_raw, mask_raw, h_raw, c_raw)):
        raise ValueError(f"Non-finite input for {mid}")
    if not np.isin(c_raw, (0., 1.)).all():
        raise ValueError(f"Non-binary C13 reference for {mid}")
    if not np.isin(mask_raw, (0., 1.)).all():
        raise ValueError(f"Invalid atom mask for {mid}")

    coords = torch.zeros(MAX_ATOMS, 3, dtype=torch.float32)
    atoms = torch.zeros(MAX_ATOMS, 4, dtype=torch.float32)
    mask = torch.zeros(MAX_ATOMS, dtype=torch.float32)
    coords[:n] = torch.from_numpy(coords_raw)
    atoms[:n] = torch.from_numpy(feat_raw)
    mask[:n] = torch.from_numpy(mask_raw)
    # Match the centering portion of ChefNMR's observed training transform;
    # avoid random rotations and translations for fixed-noise comparison.
    center = (coords * mask[:, None]).sum(0) / mask.sum().clamp_min(1.0)
    coords = (coords - center) * mask[:, None]
    condition = torch.from_numpy(np.concatenate((h_raw, c_raw)).copy())
    return dict(mid=int(mid), coords=coords, atoms=atoms, mask=mask,
                condition=condition, signature=formula_signature(group),
                n_atoms=int(mask.sum().item()))


def choose_targets(splits: dict, n_train: int, n_val: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(SEED)
    chosen = {}
    for name, count in (("train", n_train), ("val", n_val)):
        pool = splits[name]
        if name == "val":
            pool = pool[~np.isin(pool, list(EXCLUDE_VAL))]
        if count > len(pool):
            raise ValueError(f"Requested {count} {name} examples, only {len(pool)} available")
        chosen[name] = sorted(int(x) for x in rng.choice(pool, size=count, replace=False))
    return chosen


def find_donors(h5, splits, targets, items, scan_limit: int):
    """Find within-partition molecules with identical encoded atom counts."""
    rng = np.random.default_rng(SEED + 17)
    matched = {}
    for split, chosen in targets.items():
        needed = defaultdict(list)
        candidates = defaultdict(list)
        for mid in chosen:
            sig = items[mid]["signature"]
            needed[sig].append(mid)
            candidates[sig].append(mid)

        pool = rng.permutation(splits[split])
        if scan_limit > 0:
            pool = pool[:scan_limit]
        seen = set(chosen)
        # Try to find a different molecule with each requested composition.
        missing = {sig for sig, mids in needed.items() if len(candidates[sig]) < 2}
        print(f"  {split}: searching donors for {len(missing)} formula groups", flush=True)
        for j, index in enumerate(pool):
            if not missing:
                break
            candidate = int(index)
            if candidate in seen:
                continue
            sig = formula_signature(h5[key(candidate)])
            if sig not in missing:
                continue
            candidates[sig].append(candidate)
            missing.discard(sig)
            if (j + 1) % 10000 == 0:
                print(f"    scanned {j+1:,} IDs, unmatched groups={len(missing)}", flush=True)

        for mid in chosen:
            sig = items[mid]["signature"]
            alt = next((c for c in candidates[sig] if c != mid), None)
            if alt is not None:
                donor = load_item(h5, alt)
                if torch.equal(donor["condition"], items[mid]["condition"]):
                    alt = None  # Non-informative identical spectrum.
                else:
                    items[mid]["donor_condition"] = donor["condition"]
                    matched[mid] = alt
            if alt is None:
                matched[mid] = None
        nmatch = sum(matched[mid] is not None for mid in chosen)
        print(f"  {split}: matched controls={nmatch}/{len(chosen)}; "
              f"unmatched formula groups={len(missing)}", flush=True)
    return matched


def score_forward(diffusion, noisy, sigma_vec, atoms, mask, condition):
    out = diffusion.neural_network_forward(
        noisy, sigma_vec,
        network_condition_kwargs={
            "model_inputs": {
                "atom_mask": mask, "atom_one_hot": atoms, "condition": condition,
            },
            "multiplicity": 1,
            "guidance_scale": 0.0,
        },
    )
    pred = diffusion.predict_denoised_atom_coords(noisy, out, sigma_vec)
    score = out["r_update"]
    if pred.shape != noisy.shape or score.shape != noisy.shape:
        raise RuntimeError(f"Unexpected outputs {pred.shape}, {score.shape}")
    if not torch.isfinite(pred).all() or not torch.isfinite(score).all():
        raise RuntimeError("Nonfinite network predictions")
    return score, pred


def masked_rmse(pred, truth, mask):
    err2 = ((pred - truth) ** 2 * mask[..., None]).sum((1, 2))
    return torch.sqrt(err2 / (3 * mask.sum(1).clamp_min(1)))


def masked_coord_change(a, b, mask):
    return torch.sqrt(((a - b).square() * mask[..., None]).sum((1, 2)) /
                      mask.sum(1).clamp_min(1))


def masked_relative_score_change(a, b, mask):
    num = ((a - b).square() * mask[..., None]).sum((1, 2)).sqrt()
    den = (b.square() * mask[..., None]).sum((1, 2)).sqrt().clamp_min(1e-12)
    return num / den


def fixed_noisy(diffusion, batch_items, sigma_val, sigma_idx, rep):
    coords = torch.stack([it["coords"] for it in batch_items])
    mask = torch.stack([it["mask"] for it in batch_items])
    noise = torch.stack([
        torch.randn(it["coords"].shape,
                    generator=torch.Generator(device="cpu").manual_seed(
                        SEED + it["mid"] + 100000 * sigma_idx + rep
                    )) * it["mask"][:, None]
        for it in batch_items
    ])
    sigmas = torch.full((len(batch_items),), sigma_val, dtype=torch.float32)
    return coords, mask, sigmas, diffusion.interpolate(coords, noise, sigmas)


def estimate_ci(values: np.ndarray, rng, n_boot: int = 1000):
    if len(values) < 2:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(1)
    return tuple(float(x) for x in np.quantile(means, (0.025, 0.975)))


def make_summary(rows: list[dict], output: Path):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["condition"] == "correct":
            continue
        group = (row["model"], row["split"], row["sigma"], row["condition"])
        groups[group][row["molecule"]].append(row)
    rng = np.random.default_rng(SEED + 777)
    summary = []
    for (model, split, sigma, condition), molecule_rows in sorted(groups.items()):
        # Each molecule, not each noise realization, is an independent unit.
        deltas = np.asarray([
            np.mean([r["delta_rmse_alt_minus_correct"] for r in repetitions])
            for repetitions in molecule_rows.values()
        ])
        correct = np.asarray([
            np.mean([r["correct_rmse"] for r in repetitions])
            for repetitions in molecule_rows.values()
        ])
        changes = np.asarray([
            np.mean([r["coordinate_delta_rms_per_atom"] for r in repetitions])
            for repetitions in molecule_rows.values()
        ])
        lo, hi = estimate_ci(deltas, rng)
        base = correct.mean()
        summary.append(dict(
            model=model, split=split, sigma=sigma, condition=condition,
            n_molecules=len(deltas), mean_correct_rmse=float(base),
            mean_delta_rmse=float(deltas.mean()),
            relative_delta_pct=float(100 * deltas.mean() / base) if base > 0 else np.nan,
            delta_ci95_low=lo, delta_ci95_high=hi,
            fraction_molecules_correct_better=float((deltas > 0).mean()),
            mean_coordinate_change=float(changes.mean()),
        ))
    fields = list(summary[0]) if summary else []
    if fields:
        with output.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
    print("\nSUMMARY (molecule-averaged; positive delta = correct better)")
    for row in summary:
        print(f"{row['model']:9s} {row['split']:5s} sigma={row['sigma']:.4f} "
              f"{row['condition']:7s} n={row['n_molecules']:3d} "
              f"delta={row['mean_delta_rmse']:+.6f} "
              f"rel={row['relative_delta_pct']:+.3f}% "
              f"CI95=[{row['delta_ci95_low']:+.6f},{row['delta_ci95_high']:+.6f}] "
              f"better={row['fraction_molecules_correct_better']:.1%}")
    print(f"Summary CSV: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path.home() / "spec2mol-chefnmr")
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--donor-scan-limit", type=int, default=0,
                        help="Max pool IDs per split to scan for donors; 0=all")
    parser.add_argument("--smoke", action="store_true",
                        help="5+5 molecules, 1 replicate, middle sigma")
    args = parser.parse_args()
    if args.smoke:
        args.n_train, args.n_val, args.replicates = 5, 5, 1
        args.donor_scan_limit = 3000
        sigmas = [SIGMAS[1]]
        tag = "smoke"
    else:
        sigmas = list(SIGMAS)
        tag = "full"
    if args.n_train < 1 or args.n_val < 1 or args.replicates < 1 or args.batch_size < 1:
        parser.error("n-train, n-val, replicates and batch-size must all be >= 1")
    base = args.base.expanduser().resolve()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    output_dir = base / "logs"
    output_dir.mkdir(exist_ok=True, parents=True)
    out = output_dir / f"fixed_noise_benchmark_{tag}_20261028.csv"
    out_summary = output_dir / f"fixed_noise_benchmark_{tag}_summary_20261028.csv"
    out_meta = output_dir / f"fixed_noise_benchmark_{tag}_metadata_20261028.json"
    selection_dir = base / "selections/strict_clean_conformers_v1"
    manifest = json.loads((selection_dir / "manifest.json").read_text())
    splits = {name: np.load(selection_dir / f"{name}.npy", allow_pickle=False)
              for name in ("train", "val", "test")}
    if len(splits["train"]) != manifest["train_molecules"] or len(splits["val"]) != manifest["validation_molecules"]:
        raise RuntimeError("Split counts disagree with manifest")
    if len(np.intersect1d(splits["train"], splits["val"])):
        raise RuntimeError("Train/validation overlap")
    targets = choose_targets(splits, args.n_train, args.n_val)
    print("Selection:", {name: len(ids) for name, ids in targets.items()}, flush=True)
    items = {}
    with h5py.File(base / "data/molecules.h5", "r", swmr=True) as h5:
        for chosen in targets.values():
            for mid in chosen:
                items[mid] = load_item(h5, mid)
        donors = find_donors(h5, splits, targets, items, args.donor_scan_limit)

    checkpoints = {}
    for model_name, pattern in PATTERNS.items():
        hits = sorted((base / "logs").glob(pattern))
        if len(hits) != 1:
            raise RuntimeError(f"{model_name}: expected exactly 1 checkpoint, found {hits}")
        checkpoints[model_name] = hits[0]

    metadata = {
        "selection_name": manifest["selection_name"],
        "master_sha256_recorded": manifest.get("master_sha256"),
        "reference_variant_index": 0, "seed": SEED,
        "sigmas": sigmas, "replicates": args.replicates, "batch_size": args.batch_size,
        "n_train": args.n_train, "n_val": args.n_val,
        "first_conformer_only": True, "coordinate_centering": True,
        "negative_control": "same split and same encoded atom-type counts; donor != target",
        "donor_scan_limit": args.donor_scan_limit,
        "checkpoints": {m: str(p) for m, p in checkpoints.items()},
        "targets": targets,
        "matched_donors": {str(k): v for k, v in donors.items()},
        "no_actual_end_to_end_structure_elucidation_metric": True,
    }
    out_meta.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Metadata: {out_meta}", flush=True)

    # Keep these data local; never modify original HDF5 or checkpoints.
    from src.model.model import NMRTo3DStructureElucidation

    rows = []
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        for model_name, checkpoint in checkpoints.items():
            print(f"\nMODEL: {model_name.upper()} ({checkpoint.name})", flush=True)
            model = NMRTo3DStructureElucidation.load_from_checkpoint(
                str(checkpoint), map_location="cpu", strict=True
            )
            model.eval()
            with torch.inference_mode():
                model.prepare_eval()
                diffusion = model.model
                for split, chosen in targets.items():
                    for sigma_idx, sigma_val in enumerate(sigmas):
                        for rep in range(args.replicates):
                            for start in range(0, len(chosen), args.batch_size):
                                mids = chosen[start:start + args.batch_size]
                                batch_items = [items[mid] for mid in mids]
                                coords, masks, sigma_vec, noisy = fixed_noisy(
                                    diffusion, batch_items, sigma_val, sigma_idx, rep
                                )
                                atom_types = torch.stack([it["atoms"] for it in batch_items])
                                correct_cond = torch.stack([it["condition"] for it in batch_items])
                                base_score, base_pred = score_forward(
                                    diffusion, noisy, sigma_vec, atom_types, masks, correct_cond
                                )
                                noisy_error = masked_rmse(noisy, coords, masks)
                                correct_error = masked_rmse(base_pred, coords, masks)
                                alt_conditions = {
                                    "correct": (list(range(len(mids))), correct_cond),
                                    "zero": (list(range(len(mids))), torch.zeros_like(correct_cond)),
                                }
                                matched_local = [i for i, mid in enumerate(mids) if donors[mid] is not None]
                                if matched_local:
                                    alt_conditions["matched"] = (
                                        matched_local,
                                        torch.stack([items[mids[i]]["donor_condition"] for i in matched_local]),
                                    )
                                for condition, (positions, conditions) in alt_conditions.items():
                                    if condition == "correct":
                                        score_alt, pred_alt = base_score, base_pred
                                    else:
                                        sub = torch.as_tensor(positions, dtype=torch.long)
                                        score_alt, pred_alt = score_forward(
                                            diffusion, noisy[sub], sigma_vec[sub], atom_types[sub],
                                            masks[sub], conditions
                                        )
                                    sub = torch.as_tensor(positions, dtype=torch.long)
                                    errors = masked_rmse(pred_alt, coords[sub], masks[sub]).tolist()
                                    changes = masked_coord_change(pred_alt, base_pred[sub], masks[sub]).tolist()
                                    scores = masked_relative_score_change(score_alt, base_score[sub], masks[sub]).tolist()
                                    for j, i in enumerate(positions):
                                        it = batch_items[i]
                                        row = dict(
                                            model=model_name, split=split, molecule=key(it["mid"]),
                                            matched_donor=(key(donors[it["mid"]]) if donors[it["mid"]] is not None else ""),
                                            formula_signature="-".join(map(str, it["signature"])),
                                            sigma=sigma_val, replicate=rep,
                                            condition=condition, n_atoms=it["n_atoms"],
                                            noisy_rmse=float(noisy_error[i]),
                                            correct_rmse=float(correct_error[i]),
                                            condition_rmse=float(errors[j]),
                                            delta_rmse_alt_minus_correct=float(errors[j] - correct_error[i]),
                                            coordinate_delta_rms_per_atom=float(changes[j]),
                                            score_relative_l2=float(scores[j]),
                                        )
                                        writer.writerow(row)
                                        rows.append(row)
                            fp.flush()
                            print(f"  {split}: sigma={sigma_val:.4f} rep={rep+1}/{args.replicates} done", flush=True)
            del model
    print(f"\nRaw CSV: {out}")
    make_summary(rows, out_summary)
    print("\nBENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
