from pathlib import Path
import csv
import statistics

import h5py
import numpy as np
import torch

from src.model.model import NMRTo3DStructureElucidation

BASE = Path.home() / "spec2mol-chefnmr"
MAX_ATOMS = 59
IDS = ("0292058", "0295115")
REPLICATES = 2
SIGMA_QUANTILES = (0.2, 0.5, 0.8)
OUT = BASE / "logs" / "fixed_noise_denoising_20261028.csv"
PATTERNS = {
    "reference": "full_reference_*/checkpoints/*/diffusion_loss/epoch0010-loss88.19.ckpt",
    "augmented": "full_augmented_*/checkpoints/*/diffusion_loss/epoch0010-loss88.26.ckpt",
}
torch.set_num_threads(4)
torch.manual_seed(20261028)


def read_molecule(h5, mid):
    group = h5[mid]
    atom = group["atom_features"]
    coords_raw = np.asarray(atom["atom_coords"][0], dtype=np.float32)
    features_raw = np.asarray(atom["atom_one_hot"], dtype=np.float32)
    mask_raw = np.asarray(atom["atom_mask"], dtype=np.float32)
    h_raw = np.asarray(group["spectra/h_10k"][0], dtype=np.float32)
    c_raw = np.asarray(group["spectra/c_80"][0], dtype=np.float32)
    n = len(mask_raw)
    assert coords_raw.shape == (n, 3)
    assert features_raw.shape == (n, 4)
    assert (h_raw.shape, c_raw.shape) == ((10000,), (80,))
    assert 0 < n <= MAX_ATOMS and np.isfinite(coords_raw).all()
    assert np.isfinite(features_raw).all()
    assert np.isfinite(h_raw).all() and np.isfinite(c_raw).all()
    assert set(np.unique(c_raw)).issubset({0.0, 1.0})

    coords = torch.zeros(1, MAX_ATOMS, 3)
    feats = torch.zeros(1, MAX_ATOMS, 4)
    mask = torch.zeros(1, MAX_ATOMS)
    coords[0, :n] = torch.from_numpy(coords_raw)
    feats[0, :n] = torch.from_numpy(features_raw)
    mask[0, :n] = torch.from_numpy(mask_raw)

    # Match the centering portion of the observed training transform.
    center = (coords * mask[..., None]).sum(1, keepdim=True) / mask.sum().clamp_min(1)
    coords = (coords - center) * mask[..., None]

    h = torch.from_numpy(h_raw.copy()).unsqueeze(0)
    c = torch.from_numpy(c_raw.copy()).unsqueeze(0)
    return dict(coords=coords, feats=feats, mask=mask, h=h, c=c)


def carbon_relocated(c, mid):
    original = c.numpy().reshape(-1)
    occupied = np.flatnonzero(original)
    free = np.flatnonzero(original == 0)
    rng = np.random.default_rng(20261028 + int(mid))
    new_pos = rng.choice(free, len(occupied), replace=False)
    relocated = np.zeros_like(original)
    relocated[new_pos] = 1
    return torch.from_numpy(relocated).reshape(1, -1)


def conditions(mid, data):
    row = data[mid]
    h, c = row["h"], row["c"]
    other = data[IDS[1 - IDS.index(mid)]]
    options = {
        "correct": (h, c),
        "zero": (torch.zeros_like(h), torch.zeros_like(c)),
        "H1_only": (h, torch.zeros_like(c)),
        "C13_only": (torch.zeros_like(h), c),
        "swapped": (other["h"], other["c"]),
        "C_relocated": (h, carbon_relocated(c, mid)),
        "H_shift_500": (torch.roll(h, 500, dims=1), c),
    }
    return {name: torch.cat(parts, dim=1) for name, parts in options.items()}


def valid_rmse(a, b, mask):
    valid = mask.bool().unsqueeze(-1).expand_as(a)
    return torch.sqrt(((a - b)[valid] ** 2).mean()).item()


def rms_vector_change(a, b, mask):
    return torch.sqrt(((a - b)[mask.bool()] ** 2).sum(-1).mean()).item()


def relative_score_change(a, b, mask):
    diff = (a - b)[mask.bool()]
    norm = b[mask.bool()]
    return (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(norm).clamp_min(1e-12)).item()


def predict(diffusion, model_inputs, noisy, sigma):
    out = diffusion.neural_network_forward(
        noisy, sigma,
        network_condition_kwargs=dict(
            model_inputs=model_inputs, multiplicity=1, guidance_scale=0.0
        ),
    )
    coords = diffusion.predict_denoised_atom_coords(noisy, out, sigma)
    assert out["r_update"].shape == noisy.shape
    assert coords.shape == noisy.shape
    assert torch.isfinite(out["r_update"]).all()
    assert torch.isfinite(coords).all()
    return out["r_update"], coords


FIELDS = [
    "model", "molecule", "sigma", "replicate", "condition", "n_atoms",
    "noisy_rmse", "correct_rmse", "condition_rmse", "delta_rmse_alt_minus_correct",
    "coordinate_delta_rms_per_atom", "score_relative_l2",
]
CHECKPOINTS = {}
for name, pattern in PATTERNS.items():
    hits = sorted((BASE / "logs").glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"{name}: expected one checkpoint, found {len(hits)}")
    CHECKPOINTS[name] = hits[0]

with h5py.File(BASE / "data/molecules.h5", "r", swmr=True) as source:
    data = {mid: read_molecule(source, mid) for mid in IDS}

rows = []
sigmas = None
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=FIELDS)
    writer.writeheader()
    for model_name, checkpoint in CHECKPOINTS.items():
        print(f"\nMODEL: {model_name.upper()}, checkpoint: {checkpoint.name}", flush=True)
        model = NMRTo3DStructureElucidation.load_from_checkpoint(
            str(checkpoint), map_location="cpu", strict=True
        )
        model.eval()
        with torch.no_grad():
            model.prepare_eval()
            diffusion = model.model
            if sigmas is None:
                # Select values from the training distribution; keep identical
                # sigma values and noise draws for both checkpoints.
                sample = torch.as_tensor(diffusion.train_sigma_distribution(2048)).float().flatten()
                sample = sample[torch.isfinite(sample) & (sample > 0)]
                if sample.numel() < 100:
                    raise RuntimeError("Cannot sample positive training sigmas")
                sigmas = [float(torch.quantile(sample, q)) for q in SIGMA_QUANTILES]
                print("Fixed sigma quantiles:", sigmas, flush=True)

            for mid in IDS:
                item = data[mid]
                coords, mask = item["coords"], item["mask"]
                opts = conditions(mid, data)
                for i, sigma_val in enumerate(sigmas):
                    sigma = torch.tensor([sigma_val], dtype=torch.float32)
                    for rep in range(REPLICATES):
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(20261028 + int(mid) + 1000 * i + rep)
                        noise = torch.randn(coords.shape, generator=generator) * mask[..., None]
                        noisy = diffusion.interpolate(coords, noise, sigma)
                        base_inputs = dict(atom_mask=mask, atom_one_hot=item["feats"], condition=opts["correct"])
                        score_base, denoised_base = predict(diffusion, base_inputs, noisy, sigma)
                        correct_rmse = valid_rmse(denoised_base, coords, mask)
                        noisy_rmse = valid_rmse(noisy, coords, mask)

                        for condition, spectral_input in opts.items():
                            if condition == "correct":
                                score_alt, denoised_alt = score_base, denoised_base
                            else:
                                inputs = dict(atom_mask=mask, atom_one_hot=item["feats"], condition=spectral_input)
                                score_alt, denoised_alt = predict(diffusion, inputs, noisy, sigma)
                            alternative_rmse = valid_rmse(denoised_alt, coords, mask)
                            row = dict(
                                model=model_name, molecule=mid, sigma=sigma_val,
                                replicate=rep, condition=condition,
                                n_atoms=int(mask.sum().item()), noisy_rmse=noisy_rmse,
                                correct_rmse=correct_rmse, condition_rmse=alternative_rmse,
                                delta_rmse_alt_minus_correct=alternative_rmse - correct_rmse,
                                coordinate_delta_rms_per_atom=rms_vector_change(denoised_alt, denoised_base, mask),
                                score_relative_l2=relative_score_change(score_alt, score_base, mask),
                            )
                            rows.append(row)
                            writer.writerow(row)
                        fp.flush()
                    print(f"  {mid}: sigma={sigma_val:.5g} complete", flush=True)
        del model

print(f"\nCSV: {OUT}")
print("\nSUMMARY (positive delta RMSE = correct spectra performed better)")
for model_name in CHECKPOINTS:
    for condition in ("zero", "H1_only", "C13_only", "swapped", "C_relocated", "H_shift_500"):
        selected = [r for r in rows if r["model"] == model_name and r["condition"] == condition]
        deltas = [r["delta_rmse_alt_minus_correct"] for r in selected]
        responses = [r["coordinate_delta_rms_per_atom"] for r in selected]
        print(f"{model_name:10s} {condition:13s} n={len(deltas):2d} "
              f"mean_delta_RMSE={statistics.mean(deltas):+.6f} "
              f"correct_better={sum(d > 0 for d in deltas)}/{len(deltas)} "
              f"mean_coord_change={statistics.mean(responses):.6f}")
print("\nFIXED-NOISE DENOISING DIAGNOSTIC COMPLETE")
