from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.model.model import NMRTo3DStructureElucidation

torch.set_num_threads(4)

BASE = Path.home() / "spec2mol-chefnmr"
IDS = ["0292058", "0295115"]

PATTERNS = {
    "reference": (
        "full_reference_*/checkpoints/*/"
        "diffusion_loss/epoch0010-loss88.19.ckpt"
    ),
    "augmented": (
        "full_augmented_*/checkpoints/*/"
        "diffusion_loss/epoch0010-loss88.26.ckpt"
    ),
}

CHECKPOINTS = {}

for name, pattern in PATTERNS.items():
    matches = sorted((BASE / "logs").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"{name}: expected one checkpoint, found {len(matches)}"
        )
    CHECKPOINTS[name] = matches[0]


def load_spectrum(h5, mid):
    h = np.asarray(
        h5[mid]["spectra/h_10k"][0],
        dtype=np.float32
    ).copy()

    c = np.asarray(
        h5[mid]["spectra/c_80"][0],
        dtype=np.float32
    ).copy()

    assert h.shape == (10000,)
    assert c.shape == (80,)
    assert np.isfinite(h).all()
    assert np.isfinite(c).all()
    assert np.isin(c, [0.0, 1.0]).all()

    return (
        torch.from_numpy(h).unsqueeze(0),
        torch.from_numpy(c).unsqueeze(0)
    )


def relocate_carbon_peaks(c, seed):
    """
    Move every occupied carbon bin to an originally
    unoccupied position while preserving peak count.
    """
    rng = np.random.default_rng(seed)

    original = c.numpy().reshape(-1)

    occupied = np.flatnonzero(original > 0)
    available = np.flatnonzero(original == 0)

    assert len(available) >= len(occupied)

    new_positions = rng.choice(
        available,
        size=len(occupied),
        replace=False
    )

    relocated = np.zeros_like(original)
    relocated[new_positions] = 1.0

    assert np.count_nonzero(relocated) == len(occupied)
    assert np.count_nonzero(
        relocated * original
    ) == 0

    return torch.from_numpy(
        relocated.copy()
    ).reshape(1, -1)


def compare(reference, alternative):
    a = reference.reshape(-1)
    b = alternative.reshape(-1)

    difference = torch.linalg.vector_norm(a - b).item()
    norm = torch.linalg.vector_norm(a).item()

    relative = difference / max(norm, 1e-12)

    cosine = F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=-1
    ).item()

    return difference, relative, cosine


with h5py.File(BASE / "data/molecules.h5", "r", swmr=True) as h5:
    spectra = {
        mid: load_spectrum(h5, mid)
        for mid in IDS
    }


for model_name, checkpoint in CHECKPOINTS.items():

    print("\n" + "=" * 78)
    print("MODEL:", model_name.upper())
    print("=" * 78)

    model = NMRTo3DStructureElucidation.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
        strict=True
    )

    model.eval()

    with torch.no_grad():
        model.prepare_eval()

        embedder = model.model.score_model.y_embedder

        for mid in IDS:

            h, c = spectra[mid]

            c_relocated = relocate_carbon_peaks(
                c, seed=20261028 + int(mid)
            )

            conditions = {
                "correct": (h, c),

                "C_shift_7": (
                    h,
                    torch.roll(c, shifts=7, dims=1)
                ),

                "C_shift_23": (
                    h,
                    torch.roll(c, shifts=23, dims=1)
                ),

                "C_relocated": (
                    h,
                    c_relocated
                ),

                "C_zero": (
                    h,
                    torch.zeros_like(c)
                ),

                "H_shift_500": (
                    torch.roll(h, shifts=500, dims=1),
                    c
                ),

                "H_shift_1500": (
                    torch.roll(h, shifts=1500, dims=1),
                    c
                ),
            }

            embeddings = {}

            for name, (hh, cc) in conditions.items():

                combined = torch.cat([hh, cc], dim=1)

                embeddings[name] = embedder(
                    combined
                ).reshape(-1)

            reference = embeddings["correct"]

            print("\nMolecule:", mid)
            print(
                "Carbon peaks:",
                int(torch.count_nonzero(c))
            )

            print(
                f"{'Condition':<18}"
                f"{'Absolute L2':>16}"
                f"{'Relative L2':>16}"
                f"{'Cosine':>16}"
            )

            for name, embedding in embeddings.items():

                absolute, relative, cosine = compare(
                    reference,
                    embedding
                )

                print(
                    f"{name:<18}"
                    f"{absolute:>16.8f}"
                    f"{relative:>16.8f}"
                    f"{cosine:>16.8f}"
                )

    del model

print("\nPOSITION SENSITIVITY DIAGNOSTIC COMPLETE")
