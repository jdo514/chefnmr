from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from src.model.model import NMRTo3DStructureElucidation

torch.set_num_threads(4)

BASE = Path.home() / "spec2mol-chefnmr"
IDS = ["0292058", "0295115"]

CHECKPOINT_PATTERNS = {
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

for name, pattern in CHECKPOINT_PATTERNS.items():
    matches = sorted((BASE / "logs").glob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"{name}: expected one checkpoint, found {len(matches)}"
        )

    CHECKPOINTS[name] = matches[0]

spectra = {}

with h5py.File(BASE / "data/molecules.h5", "r", swmr=True) as h5:

    for mid in IDS:

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

        spectra[mid] = {
            "h": torch.from_numpy(h).unsqueeze(0),
            "c": torch.from_numpy(c).unsqueeze(0),
        }


def flatten_embedding(output):
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"Unexpected embedding output: {type(output)}"
        )

    return output.reshape(1, -1)


for model_name, checkpoint in CHECKPOINTS.items():

    print("\n" + "=" * 75)
    print("MODEL:", model_name.upper())
    print("CHECKPOINT:", checkpoint.name)
    print("=" * 75)

    model = NMRTo3DStructureElucidation.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
        strict=True
    )

    model.eval()

    with torch.no_grad():

        model.prepare_eval()

        embedder = model.model.score_model.y_embedder

        for i, mid in enumerate(IDS):

            h = spectra[mid]["h"]
            c = spectra[mid]["c"]

            other = spectra[IDS[1 - i]]

            conditions = {
                "correct": torch.cat([h, c], dim=1),

                "zero": torch.cat([
                    torch.zeros_like(h),
                    torch.zeros_like(c)
                ], dim=1),

                "H1_only": torch.cat([
                    h,
                    torch.zeros_like(c)
                ], dim=1),

                "C13_only": torch.cat([
                    torch.zeros_like(h),
                    c
                ], dim=1),

                "swapped": torch.cat([
                    other["h"],
                    other["c"]
                ], dim=1),
            }

            embeddings = {}

            for name, condition in conditions.items():

                output = embedder(condition)

                embeddings[name] = flatten_embedding(output)

            reference = embeddings["correct"]

            reference_norm = torch.linalg.vector_norm(
                reference
            )

            print("\nMolecule:", mid)
            print("Embedding dimension:", reference.shape[-1])
            print(
                "Correct embedding norm:",
                round(reference_norm.item(), 6)
            )

            print(
                f"{'Condition':<14}"
                f"{'Relative L2':>18}"
                f"{'Cosine':>18}"
                f"{'Embedding norm':>18}"
            )

            for name, embedding in embeddings.items():

                relative = (
                    torch.linalg.vector_norm(
                        embedding - reference
                    )
                    / reference_norm.clamp_min(1e-12)
                ).item()

                cosine = F.cosine_similarity(
                    embedding,
                    reference,
                    dim=-1
                ).item()

                norm = torch.linalg.vector_norm(
                    embedding
                ).item()

                print(
                    f"{name:<14}"
                    f"{relative:>18.6f}"
                    f"{cosine:>18.6f}"
                    f"{norm:>18.6f}"
                )

    del model

print("\nEMBEDDING DIAGNOSTIC COMPLETE")
