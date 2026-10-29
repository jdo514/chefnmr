from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from src.model.model import NMRTo3DStructureElucidation
from src.model.modules.utils import center_random_augmentation
from src.data.datamodule import NMRDataset

torch.set_num_threads(4)

BASE = Path.home() / "spec2mol-chefnmr"
MOLECULES = [292058, 295115]
SIGMAS = [0.25, 1.0, 3.0]

CHECKPOINTS = {
    "reference": next((BASE / "logs").glob(
        "full_reference_*/checkpoints/*/diffusion_loss/epoch0010-loss88.19.ckpt"
    )),
    "augmented": next((BASE / "logs").glob(
        "full_augmented_*/checkpoints/*/diffusion_loss/epoch0010-loss88.26.ckpt"
    )),
}

def calculate_rmse(a, b, mask):
    squared = ((a - b) ** 2).sum(dim=-1)
    return torch.sqrt((squared * mask).sum() / mask.sum()).item()

for model_name, checkpoint in CHECKPOINTS.items():

    print(f"\n{'=' * 65}")
    print("MODEL:", model_name.upper())
    print("=" * 65)

    model = NMRTo3DStructureElucidation.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
        strict=True,
    )

    model.eval()

    assert model.use_ema
    assert model.ema is not None
    assert model.ema.num_updates == 32736

    # Apply the stored EMA parameters, as during evaluation.
    with torch.no_grad():
        model.prepare_eval()

    cfg = model.cfg.dataset_args

    inp = OmegaConf.to_container(
        cfg.input_generator_addn_args,
        resolve=True,
    )

    inp["h1nmr"]["max_n_spectra"] = 1
    inp["c13nmr"]["max_n_spectra"] = 1

    inp["atom_features"] = {
        "max_n_atoms": 59,
        "max_n_atom_types": 4,
        "max_n_conformers": 3,
        "known_atoms": True,
    }

    inp["multitask_args"] = {
        "p_drop_h1nmr": 0.0,
        "p_drop_c13nmr": 0.0,
        "p_drop_both": 0.0,
        "drop_transform": "zero",
    }

    dataset = NMRDataset(
        datadir=BASE / "data",
        data_indices=np.asarray(MOLECULES, dtype=np.int64),
        input_generator="H1C13NMRSpectrum",
        input_generator_addn_args=inp,
        use_all_conformers=True,
        spectral_variant_mode="paired_expanded",
        spectral_variant_count=11,
        spectral_variant_seed=42,
    )

    for i, mol_id in enumerate(MOLECULES):

        torch.manual_seed(1729 + i)
        np.random.seed(1729 + i)

        (raw_inputs, smiles), outputs = dataset[i * 11]

        inputs = {
            key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
            for key, value in raw_inputs.items()
        }

        mask = inputs["atom_mask"]

        x0 = torch.as_tensor(
            outputs["atom_coords"],
            dtype=torch.float32
        ).unsqueeze(0)

        # Same coordinate augmentation family used for training.
        x0 = center_random_augmentation(
            x0,
            mask,
            centering=True,
            augmentation=True,
        )

        noise = torch.randn_like(x0)

        print(f"\nMolecule {mol_id:07d}")
        print("Target:", smiles)
        print("Sigma | Noisy RMSE | Denoised RMSE | Improvement")

        for sigma_value in SIGMAS:

            sigma = torch.tensor([sigma_value])

            with torch.no_grad():

                noisy = model.model.interpolate(
                    x0, noise, sigma
                )

                prediction = model.model.neural_network_forward(
                    noisy,
                    sigma,
                    network_condition_kwargs={
                        "model_inputs": inputs,
                        "multiplicity": 1,
                        "guidance_scale": 0.0,
                    },
                )

                denoised = model.model.predict_denoised_atom_coords(
                    noisy,
                    prediction,
                    sigma,
                )

                before = calculate_rmse(noisy, x0, mask)
                after = calculate_rmse(denoised, x0, mask)

            improvement = 100 * (1 - after / before)

            print(
                f"{sigma_value:5.2f} | "
                f"{before:10.3f} | "
                f"{after:13.3f} | "
                f"{improvement:+10.1f}%"
            )

    dataset.data_manager.close()

    del model

print("\nDENOISING DIAGNOSTIC COMPLETE")
