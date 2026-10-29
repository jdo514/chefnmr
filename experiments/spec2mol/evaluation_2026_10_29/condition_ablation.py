from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from src.model.model import NMRTo3DStructureElucidation
from src.model.modules.utils import center_random_augmentation
from src.data.datamodule import NMRDataset

torch.set_num_threads(4)

BASE = Path.home() / "spec2mol-chefnmr"
IDS = [292058, 295115]
SIGMAS = [3.0, 10.0]

CHECKPOINTS = {
    "reference": next((BASE / "logs").glob(
        "full_reference_*/checkpoints/*/diffusion_loss/epoch0010-loss88.19.ckpt"
    )),
    "augmented": next((BASE / "logs").glob(
        "full_augmented_*/checkpoints/*/diffusion_loss/epoch0010-loss88.26.ckpt"
    )),
}

def rmse(a, b, mask):
    sq = ((a - b) ** 2).sum(dim=-1)
    return torch.sqrt((sq * mask).sum() / mask.sum()).item()

# Read configuration without loading the complete model into memory.
checkpoint = torch.load(
    CHECKPOINTS["reference"],
    map_location="cpu",
    weights_only=False,
    mmap=True,
)

cfg = checkpoint["hyper_parameters"]["cfg"]
inp = OmegaConf.to_container(
    cfg["dataset_args"]["input_generator_addn_args"],
    resolve=True
)

del checkpoint

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
    data_indices=np.asarray(IDS, dtype=np.int64),
    input_generator="H1C13NMRSpectrum",
    input_generator_addn_args=inp,
    use_all_conformers=True,
    spectral_variant_mode="paired_expanded",
    spectral_variant_count=11,
    spectral_variant_seed=42,
)

# Prepare examples once, ensuring identical inputs for both models.
examples = []

for i, mol_id in enumerate(IDS):
    np.random.seed(4201 + i)
    (raw_inputs, smiles), outputs = dataset[i * 11]

    inputs = {
        key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
        for key, value in raw_inputs.items()
    }

    mask = inputs["atom_mask"]

    torch.manual_seed(1729 + i)

    x0 = torch.as_tensor(
        outputs["atom_coords"], dtype=torch.float32
    ).unsqueeze(0)

    x0 = center_random_augmentation(
        x0, mask, centering=True, augmentation=True
    )

    noise = torch.randn_like(x0)

    examples.append({
        "id": mol_id,
        "smiles": smiles,
        "inputs": inputs,
        "x0": x0,
        "noise": noise,
        "mask": mask,
    })

dataset.data_manager.close()

for model_name, path in CHECKPOINTS.items():

    print(f"\n{'=' * 65}")
    print("MODEL:", model_name.upper())
    print("=" * 65)

    model = NMRTo3DStructureElucidation.load_from_checkpoint(
        str(path),
        map_location="cpu",
        strict=True,
    )

    model.eval()
    assert model.use_ema and model.ema is not None

    with torch.no_grad():
        model.prepare_eval()

        for i, example in enumerate(examples):

            inputs = example["inputs"]
            mask = example["mask"]
            x0 = example["x0"]
            noise = example["noise"]

            correct = inputs["condition"]
            swapped = examples[1 - i]["inputs"]["condition"]

            conditions = {
                "correct": correct,
                "zero": torch.zeros_like(correct),
                "swapped": swapped,
            }

            print(f"\nMolecule {example['id']:07d}")
            print("Target:", example["smiles"])

            for sigma_value in SIGMAS:

                sigma = torch.tensor([sigma_value])

                noisy = model.model.interpolate(
                    x0, noise, sigma
                )

                predictions = {}
                errors = {}

                for condition_name, condition in conditions.items():

                    test_inputs = dict(inputs)
                    test_inputs["condition"] = condition

                    net_out = model.model.neural_network_forward(
                        noisy,
                        sigma,
                        network_condition_kwargs={
                            "model_inputs": test_inputs,
                            "multiplicity": 1,
                            "guidance_scale": 0.0,
                        },
                    )

                    denoised = model.model.predict_denoised_atom_coords(
                        noisy, net_out, sigma
                    )

                    predictions[condition_name] = denoised

                    errors[condition_name] = rmse(
                        denoised, x0, mask
                    )

                output_change = rmse(
                    predictions["correct"],
                    predictions["zero"],
                    mask,
                )

                print(f"\nSigma: {sigma_value}")
                print(f"  Correct spectra RMSE: {errors['correct']:.3f} Å")
                print(f"  Zero spectra RMSE:    {errors['zero']:.3f} Å")
                print(f"  Swapped spectra RMSE: {errors['swapped']:.3f} Å")

                print(
                    "  Improvement over zero:",
                    f"{errors['zero'] - errors['correct']:+.3f} Å"
                )

                print(
                    "  Output change (correct vs zero):",
                    f"{output_change:.3f} Å"
                )

    del model

print("\nCONDITIONING ABLATION COMPLETE")
