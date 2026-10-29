"""Consistent H1+C13 variant selection without changing stored HDF5.

Converted variant order: reference, aug_000, ..., aug_009.
"""
import numpy as np
from lightning.pytorch.callbacks import Callback

MODES = {"legacy_random", "reference", "paired_cyclic", "paired_random", "paired_expanded"}


def resolve_item(index, n_molecules, n_variants, mode, epoch=0, seed=42):
    """Return (molecule_position, variant_index), or None for legacy mode."""
    if mode not in MODES:
        raise ValueError(f"Invalid spectral_variant_mode: {mode}")
    if n_molecules < 1 or n_variants < 1 or epoch < 0:
        raise ValueError("Invalid number of molecules, variants, or epoch")
    index = int(index)
    total = n_molecules * n_variants if mode == "paired_expanded" else n_molecules
    if not 0 <= index < total:
        raise IndexError(index)
    if mode == "paired_expanded":
        return divmod(index, n_variants)
    if mode == "paired_cyclic":
        return index, (index + int(seed) + int(epoch)) % n_variants
    if mode == "paired_random":
        return index, int(np.random.randint(0, n_variants))
    return index, 0 if mode == "reference" else None


def select_paired_spectra(mol_dict, variant, n_variants):
    """Give both H1 and C13 the same spectral row (still shape (1, dim))."""
    spectra = mol_dict['spectra']
    for key, width in [('h_10k', 10000), ('c_80', 80)]:
        arr = spectra.get(key)
        if not isinstance(arr, np.ndarray) or arr.shape != (n_variants, width):
            raise ValueError(
                f"{key}: expected ({n_variants}, {width}), "
                f"found {getattr(arr, 'shape', None)}"
            )
    if not 0 <= variant < n_variants:
        raise IndexError(variant)
    out = dict(mol_dict)
    out['spectra'] = dict(spectra)
    out['spectra']['h_10k'] = spectra['h_10k'][variant:variant+1]
    out['spectra']['c_80'] = spectra['c_80'][variant:variant+1]
    return out


class SpectralVariantEpochCallback(Callback):
    """Tie cyclic augmentation to absolute Lightning epoch on resume."""

    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, 'set_spectral_epoch'):
            raise RuntimeError('Paired cyclic augmentation needs NMRDataModule')
        dm.set_spectral_epoch(int(trainer.current_epoch))
