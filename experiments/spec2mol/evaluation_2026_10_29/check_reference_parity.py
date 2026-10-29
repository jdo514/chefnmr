#!/usr/bin/env python3
"""Compare manual HDF5 reference rows against ChefNMR validation loader."""
from pathlib import Path
import h5py
import numpy as np
from omegaconf import OmegaConf
from src.data.datamodule import NMRDataModule

BASE = Path.home() / 'spec2mol-chefnmr'
cfg = OmegaConf.load(BASE / 'logs/full_reference_2222733.0/.hydra/config.yaml')
dm = NMRDataModule(cfg.dataset_args)
dm.prepare_data()
dm.setup('fit')
print('\n=== REFERENCE SPECTRUM / LOADER PARITY ===')
with h5py.File(BASE / 'data/molecules.h5', 'r', swmr=True) as h5:
    for mid in (292058, 295115):
        positions = np.flatnonzero(np.asarray(dm.val_indices) == mid)
        if len(positions) != 1:
            raise RuntimeError(f'{mid}: expected one validation entry; got {len(positions)}')
        (model_inputs, smiles), model_outputs = dm._val_set[int(positions[0])]
        ref = np.concatenate([
            np.asarray(h5[f'{mid:07d}']['spectra/h_10k'][0], dtype=np.float32),
            np.asarray(h5[f'{mid:07d}']['spectra/c_80'][0], dtype=np.float32),
        ])
        actual = np.asarray(model_inputs['condition'], dtype=np.float32).reshape(-1)
        if actual.shape != ref.shape:
            raise RuntimeError(f'{mid}: loader condition shape {actual.shape} != {ref.shape}')
        maximum = float(np.max(np.abs(actual - ref)))
        print(f'{mid:07d} condition_length={len(actual)} max_abs_difference={maximum:.9g}')
        print('  inputs:', {k: tuple(np.shape(v)) for k, v in model_inputs.items()})
        print('  outputs:', {k: tuple(np.shape(v)) for k, v in model_outputs.items()})
        if maximum > 1e-6:
            raise RuntimeError(f'{mid}: condition differs from manual HDF5 variant-0 concatenation')
print('REFERENCE INPUT PARITY PASS')
