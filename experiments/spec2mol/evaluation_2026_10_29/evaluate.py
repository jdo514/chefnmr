#!/usr/bin/env python3
"""GPU evaluation of ChefNMR checkpoints with the repository's original test_step/metrics.

Does NOT modify source master or frozen split. Same subset/seed per model.
Supports the frozen synthetic validation split, or an independently converted
ChefNMR-ready external HDF5 (CHON, <=59 atoms, paired H 10k / C 80, 3D coords).

Run from ~/spec2mol-chefnmr/chefnmr (imports of `src` required).
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from lightning.pytorch import Trainer,seed_everything
from src.model.model import NMRTo3DStructureElucidation
from src.data.datamodule import NMRDataset

class FixedVariantDataset(Dataset):
    """Choose the SAME spectral variant for every held-out molecule."""
    def __init__(self,base,variant,count):
        self.base=base;self.variant=variant;self.count=count
    def __len__(self):return len(self.base.data_indices)
    def __getitem__(self,i):return self.base[i*self.count+self.variant]


def selection_ids(a):
    path=a.data_dir/'molecules.h5'
    if not path.is_file():raise ValueError(f'No ChefNMR HDF5 at {path}')
    if a.dataset=='frozen-val':
        from src.data.external_split_loader import load_external_split_indices
        x=load_external_split_indices(str(a.selection),path)
        ids=[int(k) for k in x['val']]
        if len(ids)!=10575:
            raise ValueError(f'Frozen validation selection count changed: {len(ids)} (expected 10575)')
    else:
        if a.inventory is None:
            raise ValueError('Independent data require --inventory JSON from inventory.py (audited ready records)')
        j=json.loads(a.inventory.read_text())
        if j['status']!='READY' or not j.get('eligible_ids'):
            raise ValueError('External data audit not READY')
        if Path(j['path']).resolve()!=path.resolve():
            raise ValueError('Inventory HDF5 path mismatch')
        ids=[int(k) for k in j['eligible_ids']]
    rng=np.random.default_rng(a.subset_seed)
    if a.limit and a.limit<len(ids):ids=sorted(int(x) for x in rng.choice(ids,size=a.limit,replace=False))
    return ids


def validate_selected_shapes(path,ids,count,variant):
    with h5py.File(path,'r',swmr=True) as f:
        for idx in ids:
            g=f[f'{idx:07d}'];h=g['spectra/h_10k'];c=g['spectra/c_80']
            if h.shape!=(count,10000) or c.shape!=(count,80):
                raise ValueError(f'{idx}: expected paired ({count}, 10000)/({count},80), got {h.shape}/{c.shape}')
            if g['atom_features/atom_one_hot'].shape[1]!=4:raise ValueError(f'{idx}: non-CHON model input')
            if g['atom_features/atom_mask'].shape[0]>59:raise ValueError(f'{idx}: >59 atoms')
            if g['atom_features/atom_coords'].shape[0]<1:raise ValueError(f'{idx}: missing conformers')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--dataset',choices=['frozen-val','specteach','bmrb100'],required=True)
    p.add_argument('--data-dir',required=True,type=Path)
    p.add_argument('--selection',type=Path,default=None,help='Frozen selection dir, for frozen-val')
    p.add_argument('--inventory',type=Path,default=None,help='External inventory JSON, for independent datasets')
    p.add_argument('--variant',type=int,default=0,help='0=reference; 1..10=aug_000..aug_009')
    p.add_argument('--variants-per-molecule',type=int,default=11)
    p.add_argument('--limit',type=int,default=12,help='Number of molecules, 0=ALL (large GPU cost)')
    p.add_argument('--subset-seed',type=int,default=4201)
    p.add_argument('--inference-seed',type=int,default=1729)
    p.add_argument('--candidates',type=int,default=1)
    p.add_argument('--sampling-steps',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--cpus',type=int,default=2)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.dataset=='frozen-val' and a.selection is None:
        p.error('--selection required with --dataset frozen-val')
    if a.limit<0 or a.batch_size<1 or a.candidates<1 or a.candidates>20 or a.sampling_steps<1:
        p.error('Invalid evaluation size/steps/candidates')
    if not 0<=a.variant<a.variants_per_molecule:p.error('variant outside range')
    a.data_dir=a.data_dir.resolve();a.checkpoint=a.checkpoint.resolve()
    a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=True)
    if list(a.output_dir.iterdir()):raise RuntimeError('Output folder is not empty; refusing to overwrite prior evaluation')
    start=time.time();os.chdir(a.output_dir)
    ids=selection_ids(a)
    if not ids:raise RuntimeError('No eligible molecules')
    validate_selected_shapes(a.data_dir/'molecules.h5',ids,a.variants_per_molecule,a.variant)
    seed_everything(a.inference_seed,workers=True)
    model=NMRTo3DStructureElucidation.load_from_checkpoint(str(a.checkpoint),map_location='cpu',strict=True)
    if model.use_ema and model.ema is None:raise RuntimeError('EMA state missing')
    if tuple(model.dataset_args.atom_decoder)!=('C','H','O','N') or int(model.dataset_args.max_n_atoms)!=59:
        raise ValueError('Trained checkpoint incompatible with CHON / 59 atom evaluation schema')
    if str(model.dataset_args.input_generator)!='H1C13NMRSpectrum':raise ValueError('Wrong conditioning type')
    model.cfg.general.wandb='disabled'
    model.cfg.general.experiment_name=f'{a.dataset}_variant{a.variant:02d}'
    model.test_args.diffusion_samples=a.candidates
    model.test_args.num_sampling_steps=a.sampling_steps
    model.test_args.visualize_samples=0
    model.test_args.visualize_chains=0
    model.visualization_args.n_chain_frames=1
    inp=copy.deepcopy(model.dataset_args.input_generator_addn_args)
    from omegaconf import OmegaConf
    inp=OmegaConf.to_container(inp,resolve=True)
    inp['h1nmr']['max_n_spectra']=1
    inp['c13nmr']['max_n_spectra']=1
    inp['atom_features']={'max_n_atoms':59,'max_n_atom_types':4,'max_n_conformers':3,'known_atoms':True}
    inp['multitask_args']={'p_drop_h1nmr':0.,'p_drop_c13nmr':0.,'p_drop_both':0.,'drop_transform':'zero'}
    ds=NMRDataset(datadir=a.data_dir,data_indices=np.array(ids,dtype=np.int64),
                  input_generator='H1C13NMRSpectrum',input_generator_addn_args=inp,
                  use_all_conformers=True,spectral_variant_mode='paired_expanded',
                  spectral_variant_count=a.variants_per_molecule,spectral_variant_seed=42)
    fixed=FixedVariantDataset(ds,a.variant,a.variants_per_molecule)
    loader=DataLoader(fixed,batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=True)
    # CPU worker limit affects ChefNMR bond_analyzer and RMSD parallelism.
    manifest={'dataset':a.dataset,'hdf5':str(a.data_dir/'molecules.h5'),'checkpoint':str(a.checkpoint),
              'molecule_ids':[f'{i:07d}' for i in ids],'variant_index':a.variant,
              'variants_per_molecule':a.variants_per_molecule,'candidate_count':a.candidates,
              'sampling_steps':a.sampling_steps,'subset_seed':a.subset_seed,
              'inference_seed':a.inference_seed,'n':len(ids),'notes':'Same molecules and settings required for paired reference/augmented comparison. Stereo ignored for matching per ChefNMR.'}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    trainer=Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',logger=False,
                    enable_checkpointing=False,enable_progress_bar=False)
    with patch.object(os,'cpu_count',return_value=a.cpus):
        metrics=trainer.test(model,dataloaders=loader,verbose=True)
    (a.output_dir/'metrics.json').write_text(json.dumps({'results':metrics,'wall_seconds':time.time()-start,'manifest':manifest},indent=2,default=str)+'\n')
    print('CHEFNMR_EVALUATION_PASS:',json.dumps({'dataset':a.dataset,'variant':a.variant,'n':len(ids),'results':metrics}),flush=True)

if __name__=='__main__':main()
