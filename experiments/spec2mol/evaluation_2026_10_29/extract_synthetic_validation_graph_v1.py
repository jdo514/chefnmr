#!/usr/bin/env python3
"""Read-only targeted extraction of ChefNMR-S synthetic validation spectra.

Reproduces the 100-molecule validation cohort of fixed_noise_benchmark.py
(seed 20261028; train selection is drawn first, then validation) and checks
against its saved metadata when present. Uses variant index 0 and original
HDF5 atom features. DOES NOT read or export any 3D conformer coordinates.

Output uses the same dataset interface as the BMRB100 graph evaluator, plus
atom_symbols_training_order_json in its manifest for exact training parity.
Output is written atomically and never overwrites existing data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

SEED = 20261028
EXCLUDE_VAL = (292058, 295115)
DEFAULT_N_TRAIN = 100
DEFAULT_N_VAL = 100
CANON_DECODER = None  # read from source, do not assume element order


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def decode(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def pick_frozen_cohort(split_dir: Path, n: int, metadata_path: Path | None):
    train = np.load(split_dir / 'train.npy', allow_pickle=False)
    val = np.load(split_dir / 'val.npy', allow_pickle=False)
    manifest = json.loads((split_dir / 'manifest.json').read_text())
    if len(train) != 95202 or len(val) != 10575:
        raise RuntimeError(f'Frozen conformer split changed: train={len(train)}, val={len(val)}')
    if len(np.intersect1d(train, val)):
        raise RuntimeError('Frozen train and validation partitions overlap')
    if str(manifest.get('selection_name','')) not in ('strict_clean_conformers_v1', ''):
        raise RuntimeError('Unexpected selection name in manifest')
    rng = np.random.default_rng(SEED)
    rng.choice(train, size=DEFAULT_N_TRAIN, replace=False)  # must advance RNG as in fixed_noise_benchmark.py
    valpool = val[~np.isin(val, EXCLUDE_VAL)]
    chosen = sorted(map(int, rng.choice(valpool, size=DEFAULT_N_VAL, replace=False)))
    reused = False
    if metadata_path and metadata_path.is_file():
        prior = json.loads(metadata_path.read_text())
        if prior.get('n_train') != DEFAULT_N_TRAIN or prior.get('n_val') != DEFAULT_N_VAL or prior.get('seed') != SEED:
            raise RuntimeError('Prior fixed-noise metadata differs from the documented 100+100 selection')
        saved = sorted(int(x) for x in prior['targets']['val'])
        if saved != chosen:
            raise RuntimeError('Reconstructed previous validation cohort differs from saved benchmark metadata')
        reused = True
    if n > len(chosen):
        raise ValueError(f'n={n} exceeds prior 100-molecule cohort; refuse to silently change cohort')
    if n < 1:
        raise ValueError('n must be >= 1')
    chosen = chosen[:n]
    if not set(chosen).issubset(set(map(int,val))):
        raise RuntimeError('Selection is not contained in frozen validation partition')
    return chosen, manifest, reused


def extract_one(group: h5py.Group, mid: int, position: int, target_dir: Path, expected_decoder):
    if 'spectra/h_10k' not in group or 'spectra/c_80' not in group:
        raise RuntimeError(f'{mid}: missing original ChefNMR spectral arrays')
    hds, cds = group['spectra/h_10k'], group['spectra/c_80']
    if hds.ndim != 2 or hds.shape[1] != 10000 or hds.shape[0] < 1:
        raise RuntimeError(f'{mid}: unexpected proton spectra shape {hds.shape}')
    if cds.ndim != 2 or cds.shape[1] != 80 or cds.shape[0] != hds.shape[0]:
        raise RuntimeError(f'{mid}: unexpected carbon spectra shape {cds.shape}')
    h = np.asarray(hds[0], dtype=np.float32)
    c = np.asarray(cds[0], dtype=np.float32)
    af = group['atom_features']
    atom_mask = np.asarray(af['atom_mask'][()], dtype=np.float32)
    onehot = np.asarray(af['atom_one_hot'][()], dtype=np.float32)
    decoder = [decode(x) for x in af['atom_decoder'][()]]
    if expected_decoder is not None and decoder != expected_decoder:
        raise RuntimeError(f'{mid}: atom decoder differs from prior molecule: {decoder} vs {expected_decoder}')
    if not (onehot.ndim == 2 and onehot.shape[1] == len(decoder) and
            atom_mask.shape == (onehot.shape[0],)):
        raise RuntimeError(f'{mid}: incompatible atom feature arrays')
    if not (np.isfinite(h).all() and np.isfinite(c).all() and np.isfinite(onehot).all()
            and np.isfinite(atom_mask).all()):
        raise RuntimeError(f'{mid}: nonfinite source values')
    if not (np.isin(c,[0,1]).all() and np.isin(atom_mask,[0,1]).all()):
        raise RuntimeError(f'{mid}: invalid carbon encoding or mask')
    if not np.allclose(onehot[atom_mask.astype(bool)].sum(axis=1), 1.,atol=1e-6):
        raise RuntimeError(f'{mid}: invalid one-hot atom features')
    native_symbols = [decoder[int(i)] for i in np.argmax(onehot[atom_mask.astype(bool)], axis=1)]
    if not (0 < len(native_symbols) <= 59 and set(native_symbols).issubset({'C','H','N','O'})):
        raise RuntimeError(f'{mid}: incompatible atom features {native_symbols}')
    smiles = decode(group.attrs['smiles'])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f'{mid}: invalid source SMILES {smiles}')
    molh = Chem.AddHs(mol)
    from collections import Counter
    expected_formula = Counter(a.GetSymbol() for a in molh.GetAtoms())
    actual_formula = Counter(native_symbols)
    if actual_formula != expected_formula:
        raise RuntimeError(f'{mid}: source atom features do not match reference molecular formula: {actual_formula} != {expected_formula}')
    # Chemically valid graph reference only; 2D drawing coordinates are NOT model inputs.
    rdDepictor.Compute2DCoords(mol)
    sdf_path = target_dir / 'structures' / f'{mid:07d}.sdf'
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    checkmol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if checkmol is None or Chem.MolToSmiles(checkmol,isomericSmiles=False) != Chem.MolToSmiles(mol,isomericSmiles=False):
        raise RuntimeError(f'{mid}: SDF roundtrip mismatch')
    return h,c,decoder,dict(
        pilot_position=position,
        dataset_row=int(mid),
        dataset_id=f'{mid:07d}',
        smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol,isomericSmiles=False),
        reference_structure=f'structures/{mid:07d}.sdf',
        atom_symbols_training_order_json=json.dumps(native_symbols,separators=(',',':')),
        atom_decoder_source_json=json.dumps(decoder,separators=(',',':')),
        source_variant_index=0,
        frozen_split='strict_clean_conformers_v1/val',
        n_atoms_with_h=len(native_symbols),
        h1_maxabs=float(np.max(np.abs(h))),
        c13_occupied_bins=int(np.count_nonzero(c)),
    )


def run(args):
    base = args.base.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else (base / 'data/external_evaluation/synthetic_val100_graph_v1').resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists, refusing overwrite: {output}')
    split_dir = base / 'selections/strict_clean_conformers_v1'
    meta = base / 'logs/fixed_noise_benchmark_full_metadata_20261028.json'
    selected, selection_manifest, validated = pick_frozen_cohort(split_dir,args.n,meta)
    if not args.allow_missing_prior_metadata and not validated:
        raise RuntimeError(f'Prior benchmark metadata not found: {meta}; use --allow-missing-prior-metadata to reproduce deterministic selection without independent matching proof')
    h5path = base/'data/molecules.h5'
    if not h5path.is_file():
        raise FileNotFoundError(h5path)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.synthetic_val_build_',dir=output.parent))
    done = False
    try:
        for name in ('spectra','structures','validation'):
            (work/name).mkdir()
        hs,cs,records=[],[],[]
        decoder0=None
        with h5py.File(h5path,'r',swmr=True) as h5:
            for pos,mid in enumerate(selected):
                key=f'{mid:07d}'
                if key not in h5:
                    raise KeyError(f'Frozen validation ID missing in master HDF5: {key}')
                h,c,decoder,rec=extract_one(h5[key],mid,pos,work,decoder0)
                if decoder0 is None:
                    decoder0=decoder
                hs.append(h);cs.append(c);records.append(rec)
                if pos<5 or pos+1==len(selected) or (pos+1)%25==0:
                    print(f'{pos+1:3d}/{len(selected)} ID={key} H={h.shape} Cbins={rec["c13_occupied_bins"]} atoms={rec["n_atoms_with_h"]}',flush=True)
        h_arr=np.stack(hs).astype(np.float32)
        c_arr=np.stack(cs).astype(np.float32)
        cond=np.concatenate([h_arr,c_arr],axis=1)
        np.save(work/'spectra/h1_10000.npy',h_arr,allow_pickle=False)
        np.save(work/'spectra/c13_80.npy',c_arr,allow_pickle=False)
        np.save(work/'spectra/condition_10080.npy',cond,allow_pickle=False)
        with (work/'manifest.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        report={
            'schema':'synthetic_validation_chefnmr_graph_v1',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'frozen_selection':'strict_clean_conformers_v1',
            'train_count':95202,'val_count':10575,
            'n_molecules':len(records),'seed':SEED,
            'prior_selection_reproduced_and_metadata_verified':validated,
            'prior_fixed_noise_metadata_path':str(meta) if validated else None,
            'source_hdf5':str(h5path),
            'source_reference_variant_index':0,
            'source_atom_decoder':decoder0,
            'source_group_keys': [f'{mid:07d}' for mid in selected],
            'prohibited_source_data':'atom_features/atom_coords not opened; SDF 2D only for graph scoring',
            'source_arrays':['spectra/h_10k[0]','spectra/c_80[0]','atom_features/atom_one_hot','atom_features/atom_mask','atom_features/atom_decoder','attrs/smiles'],
            'native_atom_order':'The manifest preserves exact HDF5 atom-feature order; evaluator v2 must consume atom_symbols_training_order_json',
            'benchmark_role':'Frozen training-validation population, in-distribution; not an independent test split',
            'checks':dict(split_membership=True,source_spectra_finite=True,source_c13_binary=True,
                          source_formula_atom_features_agree=True,graph_sdf_roundtrip=True),
        }
        (work/'validation/extraction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        (work/'README.md').write_text(
            '# ChefNMR-S synthetic validation graph identification benchmark\n\n'
            'Fixed cohort from fixed_noise_benchmark.py, seed 20261028; original reference variant 0.\n'
            'No 3D reference coordinates have been exported or opened.\n'
            'The manifest contains atom_symbols_training_order_json for exact training-atom ordering.\n'
            'Run using the patched graph evaluator v2, which consumes this column.\n'
            'This cohort was used for validation/checkpoint selection and is not a new independent test partition.\n'
        )
        with (work/'checksums.sha256').open('w') as f:
            for path in sorted(x for x in work.rglob('*') if x.is_file() and x.name!='checksums.sha256'):
                f.write(f'{checksum(path)}  {path.relative_to(work).as_posix()}\n')
        if args.dry_run:
            print('DRY RUN: all source checks and conversion passed; temporary outputs discarded.',flush=True)
        else:
            work.rename(output)
            done=True
            print(f'OUTPUT: {output}',flush=True)
            print(f'H1 {h_arr.shape} C13 {c_arr.shape} condition {cond.shape}',flush=True)
            print('SYNTHETIC VALIDATION EXTRACTION PASS',flush=True)
    finally:
        if not done:
            shutil.rmtree(work,ignore_errors=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path.home()/'spec2mol-chefnmr')
    p.add_argument('--output',type=Path)
    p.add_argument('--n',type=int,default=100,help='Up to first 100 IDs from the frozen previously used denoising cohort')
    p.add_argument('--allow-missing-prior-metadata',action='store_true',help='Allow exact selection-algorithm reproduction without saved metadata verification')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    run(args)


if __name__=='__main__':
    main()
