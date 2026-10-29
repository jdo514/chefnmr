#!/usr/bin/env python3
"""CPU-only diagnostic: benchmark ChefNMR's SAME bond-reconstruction function on
known-correct 3D conformers from the frozen synthetic validation cohort.

Run on indium (from ~/spec2mol-chefnmr):
  export PYTHONPATH="$PWD/chefnmr${PYTHONPATH:+:$PYTHONPATH}"
  python launch/chefnmr_eval/diagnose_chefnmr_graph_reconstruction_v1.py \
    --base "$PWD" --limit 20 --conformers 1

No GPU/model checkpoint used. No full HDF5 scan. Original molecule coordinates
are used ONLY in this separate reconstruction diagnostic, never for sampling.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem


def decode(x):
    return x.decode('utf-8') if isinstance(x, bytes) else str(x)


def canonical(smiles):
    if not smiles:
        return None
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def nearest_neighbor_stats(coords, mask):
    xyz = np.asarray(coords, dtype=np.float64)[np.asarray(mask).astype(bool)]
    if len(xyz) < 2 or not np.isfinite(xyz).all():
        return dict(nearest_median=None, nearest_over_2A=None, nearest_under_0p5A=None)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    near = d.min(axis=1)
    return dict(nearest_median=float(np.median(near)),
                nearest_over_2A=float(np.mean(near > 2.0)),
                nearest_under_0p5A=float(np.mean(near < 0.5)))


def summarize_generated_candidates(path):
    if not path.is_file():
        print('CANDIDATE CSV not found (optional):', path)
        return
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    breakdown = Counter()
    for r in rows:
        s = bool(str(r.get('predicted_smiles', '')).strip())
        f = bool(str(r.get('predicted_largest_fragment', '')).strip())
        key = ('full_graph' if s else 'fragment_only' if f else 'no_graph')
        breakdown[(r['model'], r['condition'], r['atom_order'], key)] += 1
    groups = sorted({k[:3] for k in breakdown})
    print('\n=== BMRB100 CANDIDATE RECONSTRUCTION OUTCOMES ===')
    print('model,condition,atom_order,full_graph,fragment_only,no_graph,n')
    for group in groups:
        count = [breakdown[(*group, state)] for state in ('full_graph','fragment_only','no_graph')]
        print(','.join([*group,*(str(x) for x in count),str(sum(count))]))
    print('NOTE: no_graph conflates invalid bond determination, timeouts, and exceptions.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', type=Path, default=Path.home()/'spec2mol-chefnmr')
    ap.add_argument('--limit', type=int, default=20, help='First N frozen validation molecules, 1..100')
    ap.add_argument('--conformers', type=int, default=1, help='Number of known reference conformers per molecule (1..3)')
    ap.add_argument('--timeout', type=int, default=4, help='Use identical RDKit timeout as pilot (seconds)')
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    if args.limit < 1 or args.limit > 100 or args.conformers < 1 or args.conformers > 3 or args.timeout < 0:
        ap.error('Require 1 <= limit <= 100, 1 <= conformers <= 3, timeout >= 0')
    base = args.base.expanduser().resolve()
    manifest = base/'data/external_evaluation/synthetic_val100_graph_v1/manifest.csv'
    h5path = base/'data/molecules.h5'
    out = (args.output or (base/'reports/chefnmr_truth_geometry_reconstruction_v1.csv')).expanduser().resolve()
    if not manifest.exists() or not h5path.exists():
        raise SystemExit(f'Missing manifest or source archive: {manifest} | {h5path}')

    # Must use the exact same implementation used by the graph identification pilot.
    from src.evaluation.bond_analyzer import atom_features_to_smiles_one
    with manifest.open(newline='') as f:
        records = list(csv.DictReader(f))
    if len(records) != 100:
        raise RuntimeError(f'Expected frozen 100-molecule selection, found {len(records)}')
    selected = records[:args.limit]
    rows = []
    print('=== KNOWN-CORRECT COORDINATES -> RDKit GRAPH RECONSTRUCTION ===', flush=True)
    with h5py.File(h5path,'r') as h:
        for i, rec in enumerate(selected, 1):
            mid = rec['dataset_id']
            if mid not in h:
                raise RuntimeError(f'Molecule missing from HDF5: {mid}')
            group = h[mid]
            af = group['atom_features']
            onehot = np.asarray(af['atom_one_hot'][()], dtype=np.float32)
            mask = np.asarray(af['atom_mask'][()], dtype=np.float32)
            decoder = [decode(x) for x in af['atom_decoder'][()]]
            coords = np.asarray(af['atom_coords'][()], dtype=np.float32)
            if decoder != ['C', 'H', 'O', 'N']:
                raise RuntimeError(f'{mid}: decoder mismatch {decoder}')
            if coords.ndim != 3 or coords.shape[1:] != (len(mask),3):
                raise RuntimeError(f'{mid}: wrong conformer shape {coords.shape}')
            expected = canonical(decode(group.attrs['smiles']))
            if expected != canonical(rec['smiles']):
                raise RuntimeError(f'{mid}: reference SMILES mismatch')
            if not np.isfinite(coords).all():
                raise RuntimeError(f'{mid}: non-finite conformers')
            for conf in range(min(args.conformers, len(coords))):
                exception = ''
                try:
                    _, predicted, fragment = atom_features_to_smiles_one((
                        conf, coords[conf], onehot, mask, 'spectrabase', False, True,
                        decoder, args.timeout))
                except Exception as e:
                    predicted = fragment = None
                    exception = f'{type(e).__name__}: {str(e)[:180]}'
                stat = nearest_neighbor_stats(coords[conf], mask)
                row = dict(molecule_id=mid, conformer_index=conf, n_atoms=int(mask.sum()),
                           target_smiles=expected, reconstructed_smiles=predicted or '',
                           reconstructed_fragment=fragment or '',
                           full_graph=int(bool(predicted)), fragment_only=int(not predicted and bool(fragment)),
                           exact_graph_match=int(canonical(predicted)==expected),
                           exception=exception, **stat)
                rows.append(row)
            if i <= 5 or i % 10 == 0 or i == len(selected):
                succ=sum(r['exact_graph_match'] for r in rows)
                print(f'{i}/{len(selected)} molecules, {len(rows)} reference conformers checked, exact={succ}', flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    full=sum(r['full_graph'] for r in rows)
    fragments=sum(r['fragment_only'] for r in rows)
    exact=sum(r['exact_graph_match'] for r in rows)
    print('\n=== KNOWN-CORRECT GEOMETRY SUMMARY ===')
    print(json.dumps(dict(n_reference_conformers=len(rows), full_graph=full,
                          fragment_only=fragments, no_graph=len(rows)-full-fragments,
                          exact_recovery=exact,
                          exact_recovery_fraction=exact/len(rows),
                          output=str(out)),indent=2))
    print('NOTE: Ground-truth coordinates were read solely for the reconstruction QA.')
    summarize_generated_candidates(base/'logs/bmrb100_graph_eval_2223655.0/full/candidates.csv')

if __name__=='__main__':
    main()
