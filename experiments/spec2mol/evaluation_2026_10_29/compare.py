#!/usr/bin/env python3
"""Paired evaluation summaries for two ChefNMR models on exactly matched test inputs.
Reads *actual structure predictions*, NOT diffusion-loss checkpoint names.
Original ChefNMR exact-matching ignores stereochemistry.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from rdkit import Chem


def canon(smiles):
    if not isinstance(smiles,str) or smiles in ('None',''):
        return None
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)


def read_run(folder):
    m=json.loads((folder/'manifest.json').read_text())
    met=json.loads((folder/'metrics.json').read_text())
    output=folder/'samples'/f"{m['dataset']}_variant{m['variant_index']:02d}"/'target_predicted_similarity.csv'
    if not output.is_file():raise ValueError(f'Missing ChefNMR structure predictions {output}')
    with output.open(newline='') as f:rows=list(csv.DictReader(f))
    n=m['n'];k=m['candidate_count']
    if len(rows)!=n*k:raise ValueError(f'Candidate count mismatch: {len(rows)} rows, expected {n*k}')
    correct=[];valid=[];sims=[]
    for i in range(n):
        rr=rows[i*k:(i+1)*k]
        target=[canon(r['target_smiles']) for r in rr]
        if len(set(target))!=1 or target[0] is None:
            raise ValueError(f'Target SMILES inconsistent at position {i}')
        pred=[canon(r['predicted_smiles']) for r in rr]
        correct.append(int(target[0] in pred))
        valid.append(sum(x is not None for x in pred)/k)
        x=[]
        for r in rr:
            try:x.append(float(r['similarity']))
            except (ValueError,TypeError):x.append(0.0)
        sims.append(max(x))
    return {'folder':str(folder),'manifest':m,'original_metrics':met.get('results'),
            'correct':np.array(correct,dtype=int),'valid':np.array(valid,dtype=float),
            'similarity':np.array(sims,dtype=float)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--augmented',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    a=read_run(args.reference);b=read_run(args.augmented)
    shared=('dataset','hdf5','molecule_ids','variant_index','variants_per_molecule',
            'candidate_count','sampling_steps','subset_seed','inference_seed','n')
    for key in shared:
        if a['manifest'][key]!=b['manifest'][key]:
            raise ValueError(f'NOT PAIRED: {key} mismatch. Refusing misleading comparison.')
    n=len(a['correct']);rng=np.random.default_rng(2408)
    delta=b['correct']-a['correct']
    bs=[]
    for _ in range(args.bootstrap):
        idx=rng.integers(0,n,size=n)
        bs.append(float(delta[idx].mean()*100))
    report={
        'dataset':a['manifest']['dataset'],'variant_index':a['manifest']['variant_index'],
        'n_test_molecules':n,'candidates_per_molecule':a['manifest']['candidate_count'],
        'ref_match_percent':round(float(100*a['correct'].mean()),4),
        'aug_match_percent':round(float(100*b['correct'].mean()),4),
        'aug_minus_ref_percentage_points':round(float(100*delta.mean()),4),
        'paired_bootstrap_95pct_interval_percentage_points':[round(float(x),4) for x in np.quantile(bs,[.025,.975])],
        'reference_valid_fraction':round(float(a['valid'].mean()),4),
        'augmented_valid_fraction':round(float(b['valid'].mean()),4),
        'reference_mean_max_tanimoto':round(float(a['similarity'].mean()),4),
        'augmented_mean_max_tanimoto':round(float(b['similarity'].mean()),4),
        'reference_output':str(args.reference),
        'augmented_output':str(args.augmented),
        'notes':'Exact connectivity match ignores stereochemistry, as in original ChefNMR. Paired molecule-level bootstrap estimates sampling uncertainty only, not variation across training seeds.'
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PAIRED_COMPARISON_PASS',json.dumps(report,indent=2))

if __name__=='__main__':main()
