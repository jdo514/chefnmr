#!/usr/bin/env python3
"""CPU-only, read-only comparison of converted SpecTeach graphs against FROZEN train selection.

Uses strict_clean_conformers_v1/train.npy (correct trained 95,202 IDs) plus
strict_clean_v1/split_assignments.csv (SMILES/scaffold lookup). Does not read
42.7 GB HDF5. An old split_assignment CSV is only a lookup; membership always
comes from the actual frozen training .npy file. Summaries are output elsewhere.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def canon(s):
    mol=Chem.MolFromSmiles(str(s)) if s else None
    if mol is None:return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol,isomericSmiles=False)

def murcko(s):
    mol=Chem.MolFromSmiles(s)
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return canon(scaffold) if scaffold else ''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--converted',type=Path,required=True)
    p.add_argument('--train-ids',type=Path,required=True)
    p.add_argument('--assignments',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    train={int(x) for x in np.load(args.train_ids,allow_pickle=False)}
    refs={}; used=0
    with args.assignments.open(newline='') as f:
        for row in csv.DictReader(f):
            try: idnum=int(row['molecule_id'])
            except (ValueError,KeyError):continue
            if idnum in train:
                refs[idnum]=row
                used+=1
    missing=train-set(refs)
    if missing:
        raise RuntimeError(f'FROZEN TRAIN LOOKUP INCOMPLETE: {len(missing)} IDs missing from {args.assignments}; do not infer leakage clearance')
    train_exact=set()
    train_scaf=set()
    for rec in refs.values():
        smi=canon(rec.get('canonical_smiles',''))
        if smi: train_exact.add(smi)
        s=rec.get('murcko_scaffold_smiles','')
        if s:
            sc=canon(s)
            if sc: train_scaf.add(sc)
    rows=[]
    with (args.converted/'manifest.csv').open(newline='') as f:
        for rec in csv.DictReader(f):
            smi=canon(rec['smiles'])
            if smi is None:raise RuntimeError('Invalid converted reference smiles')
            sc=murcko(smi)
            hit_exact=smi in train_exact
            hit_scaf=bool(sc) and sc in train_scaf
            id_collision=int(rec['source_molecule_id']) in train
            rows.append(dict(dataset_id=rec['dataset_id'],source_molecule_id=rec['source_molecule_id'],
                            canonical_smiles=smi,murcko_scaffold_smiles=sc,
                            exact_train_structure_overlap=int(hit_exact),
                            train_murcko_scaffold_overlap=int(hit_scaf),
                            source_id_present_in_train=int(id_collision),
                            acyclic_scaffold_not_tested=int(not sc)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(train_ids=len(train),train_ids_with_lookup=len(refs),n_evaluation_molecules=len(rows),
      n_exact_train_structure_overlap=sum(r['exact_train_structure_overlap'] for r in rows),
      n_murcko_train_scaffold_overlap=sum(r['train_murcko_scaffold_overlap'] for r in rows),
      n_source_id_collision=sum(r['source_id_present_in_train'] for r in rows),
      n_acyclic_scaffold_not_tested=sum(r['acyclic_scaffold_not_tested'] for r in rows),
      limitation='For acyclic molecules empty Murcko scaffolds are not compared; topology-based leakage remains unverified.',
      pass_no_exact_or_ring_scaffold_overlap=not any(r['exact_train_structure_overlap'] or r['train_murcko_scaffold_overlap'] for r in rows))
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not summary['pass_no_exact_or_ring_scaffold_overlap']:
        print('LEAKAGE REVIEW REQUIRED: do not claim independent evaluation until resolved.')
        return 2
    print('EXACT / RING-SCAFFOLD TRAIN OVERLAP AUDIT PASS; ACYCLIC TOPOLOGY AUDIT STILL REQUIRED IF RELEVANT')
    return 0
if __name__=='__main__':
    raise SystemExit(main())
