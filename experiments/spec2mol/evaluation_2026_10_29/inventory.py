#!/usr/bin/env python3
"""Read-only audit for external ChefNMR-ready HDF5 or raw ZIP contents.
Does not convert raw experimental peaks/FIDs to spectra: that requires source-specific processing.
"""
import argparse
import json
from pathlib import Path
import zipfile
from collections import Counter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--limit',type=int,default=0,help='0 audits all molecules')
    p.add_argument('--output',default=None)
    a=p.parse_args()
    path=Path(a.path)
    if not path.exists():raise SystemExit('MISSING_FILE: '+str(path))
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            rows=[{'name':x.filename,'bytes':x.file_size} for x in z.infolist()]
        res={'archive':str(path),'members':len(rows),'first_80':rows[:80],'status':'RAW_ZIP_REQUIRES_SCHEMA_ADAPTER'}
    else:
        import h5py
        import numpy as np
        if path.is_dir():path=path/'molecules.h5'
        stats=Counter();valid=[];invalid=[];variants=Counter()
        with h5py.File(path,'r',swmr=True) as f:
            ids=sorted([k for k in f.keys() if k.isdigit()])
            for idx,k in enumerate(ids):
                if a.limit and idx>=a.limit:break
                try:
                    mol=f[k]
                    sp=mol['spectra']; at=mol['atom_features']
                    h=sp['h_10k']; c=sp['c_80']; coords=at['atom_coords']; mask=at['atom_mask']; one=at['atom_one_hot']
                    if h.ndim!=2 or h.shape[1]!=10000:raise ValueError('h1_grid_not_10000')
                    if c.ndim!=2 or c.shape[1]!=80:raise ValueError('c13_grid_not_80')
                    if h.shape[0]<1 or c.shape[0]<1 or h.shape[0]!=c.shape[0]:raise ValueError('unpaired_or_absent_spectra')
                    if one.ndim!=2 or one.shape[1]!=4:raise ValueError('non_CHON_or_incompatible_atom_one_hot')
                    if mask.shape[0]>59 or one.shape[0]>59:raise ValueError('too_many_atoms')
                    if coords.ndim!=3 or coords.shape[0]<1 or coords.shape[2]!=3:raise ValueError('missing_3d_conformers')
                    if not np.isfinite(h[:]).all() or not np.isfinite(c[:]).all():raise ValueError('nonfinite_spectra')
                    if not np.isfinite(coords[:]).all():raise ValueError('nonfinite_3d_coords')
                    if not np.isfinite(one[:]).all() or not np.isfinite(mask[:]).all():raise ValueError('nonfinite_atom_features')
                    if not mol.attrs.get('smiles'):raise ValueError('missing_smiles')
                    valid.append(k);variants[str(h.shape[0])]+=1
                except Exception as ex:
                    why=str(ex).split(':')[0];stats[why]+=1
                    if len(invalid)<30:invalid.append({'molecule_id':k,'reason':why})
        res={'path':str(path),'audited':len(valid)+sum(stats.values()),'eligible':len(valid),'eligible_ids':valid,'variants_per_molecule':dict(variants),'excluded_reasons':dict(stats),'examples_excluded':invalid,'status':'READY' if valid else 'NO_ELIGIBLE_MOLECULES'}
    print(json.dumps({k:v for k,v in res.items() if k!='eligible_ids'},indent=2))
    if a.output:
        dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(res,indent=2)+'\n')
    if res['status'] not in ('READY',):
        print('INVENTORY_NOT_READY: source-specific experimental conversion / compatibility checks still needed')

if __name__=='__main__':main()
