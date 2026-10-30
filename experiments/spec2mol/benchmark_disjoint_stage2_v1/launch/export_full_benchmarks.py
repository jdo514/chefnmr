#!/usr/bin/env python3
"""Export FULL benchmark structure identities, not converted/pilot subsets.
Reads official 100-row BMRB100 CSV and official 238-molecule SpecTeach HDF5.
No spectra are processed. This can run on the Ubuntu laptop.
"""
import argparse, csv, json, hashlib
from pathlib import Path
import h5py

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--bmrb-manifest',required=True,type=Path)
    a.add_argument('--specteach-h5',required=True,type=Path)
    a.add_argument('--output',required=True,type=Path)
    o=a.parse_args()
    if o.output.exists():raise FileExistsError(f'Output exists; refusing overwrite: {o.output}')
    with o.bmrb_manifest.open(newline='') as f:raw=list(csv.DictReader(f))
    if len(raw)!=100:raise RuntimeError(f'BMRB100 must be full 100-row manifest, got {len(raw)}')
    if not all('smiles' in r and r['smiles'].strip() for r in raw):raise RuntimeError('BMRB100 missing SMILES')
    bmrb=[dict(source_id=str(r.get('dataset_id') or r.get('dataset_row') or i),smiles=r['smiles'].strip()) for i,r in enumerate(raw)]
    teach=[]
    with h5py.File(o.specteach_h5,'r') as f:
        for k in sorted(f.keys()):
            g=f[k]
            if not isinstance(g,h5py.Group) or 'smiles' not in g.attrs:continue
            s=g.attrs['smiles'];s=s.decode() if isinstance(s,bytes) else str(s)
            if not s:raise RuntimeError(f'SpecTeach {k}: empty smiles')
            teach.append(dict(source_id=k,smiles=s))
    if len(teach)!=238:raise RuntimeError(f'SpecTeach must be full 238 structures, got {len(teach)} (not 213-compatible subset)')
    o.output.mkdir(parents=True)
    for name,records in (('bmrb100_full',bmrb),('specteach_full',teach)):
        with (o.output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['source_id','smiles']);w.writeheader();w.writerows(records)
    m=dict(bmrb100_count=len(bmrb),specteach_count=len(teach),
           bmrb100_source=str(o.bmrb_manifest.resolve()),bmrb100_source_sha256=sha(o.bmrb_manifest),
           specteach_source=str(o.specteach_h5.resolve()),specteach_source_bytes=o.specteach_h5.stat().st_size,
           files={n:sha(o.output/n) for n in ('bmrb100_full.csv','specteach_full.csv')})
    (o.output/'reference_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print('FULL BENCHMARK REFERENCE EXPORT PASS: BMRB100 100; SpecTeach 238')
    print('Output:',o.output)
if __name__=='__main__':main()
