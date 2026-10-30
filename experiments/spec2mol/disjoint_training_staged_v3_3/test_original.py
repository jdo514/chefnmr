#!/usr/bin/env python3
"""Synthetic-only, CPU-only integration tests for split builder and submission generator."""
import csv,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,h5py
THIS=Path(__file__).resolve().parent

def run(*args):
    p=subprocess.run([sys.executable,*map(str,args)],text=True,capture_output=True)
    if p.returncode:raise AssertionError(f'command failed {args}: {p.stdout}\n{p.stderr}')
    return p.stdout

def main():
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);m=d/'molecules.h5'
        all_smiles=['CC','CCC','C=C','O=O','C#N','CCO','OCC','N#N']
        with h5py.File(m,'w') as h:
            h.create_dataset('valid_indices_h',data=np.arange(8,dtype='i4'))
            h.create_dataset('valid_indices_c',data=np.arange(8,dtype='i4'))
            spl=h.create_group('split_indices')
            for name,idx in [('train',[0,1,2,3,4,5]),('val',[6,7]),('test',[])]:spl.create_dataset(name,data=np.array(idx,dtype='i4'))
            spl.attrs['sigma_data']=2.02;spl.attrs['max_n_atoms']=59
            for i,smi in enumerate(all_smiles):
                g=h.create_group(f'{i:07d}');g.attrs['smiles']=smi
                spec=g.create_group('spectra');spec.create_dataset('h_10k',shape=(11,10000),dtype='f4');spec.create_dataset('c_80',shape=(11,80),dtype='f4')
                atom=g.create_group('atom_features');atom.create_dataset('atom_coords',shape=(1,12,3),dtype='f4');atom.create_dataset('atom_one_hot',shape=(12,4),dtype='f4');atom.create_dataset('atom_mask',shape=(12,),dtype='f4')
        for name,n,smi in [('bmrb',100,'N#N'),('specteach',238,'O=O')]:
            with (d/(name+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['source_id','smiles']);w.writeheader();w.writerows(dict(source_id=i,smiles=smi) for i in range(n))
        old=np.asarray([0,1,5],dtype='i4')
        np.save(d/'old_train.npy',old)
        out=d/'sel'
        output=run(THIS/'build_molecule_disjoint.py','--master',m,'--bmrb',d/'bmrb.csv','--specteach',d/'specteach.csv','--output',out,'--expected-master',8,'--previous-train',d/'old_train.npy','--expected-previous-train',3)
        s=json.loads((out/'manifest.json').read_text())
        assert s['train_molecules']+s['validation_molecules']==6,s
        assert s['excluded_reasons']['external_benchmark_identity']==2,s
        assert not set(np.load(out/'train.npy'))&set(np.load(out/'val.npy'))
        # This deliberately has duplicate canonical identities ('CCO'/'OCC') which may not split.
        assert set([0,1,5,6]).issubset(set(np.load(out/'train.npy')))
        assert not set([0,1,5,6]).intersection(set(np.load(out/'val.npy')))
        for t in ('train','val','test'):
            assert len(np.load(out/(t+'.npy')))==s[t+'_molecules'] if t != 'val' else len(np.load(out/'val.npy'))==s['validation_molecules']
        p=subprocess.run([sys.executable,str(THIS/'build_molecule_disjoint.py'),'--master',str(m),'--bmrb',str(d/'bmrb.csv'),'--specteach',str(d/'specteach.csv'),'--output',str(out),'--expected-master','8','--previous-train',str(d/'old_train.npy'),'--expected-previous-train','3'],capture_output=True)
        assert p.returncode!=0
        print('CPU SYNTHETIC DATASET TESTS PASS')
if __name__=='__main__':main()
