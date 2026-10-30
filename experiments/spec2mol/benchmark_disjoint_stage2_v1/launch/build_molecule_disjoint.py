#!/usr/bin/env python3
"""Create ChefNMR external train/val manifests with full BMRB100+SpecTeach identity exclusion.
- 100 BMRB100 and 238 official SpecTeach reference structures are mandatory.
- Scaffold overlap is ALLOWED; stereo-agnostic canonical molecular identity is disjoint.
- Identity groups never straddle train and validation.
- Reads HDF5 metadata/shapes in read-only mode; never changes production HDF5.
"""
import argparse, csv, hashlib, json, os, shutil, tempfile
from collections import Counter, defaultdict
from pathlib import Path
import h5py
import numpy as np
from rdkit import Chem, rdBase

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def identity(s):
    mol=Chem.MolFromSmiles(s)
    if mol is None:raise ValueError(f'Invalid SMILES: {s[:100]!r}')
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol,isomericSmiles=False)

def refs(path,expected):
    with path.open(newline='') as f: records=list(csv.DictReader(f))
    if len(records)!=expected:raise RuntimeError(f'{path}: expected FULL {expected} rows, got {len(records)}')
    if 'smiles' not in (records[0] if records else {}):raise RuntimeError(f'{path}: missing smiles column')
    keys=set()
    for j,row in enumerate(records):
        if not row['smiles'].strip():raise RuntimeError(f'{path}: blank SMILES at row {j}')
        keys.add(identity(row['smiles'].strip()))
    return keys,len(records)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master',type=Path,required=True,help='read-only ChefNMR data/molecules.h5')
    p.add_argument('--bmrb',type=Path,required=True,help='FULL 100-row reference CSV')
    p.add_argument('--specteach',type=Path,required=True,help='FULL 238-row reference CSV')
    p.add_argument('--output',type=Path,required=True,help='new selection dir; must not exist')
    p.add_argument('--previous-train',type=Path,required=True,help='original strict_clean_conformers_v1/train.npy; forces stage1-seen identities into TRAIN')
    p.add_argument('--expected-previous-train',type=int,default=95202,help='Only change for synthetic tests')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-fraction',type=float,default=0.90)
    p.add_argument('--expected-master',type=int,default=142277)
    a=p.parse_args()
    if not 0.5<=a.train_fraction<=0.99:raise ValueError('train fraction must be 0.5..0.99')
    if a.output.exists():raise FileExistsError(f'Refusing to overwrite {a.output}')
    if not a.master.is_file():raise FileNotFoundError(a.master)
    if not a.previous_train.is_file():raise FileNotFoundError(a.previous_train)
    old_train=np.load(a.previous_train,allow_pickle=False)
    if old_train.ndim!=1 or old_train.dtype.kind not in 'iu' or len(old_train)!=a.expected_previous_train or len(set(map(int,old_train)))!=len(old_train):
        raise RuntimeError('Original training selection size/identity mismatch')
    old_train_ids=set(int(x) for x in old_train)
    for ref in (a.bmrb,a.specteach):
        if not ref.is_file():raise FileNotFoundError(ref)
    b,n_b=refs(a.bmrb,100);s,n_s=refs(a.specteach,238)
    benchmark=b|s
    print(f'BENCHMARKS: BMRB100={n_b} rows ({len(b)} identities); SpecTeach={n_s} rows ({len(s)} identities); combined={len(benchmark)} identities',flush=True)
    excluded=Counter();groups=defaultdict(list); excluded_rows=[]
    with h5py.File(a.master,'r') as h:
        valid_h={int(x) for x in h['valid_indices_h'][()]}
        valid_c={int(x) for x in h['valid_indices_c'][()]}
        ids=sorted(valid_h & valid_c)
        if len(ids)!=a.expected_master:
            raise RuntimeError(f'Expected {a.expected_master} paired compatible indices, found {len(ids)}; inspect source before proceeding')
        for idx,mid in enumerate(ids):
            key=f'{mid:07d}'
            reason=None
            if key not in h:reason='missing_molecule_record'
            else:
                g=h[key]
                if 'smiles' not in g.attrs:reason='missing_smiles'
                elif 'spectra/h_10k' not in g or 'spectra/c_80' not in g:reason='missing_paired_spectra'
                elif g['spectra/h_10k'].shape!=(11,10000) or g['spectra/c_80'].shape!=(11,80):reason='incorrect_variant_shape'
                elif 'atom_features/atom_coords' not in g or 'atom_features/atom_one_hot' not in g or 'atom_features/atom_mask' not in g:reason='missing_atom_features'
                else:
                    cr=g['atom_features/atom_coords'].shape
                    oh=g['atom_features/atom_one_hot'].shape
                    mask=g['atom_features/atom_mask'].shape
                    if len(cr)!=3 or cr[0]<1 or cr[2]!=3:reason='zero_or_invalid_conformers'
                    elif len(oh)!=2 or oh[1]!=4 or oh[0]<1 or oh[0]>59 or mask!=(oh[0],):reason='incompatible_CHON_atom_features'
            if reason is None:
                raw=g.attrs['smiles'];raw=raw.decode() if isinstance(raw,bytes) else str(raw)
                try:canonical=identity(raw)
                except ValueError:reason='invalid_smiles'
                else:
                    if canonical in benchmark:reason='external_benchmark_identity'
                    else:groups[canonical].append(mid)
            if reason:
                excluded[reason]+=1
                excluded_rows.append((mid,reason))
            if idx and idx%20000==0:
                print(f'Master metadata audited: {idx:,}/{len(ids):,}',flush=True)
    if not groups:raise RuntimeError('No eligible molecules')
    # CONTINUATION CRITICAL: a checkpoint already trained on 95,202 molecules.
    # None of those identities (including equivalent SMILES) may enter new validation.
    known_ids={mid for v in groups.values() for mid in v}
    missing_old=old_train_ids-known_ids
    if missing_old:
        raise RuntimeError(f'{len(missing_old)} originally trained molecules absent from candidate pool; first={min(missing_old)}')
    old_train_keys={key for key, mids in groups.items() if any(mid in old_train_ids for mid in mids)}
    val_candidates=sorted(set(groups)-old_train_keys)
    rng=np.random.default_rng(a.seed)
    rng.shuffle(val_candidates)
    n_val_groups=max(1,round((1-a.train_fraction)*len(groups)))
    if len(val_candidates)<n_val_groups:
        raise RuntimeError(f'Only {len(val_candidates)} never-trained groups for {n_val_groups} validation groups')
    val_keys=set(val_candidates[:n_val_groups]);train_keys=set(groups)-val_keys
    if not old_train_keys.issubset(train_keys):raise RuntimeError('Stage1-trained identities leaked to validation')
    if not train_keys or not val_keys or train_keys&val_keys:raise RuntimeError('Identity split invalid')
    train=np.sort(np.asarray([mid for key in train_keys for mid in groups[key]],dtype=np.int32))
    val=np.sort(np.asarray([mid for key in val_keys for mid in groups[key]],dtype=np.int32))
    test=np.empty((0,),dtype=np.int32)
    assert not set(train).intersection(set(val))
    assert not (benchmark & train_keys) and not (benchmark & val_keys)
    if len(train)+len(val)+sum(excluded.values())!=len(ids):raise RuntimeError('Conservation mismatch')
    # The existing loader verifies this against metadata file if present.
    checksum_file=a.master.with_name('molecules.h5.sha256')
    master_sha=checksum_file.read_text().split()[0] if checksum_file.is_file() else None
    work=Path(tempfile.mkdtemp(prefix='._benchmark_disjoint_',dir=a.output.parent))
    created=False
    try:
        for name,arr in [('train',train),('val',val),('test',test)]:np.save(work/f'{name}.npy',arr,allow_pickle=False)
        with (work/'excluded.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['molecule_id','reason']);w.writerows(excluded_rows)
        manifest=dict(selection_name=a.output.name,definition='benchmark-molecule-disjoint / scaffold-overlap-allowed',
            train_molecules=int(len(train)),validation_molecules=int(len(val)),test_molecules=0,
            total_molecules=int(len(train)+len(val)),
            eligible_master_ids=len(ids),benchmarks={'BMRB100':{'rows':n_b,'unique_identities':len(b),'source_sha256':sha(a.bmrb)},
                'SpecTeach':{'rows':n_s,'unique_identities':len(s),'source_sha256':sha(a.specteach)}},
            combined_benchmark_identities=len(benchmark),excluded_reasons=dict(excluded),
            unique_train_structures=len(train_keys),unique_validation_structures=len(val_keys),
            previous_training_molecules=len(old_train_ids),previous_training_ids_sha256=sha(a.previous_train),
            previously_trained_validation_overlap=0,
            train_fraction_groups=a.train_fraction,seed=a.seed,master_sha256=master_sha,
            rdkit_version=rdBase.rdkitVersion,
            identity_rule='RDKit canonical isomericSmiles=False after Chem.RemoveStereochemistry; no scaffold exclusion',
            caution='Protonation/tautomer/salt equivalents not exhaustively resolved by canonical SMILES. Review benchmark overlaps before declaring full chemical identity disjointness.',
            split_file_sha256={name:sha(work/f'{name}.npy') for name in ('train','val','test')},
            audit={'benchmark_exact_structure_overlap_train':0,'benchmark_exact_structure_overlap_val':0,
                   'train_val_structure_identity_overlap':0,'previously_trained_validation_overlap':0},
            notes='Primary benchmark exclusion uses all 100 BMRB100 + 238 SpecTeach source reference molecules; not pilot subsets. HDF5 unchanged.')
        (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (work/'audit.json').write_text(json.dumps(dict(audit=manifest['audit'],excluded_reasons=dict(excluded),
            train=len(train),val=len(val),master=len(ids),benchmark_rows=338),indent=2)+'\n')
        work.rename(a.output);created=True
        print(json.dumps({'output':str(a.output),'train':len(train),'validation':len(val),
            'excluded_reasons':dict(excluded),'benchmark_overlap':0,
            'train_val_identity_overlap':0,'no_scaffold_filter':True},indent=2),flush=True)
        print('BENCHMARK-MOLECULE-DISJOINT SPLIT BUILD PASS',flush=True)
    finally:
        if not created:shutil.rmtree(work,ignore_errors=True)
if __name__=='__main__':main()
