#!/usr/bin/env python3
"""Generate independent 1- and/or 2-GPU HTCondor candidates (DOES NOT SUBMIT)."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,default=Path.home()/'spec2mol-chefnmr')
    p.add_argument('--campaign',default='extended_augmented_disjoint_v1')
    p.add_argument('--gpus',type=int,nargs='+',choices=(1,2),default=[1,2])
    p.add_argument('--hours',type=float,default=43)
    a=p.parse_args()
    if not 0<a.hours<=43:raise ValueError('Use 43h or less for 48h HTCondor jobs')
    base=a.base.expanduser().resolve();repo=base/'chefnmr'
    work=base/'launch/disjoint_training_v1'
    selection=base/'selections/benchmark_molecule_disjoint_v1'
    manifest_path=selection/'manifest.json'
    if not manifest_path.is_file():raise RuntimeError(f'Build and audit new dataset selection first: {manifest_path}')
    m=json.loads(manifest_path.read_text())
    if m['audit']!={'benchmark_exact_structure_overlap_train':0,
        'benchmark_exact_structure_overlap_val':0,'train_val_structure_identity_overlap':0,
        'previously_trained_validation_overlap':0}:
        raise RuntimeError('Leakage audit not PASS')
    if m['benchmarks']['BMRB100']['rows']!=100 or m['benchmarks']['SpecTeach']['rows']!=238:
        raise RuntimeError('Full benchmark coverage not verified')
    previous_train=base/'selections/strict_clean_conformers_v1/train.npy'
    if not previous_train.is_file() or sha(previous_train)!=m['previous_training_ids_sha256']:
        raise RuntimeError('Original 95,202 molecule train selection has changed')
    for k in ('train','val','test'):
        if sha(selection/(k+'.npy'))!=m['split_file_sha256'][k]:
            raise RuntimeError('Split checksum mismatch: '+k)
    if not (work/'train_disjoint.py').exists() or not (work/'run_disjoint.sh').exists():
        raise FileNotFoundError('Install all package files in launch/disjoint_training_v1')
    original=base/'logs/full_augmented_2222972.0/checkpoints/spec2mol-augmented-seed42/diffusion_loss/last.ckpt'
    if original.stat().st_size<100*1024**2:raise RuntimeError('Original 11-epoch checkpoint missing or undersized')
    current=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if dirty:raise RuntimeError('ChefNMR source tracked files dirty: need clean pinned checkout')
    # Ensure cross-campaign code version traceability, but do not modify Git checkout.
    root=base/'logs'/a.campaign
    if root.exists():raise FileExistsError(f'Campaign exists: {root}; refusing overwrite')
    root.mkdir(parents=True)
    (root/'condor').mkdir()
    metadata=dict(campaign=a.campaign,segments=1,target_epochs=10000,segment_hours=a.hours,
        original_checkpoint=str(original),chef_commit=current,base=str(base),repo=str(repo),
        dataset_name='benchmark_molecule_disjoint_v1',split_manifest_sha256=sha(manifest_path),
        train_script_sha256=sha(work/'train_disjoint.py'),launcher_script_sha256=sha(work/'run_disjoint.sh'),
        train_molecules=m['train_molecules'],validation_molecules=m['validation_molecules'],
        protocol='stage2_resume_original_strict_clean_11epoch_checkpoint_then_train_on_molecule_disjoint')
    (root/'campaign.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for g in sorted(set(a.gpus)):
        f=root/'condor'/f'candidate_{g}gpu.sub'
        lines=[ 'universe = vanilla', 'executable = /bin/bash',
            f'arguments = {work / "run_disjoint.sh"} {a.campaign} 1 {g}',
            f'initialdir = {base}',f'request_gpus = {g}',
            f'request_cpus = {4 if g==2 else 2}',
            f'request_memory = {12 if g==2 else 8}GB',
            'request_disk = 3GB','gpus_minimum_memory = 38000',
            '+Production = True','+MaxRuntime = 172800',
            'should_transfer_files = NO','transfer_executable = FALSE',
            f'log = {root / "condor" / f"candidate_{g}gpu.log"}',
            f'output = {root / "condor" / f"candidate_{g}gpu.out"}',
            f'error = {root / "condor" / f"candidate_{g}gpu.err"}',
            'queue 1','']
        f.write_text('\n'.join(lines))
        print('DRYRUN: condor_submit -dry-run /tmp/disjoint_'+str(g)+'g.ad '+str(f))
        print('SUBMIT: condor_submit '+str(f))
    print('DATASET CHECKS PASS; campaign created:',root)
    print('PINNED',current)
    print('TRAIN',m['train_molecules'],'VAL',m['validation_molecules'])
    print('Both candidates share ONE campaign directory and flock lock; first GPU allocation wins.')
    print('No job has been submitted by this script.')
if __name__=='__main__':main()
