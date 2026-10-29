#!/usr/bin/env python3
"""Read-only targeted source/schema audit for frozen synthetic validation and SpecTeach.
Designed to emit a compact JSON report without traversing enormous production files.
"""
import argparse, json, os, csv, hashlib
from pathlib import Path

def summarize_h5(path):
    import h5py
    out={'path':str(path),'size_bytes':path.stat().st_size,'datasets':[],'groups':[]}
    with h5py.File(path,'r') as h:
        def visitor(name,obj):
            if len(out['datasets'])>=35 or len(out['groups'])>=15: return 1
            if isinstance(obj,h5py.Dataset) and len(out['datasets'])<35:
                out['datasets'].append({'path':name,'shape':list(obj.shape),'dtype':str(obj.dtype)})
            elif isinstance(obj,h5py.Group) and len(out['groups'])<15:
                out['groups'].append(name)
        h.visititems(visitor)
        out['root_keys']=list(__import__('itertools').islice(h.keys(), 30))
        if out['root_keys']:
            k=out['root_keys'][0]
            if isinstance(h[k],h5py.Group):
                out['first_record_keys']=list(h[k].keys())[:40]
                for sub in ('spectra','atom_features','molecule'):
                    if sub in h[k] and isinstance(h[k][sub],h5py.Group):
                        out['first_record_'+sub+'_keys']=list(h[k][sub].keys())[:30]
    return out

def preview(path):
    suffix=path.suffix.lower()
    rec={'path':str(path),'size_bytes':path.stat().st_size}
    try:
        if suffix in ('.h5','.hdf5'): rec.update(summarize_h5(path))
        elif suffix=='.npy':
            import numpy as np
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            rec.update(shape=list(a.shape),dtype=str(a.dtype))
        elif suffix=='.npz':
            import numpy as np
            with np.load(path,allow_pickle=False) as f:rec['arrays']=[{'key':k,'shape':list(f[k].shape),'dtype':str(f[k].dtype)} for k in f.files[:20]]
        elif suffix=='.csv':
            with path.open(errors='replace',newline='') as f:
                reader=csv.reader(f); rec['columns']=next(reader,[])[:60]; rec['sample']=next(reader,[])[:16]
        elif suffix in ('.json','.yaml','.yml','.toml','.txt','.md'):
            rec['first_lines']=path.read_text(errors='replace').splitlines()[:8]
    except Exception as e:rec['inspect_error']=repr(e)
    return rec

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-files',type=int,default=60)
    args=ap.parse_args()
    root=args.root.expanduser().resolve();report={'root':str(root),'targets':{}}
    targets={'specteach':['data/external','data/processed','selections','reports'],
      'frozen_split':['selections','data','configs'],
      'production_master':['data','archives','selections']}
    patterns={'specteach':('specteach','spec_teach','spec-teach'),
      'frozen_split':('strict_clean_conformers_v1','split','selection','val_ids'),
      'production_master':('chefnmr','master','molecules.h5')}
    extensions={'.h5','.hdf5','.npz','.npy','.csv','.json','.yaml','.yml','.sdf','.smi','.pkl','.txt','.md'}
    for name,bases in targets.items():
        found=[]
        for base in bases:
            d=root/base
            if not d.exists():continue
            for current,dirs,files in os.walk(d):
                depth=len(Path(current).relative_to(d).parts)
                dirs[:]=[x for x in dirs if not x.startswith('.') and x not in ('__pycache__','venv','.git')]
                if depth>=5:dirs[:]=[]
                for file in files:
                    p=Path(current)/file
                    if p.suffix.lower() not in extensions:continue
                    pathstr=str(p.relative_to(root)).lower()
                    if any(s in pathstr for s in patterns[name]):
                        found.append(p)
                    if len(found)>=args.max_files:break
                if len(found)>=args.max_files:break
            if len(found)>=args.max_files:break
        report['targets'][name]={'files':[preview(x) for x in found[:args.max_files]],'count_capped':len(found)>=args.max_files}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,default=str))
    print('AUDIT OUTPUT:',args.output)
    for k,v in report['targets'].items():
        print(k,':',len(v['files']),'candidate files')
        for f in v['files'][:12]:
            print(' ',f['path'].replace(str(root)+'/',''),f.get('shape',''),f.get('root_keys',''),f.get('columns',[])[:10])
    print('READ-ONLY AUDIT COMPLETE')
if __name__=='__main__':main()
