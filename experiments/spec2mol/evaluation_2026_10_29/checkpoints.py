#!/usr/bin/env python3
"""Audit trusted ChefNMR Lightning checkpoints and optionally reload the full model.
Run --load on NMRbox; loading a checkpoint requires several GiB RAM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',action='append',required=True,help='Path to best reference / augmented .ckpt; repeat')
    ap.add_argument('--load',action='store_true',help='Actually instantiate ChefNMR model from trusted checkpoint; requires Torch and ChefNMR imports')
    ap.add_argument('--hash',action='store_true',help='Calculate SHA256 (large files; can take time)')
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    out=[]
    for path in map(Path,args.checkpoint):
        if not path.is_file():
            raise SystemExit(f'CHECKPOINT_MISSING: {path}')
        d={'path':str(path.resolve()),'bytes':path.stat().st_size,'zip_format':zipfile.is_zipfile(path)}
        if not d['zip_format']:
            raise SystemExit(f'CHECKPOINT_BAD_FORMAT: {path}')
        if args.hash:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            d['sha256']=h.hexdigest()
        if args.load:
            import torch
            from src.model.model import NMRTo3DStructureElucidation
            m=NMRTo3DStructureElucidation.load_from_checkpoint(str(path.resolve()),map_location='cpu',strict=True)
            d['reload']='PASS'
            d['checkpoint_epoch']=m.loaded_ckpt_epoch
            d['trainable_parameters']=sum(p.numel() for p in m.parameters() if p.requires_grad)
            d['use_ema']=bool(m.use_ema)
            d['ema_restored']=m.ema is not None
            d['conditioning']=str(m.dataset_args.input_generator)
            d['max_n_atoms']=int(m.dataset_args.max_n_atoms)
            d['atom_decoder']=list(m.dataset_args.atom_decoder)
            d['spectral_variant_mode']=str(m.dataset_args.spectral_variant_mode)
            if m.use_ema and m.ema is None:
                raise SystemExit('CHECKPOINT_EMA_MISSING: trained model requested EMA but no EMA weights restored')
            del m
        out.append(d)
        print('CHECKPOINT_AUDIT:',json.dumps(d,sort_keys=True),flush=True)
    if args.output:
        op=Path(args.output);op.parent.mkdir(parents=True,exist_ok=True)
        op.write_text(json.dumps(out,indent=2)+'\n')
    print('CHECKPOINT_AUDIT_PASS',flush=True)

if __name__=='__main__':main()
