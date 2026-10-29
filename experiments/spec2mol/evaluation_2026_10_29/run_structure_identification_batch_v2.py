#!/usr/bin/env python3
"""Structure-identification inference using frozen ChefNMR-S checkpoints (v2).

One GPU job can process multiple experimental datasets, both models, several
spectral ablations, and K stochastic candidates per molecule. This evaluator
intentionally NEVER passes reference coordinates/bonds to the diffusion model.
Reference structure is used only for atom composition, target identity and scoring.

Example:
  python run_structure_identification_batch.py --preflight \
    --dataset bmrb100=/path/to/bmrb100_chefnmr_pilot_v1
  python run_structure_identification_batch.py \
    --dataset bmrb100=/path/to/bmrb100_chefnmr_pilot_v1 \
    --output logs/structure_identification_v1 --candidates 10

Scientific limitation: atom ordering derived from a reference SDF may encode
some connectivity; use --atom-orders native,shuffled to probe this sensitivity.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np

SEED = 20261029
CKPT_PATTERNS = {
    'reference': 'logs/full_reference_*/checkpoints/*/diffusion_loss/epoch0010-loss88.19.ckpt',
    'augmented': 'logs/full_augmented_*/checkpoints/*/diffusion_loss/epoch0010-loss88.26.ckpt',
}
RAW_FIELDS = [
    'dataset','molecule_id','dataset_row','model','condition','atom_order',
    'candidate','seed','guidance','steps','target_smiles','target_canonical',
    'predicted_smiles','predicted_largest_fragment','valid_full','valid_fragment',
    'exact_graph_match','formula_match','tanimoto','n_atoms'
]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_args(values):
    result = []
    for entry in values:
        if '=' not in entry:
            raise ValueError('Use --dataset name=/absolute/path')
        name, path = entry.split('=', 1)
        if not name or any(c in name for c in '/\\, '):
            raise ValueError('Dataset name must be a simple identifier')
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        result.append((name, root))
    if len(result) != len(set(name for name, _ in result)):
        raise ValueError('Duplicate dataset names')
    return result


def verify_checksums(root):
    checksum_file = root / 'checksums.sha256'
    if not checksum_file.exists():
        raise RuntimeError(f'Missing checksums.sha256: {root}')
    checks = 0
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        checksum, rel = line.split(None, 1)
        path = (root / rel.strip()).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f'Bad/missing checked file: {path}')
        if file_sha256(path) != checksum:
            raise RuntimeError(f'Checksum mismatch: {path}')
        checks += 1
    if not checks:
        raise RuntimeError(f'Empty checksums: {root}')
    return checks


def load_datasets(dataset_args, max_n_atoms=59):
    from rdkit import Chem
    loaded = []
    for name, root in dataset_args:
        nchecked = verify_checksums(root)
        with (root / 'manifest.csv').open(newline='') as f:
            records = list(csv.DictReader(f))
        h = np.load(root / 'spectra/h1_10000.npy', allow_pickle=False)
        c = np.load(root / 'spectra/c13_80.npy', allow_pickle=False)
        cond = np.load(root / 'spectra/condition_10080.npy', allow_pickle=False)
        n = len(records)
        if not (h.shape == (n, 10000) and c.shape == (n, 80) and
                cond.shape == (n, 10080)):
            raise RuntimeError(f'{name}: spectral dimensions inconsistent')
        if not (h.dtype == c.dtype == cond.dtype == np.float32):
            raise RuntimeError(f'{name}: expected float32 arrays')
        if not (np.isfinite(h).all() and np.isfinite(c).all() and
                np.isfinite(cond).all()):
            raise RuntimeError(f'{name}: nonfinite spectrum')
        if not np.isin(c, [0, 1]).all() or not np.array_equal(cond, np.concatenate([h,c], axis=1)):
            raise RuntimeError(f'{name}: nonbinary C13 or combined condition mismatch')
        ids = set()
        for i, rec in enumerate(records):
            mid = rec['dataset_id']
            if mid in ids: raise RuntimeError(f'{name}: duplicate {mid}')
            ids.add(mid)
            if int(rec.get('pilot_position', i)) != i:
                raise RuntimeError(f'{name}: row position mismatch: {mid}')
            sdf_rel = rec.get('reference_structure', f'structures/{mid}.sdf')
            sdf_path = (root / sdf_rel).resolve()
            if not sdf_path.is_relative_to(root):
                raise RuntimeError('SDF path outside dataset root')
            mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
            reference = Chem.MolFromSmiles(rec['smiles'])
            if mol is None or reference is None:
                raise RuntimeError(f'{name}/{mid}: invalid reference structure')
            canonical = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)
            target_canon = Chem.MolToSmiles(reference, isomericSmiles=False)
            if canonical != target_canon:
                raise RuntimeError(f'{name}/{mid}: target SMILES/SDF mismatch')
            molh = Chem.AddHs(mol)
            graph_symbols = [atom.GetSymbol() for atom in molh.GetAtoms()]
            # Synthetic-validation records preserve exactly the atom-feature order
            # encoded in the original training HDF5. For experimental SDFs the
            # source-graph order remains the default. In neither case are bonds
            # or reference coordinates passed to the model.
            if rec.get('atom_symbols_training_order_json', '').strip():
                symbols = json.loads(rec['atom_symbols_training_order_json'])
                from collections import Counter
                if Counter(symbols) != Counter(graph_symbols):
                    raise RuntimeError(f'{name}/{mid}: frozen atom order changes formula')
            else:
                symbols = graph_symbols
            if not (0 < len(symbols) <= max_n_atoms) or not set(symbols).issubset({'C','H','N','O'}):
                raise RuntimeError(f'{name}/{mid}: unsupported formula / atom count')
            loaded.append(dict(dataset=name, id=mid, dataset_row=rec.get('dataset_row',''),
                               condition=cond[i].copy(), symbols=symbols,
                               target_smiles=rec['smiles'], canonical=target_canon,
                               warnings=';'.join(k for k in ['h1_signal_gt10ppm_review_flag']
                                                 if rec.get(k, '') == '1')))
        print(f'DATASET {name}: {n} rows, checksums={nchecked}, integrity=PASS', flush=True)
    return loaded


def checkpoint_paths(base):
    result = {}
    for model, pattern in CKPT_PATTERNS.items():
        hits = sorted(base.glob(pattern))
        if len(hits) != 1:
            raise RuntimeError(f'{model}: expected one checkpoint for {pattern}; found {hits}')
        result[model] = hits[0]
    return result


def atom_features(item, decoder, max_n_atoms, order, device):
    import torch
    syms = list(item['symbols'])
    if order == 'shuffled':
        # Control depends only on composition and deterministic molecule ID seed.
        seed = int(hashlib.sha256((item['dataset']+'|'+item['id']).encode()).hexdigest()[:8],16)
        rng = np.random.default_rng(seed)
        rng.shuffle(syms)
    if not set(syms).issubset(set(decoder)):
        raise RuntimeError(f"Unknown decoder type: {set(syms)-set(decoder)}")
    arr = np.zeros((max_n_atoms, len(decoder)), np.float32)
    mask = np.zeros((max_n_atoms,), np.float32)
    for i, atom in enumerate(syms):
        arr[i, decoder.index(atom)] = 1.
        mask[i] = 1.
    return (torch.from_numpy(arr)[None].to(device),
            torch.from_numpy(mask)[None].to(device))


def condition_tensor(condition, mode, device):
    import torch
    cond = np.asarray(condition, dtype=np.float32).copy()
    if mode == 'zero':
        cond[:] = 0
    elif mode == 'h1_only':
        cond[10000:] = 0
    elif mode == 'c13_only':
        cond[:10000] = 0
    elif mode != 'experimental':
        raise ValueError(mode)
    return torch.from_numpy(cond)[None].to(device)


def canon_smiles(smiles):
    from rdkit import Chem
    if not smiles: return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def score_candidate(target_smiles, prediction, fragment):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    target = Chem.MolFromSmiles(target_smiles)
    pred = Chem.MolFromSmiles(prediction) if prediction else None
    frag = Chem.MolFromSmiles(fragment) if fragment else None
    truth = canon_smiles(target_smiles)
    canon = canon_smiles(prediction)
    if pred is None:
        return int(False), int(frag is not None), int(False), int(False), 0.
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    tanimoto = DataStructs.TanimotoSimilarity(fpgen.GetFingerprint(target), fpgen.GetFingerprint(pred))
    formula = rdMolDescriptors.CalcMolFormula(pred) == rdMolDescriptors.CalcMolFormula(target)
    return int(True), int(frag is not None), int(canon == truth), int(formula), float(tanimoto)


def summarize(rows, out):
    grouped = defaultdict(list)
    for row in rows:
        key = (row['dataset'], row['model'], row['condition'], row['atom_order'], row['molecule_id'])
        grouped[key].append(row)
    per_molecule = []
    for key, vals in sorted(grouped.items()):
        vals.sort(key=lambda x: int(x['candidate']))
        exact = np.asarray([int(x['exact_graph_match']) for x in vals], dtype=int)
        sims = np.asarray([float(x['tanimoto']) for x in vals], dtype=float)
        entry = dict(zip(['dataset','model','condition','atom_order','molecule_id'], key))
        entry.update(n_candidates=len(vals), top1_exact=int(exact[0]),
                     oracle_top5=int(exact[:5].any()), oracle_top10=int(exact[:10].any()),
                     oracle_topk=int(exact.any()), best_tanimoto=float(sims.max()),
                     first_tanimoto=float(sims[0]), valid_candidate_fraction=float(np.mean([int(x['valid_full']) for x in vals])))
        per_molecule.append(entry)
    if not per_molecule:
        return
    with (out/'per_molecule.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(per_molecule[0])); w.writeheader();w.writerows(per_molecule)
    groups=defaultdict(list)
    for row in per_molecule:
        groups[(row['dataset'],row['model'],row['condition'],row['atom_order'])].append(row)
    agg=[]
    for k,vals in sorted(groups.items()):
        item=dict(zip(['dataset','model','condition','atom_order'],k))
        item.update(n_molecules=len(vals))
        for attr in ['top1_exact','oracle_top5','oracle_top10','oracle_topk','best_tanimoto','first_tanimoto','valid_candidate_fraction']:
            item[attr]=float(np.mean([v[attr] for v in vals]))
        agg.append(item)
        print(f"{item['dataset']} {item['model']} {item['condition']} {item['atom_order']}: "
              f"n={item['n_molecules']} top1={item['top1_exact']:.1%} oracle@K={item['oracle_topk']:.1%}",flush=True)
    with (out/'summary.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
    # Molecule-paired reference vs augmented, bootstrap over molecules.
    pairs=defaultdict(dict)
    for r in per_molecule:
        pairs[(r['dataset'],r['condition'],r['atom_order'],r['molecule_id'])][r['model']] = r
    effect_groups=defaultdict(list)
    for (dataset,condition,atom_order,mid), d in pairs.items():
        if set(d)=={'reference','augmented'}:
            effect_groups[(dataset,condition,atom_order)].append((mid,d))
    effects=[]
    rng=np.random.default_rng(SEED+11)
    for group, vals in sorted(effect_groups.items()):
        for metric in ['top1_exact','oracle_top5','oracle_top10','oracle_topk','best_tanimoto']:
            differences=np.array([d['augmented'][metric]-d['reference'][metric] for mid,d in vals],float)
            n=len(differences)
            boots=differences[rng.integers(0,n,(10000,n))].mean(axis=1) if n else np.array([])
            lo,hi=np.quantile(boots,[0.025,0.975]) if n else (float('nan'),float('nan'))
            effects.append(dict(dataset=group[0],condition=group[1],atom_order=group[2],
                                metric=metric,n_molecules=n,aug_minus_ref=float(differences.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    if effects:
        with (out/'paired_effects.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path.home()/'spec2mol-chefnmr')
    p.add_argument('--dataset',action='append',required=True,help='NAME=/absolute/path/to/prepared_dataset')
    p.add_argument('--output',type=Path,default=Path('logs/structure_identification_v1'))
    p.add_argument('--candidates',type=int,default=10)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--modes',default='experimental,zero,h1_only,c13_only',help='experimental means full paired spectrum, also for synthetic validation')
    p.add_argument('--atom-orders',default='native,shuffled')
    p.add_argument('--guidance',type=float,default=None,help='Override checkpoint guidance; default preserves each checkpoint')
    p.add_argument('--limit',type=int,default=0,help='Restrict to first N molecules per dataset (smoke)')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--save-coordinates',action='store_true')
    args=p.parse_args()
    if args.candidates<=0 or args.steps<2 or args.limit<0: p.error('Bad sampling parameters')
    modes=args.modes.split(',')
    orders=args.atom_orders.split(',')
    if not set(modes).issubset({'experimental','zero','h1_only','c13_only'}) or not modes:
        p.error('Invalid modes')
    if not set(orders).issubset({'native','shuffled'}) or not orders:
        p.error('Invalid atom-orders')
    base=args.base.expanduser().resolve()
    datasets=resolve_dataset_args(args.dataset)
    items=load_datasets(datasets)
    if args.limit: items=[i for i in items if sum(x['dataset']==i['dataset'] for x in items[:items.index(i)]) < args.limit]
    checkpoints=checkpoint_paths(base)
    print('CHECKPOINTS:', {k:str(v) for k,v in checkpoints.items()},flush=True)
    print('Molecules:',len(items),'modes:',modes,'orders:',orders,'candidates:',args.candidates,'steps:',args.steps,flush=True)
    if args.preflight:
        print('STRUCTURE IDENTIFICATION PREFLIGHT PASS')
        return
    import torch
    from src.model.model import NMRTo3DStructureElucidation
    from src.evaluation.bond_analyzer import atom_features_to_smiles_one
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('Full inference requires GPU (preflight does not)')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    out=args.output.resolve()
    existing=out/'candidates.csv'
    if existing.exists(): raise FileExistsError(f'Refusing to overwrite {existing}; specify new output directory')
    (out/'run_config.json').write_text(json.dumps(dict(
        seed=SEED,datasets={n:str(root) for n,root in datasets},
        checkpoints={name:dict(path=str(path),sha256=file_sha256(path)) for name,path in checkpoints.items()},
        modes=modes,atom_orders=orders,candidates=args.candidates,steps=args.steps,
        guidance_override=args.guidance,limit=args.limit,
        data_used_for_model_inputs=['atom types from frozen HDF5 order if present, otherwise SDF formula','full paired spectrum or specified ablation','mask'],
        forbidden_model_inputs=['SDF 2D coordinates','reference bonds','reference SMILES','HDF5 conformer coordinates'],
        study_limitations=['Reference 2D SDF atom order may encode structural information',
                           'BMRB100 used previously in augmentation calibration',
                           'Top-K is oracle inclusion, not a validated ranking'],
    ),indent=2)+'\n')
    if args.save_coordinates: (out/'coordinates').mkdir()
    rows=[]
    t0=time.monotonic()
    with existing.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=RAW_FIELDS);writer.writeheader()
        for modelname,checkpoint in checkpoints.items():
            print('LOAD',modelname,checkpoint,flush=True)
            net=NMRTo3DStructureElucidation.load_from_checkpoint(str(checkpoint),map_location='cpu',strict=True)
            net.eval()
            if bool(net.use_ema):
                if net.ema is None: raise RuntimeError(f'{modelname}: EMA missing, abort')
                net.prepare_eval()  # copy EMA shadow weights while on CPU
            net=net.to(device).eval()
            decoder=list(net.dataset_args.atom_decoder)
            max_atoms=int(net.dataset_args.max_n_atoms)
            if bool(net.dataset_args.remove_h): raise RuntimeError('Model remove_h=True not supported')
            if max_atoms!=59: raise RuntimeError(f'Unexpected max_n_atoms {max_atoms}')
            if args.guidance is not None: net.model.guidance_scale=args.guidance
            guidance=float(net.model.guidance_scale)
            print('DECODER',modelname,decoder,'guidance',guidance,flush=True)
            for item in items:
                for order in orders:
                    atoms, mask=atom_features(item,decoder,max_atoms,order,device)
                    for mode in modes:
                        cond=condition_tensor(item['condition'],mode,device)
                        # Same candidate noise draws for each model and spectral control.
                        # Different molecular IDs and atom orders are independent seeds.
                        uid=item['dataset']+'|'+item['id']+'|'+order
                        seed=(SEED+int(hashlib.sha256(uid.encode()).hexdigest()[:8],16))%(2**31-1)
                        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                        with torch.inference_mode():
                            coords,_=net.model.sample(
                                model_inputs={'atom_one_hot':atoms,'atom_mask':mask,'condition':cond},
                                num_sampling_steps=args.steps,multiplicity=args.candidates,n_chain_frames=1)
                        pred_coords=coords.detach().cpu().numpy().astype(np.float32)
                        atom_np=atoms[0].detach().cpu().numpy()
                        mask_np=mask[0].detach().cpu().numpy()
                        if args.save_coordinates:
                            fn=f"{item['dataset']}_{item['id']}_{modelname}_{mode}_{order}.npz"
                            np.savez_compressed(out/'coordinates'/fn,coords=pred_coords,atom_one_hot=atom_np,mask=mask_np)
                        for j in range(args.candidates):
                            try:
                                _,smiles,frag=atom_features_to_smiles_one((
                                    j,pred_coords[j],atom_np,mask_np,str(net.dataset_args.name),
                                    False,True,decoder,4))
                            except Exception as exc:
                                print('WARN reconstruction',item['id'],j,type(exc).__name__,str(exc)[:150],flush=True)
                                smiles=frag=None
                            validity,validfrag,exact,formula_match,tani=score_candidate(item['target_smiles'],smiles,frag)
                            rec=dict(dataset=item['dataset'],molecule_id=item['id'],dataset_row=item['dataset_row'],
                                model=modelname,condition=mode,atom_order=order,candidate=j+1,seed=seed,
                                guidance=guidance,steps=args.steps,target_smiles=item['target_smiles'],
                                target_canonical=item['canonical'],predicted_smiles=smiles or '',
                                predicted_largest_fragment=frag or '',valid_full=validity,
                                valid_fragment=validfrag,exact_graph_match=exact,formula_match=formula_match,
                                tanimoto=tani,n_atoms=len(item['symbols']))
                            writer.writerow(rec);rows.append(rec)
                        f.flush()
                        print(f"{modelname} {item['dataset']}/{item['id']} {order} {mode}: {args.candidates} candidates",flush=True)
            del net
            torch.cuda.empty_cache()
    summarize(rows,out)
    print('RUN COMPLETE',f'elapsed_s={time.monotonic()-t0:.0f}',flush=True)

if __name__=='__main__':
    main()
