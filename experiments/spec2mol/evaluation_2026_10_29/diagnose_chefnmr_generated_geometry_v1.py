#!/usr/bin/env python3
"""Read-only CPU geometry QC for the completed ChefNMR BMRB100 pilot.

Compares generated molecular coordinates to reference conformers from the
*frozen synthetic-validation* selection. Reference molecules are a geometry
sanity baseline, not molecule-matched BMRB100 experimental structures.

Inputs (default, relative to --base):
  logs/bmrb100_graph_eval_2223655.0/full/coordinates/*.npz
  logs/bmrb100_graph_eval_2223655.0/full/candidates.csv
  data/molecules.h5
  data/external_evaluation/synthetic_val100_graph_v1/manifest.csv

Outputs (default):
  reports/chefnmr_generated_geometry_v1/candidate_geometry.csv
  reports/chefnmr_generated_geometry_v1/reference_geometry.csv
  reports/chefnmr_generated_geometry_v1/group_summary.csv
  reports/chefnmr_generated_geometry_v1/summary.json

No checkpoints, GPU, bond changes, or output changes to original evaluation.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty results")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def geometry_stats(coords, mask, symbols):
    coords = np.asarray(coords, dtype=np.float64)
    valid = np.asarray(mask).astype(bool)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != len(valid):
        raise ValueError(f"Coordinate/mask shape mismatch: {coords.shape} vs {valid.shape}")
    if len(symbols) != len(valid):
        raise ValueError("Symbol/mask length mismatch")
    xyz = coords[valid]
    syms = np.asarray(symbols, dtype=str)[valid]
    n = len(xyz)
    if n < 2:
        return {"n_atoms": n, "finite": int(np.isfinite(xyz).all()), "invalid": 1}
    if not np.isfinite(xyz).all():
        return {"n_atoms": n, "finite": 0, "invalid": 1}
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    near = np.min(distances, axis=1)
    pairs = distances[np.triu_indices(n, k=1)]
    center = np.mean(xyz, axis=0)
    rg = float(np.sqrt(np.mean(np.sum((xyz - center)**2, axis=1))))
    heavy = syms != "H"
    hydrogen = syms == "H"
    # The H-to-heavy distance is a useful geometric sanity measure but not a
    # bond assignment; molecules with no heavy atoms use NaN.
    if np.any(heavy) and np.any(hydrogen):
        h_heavy = np.linalg.norm(xyz[hydrogen, None, :] - xyz[None, heavy, :], axis=2)
        h_heavy_nearest = np.min(h_heavy, axis=1)
        h_heavy_med = float(np.median(h_heavy_nearest))
        h_heavy_over2 = float(np.mean(h_heavy_nearest > 2.0))
    else:
        h_heavy_med = float('nan')
        h_heavy_over2 = float('nan')
    return {
        "n_atoms": int(n), "finite": 1, "invalid": 0,
        "nearest_min_A": float(np.min(near)),
        "nearest_q10_A": float(np.quantile(near, .10)),
        "nearest_median_A": float(np.median(near)),
        "nearest_q90_A": float(np.quantile(near, .90)),
        "nearest_max_A": float(np.max(near)),
        "fraction_atoms_nn_lt_0p55A": float(np.mean(near < 0.55)),
        "fraction_atoms_nn_gt_2p2A": float(np.mean(near > 2.2)),
        "has_collision_lt_0p55A": int(np.any(near < 0.55)),
        "has_isolated_nn_gt_2p2A": int(np.any(near > 2.2)),
        "n_pairs_lt_0p55A": int(np.sum(pairs < 0.55)),
        "radius_gyration_A": rg,
        "max_pair_distance_A": float(np.max(pairs)),
        "h_to_heavy_nn_median_A": h_heavy_med,
        "h_fraction_heavy_nn_gt_2A": h_heavy_over2,
    }


def decode_symbols(onehot, decoder):
    x = np.asarray(onehot)
    if x.ndim != 2 or x.shape[1] != len(decoder):
        raise ValueError(f"Bad atom encoding shape {x.shape} for decoder {decoder}")
    return np.asarray(decoder)[np.argmax(x, axis=1)]


def generated(base, run, csvpath):
    candidates = read_csv(csvpath)
    expected_fields = ["dataset", "molecule_id", "model", "condition", "atom_order", "candidate", "valid_full", "predicted_smiles", "predicted_largest_fragment"]
    missing_fields = [x for x in expected_fields if x not in candidates[0]]
    if missing_fields:
        raise RuntimeError(f"Missing candidate fields: {missing_fields}")
    candidate_map = {}
    for r in candidates:
        key = (r["dataset"],r["molecule_id"],r["model"],r["condition"],r["atom_order"],int(r["candidate"]))
        if key in candidate_map:
            raise RuntimeError(f"Duplicate candidate key {key}")
        candidate_map[key] = r
    coords_dir = run/"coordinates"
    files = sorted(coords_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No coordinate NPZ files in {coords_dir}")
    outputs = []
    matched = set()
    for nf,p in enumerate(files,1):
        # Reconstruct file names from CSV keys rather than parsing potentially
        # ambiguous molecule names from the filename.
        # Example: bmrb100_BMRB0002_augmented_c13_only_native.npz
        matching = [r for r in candidates if p.name == f"{r['dataset']}_{r['molecule_id']}_{r['model']}_{r['condition']}_{r['atom_order']}.npz"]
        if not matching:
            raise RuntimeError(f"No candidate rows matching coordinates file {p.name}")
        example = matching[0]
        with np.load(p, allow_pickle=False) as z:
            if set(z.files) != {"coords","atom_one_hot","mask"}:
                raise RuntimeError(f"Unexpected NPZ keys in {p}: {z.files}")
            xyz = z["coords"]
            oh = z["atom_one_hot"]
            mask = z["mask"]
        if xyz.ndim != 3 or xyz.shape[1:] != (len(mask), 3):
            raise RuntimeError(f"Bad coordinates shape {xyz.shape} in {p}")
        syms = decode_symbols(oh, ["C","H","O","N"])
        for ci in range(xyz.shape[0]):
            key = (example["dataset"],example["molecule_id"],example["model"],example["condition"],example["atom_order"],ci+1)
            row = candidate_map.get(key)
            if row is None:
                raise RuntimeError(f"Missing candidate CSV row for {key}")
            matched.add(key)
            complete = bool(row["predicted_smiles"].strip())
            fragment = bool(row["predicted_largest_fragment"].strip())
            outcome = "full_graph" if complete else "fragment_only" if fragment else "no_graph"
            metrics = geometry_stats(xyz[ci],mask,syms)
            outputs.append({
                "dataset":row["dataset"], "molecule_id":row["molecule_id"],
                "model":row["model"],"condition":row["condition"],
                "atom_order":row["atom_order"],"candidate":int(row["candidate"]),
                "reconstruction_outcome":outcome,
                "exact_graph_match":int(row["exact_graph_match"]),
                **metrics,
            })
        if nf<=3 or nf%80==0 or nf==len(files):
            print(f"Read {nf}/{len(files)} NPZ files, {len(outputs)} candidate geometries",flush=True)
    if matched != set(candidate_map):
        raise RuntimeError(f"CSV has {len(set(candidate_map)-matched)} candidates missing coordinate files")
    return outputs


def references(base, count):
    manifest = base/"data/external_evaluation/synthetic_val100_graph_v1/manifest.csv"
    h5path = base/"data/molecules.h5"
    if not manifest.is_file() or not h5path.is_file():
        raise FileNotFoundError(f"Missing synthetic reference manifest or HDF5: {manifest}, {h5path}")
    selected = read_csv(manifest)[:count]
    if len(selected)<count:
        raise ValueError(f"Only {len(selected)} synthetic reference molecules available, requested {count}")
    rows=[]
    with h5py.File(h5path, "r") as h:
        for record in selected:
            mid=record["dataset_id"]
            group=h[mid]["atom_features"]
            xyz=group["atom_coords"][0]
            mask=group["atom_mask"][()]
            onehot=group["atom_one_hot"][()]
            decoder=[x.decode() if isinstance(x,bytes) else str(x) for x in group["atom_decoder"][()]]
            syms=decode_symbols(onehot,decoder)
            rows.append({"molecule_id":mid, "source":"synthetic_reference_conformer_0",**geometry_stats(xyz,mask,syms)})
    return rows


def med(rows,k):
    vals=[float(r[k]) for r in rows if k in r and not (isinstance(r[k],float) and np.isnan(r[k]))]
    return float(np.median(vals)) if vals else None


def groupsummary(rows, reference):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["model"],r["condition"],r["atom_order"],r["reconstruction_outcome"])].append(r)
    out=[]
    for k,g in sorted(groups.items()):
        out.append({"model":k[0],"condition":k[1],"atom_order":k[2],"reconstruction_outcome":k[3],
                    "n_candidates":len(g),
                    "median_nn_A":med(g,"nearest_median_A"),
                    "median_h_to_heavy_A":med(g,"h_to_heavy_nn_median_A"),
                    "median_radius_gyration_A":med(g,"radius_gyration_A"),
                    "fraction_candidates_with_collision":med_fraction(g,"has_collision_lt_0p55A"),
                    "fraction_candidates_with_isolated_atom":med_fraction(g,"has_isolated_nn_gt_2p2A"),
                    "median_fraction_atoms_nn_gt_2p2A":med(g,"fraction_atoms_nn_gt_2p2A"),
                    })
    return out


def med_fraction(rows,k):
    valid=[float(r[k]) for r in rows if k in r]
    return float(np.mean(valid)) if valid else None


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base',type=Path,default=Path.home()/"spec2mol-chefnmr")
    p.add_argument('--run',type=Path,default=None)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--reference-count',type=int,default=20)
    a=p.parse_args()
    base=a.base.expanduser().resolve()
    run=(a.run if a.run is not None else base/"logs/bmrb100_graph_eval_2223655.0/full").expanduser().resolve()
    out=(a.output if a.output is not None else base/"reports/chefnmr_generated_geometry_v1").expanduser().resolve()
    if a.reference_count<1 or a.reference_count>100:
        p.error('--reference-count must be 1..100')
    print('=== GENERATED GEOMETRY ===',flush=True)
    gen=generated(base,run,run/"candidates.csv")
    print('\n=== REFERENCE GEOMETRY ===',flush=True)
    refs=references(base,a.reference_count)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"candidate_geometry.csv",gen)
    write_csv(out/"reference_geometry.csv",refs)
    groups=groupsummary(gen,refs)
    write_csv(out/"group_summary.csv",groups)
    def headline(g):
        return {"n":len(g),"median_nearest_A":med(g,"nearest_median_A"),
                "median_h_to_heavy_A":med(g,"h_to_heavy_nn_median_A"),
                "median_radius_gyration_A":med(g,"radius_gyration_A"),
                "fraction_candidate_collision":med_fraction(g,"has_collision_lt_0p55A"),
                "fraction_candidate_isolation":med_fraction(g,"has_isolated_nn_gt_2p2A"),
                "median_fraction_atoms_isolated":med(g,"fraction_atoms_nn_gt_2p2A")}
    report={"source_run":str(run),"reference_note":"Reference conformers are synthetic validation molecules, NOT matched experimental BMRB100 molecules.",
            "reference":headline(refs),"all_generated":headline(gen),
            "generated_by_reconstruction_outcome":{
                outcome:headline([r for r in gen if r["reconstruction_outcome"]==outcome])
                for outcome in ("full_graph","fragment_only","no_graph")},
            "primary_full_NMR_native":{
                model:headline([r for r in gen if r["model"]==model and r["condition"]=="experimental" and r["atom_order"]=="native"])
                for model in ("reference","augmented")},
            "saved_csvs":["candidate_geometry.csv","reference_geometry.csv","group_summary.csv"]}
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print('\n=== GEOMETRY DIAGNOSTIC SUMMARY ===')
    print(json.dumps(report,indent=2))
    print('\nGEOMETRY DIAGNOSTIC PASS: original data read-only; generated coordinates not modified')

if __name__=='__main__':
    main()
