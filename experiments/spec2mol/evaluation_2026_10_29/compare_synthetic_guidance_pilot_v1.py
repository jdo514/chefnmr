#!/usr/bin/env python3
"""Analyze paired synthetic ChefNMR-S guidance 0 versus 2 pilot.

Uses one record per molecule in each model x condition x guidance group.
The reported top-K is oracle inclusion, not confidence-based ranking.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

METRICS = ["top1_exact", "oracle_top5", "oracle_top10", "best_tanimoto", "valid_candidate_fraction"]
EXPECTED = {"reference", "augmented"}

def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def to_float(v):
    return float(v)

def write_rows(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def summarize_by_group(rows, keys, n_expected):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    output=[]
    for group, vals in sorted(groups.items()):
        ids={v["molecule_id"] for v in vals}
        if len(vals)!=n_expected or len(ids)!=n_expected:
            raise ValueError(f"Unequal molecule cohort in {group}: {len(vals)} rows, {len(ids)} distinct IDs")
        record=dict(zip(keys,group)); record["n_molecules"]=len(vals)
        record.update({metric:float(np.mean([to_float(v[metric]) for v in vals])) for metric in METRICS})
        output.append(record)
    return output

def paired_comparison(rows, contrast, keys, lhs, rhs, n_expected, rng):
    """Compare lhs - rhs, matched on molecule ID for each named group."""
    groups=defaultdict(lambda:defaultdict(dict))
    for r in rows:
        k=tuple(r[x] for x in keys)
        arm=r[contrast]
        if r["molecule_id"] in groups[k][arm]:
            raise ValueError(f"Duplicate molecule ID in {contrast}={arm} {k}")
        groups[k][arm][r["molecule_id"]]=r
    output=[]
    for group,arms in sorted(groups.items()):
        if set(arms)!={lhs,rhs}:
            raise ValueError(f"Expected {lhs}/{rhs}, got {sorted(arms)} for {group}")
        l,r=arms[lhs],arms[rhs]
        if set(l)!=set(r) or len(l)!=n_expected:
            raise ValueError(f"Molecule identity mismatch in {group}: {len(l)}, {len(r)}")
        for metric in METRICS:
            ids=sorted(l)
            diffs=np.array([to_float(l[mid][metric])-to_float(r[mid][metric]) for mid in ids])
            boots=diffs[rng.integers(0,len(diffs),size=(10000,len(diffs)))].mean(axis=1)
            lo,hi=np.quantile(boots,[0.025,0.975])
            record=dict(zip(keys,group))
            record.update(metric=metric,n_molecules=len(ids),difference=float(np.mean(diffs)),
                          ci95_low=float(lo),ci95_high=float(hi),contrast=f"{contrast}: {lhs} - {rhs}")
            output.append(record)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True,help="Job directory containing guidance0/ and guidance2/")
    p.add_argument("--expected-molecules",type=int,default=10)
    args=p.parse_args()
    root=args.run.resolve(); rng=np.random.default_rng(20261029)
    rows=[]
    for guidance in ("0","2"):
        sub=root/f"guidance{guidance}"
        data=read_rows(sub/"per_molecule.csv")
        candidate_data=read_rows(sub/"candidates.csv")
        expected_candidates=args.expected_molecules*2*2*10
        if len(candidate_data)!=expected_candidates:
            raise ValueError(f"{sub}: expected {expected_candidates} candidates, got {len(candidate_data)}")
        for r in data:
            if r["atom_order"]!="native" or r["condition"] not in {"experimental","zero"} or r["model"] not in EXPECTED:
                raise ValueError(f"Unexpected evaluation group: {r}")
            r["guidance"]=guidance
            rows.append(r)
    if len(rows)!=args.expected_molecules*2*2*2:
        raise ValueError(f"Expected {args.expected_molecules*8} per-molecule rows, got {len(rows)}")
    models={r["model"] for r in rows}
    modes={r["condition"] for r in rows}
    gids={r["guidance"] for r in rows}
    if models!=EXPECTED or modes!={"experimental","zero"} or gids!={"0","2"}:
        raise ValueError(f"Unexpected arms models={models} conditions={modes} guidance={gids}")
    manifest={r["molecule_id"] for r in rows}
    if len(manifest)!=args.expected_molecules:
        raise ValueError(f"Expected same {args.expected_molecules} molecular IDs, got {len(manifest)}")
    summary=summarize_by_group(rows,["guidance","model","condition"],args.expected_molecules)
    effect_guidance=paired_comparison(rows,"guidance",["model","condition"],"0","2",args.expected_molecules,rng)
    effect_condition=paired_comparison(rows,"condition",["guidance","model"],"experimental","zero",args.expected_molecules,rng)
    effect_model=paired_comparison(rows,"model",["guidance","condition"],"augmented","reference",args.expected_molecules,rng)
    write_rows(root/"cross_guidance_summary.csv",summary)
    write_rows(root/"guidance_effects.csv",effect_guidance)
    write_rows(root/"nmr_condition_effects.csv",effect_condition)
    write_rows(root/"augmentation_effects.csv",effect_model)
    manifest_sorted=sorted(manifest)
    (root/"analysis_manifest.json").write_text(json.dumps(dict(
        input_dir=str(root),models=sorted(models),guidance=[0,2],
        conditions=sorted(modes),molecule_ids=manifest_sorted,
        generated_candidate_count=2*expected_candidates,
        candidate_generation="50 diffusion steps, 10 candidates per molecule and condition",
        note="Pilot of 10 validation molecules; results not representative or independent test performance",
    ),indent=2)+"\n")
    print("=== SYNTHETIC GUIDANCE PILOT: COMPARATIVE SUMMARY ===")
    for r in summary:
        print(f"guidance={r['guidance']} {r['model']:<10} {r['condition']:<13} "
              f"top1={r['top1_exact']:.1%} oracle10={r['oracle_top10']:.1%} "
              f"valid={r['valid_candidate_fraction']:.1%} bestT={r['best_tanimoto']:.3f}")
    print("\n=== GUIDANCE EFFECT: guidance 0 minus guidance 2 (paired) ===")
    for r in effect_guidance:
        if r["metric"] in {"oracle_top10","valid_candidate_fraction","best_tanimoto"}:
            print(f"{r['model']:<10} {r['condition']:<13} {r['metric']:<24} "
                  f"delta={r['difference']:+.4f} CI95=[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]")
    print("\n=== FULL NMR EFFECT: full minus zero spectrum (paired) ===")
    for r in effect_condition:
        if r["metric"] in {"oracle_top10","valid_candidate_fraction","best_tanimoto"}:
            print(f"g={r['guidance']} {r['model']:<10} {r['metric']:<24} "
                  f"delta={r['difference']:+.4f} CI95=[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]")
    print("\n=== AUGMENTATION EFFECT: augmented minus reference (paired) ===")
    for r in effect_model:
        if r["condition"]=="experimental" and r["metric"] in {"oracle_top10","valid_candidate_fraction","best_tanimoto"}:
            print(f"g={r['guidance']} {r['metric']:<24} delta={r['difference']:+.4f} "
                  f"CI95=[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]")
    print("\nANALYSIS PASS; results saved under",root)

if __name__=="__main__": main()
