"""
Aplica dos políticas de filtrado de pseudoetiquetas (PE) sobre las detecciones
del ensemble multi-prompt DINO y genera los JSONs filtrados listos para evaluar.

Políticas:

    conservadora :  score >= 0.25  AND  class_name in
                    {Aircraft, Building, Maritime Vessel, Storage Tank}
                    Las 5 clases restantes quedan excluidas (sin PE).

    balanceada   :  score >= 0.15  AND  n_cluster >= 3
                    Sin restricción de clase. Reduce singletons y cajas de
                    baja confianza.

Uso:
    python scripts/apply_pe_filter_policies.py \
        --detections results/dino/single_term_aggregated_uf_score_weighted/detection_results.json \
        --output_dir results/dino/pe_policies/
"""

import argparse
import json
from pathlib import Path

CONSERVATIVE_CLASSES = {"Aircraft", "Building", "Maritime Vessel", "Storage Tank"}
CONSERVATIVE_SCORE = 0.25

INTERMEDIATE_CLASSES = {
    "Aircraft", "Building", "Maritime Vessel", "Light Vehicle", "Storage Tank"
}
INTERMEDIATE_SCORE = 0.15
INTERMEDIATE_NCLUSTER = 3

BALANCED_SCORE = 0.15
BALANCED_NCLUSTER = 3


def apply_conservative(detections):
    return [
        d for d in detections
        if d.get("class_name") in CONSERVATIVE_CLASSES
        and float(d.get("score", 0.0)) >= CONSERVATIVE_SCORE
    ]


def apply_intermediate(detections):
    return [
        d for d in detections
        if d.get("class_name") in INTERMEDIATE_CLASSES
        and float(d.get("score", 0.0)) >= INTERMEDIATE_SCORE
        and int(d.get("n_cluster", 1)) >= INTERMEDIATE_NCLUSTER
    ]


def apply_balanced(detections):
    return [
        d for d in detections
        if float(d.get("score", 0.0)) >= BALANCED_SCORE
        and int(d.get("n_cluster", 1)) >= BALANCED_NCLUSTER
    ]


def _per_class_counts(filtered):
    from collections import Counter
    return Counter(d.get("class_name", "_unknown") for d in filtered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.detections} ...")
    with args.detections.open() as fh:
        detections = json.load(fh)
    print(f"  N = {len(detections):,}")

    print(f"\nApplying conservative (score>={CONSERVATIVE_SCORE}, cls in {sorted(CONSERVATIVE_CLASSES)}) ...")
    cons = apply_conservative(detections)
    print(f"  N filtered = {len(cons):,}")
    for cls, n in sorted(_per_class_counts(cons).items()):
        print(f"    {cls}: {n:,}")

    print(f"\nApplying intermediate (score>={INTERMEDIATE_SCORE}, n_cluster>={INTERMEDIATE_NCLUSTER}, cls in {sorted(INTERMEDIATE_CLASSES)}) ...")
    inter = apply_intermediate(detections)
    print(f"  N filtered = {len(inter):,}")
    for cls, n in sorted(_per_class_counts(inter).items()):
        print(f"    {cls}: {n:,}")

    print(f"\nApplying balanced (score>={BALANCED_SCORE}, n_cluster>={BALANCED_NCLUSTER}) ...")
    bal = apply_balanced(detections)
    print(f"  N filtered = {len(bal):,}")
    for cls, n in sorted(_per_class_counts(bal).items()):
        print(f"    {cls}: {n:,}")

    out_cons = args.output_dir / "conservative" / "detection_results.json"
    out_inter = args.output_dir / "intermediate" / "detection_results.json"
    out_bal = args.output_dir / "balanced" / "detection_results.json"
    for p in [out_cons, out_inter, out_bal]:
        p.parent.mkdir(parents=True, exist_ok=True)
    with out_cons.open("w") as fh:
        json.dump(cons, fh)
    with out_inter.open("w") as fh:
        json.dump(inter, fh)
    with out_bal.open("w") as fh:
        json.dump(bal, fh)
    print(f"\nSaved:\n  {out_cons}\n  {out_inter}\n  {out_bal}")


if __name__ == "__main__":
    main()
