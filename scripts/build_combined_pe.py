"""
Aplica políticas de filtrado de pseudoetiquetas para DINO y Rex, cada una con
sus propias políticas optimizadas según la métrica que mejor funciona en
cada detector.

DINO (mejor métrica: score):
    conservadora : score >= 0.25  AND  cls ∈ {Aircraft, Building, Maritime Vessel, Storage Tank}
    intermedia   : score >= 0.15  AND  n_cluster >= 3  AND
                   cls ∈ {Aircraft, Building, Maritime Vessel, Light Vehicle, Storage Tank}
    balanceada   : score >= 0.15  AND  n_cluster >= 3   (todas las clases)

Rex (mejor métrica: n_cluster / contrib_sets):
    conservadora : n_cluster >= 5  AND  cls ≠ Tower & Pylon
    intermedia   : n_cluster >= 3  AND  cls ≠ Tower & Pylon
    balanceada   : n_cluster >= 2

Opcional (simulación combinada, no es la política final):
    combined_sim : union de DINO conservadora + Rex conservadora (sin dedup),
                   para tener una referencia exploratoria del techo precision×cobertura.

Uso:
    python scripts/build_combined_pe.py \
        --dino results/dino/single_term_aggregated_uf_score_weighted/detection_results.json \
        --rex  results/rexomni/single_term_aggregated_borda_real/detection_results.json \
        --output_dir results/combined_pe/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ── Políticas DINO ────────────────────────────────────────────────────────────
DINO_CONS_CLASSES = {"Aircraft", "Building", "Maritime Vessel", "Storage Tank"}
DINO_INTER_CLASSES = DINO_CONS_CLASSES | {"Light Vehicle"}


def dino_conservative(dets):
    return [
        d for d in dets
        if d.get("class_name") in DINO_CONS_CLASSES
        and float(d.get("score", 0.0)) >= 0.25
    ]


def dino_intermediate(dets):
    return [
        d for d in dets
        if d.get("class_name") in DINO_INTER_CLASSES
        and float(d.get("score", 0.0)) >= 0.15
        and int(d.get("n_cluster", 1)) >= 3
    ]


def dino_balanced(dets):
    return [
        d for d in dets
        if float(d.get("score", 0.0)) >= 0.15
        and int(d.get("n_cluster", 1)) >= 3
    ]


# ── Políticas Rex (basadas en n_cluster / contrib_sets, métricas óptimas) ───
REX_EXCLUDE = {"Tower & Pylon"}


def rex_conservative(dets):
    """contrib_sets ≥ 4: al menos 4 de los 5 prompt sets de Rex coinciden en la caja.
    Excluye Tower & Pylon que ningún detector resuelve."""
    return [
        d for d in dets
        if d.get("class_name") not in REX_EXCLUDE
        and len(d.get("contributing_sets") or []) >= 4
    ]


def rex_intermediate(dets):
    """n_cluster ≥ 3: clusters de ≥3 detecciones físicas. Excluye Tower & Pylon."""
    return [
        d for d in dets
        if d.get("class_name") not in REX_EXCLUDE
        and int(d.get("n_cluster", 1)) >= 3
    ]


def rex_balanced(dets):
    """n_cluster ≥ 2: descarta solo singletons. Mantiene todas las clases."""
    return [d for d in dets if int(d.get("n_cluster", 1)) >= 2]


# ── Helper ────────────────────────────────────────────────────────────────────
def _per_class_counts(filtered):
    from collections import Counter
    return dict(Counter(d.get("class_name", "_unknown") for d in filtered))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--rex", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--include_sim", action="store_true",
                        help="Genera también la simulación combinada DINO_cons ∪ Rex_cons")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DINO {args.dino} ...")
    with args.dino.open() as fh:
        dino = json.load(fh)
    print(f"  N = {len(dino):,}")

    print(f"Loading Rex {args.rex} ...")
    with args.rex.open() as fh:
        rex = json.load(fh)
    print(f"  N = {len(rex):,}")

    outputs = {
        "dino_conservative": dino_conservative(dino),
        "dino_intermediate": dino_intermediate(dino),
        "dino_balanced":     dino_balanced(dino),
        "rex_conservative":  rex_conservative(rex),
        "rex_intermediate":  rex_intermediate(rex),
        "rex_balanced":      rex_balanced(rex),
    }

    if args.include_sim:
        outputs["sim_dino_cons_plus_rex_cons"] = (
            dino_conservative(dino) + rex_conservative(rex)
        )

    for name, dets in outputs.items():
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "detection_results.json"
        with out.open("w") as fh:
            json.dump(dets, fh)
        counts = _per_class_counts(dets)
        print(f"\n=== {name}: N={len(dets):,} ===")
        for cls, n in sorted(counts.items()):
            print(f"  {cls}: {n:,}")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
