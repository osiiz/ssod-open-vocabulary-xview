"""
Constrúe o conxunto de zonas de exclusión Rex-Omni a partir das deteccións
do ensemble final (single-term + Borda) e das deteccións xa filtradas pola
política operativa que estea en uso (rex_conservative, rex_intermediate ou
rex_balanced).

Para Rex non se usa un limiar inferior sobre o score (xa que o score
autoregresivo non é discriminante); as zonas son todas as deteccións do
ensemble que non foron seleccionadas pola política. Cada detección rexeitada
emítese como caixa nun JSON estilo COCO results, listo para ser inxerido
como anotación con `category_id = -1` polo build_pe_dataset.

O parámetro opcional ``--score_floor`` permite reutilizar o script con outras
fontes de vocabulario aberto nas que o score si é discriminante (p.ex.
Grounding DINO): nese caso a zona é toda detección rexeitada pola política
\\emph{e} con score por riba do piso, de modo que a cola de baixísima confianza
non se converte en zona (o que cubriría case toda a imaxe).

Uso (Rex, sen piso de score):
    python scripts/build_rex_exclusion_zones.py \\
        --raw_ensemble  ./results/rexomni/single_term_aggregated_borda_real/detection_results.json \\
        --policy        ./results/pe_policies/rex_intermediate/detection_results.json \\
        --output        ./results/pe_policies/rex_intermediate/exclusion_zones.json

Uso (DINO, piso de score 0.12):
    python scripts/build_rex_exclusion_zones.py \\
        --raw_ensemble  ./results/dino/single_term_aggregated_borda_real/detection_results.json \\
        --policy        ./results/pe_policies/dino_intermediate/detection_results.json \\
        --score_floor   0.12 \\
        --output        ./results/pe_policies/dino_intermediate/exclusion_zones.json
"""
import argparse
import json
from pathlib import Path


def _det_key(det: dict) -> tuple:
    """Identificador único dunha detección: (image_id, category_id, bbox tupla)."""
    return (det["image_id"], det["category_id"], tuple(det["bbox"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_ensemble", type=Path, required=True,
                    help="JSON do ensemble Rex completo (single-term + Borda).")
    ap.add_argument("--policy", type=Path, required=True,
                    help="JSON da política operativa Rex (subconxunto das anteriores).")
    ap.add_argument("--output", type=Path, required=True,
                    help="JSON de saída coas deteccións rexeitadas (zonas de exclusión).")
    ap.add_argument("--score_floor", type=float, default=None,
                    help="Piso de score: só se converten en zona as deteccións "
                         "rexeitadas con score >= este valor. Por defecto (None) "
                         "non se aplica (comportamento Rex).")
    args = ap.parse_args()

    with args.raw_ensemble.open() as fh:
        raw = json.load(fh)
    with args.policy.open() as fh:
        policy = json.load(fh)

    policy_keys = {_det_key(d) for d in policy}
    exclusion = [d for d in raw if _det_key(d) not in policy_keys]
    if args.score_floor is not None:
        exclusion = [d for d in exclusion if d.get("score", 0.0) >= args.score_floor]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(exclusion, fh)

    print(f"Raw ensemble: {len(raw):,} deteccións.")
    print(f"Política: {len(policy):,} deteccións.")
    print(f"Zonas de exclusión: {len(exclusion):,} deteccións ({100.0 * len(exclusion) / max(len(raw), 1):.1f}% do ensemble).")
    print(f"Gardado en {args.output}.")


if __name__ == "__main__":
    main()
