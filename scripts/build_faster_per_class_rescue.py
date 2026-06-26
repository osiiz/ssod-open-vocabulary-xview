"""
Constrúe unha política de pseudo-etiquetado per-clase sobre Faster10:
  - Para cada clase, manter as deteccións con score >= tau_global.
  - Se iso deixa menos de min_n deteccións nesa clase, baixar o threshold só
    o necesario para alcanzar min_n (ou conservar todas as deteccións dispoñibles
    se hai menos de min_n en total).

O criterio é SSOD-lexítimo: depende exclusivamente dos scores raw do detector
base e do número de instancias da clase no train_sampled (visible para o
escenario SSOD), nunca do GT do conxunto non etiquetado.

Uso:
    python scripts/build_faster_per_class_rescue.py \\
        --raw          ./results/ssod/faster10_unlabeled_raw/detection_results.json \\
        --sampled_ann  ./results/preprocess/tile_images/train_sampled/COCO_annotations.json \\
        --tau          0.7 \\
        --min_n        300 \\
        --output       ./results/pe_policies/faster_rescue300/detection_results.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True,
                    help="JSON con deteccións Faster10 raw (formato COCO results).")
    ap.add_argument("--sampled_ann", type=Path, required=True,
                    help="COCO annotations do train_sampled (para nomes de categoría).")
    ap.add_argument("--tau", type=float, default=0.7,
                    help="Umbral global por defecto (default 0.7).")
    ap.add_argument("--min_n", type=int, default=300,
                    help="Mínimo de deteccións por clase a conservar.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--exclusion_low_thresh", type=float, default=0.05,
                    help="Score mínimo das deteccións descartadas que se emiten "
                         "como zonas de exclusión (default 0.05). Use 0 para "
                         "incluír todas as descartadas; use >= max(tau_eff) "
                         "para non emitir zonas.")
    ap.add_argument("--exclusion_output", type=Path, default=None,
                    help="Path de saída para as zonas de exclusión (JSON list "
                         "estilo COCO results). Se non se indica, "
                         "<output_dir>/exclusion_zones.json.")
    args = ap.parse_args()

    with args.raw.open() as fh:
        raw = json.load(fh)
    with args.sampled_ann.open() as fh:
        coco = json.load(fh)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Agrupar deteccións por clase e ordenar por score descendente
    by_class: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for i, d in enumerate(raw):
        by_class[d["category_id"]].append((d["score"], i))
    for cid in by_class:
        by_class[cid].sort(key=lambda x: -x[0])

    keep_idx: list[int] = []
    per_class_stats: dict[str, dict] = {}
    for cid, scored in by_class.items():
        n_high = sum(1 for s, _ in scored if s >= args.tau)
        if n_high >= args.min_n:
            n_keep = n_high
            tau_eff = args.tau
        elif len(scored) >= args.min_n:
            n_keep = args.min_n
            tau_eff = scored[args.min_n - 1][0]
        else:
            n_keep = len(scored)
            tau_eff = scored[-1][0] if scored else 1.0
        for _, idx in scored[:n_keep]:
            keep_idx.append(idx)
        per_class_stats[cat_names.get(cid, str(cid))] = {
            "n_high_at_tau_global": n_high,
            "n_kept": n_keep,
            "tau_effective": round(float(tau_eff), 4),
        }

    kept_dets = [raw[i] for i in keep_idx]

    # Zonas de exclusión: deteccións descartadas (score >= exclusion_low_thresh)
    # que non quedaron na política. Excluímos do rescate por tau_effective per-clase:
    # unha detección rexeitada é a que (a) non está nas keep_idx e (b) cumpre
    # score >= exclusion_low_thresh. Tower & Pylon (sin política) contribúe coas
    # súas raw que cumpran o limiar inferior.
    keep_idx_set = set(keep_idx)
    exclusion_dets = [
        raw[i]
        for i, d in enumerate(raw)
        if i not in keep_idx_set and d["score"] >= args.exclusion_low_thresh
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(kept_dets, fh)

    exclusion_path = args.exclusion_output or (args.output.parent / "exclusion_zones.json")
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusion_path.open("w") as fh:
        json.dump(exclusion_dets, fh)

    stats_path = args.output.parent / "policy_stats.json"
    with stats_path.open("w") as fh:
        json.dump({
            "tau_global": args.tau,
            "min_n": args.min_n,
            "exclusion_low_thresh": args.exclusion_low_thresh,
            "total_raw": len(raw),
            "total_kept": len(kept_dets),
            "total_exclusion_zones": len(exclusion_dets),
            "per_class": per_class_stats,
        }, fh, indent=2)
    print(f"Política gardada en {args.output} ({len(kept_dets)} de {len(raw)} deteccións).")
    print(f"Zonas de exclusión en {exclusion_path} ({len(exclusion_dets)} deteccións con "
          f"score >= {args.exclusion_low_thresh} rexeitadas pola política).")
    print(f"Stats en {stats_path}")


if __name__ == "__main__":
    main()
