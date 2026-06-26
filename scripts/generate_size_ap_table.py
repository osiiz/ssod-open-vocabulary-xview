"""Genera tabla comparativa de AP/PR por tamaño de objeto para los detectores OV.

Modelos incluidos:
  - DINO ensemble argmax (class-aware y class-agnostic) → métricas COCO AP
  - Rex ensemble (class-aware y class-agnostic) → métricas P/R (sin scores)
    con desglose por tamaño (P50_small/medium/large, R50_small/medium/large).

Nota sobre class-aware vs class-agnostic en Rex:
  - class-aware:    detecciones con categorías originales de Rex.
  - class-agnostic: detecciones re-etiquetadas con la categoría del GT más cercano
                    (IoU ≥ 0.1), igual que hace ov_coco_eval internamente.
  El detection_results.json guardado en ambas carpetas es idéntico (original Rex);
  la diferencia se recomputa aquí correctamente.

Uso:
    python scripts/generate_size_ap_table.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.ov_coco_eval import _relabel_by_gt_matching  # noqa: E402

MODELS_AP = [
    {
        "name": "DINO ensemble argmax 5 prompts (class-aware)",
        "path": "results/dino/ensemble_argmax_aggregated/eval_class-aware/metrics.json",
    },
    {
        "name": "DINO ensemble argmax 5 prompts (class-agnostic)",
        "path": "results/dino/ensemble_argmax_aggregated/eval_class-agnostic/metrics.json",
    },
    {
        "name": "DINO ensemble simple+original (class-aware)",
        "path": "results/dino/ensemble_argmax_so_aggregated/eval_class-aware/metrics.json",
    },
    {
        "name": "DINO ensemble simple+original (class-agnostic)",
        "path": "results/dino/ensemble_argmax_so_aggregated/eval_class-agnostic/metrics.json",
    },
    {
        "name": "DINO ensemble synonyms (class-aware)",
        "path": "results/dino/ensemble_argmax_synonyms_aggregated/eval_class-aware/metrics.json",
    },
    {
        "name": "DINO ensemble synonyms (class-agnostic)",
        "path": "results/dino/ensemble_argmax_synonyms_aggregated/eval_class-agnostic/metrics.json",
    },
    {
        "name": "DINO ensemble simple+original+synonyms (class-aware)",
        "path": "results/dino/ensemble_argmax_sos_aggregated/eval_class-aware/metrics.json",
    },
    {
        "name": "DINO ensemble simple+original+synonyms (class-agnostic)",
        "path": "results/dino/ensemble_argmax_sos_aggregated/eval_class-agnostic/metrics.json",
    },
]

# Para Rex usamos las detecciones originales (pre ov_coco_eval) para poder
# replicar el relabeling correctamente en el cálculo por tamaño.
# Cada entrada lleva su propio raw_dets_path.
MODELS_PR = [
    {
        "name": "Rex ensemble 5 prompts (class-aware)",
        "metrics_path": "results/rexomni/ensemble_class-aware/metrics.json",
        "raw_dets": "results/rexomni/ensemble/detection_results.json",
        "mode": "aware",
    },
    {
        "name": "Rex ensemble 5 prompts (class-agnostic)",
        "metrics_path": "results/rexomni/ensemble_class-agnostic/metrics.json",
        "raw_dets": "results/rexomni/ensemble/detection_results.json",
        "mode": "agnostic",
    },
    {
        "name": "Rex ensemble synonyms (class-aware)",
        "metrics_path": "results/rexomni/ensemble_synonyms_aggregated/eval_class-aware/metrics.json",
        "raw_dets": "results/rexomni/ensemble_synonyms_aggregated/detection_results.json",
        "mode": "aware",
    },
    {
        "name": "Rex ensemble synonyms (class-agnostic)",
        "metrics_path": "results/rexomni/ensemble_synonyms_aggregated/eval_class-agnostic/metrics.json",
        "raw_dets": "results/rexomni/ensemble_synonyms_raw/detection_results.json",
        "mode": "agnostic",
    },
    {
        "name": "Rex ensemble simple+original+synonyms (class-aware)",
        "metrics_path": "results/rexomni/ensemble_sos_aggregated/eval_class-aware/metrics.json",
        "raw_dets": "results/rexomni/ensemble_sos_aggregated/detection_results.json",
        "mode": "aware",
    },
    {
        "name": "Rex ensemble simple+original+synonyms (class-agnostic)",
        "metrics_path": "results/rexomni/ensemble_sos_aggregated/eval_class-agnostic/metrics.json",
        "raw_dets": "results/rexomni/ensemble_sos_raw/detection_results.json",
        "mode": "agnostic",
    },
    {
        "name": "Rex ensemble 5 prompts+synonyms (class-aware)",
        "metrics_path": "results/rexomni/ensemble_full_synonyms_aggregated/eval_class-aware/metrics.json",
        "raw_dets": "results/rexomni/ensemble_full_synonyms_aggregated/detection_results.json",
        "mode": "aware",
    },
    {
        "name": "Rex ensemble 5 prompts+synonyms (class-agnostic)",
        "metrics_path": "results/rexomni/ensemble_full_synonyms_aggregated/eval_class-agnostic/metrics.json",
        "raw_dets": "results/rexomni/ensemble_full_synonyms_aggregated/detection_results.json",
        "mode": "agnostic",
    },
]

ANN_FILE = "results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json"

AREA_RANGES = {
    "small": [0, 1024],
    "medium": [1024, 9216],
    "large": [9216, 1e10],
}
IOU50_IDX = 0


def load(path: str) -> dict | list | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(v) -> str:
    if v is None or v == -1.0:
        return "  —  "
    return f"{v:.4f}"


def _pr_from_eval_imgs(eval_imgs: list, iou_idx: int) -> tuple[float, float]:
    tp = fp = fn = 0
    for e in eval_imgs:
        if e is None:
            continue
        dtm = e["dtMatches"]
        dtig = e["dtIgnore"]
        gtm = e["gtMatches"]
        gtig = e["gtIgnore"]
        if dtm.shape[1] > 0:
            tp += int(np.sum((dtm[iou_idx] > 0) & ~dtig[iou_idx].astype(bool)))
            fp += int(np.sum((dtm[iou_idx] == 0) & ~dtig[iou_idx].astype(bool)))
        if gtm.shape[1] > 0:
            fn += int(np.sum((gtm[iou_idx] == 0) & ~gtig.astype(bool)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return p, r


def compute_pr_by_size(
    coco_gt: COCO, detections: list, mode: str, max_dets: int = 1500
) -> dict:
    """Calcula P50/R50 para small/medium/large.

    mode='aware':    usa category_ids originales de Rex.
    mode='agnostic': re-etiqueta detecciones con GT más cercano (mismo que ov_coco_eval).
    """
    if mode == "agnostic":
        dets_eval = _relabel_by_gt_matching(detections, coco_gt, iou_thresh=0.1)
    else:
        dets_eval = detections

    coco_dt = coco_gt.loadRes(dets_eval)
    result = {}
    for size_label, area_rng in AREA_RANGES.items():
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.areaRng = [[0, 1e10], area_rng]
        ev.params.areaRngLbl = ["all", size_label]
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate()
        ev.accumulate()
        filtered = [e for e in ev.evalImgs if e is not None and e["aRng"] == area_rng]
        p, r = _pr_from_eval_imgs(filtered, IOU50_IDX)
        result[f"P50_{size_label}"] = round(p, 4)
        result[f"R50_{size_label}"] = round(r, 4)
    return result


# Columnas en orden: AP50, pequeño→grande para P, pequeño→grande para R
AP_COLS = [
    "AP50",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_small",
    "AR_medium",
    "AR_large",
]
PR_GLOBAL_COLS = ["P50", "P50:95", "P75", "R50", "R50:95", "R75"]
PR_SIZE_COLS = [
    "P50_small",
    "P50_medium",
    "P50_large",
    "R50_small",
    "R50_medium",
    "R50_large",
]


def print_ap_table(rows: list[dict]) -> str:
    col_w = 10
    name_w = 42
    header = f"{'Modelo':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in AP_COLS)
    sep = "-" * len(header)
    lines = ["\n=== DINO ensemble — métricas COCO AP ===", header, sep]
    for row in rows:
        line = f"{row['name']:<{name_w}}"
        for c in AP_COLS:
            line += f"{fmt(row['metrics'].get(c)):>{col_w}}"
        lines.append(line)
    return "\n".join(lines)


def print_pr_table(rows: list[dict]) -> str:
    col_w = 10
    name_w = 30
    all_cols = PR_GLOBAL_COLS + PR_SIZE_COLS
    header = f"{'Modelo':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in all_cols)
    sep = "-" * len(header)
    lines = ["\n=== Rex ensemble — métricas P/R (sin scores) ===", header, sep]
    for row in rows:
        m = {**row["metrics"], **row.get("size_metrics", {})}
        line = f"{row['name']:<{name_w}}"
        for c in all_cols:
            line += f"{fmt(m.get(c)):>{col_w}}"
        lines.append(line)
    return "\n".join(lines)


def to_markdown_ap(rows: list[dict]) -> str:
    lines = ["## DINO ensemble — métricas COCO AP\n"]
    lines.append(
        "| Modelo | AP50 | AP_small | AP_medium | AP_large | AR_small | AR_medium | AR_large |"
    )
    lines.append(
        "|--------|------|----------|-----------|----------|----------|-----------|----------|"
    )
    for row in rows:
        m = row["metrics"]
        vals = [fmt(m.get(c)) for c in AP_COLS]
        lines.append(f"| {row['name']} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def to_markdown_pr(rows: list[dict]) -> str:
    lines = ["\n## Rex ensemble — métricas P/R (sin scores de confianza)\n"]
    lines.append(
        "| Modelo | P50 | P50:95 | P75 | R50 | R50:95 | R75 | P50_small | P50_medium | P50_large | R50_small | R50_medium | R50_large |"
    )
    lines.append(
        "|--------|-----|--------|-----|-----|--------|-----|-----------|------------|-----------|-----------|------------|-----------|"
    )
    for row in rows:
        m = {**row["metrics"], **row.get("size_metrics", {})}
        cols = PR_GLOBAL_COLS + PR_SIZE_COLS
        vals = [fmt(m.get(c)) for c in cols]
        lines.append(f"| {row['name']} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _size_cache_path(metrics_path: str) -> Path:
    """Cache de métricas por tamaño, co-ubicado junto a metrics.json."""
    return Path(metrics_path).parent / "size_metrics.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="docs/results_reports/ap_by_size_comparison"
    )
    parser.add_argument("--max_dets", type=int, default=1500)
    parser.add_argument(
        "--recompute", action="store_true",
        help="Ignorar caché y recalcular métricas por tamaño de Rex"
    )
    args = parser.parse_args()

    ap_rows = []
    for m in MODELS_AP:
        metrics = load(m["path"])
        if metrics is None:
            print(f"[SKIP] {m['name']}: {m['path']} no encontrado")
            continue
        ap_rows.append({"name": m["name"], "metrics": metrics})

    coco_gt = None  # carga diferida: solo si hay alguna entrada sin caché

    _rex_dets_cache: dict[str, list] = {}

    pr_rows = []
    for m in MODELS_PR:
        global_metrics = load(m["metrics_path"])
        if global_metrics is None:
            print(f"[SKIP] {m['name']}: {m['metrics_path']} no encontrado")
            continue

        cache_path = _size_cache_path(m["metrics_path"])
        if cache_path.exists() and not args.recompute:
            size_metrics = json.loads(cache_path.read_text())
            print(f"[cache] {m['name']}")
        else:
            if coco_gt is None:
                print(f"Cargando GT: {ANN_FILE}")
                coco_gt = COCO(ANN_FILE)
            raw_path = m["raw_dets"]
            if raw_path not in _rex_dets_cache:
                dets = load(raw_path)
                if dets is None:
                    print(f"[SKIP] {m['name']}: {raw_path} no encontrado")
                    continue
                _rex_dets_cache[raw_path] = dets
                print(f"Cargando detecciones Rex: {raw_path} ({len(dets):,} dets)")
            print(f"Calculando P/R por tamaño para {m['name']} (mode={m['mode']})...")
            size_metrics = compute_pr_by_size(
                coco_gt, _rex_dets_cache[raw_path], m["mode"], args.max_dets
            )
            cache_path.write_text(json.dumps(size_metrics, indent=2))
            print(f"  → {size_metrics} [guardado en {cache_path}]")

        pr_rows.append(
            {"name": m["name"], "metrics": global_metrics, "size_metrics": size_metrics}
        )

    print(print_ap_table(ap_rows))
    print(print_pr_table(pr_rows))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    md = to_markdown_ap(ap_rows) + "\n" + to_markdown_pr(pr_rows)
    (out.with_suffix(".md")).write_text(md)
    print(f"\nMarkdown guardado en {out.with_suffix('.md')}")

    combined = {
        "dino": [{"name": r["name"], "metrics": r["metrics"]} for r in ap_rows],
        "rex": [
            {
                "name": r["name"],
                "metrics": {**r["metrics"], **r.get("size_metrics", {})},
            }
            for r in pr_rows
        ],
    }
    (out.with_suffix(".json")).write_text(json.dumps(combined, indent=2))
    print(f"JSON guardado en {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
