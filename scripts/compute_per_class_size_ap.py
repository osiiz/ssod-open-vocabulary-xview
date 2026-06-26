"""Calcula AP@50 (DINO) o P50/R50 (Rex) por categoría × tamaño de objeto.

Tamaños COCO:
  small  : area <  1024 px²
  medium : area in [1024, 9216) px²
  large  : area >= 9216 px²

Para DINO (mode=ap): AP@IoU=0.50 por (categoría, tamaño) usando COCOeval estándar.
Para Rex (mode=pr): P50 y R50 por (categoría, tamaño) extrayendo TP/FP/FN de evalImgs,
  igual que _pr_from_eval_imgs en ov_coco_eval.py.
  Añadir --relabel para modo class-agnostic (re-etiqueta dets con GT más cercano).

Uso:
    # DINO ensemble argmax (class-aware)
    python scripts/compute_per_class_size_ap.py \
        --detections results/dino/ensemble_argmax_aggregated/eval_class-aware/detection_results.json \
        --ann_file results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \
        --mode ap \
        --output docs/results_reports/per_class_per_size_dino_aware.json

    # Rex ensemble (class-aware)
    python scripts/compute_per_class_size_ap.py \
        --detections results/rexomni/ensemble/detection_results.json \
        --ann_file results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \
        --mode pr \
        --output docs/results_reports/per_class_per_size_rex_aware.json

    # Rex ensemble (class-agnostic)
    python scripts/compute_per_class_size_ap.py \
        --detections results/rexomni/ensemble/detection_results.json \
        --ann_file results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \
        --mode pr --relabel \
        --output docs/results_reports/per_class_per_size_rex_agnostic.json
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

AREA_RANGES = {
    "all": [0, 1e10],
    "small": [0, 1024],
    "medium": [1024, 9216],
    "large": [9216, 1e10],
}
IOU50_IDX = 0  # iouThrs[0] = 0.50 in COCOeval default


def _pr_from_eval_imgs(eval_imgs: list, iou_idx: int) -> tuple[float, float]:
    """TP/FP/FN de evalImgs para un umbral IoU → (precision, recall)."""
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


def run_coco_eval(
    coco_gt: COCO, coco_dt, cat_id: int, area_rng: list, max_dets: int
) -> COCOeval:
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.catIds = [cat_id]
    ev.params.areaRng = [AREA_RANGES["all"], area_rng]
    ev.params.areaRngLbl = ["all", "filtered"]
    ev.params.maxDets = [1, 10, max_dets]  # pycocotools requires at least 3 entries
    ev.evaluate()
    ev.accumulate()
    return ev


def ap50_from_eval(ev: COCOeval) -> float:
    """AP@IoU=0.50 para el segundo areaRng (índice 1)."""
    # precision shape: [T, R, K, A, M] donde A=2 (all, filtered)
    # T=10 iou thresholds, R=101 recall points, K=1 cat, A=2, M=1 det
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    if prec.size == 0:
        return -1.0
    # iou=0.50 → índice 0; area=filtered → índice 1; maxDets=last → índice -1
    p = prec[0, :, 0, 1, -1]
    p = p[p >= 0]
    return float(np.mean(p)) if len(p) > 0 else -1.0


def pr50_from_eval(ev: COCOeval) -> tuple[float, float]:
    """P50 y R50 para el segundo areaRng (índice 1) filtrando evalImgs."""
    area_rng = ev.params.areaRng[1]
    filtered = [e for e in ev.evalImgs if e is not None and e["aRng"] == area_rng]
    return _pr_from_eval_imgs(filtered, iou_idx=IOU50_IDX)


def compute_ap_per_class_size(coco_gt: COCO, detections: list, max_dets: int) -> dict:
    """AP@50 por categoría × tamaño."""
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    cat_ids = coco_gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    result = {}
    for cat_id in cat_ids:
        cat_name = cat_names[cat_id]
        result[cat_name] = {}
        for size_label, area_rng in AREA_RANGES.items():
            ev = run_coco_eval(coco_gt, coco_dt, cat_id, area_rng, max_dets)
            ap = ap50_from_eval(ev)
            result[cat_name][size_label] = round(ap, 4)
        print(f"  {cat_name}: {result[cat_name]}")
    return result


def compute_pr_per_class_size(coco_gt: COCO, detections: list, max_dets: int) -> dict:
    """P50 y R50 por categoría × tamaño."""
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    cat_ids = coco_gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    result = {}
    for cat_id in cat_ids:
        cat_name = cat_names[cat_id]
        result[cat_name] = {}
        for size_label, area_rng in AREA_RANGES.items():
            ev = run_coco_eval(coco_gt, coco_dt, cat_id, area_rng, max_dets)
            p, r = pr50_from_eval(ev)
            result[cat_name][size_label] = {"P50": round(p, 4), "R50": round(r, 4)}
        print(
            f"  {cat_name}: { {sz: result[cat_name][sz] for sz in ('small','medium','large')} }"
        )
    return result


def print_ap_table(result: dict) -> None:
    print(f"\n{'Clase':<22} {'all':>7} {'small':>7} {'medium':>8} {'large':>7}")
    print("-" * 55)
    for cat, sizes in result.items():
        row = f"{cat:<22}"
        for sz in ("all", "small", "medium", "large"):
            v = sizes.get(sz, -1)
            row += f"  {v:5.3f}" if v >= 0 else "      —"
        print(row)


def print_pr_table(result: dict) -> None:
    print(
        f"\n{'Clase':<22} {'P50_all':>8} {'R50_all':>8} {'P50_sm':>8} {'R50_sm':>8} {'P50_md':>8} {'R50_md':>8} {'P50_lg':>8} {'R50_lg':>8}"
    )
    print("-" * 90)
    for cat, sizes in result.items():
        row = f"{cat:<22}"
        for sz in ("all", "small", "medium", "large"):
            d = sizes.get(sz, {})
            p, r = d.get("P50", -1), d.get("R50", -1)
            row += f"  {p:5.3f}  {r:5.3f}"
        print(row)


def to_markdown_ap(result: dict) -> str:
    lines = [
        "| Clase | AP50_all | AP50_small | AP50_medium | AP50_large |",
        "|-------|----------|------------|-------------|------------|",
    ]
    for cat, sizes in result.items():
        vals = [
            f"{sizes.get(sz, -1):.3f}" for sz in ("all", "small", "medium", "large")
        ]
        lines.append(f"| {cat} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def to_markdown_pr(result: dict) -> str:
    lines = [
        "| Clase | P50_all | R50_all | P50_small | R50_small | P50_medium | R50_medium | P50_large | R50_large |",
        "|-------|---------|---------|-----------|-----------|------------|------------|-----------|-----------|",
    ]
    for cat, sizes in result.items():
        vals = []
        for sz in ("all", "small", "medium", "large"):
            d = sizes.get(sz, {})
            vals += [f"{d.get('P50',-1):.3f}", f"{d.get('R50',-1):.3f}"]
        lines.append(f"| {cat} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--ann_file", required=True)
    parser.add_argument(
        "--mode",
        choices=["ap", "pr"],
        required=True,
        help="ap: AP@50 (DINO); pr: P50/R50 (Rex sin scores)",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Re-etiqueta dets con GT más cercano (class-agnostic, mode=ap y mode=pr)",
    )
    parser.add_argument("--max_dets", type=int, default=1500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Cargando GT: {args.ann_file}")
    coco_gt = COCO(args.ann_file)

    print(f"Cargando detecciones: {args.detections}")
    detections = json.loads(Path(args.detections).read_text())
    print(f"  {len(detections):,} detecciones")

    if args.relabel:
        print("  Aplicando relabeling class-agnostic (IoU≥0.1)...")
        detections = _relabel_by_gt_matching(detections, coco_gt, iou_thresh=0.1)
        print(f"  {len(detections):,} detecciones después del relabeling")

    if args.mode == "ap":
        print("\nCalculando AP@50 por clase × tamaño...")
        result = compute_ap_per_class_size(coco_gt, detections, args.max_dets)
        print_ap_table(result)
        md = to_markdown_ap(result)
    else:
        print("\nCalculando P50/R50 por clase × tamaño...")
        result = compute_pr_per_class_size(coco_gt, detections, args.max_dets)
        print_pr_table(result)
        md = to_markdown_pr(result)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nJSON guardado en {out}")

    md_path = out.with_suffix(".md")
    md_path.write_text(md)
    print(f"Markdown guardado en {md_path}")


if __name__ == "__main__":
    main()
