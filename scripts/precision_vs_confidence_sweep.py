"""
Calcula curvas de PRECISIÓN (no AP) en función del umbral de varias métricas
de confianza, por categoría, sobre las detecciones de un ensemble.

Métricas de confianza incluidas (por defecto las 7; se pueden seleccionar con --metrics):
    score             : media de scores de los miembros del cluster que votaron
                        a la clase ganadora (no media de TODO el cluster)     (↑ mejor)
    n_cluster         : nº de miembros del cluster que votaron a la clase
                        ganadora (no nº total de detecciones físicas)         (↑ mejor)
    contrib_sets      : nº de sets distintos que contribuyeron una detección
                        de la clase ganadora ∈ [1,5]                          (↑ mejor)
    class_uncertainty : entropía del vector de votos (cuantizada)      (↓ mejor)
    class_vote_margin : top1−top2 del vector class_votes normalizado   (↑ mejor)
    loc_uncertainty   : media de bbox_std                              (↓ mejor)
    max_bbox_std      : max de bbox_std                                (↓ mejor)

Pipeline:
    1. Matching greedy detección→GT (una sola pasada, IoU ≥ 0.5 por categoría).
       Cada detección queda anotada con is_tp ∈ {0,1}.
    2. Para cada (categoría, métrica, threshold): filtra detecciones de esa
       categoría por la métrica y calcula precisión = TP / (TP + FP).

Salida:
    JSON con todo el sweep + 1 PNG por categoría (líneas = métricas) +
    1 PNG por métrica (precisión macro-avg + líneas por clase) + summary.md

Uso:
    python scripts/precision_vs_confidence_sweep.py \
        --detections results/dino/single_term_aggregated_uf_score_weighted/detection_results.json \
        --gt_ann_file results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \
        --output_dir docs/results_reports/precision_vs_confidence/
"""

import argparse
import contextlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # backend sin display X
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO


# ---------------------------------------------------------------------------
# Geometría y matching
# ---------------------------------------------------------------------------


def _iou_xywh(box: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """IoU de una caja [x,y,w,h] contra un array (M,4) de cajas en el mismo formato."""
    if gts.size == 0:
        return np.empty(0, dtype=float)
    bx, by, bw, bh = box
    gx, gy, gw, gh = gts.T
    inter_x1 = np.maximum(bx, gx)
    inter_y1 = np.maximum(by, gy)
    inter_x2 = np.minimum(bx + bw, gx + gw)
    inter_y2 = np.minimum(by + bh, gy + gh)
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    a_box = bw * bh
    a_gts = gw * gh
    union = a_box + a_gts - inter
    out = np.where(union > 0, inter / union, 0.0)
    return out


def _match_detections_to_gt(
    detections: list[dict],
    coco_gt: COCO,
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, list[int]]:
    """
    Matching greedy una sola vez. Para cada (image_id, category_id):
        1. Orden por score desc.
        2. Para cada det, asignar al GT libre con mayor IoU (si IoU >= iou_thresh).

    Devuelve:
        is_tp     : np.ndarray[bool] de longitud len(detections)
        gt_counts : list de nº GT por category_id (índice por orden de coco.getCatIds())
                    (no se usa para precisión, sólo se reporta como referencia)
    """
    n = len(detections)
    is_tp = np.zeros(n, dtype=bool)

    # Indexar detections por (image_id, category_id) → lista de índices globales
    by_img_cat: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, d in enumerate(detections):
        by_img_cat[(d["image_id"], d["category_id"])].append(i)

    # GTs por (image_id, category_id) → array (M,4) en formato xywh
    gt_boxes_by_img_cat: dict[tuple[int, int], np.ndarray] = {}
    all_img_ids = coco_gt.getImgIds()
    for img_id in all_img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        per_cat: dict[int, list[list[float]]] = defaultdict(list)
        for a in anns:
            per_cat[a["category_id"]].append(a["bbox"])
        for cid, boxes in per_cat.items():
            gt_boxes_by_img_cat[(img_id, cid)] = np.asarray(boxes, dtype=float)

    # Recorrer cada (img, cat) y emparejar
    for (img_id, cat_id), idxs in by_img_cat.items():
        if cat_id < 0:
            # category_id desconocida (e.g. -1) → todas FP
            continue
        gts = gt_boxes_by_img_cat.get((img_id, cat_id))
        if gts is None or gts.shape[0] == 0:
            continue

        # ordenar por score desc
        scores = np.array([detections[i]["score"] for i in idxs], dtype=float)
        order = np.argsort(-scores)
        sorted_idxs = [idxs[k] for k in order]

        gt_used = np.zeros(gts.shape[0], dtype=bool)
        for det_global in sorted_idxs:
            box = np.asarray(detections[det_global]["bbox"], dtype=float)
            ious = _iou_xywh(box, gts)
            ious_masked = np.where(gt_used, -1.0, ious)
            best_gt = int(np.argmax(ious_masked))
            if ious_masked[best_gt] >= iou_thresh:
                is_tp[det_global] = True
                gt_used[best_gt] = True

    gt_counts = []
    for cid in coco_gt.getCatIds():
        ann_ids = coco_gt.getAnnIds(catIds=[cid])
        gt_counts.append(len(ann_ids))
    return is_tp, gt_counts


# ---------------------------------------------------------------------------
# Métricas de confianza por detección
# ---------------------------------------------------------------------------


def _confidence_value(det: dict, metric: str) -> float:
    if metric == "score":
        return float(det.get("score", 0.0))
    if metric == "n_cluster":
        return float(det.get("n_cluster", 1))
    if metric == "contrib_sets":
        cs = det.get("contributing_sets") or []
        return float(len(cs))
    if metric == "class_uncertainty":
        return float(det.get("class_uncertainty", 0.0))
    if metric == "class_vote_margin":
        votes = det.get("class_votes") or {}
        if not votes:
            return 1.0  # singleton: top1=1, top2=0 → margin=1
        total = sum(votes.values())
        if total <= 0:
            return 0.0
        vals_sorted = sorted(votes.values(), reverse=True)
        top1 = vals_sorted[0] / total
        top2 = (vals_sorted[1] / total) if len(vals_sorted) >= 2 else 0.0
        return top1 - top2
    if metric == "borda_margin":
        bs = det.get("borda_scores") or {}
        if not bs:
            return 1.0  # singleton: vector colapsa al ganador
        vals_sorted = sorted(bs.values(), reverse=True)
        top1 = float(vals_sorted[0])
        top2 = float(vals_sorted[1]) if len(vals_sorted) >= 2 else 0.0
        return top1 - top2
    if metric == "loc_uncertainty":
        return float(det.get("loc_uncertainty", 0.0))
    if metric == "max_bbox_std":
        bs = det.get("bbox_std") or [0, 0, 0, 0]
        return float(max(bs))
    raise ValueError(f"Métrica desconocida: {metric}")


# nombre → (mayor_es_mejor, etiqueta_display, color, grid_thresholds)
METRICS_META: dict[str, tuple[bool, str, str, np.ndarray]] = {
    "score":              (True,  "score (mean cluster)",  "#1f77b4",
                           np.round(np.arange(0.05, 0.91, 0.025), 4)),
    "n_cluster":          (True,  "n_cluster",             "#ff7f0e",
                           np.arange(1, 11)),
    "contrib_sets":       (True,  "contrib_sets (1-5)",    "#2ca02c",
                           np.arange(1, 6)),
    "class_uncertainty":  (False, "class_uncertainty",     "#d62728",
                           np.round(np.linspace(0.0, 2.2, 12), 3)),
    "class_vote_margin":  (True,  "class_vote_margin",     "#9467bd",
                           np.round(np.linspace(0.0, 1.0, 11), 3)),
    "borda_margin":       (True,  "borda_margin",          "#e377c2",
                           np.round(np.linspace(0.0, 1.0, 11), 3)),
    "loc_uncertainty":    (False, "loc_uncertainty",       "#8c564b",
                           np.round(np.linspace(0.0, 0.5, 11), 3)),
    "max_bbox_std":       (False, "max_bbox_std",          "#17becf",
                           np.round(np.linspace(0.0, 1.0, 11), 3)),
}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _sweep(
    detections: list[dict],
    is_tp: np.ndarray,
    cat_id_to_name: dict[int, str],
    metrics: list[str],
) -> dict:
    """
    Devuelve:
        {
            metric_name: {
                "thresholds": [...],
                "direction":  "higher_better" | "lower_better",
                "per_class": {
                    class_name: {
                        "precision": [...],
                        "n_dets":    [...],
                        "n_tp":      [...],
                    }
                },
                "macro": {
                    "precision": [...],  # macro-avg sobre clases con n_dets > 0
                    "n_classes_active": [...],
                }
            }
        }
    """
    # Pre-extraer per-detection: cat_name, todas las métricas
    cat_names = [cat_id_to_name.get(d["category_id"], "_unknown") for d in detections]
    metric_values: dict[str, np.ndarray] = {
        m: np.array([_confidence_value(d, m) for d in detections], dtype=float)
        for m in metrics
    }

    classes_present = sorted(set(cat_names) - {"_unknown"})

    result: dict = {}
    for m in metrics:
        higher, _, _, grid = METRICS_META[m]
        direction = "higher_better" if higher else "lower_better"
        vals = metric_values[m]
        is_tp_arr = is_tp

        per_class = {}
        for c in classes_present:
            cls_mask = np.array([cn == c for cn in cat_names], dtype=bool)
            cls_vals = vals[cls_mask]
            cls_tp = is_tp_arr[cls_mask]

            precs, n_dets, n_tp = [], [], []
            for t in grid:
                if higher:
                    keep = cls_vals >= t
                else:
                    keep = cls_vals <= t
                n_keep = int(keep.sum())
                n_tp_keep = int(cls_tp[keep].sum())
                n_dets.append(n_keep)
                n_tp.append(n_tp_keep)
                precs.append(float(n_tp_keep / n_keep) if n_keep > 0 else float("nan"))

            per_class[c] = {"precision": precs, "n_dets": n_dets, "n_tp": n_tp}

        # macro-avg sobre clases con n_dets > 0
        macro_prec = []
        n_active = []
        for ti in range(len(grid)):
            ps = [
                per_class[c]["precision"][ti]
                for c in classes_present
                if per_class[c]["n_dets"][ti] > 0
            ]
            if ps:
                macro_prec.append(float(np.mean(ps)))
                n_active.append(len(ps))
            else:
                macro_prec.append(float("nan"))
                n_active.append(0)

        result[m] = {
            "thresholds": [float(t) for t in grid],
            "direction": direction,
            "per_class": per_class,
            "macro": {"precision": macro_prec, "n_classes_active": n_active},
        }

    return result


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------


def _plot_per_class(sweep: dict, classes: list[str], output_dir: Path) -> None:
    """Una PNG por categoría: eje X = threshold normalizado [0,1], 1 línea por métrica."""
    for c in classes:
        fig, ax = plt.subplots(figsize=(9, 5))
        for m, m_data in sweep.items():
            _, label, color, _ = METRICS_META[m]
            thr = np.array(m_data["thresholds"], dtype=float)
            prec = np.array(m_data["per_class"][c]["precision"], dtype=float)
            n = np.array(m_data["per_class"][c]["n_dets"], dtype=int)

            # Normalizar threshold a [0,1] (para superponer escalas)
            t_norm = (thr - thr.min()) / max(thr.max() - thr.min(), 1e-9)
            if m_data["direction"] == "lower_better":
                # Invertir: t_norm bajo = umbral permisivo, t_norm alto = umbral estricto
                t_norm = 1.0 - t_norm

            mask = n > 0
            ax.plot(
                t_norm[mask], prec[mask],
                marker="o", markersize=4, linewidth=1.6,
                color=color, label=label,
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("threshold normalizado (0=permisivo, 1=estricto)", fontsize=10)
        ax.set_ylabel("precisión = TP / (TP + FP)", fontsize=10)
        ax.set_title(f"Precisión vs threshold por métrica  —  clase: {c}", fontsize=11)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

        safe = c.replace(" ", "_").replace("&", "and").replace("/", "_")
        out = output_dir / f"per_class_{safe}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"PNG: {out}")


def _plot_per_metric(sweep: dict, classes: list[str], output_dir: Path) -> None:
    """Una PNG por métrica con:
       - subplot izq: precisión macro-avg vs threshold (línea negra gruesa)
       - subplot der: precisión por clase vs threshold (1 línea por clase)
    """
    cmap = plt.get_cmap("tab10")
    class_colors = {c: cmap(i % 10) for i, c in enumerate(classes)}

    for m, m_data in sweep.items():
        higher, label, _, _ = METRICS_META[m]
        thr = np.array(m_data["thresholds"], dtype=float)

        fig, (ax_macro, ax_class) = plt.subplots(1, 2, figsize=(14, 5))

        # macro
        macro = np.array(m_data["macro"]["precision"], dtype=float)
        ax_macro.plot(thr, macro, marker="o", color="black", linewidth=2.0)
        ax_macro.set_title(f"{label}  —  precisión macro-avg (9 clases)", fontsize=11)
        ax_macro.set_xlabel(f"threshold {label} ({'↑' if higher else '↓'} mejor)", fontsize=10)
        ax_macro.set_ylabel("precisión macro-avg", fontsize=10)
        ax_macro.set_ylim(-0.02, 1.02)
        ax_macro.grid(linestyle="--", alpha=0.4)

        # per-class
        for c in classes:
            prec = np.array(m_data["per_class"][c]["precision"], dtype=float)
            n = np.array(m_data["per_class"][c]["n_dets"], dtype=int)
            mask = n > 0
            ax_class.plot(
                thr[mask], prec[mask], marker="o", markersize=3.5,
                linewidth=1.4, color=class_colors[c], label=c,
            )
        ax_class.set_title(f"{label}  —  precisión por clase", fontsize=11)
        ax_class.set_xlabel(f"threshold {label} ({'↑' if higher else '↓'} mejor)", fontsize=10)
        ax_class.set_ylabel("precisión", fontsize=10)
        ax_class.set_ylim(-0.02, 1.02)
        ax_class.grid(linestyle="--", alpha=0.4)
        ax_class.legend(fontsize=7, loc="best", ncol=2)

        out = output_dir / f"per_metric_{m}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"PNG: {out}")


def _write_summary(sweep: dict, classes: list[str], md_path: Path) -> None:
    lines = [
        "# Precisión vs threshold de confianza, por categoría",
        "",
        "Mejor threshold por (clase, métrica) según precisión, considerando solo umbrales",
        "donde sobreviven ≥ 10 detecciones (filtros con menos detecciones dan precisiones",
        "ruidosas y poco accionables).",
        "",
    ]
    for m, m_data in sweep.items():
        _, label, _, _ = METRICS_META[m]
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Clase | mejor threshold | precisión | n_dets | n_tp |")
        lines.append("|---|---:|---:|---:|---:|")
        thr = np.array(m_data["thresholds"], dtype=float)
        for c in classes:
            prec = np.array(m_data["per_class"][c]["precision"], dtype=float)
            n = np.array(m_data["per_class"][c]["n_dets"], dtype=int)
            tp = np.array(m_data["per_class"][c]["n_tp"], dtype=int)
            valid = n >= 10
            if not valid.any():
                lines.append(f"| {c} | — | — | — | — |")
                continue
            valid_idx = np.where(valid)[0]
            best = valid_idx[np.nanargmax(prec[valid_idx])]
            lines.append(
                f"| {c} | {thr[best]:.3f} | {prec[best]:.4f} | "
                f"{int(n[best]):,} | {int(tp[best]):,} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD : {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curvas de precisión vs threshold de confianza por categoría."
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--gt_ann_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--metrics", nargs="*", default=list(METRICS_META),
        help="Subconjunto de métricas a evaluar (por defecto las 7)",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading detections from {args.detections} ...")
    with args.detections.open() as fh:
        detections = json.load(fh)
    print(f"  {len(detections):,} detections")

    print(f"Loading GT from {args.gt_ann_file} ...")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(args.gt_ann_file))
    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    print(f"  Categorías: {cat_id_to_name}")

    print(f"Matching detections → GT (greedy, IoU≥{args.iou_thresh}) ...")
    is_tp, gt_counts = _match_detections_to_gt(detections, coco_gt, args.iou_thresh)
    print(f"  TP: {int(is_tp.sum()):,} / {len(detections):,} ({is_tp.mean()*100:.2f}%)")

    print("Sweeping thresholds ...")
    sweep = _sweep(detections, is_tp, cat_id_to_name, args.metrics)

    # ¿Qué clases aparecen?
    classes_present = sorted({cat_id_to_name[d["category_id"]]
                              for d in detections
                              if d["category_id"] in cat_id_to_name})

    # JSON
    out_json = args.output_dir / "sweep_results.json"
    with out_json.open("w") as fh:
        json.dump({
            "iou_thresh": args.iou_thresh,
            "n_detections": len(detections),
            "n_tp_total": int(is_tp.sum()),
            "gt_counts_per_cat_id": dict(zip(coco_gt.getCatIds(), gt_counts)),
            "cat_id_to_name": cat_id_to_name,
            "metrics": sweep,
        }, fh, indent=2)
    print(f"JSON: {out_json}")

    # Plots
    _plot_per_class(sweep, classes_present, args.output_dir)
    _plot_per_metric(sweep, classes_present, args.output_dir)

    # Summary MD
    _write_summary(sweep, classes_present, args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
