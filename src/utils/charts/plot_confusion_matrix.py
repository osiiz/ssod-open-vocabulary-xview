import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import os


def compute_iou(boxA, boxB):
    """Calcula el IoU entre dos cajas en formato COCO [x, y, w, h]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)


def _best_match_detection(det, gts, unmatched_gt, iou_thresh, require_same_class=False):
    """Return the index of the best unmatched GT for this detection, or -1 if none."""
    best_iou = 0.0
    best_gt_idx = -1
    dt_cat = det["category_id"]

    for i, gt in enumerate(gts):
        if i not in unmatched_gt:
            continue
        if require_same_class and gt["category_id"] != dt_cat:
            continue
        iou = compute_iou(det["bbox"], gt["bbox"])
        if iou >= iou_thresh and iou > best_iou:
            best_iou = iou
            best_gt_idx = i

    return best_gt_idx


def build_confusion_matrix(gt_data, dt_data, score_thresh, iou_thresh):
    # Mapear categorías y añadir el fondo (Background)
    categories = {cat["id"]: cat["name"] for cat in gt_data["categories"]}

    # Creamos un ID virtual para el fondo (Background)
    bg_id = max(categories.keys()) + 1
    categories[bg_id] = "Background"

    cat_ids = sorted(list(categories.keys()))
    num_classes = len(cat_ids)
    cat_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    # Matriz de confusión [Ground Truth, Detección]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Organizar GT por imagen
    gt_by_img = defaultdict(list)
    for ann in gt_data["annotations"]:
        gt_by_img[ann["image_id"]].append(ann)

    # Organizar DT por imagen y filtrar por confianza
    dt_by_img = defaultdict(list)
    for det in dt_data:
        if det["score"] >= score_thresh:
            dt_by_img[det["image_id"]].append(det)

    # Procesar cada imagen
    for img_id, gts in gt_by_img.items():
        dts = dt_by_img.get(img_id, [])

        # Ordenar detecciones por score descendente (las mejores primero)
        dts = sorted(dts, key=lambda x: x["score"], reverse=True)

        unmatched_gt = set(range(len(gts)))
        matched_dt = set()

        # Paso 1: matches de misma clase (TP puros por clase)
        for dt_idx, dt in enumerate(dts):
            gt_idx = _best_match_detection(
                dt,
                gts,
                unmatched_gt,
                iou_thresh=iou_thresh,
                require_same_class=True,
            )
            if gt_idx != -1:
                gt_cat = gts[gt_idx]["category_id"]
                dt_cat = dt["category_id"]
                confusion_matrix[cat_to_idx[gt_cat], cat_to_idx[dt_cat]] += 1
                unmatched_gt.remove(gt_idx)
                matched_dt.add(dt_idx)

        # Paso 2: detecciones restantes se intentan casar con cualquier clase (confusiones)
        for dt_idx, dt in enumerate(dts):
            if dt_idx in matched_dt:
                continue

            gt_idx = _best_match_detection(
                dt,
                gts,
                unmatched_gt,
                iou_thresh=iou_thresh,
                require_same_class=False,
            )

            if gt_idx != -1:
                gt_cat = gts[gt_idx]["category_id"]
                dt_cat = dt["category_id"]
                confusion_matrix[cat_to_idx[gt_cat], cat_to_idx[dt_cat]] += 1
                unmatched_gt.remove(gt_idx)
            else:
                # Falso Positivo: detección sin GT que cumpla IoU
                dt_cat = dt["category_id"]
                confusion_matrix[cat_to_idx[bg_id], cat_to_idx[dt_cat]] += 1

        # Falsos Negativos: GTs no detectados
        for gt_idx in unmatched_gt:
            gt_cat = gts[gt_idx]["category_id"]
            confusion_matrix[cat_to_idx[gt_cat], cat_to_idx[bg_id]] += 1

    class_names = [categories[c] for c in cat_ids]
    bg_idx = cat_to_idx[bg_id]

    # Normalizar por filas para inspección visual
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    confusion_matrix_norm = confusion_matrix / row_sums

    tp = int(np.trace(confusion_matrix[:bg_idx, :bg_idx]))
    fn = int(confusion_matrix[:bg_idx, bg_idx].sum())
    fp = int(confusion_matrix[bg_idx, :bg_idx].sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "confusion_matrix": confusion_matrix,
        "confusion_matrix_norm": confusion_matrix_norm,
        "class_names": class_names,
        "bg_idx": bg_idx,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def _render_confusion_plot(result, output_image, score_thresh, iou_thresh):
    confusion_matrix = result["confusion_matrix"]
    confusion_matrix_norm = result["confusion_matrix_norm"]
    class_names = result["class_names"]
    precision = result["precision"]
    recall = result["recall"]

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        confusion_matrix_norm,
        annot=confusion_matrix,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 10},
    )

    plt.xlabel("Predicción (Detección)", fontsize=14, fontweight="bold")
    plt.ylabel("Verdad (Ground Truth)", fontsize=14, fontweight="bold")
    plt.title(
        (
            f"Matriz de Confusión de Detección (score >= {score_thresh:.2f}, IoU >= {iou_thresh:.2f})\n"
            f"Precisión@op={precision:.3f} | Recall@op={recall:.3f}"
        ),
        fontsize=16,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_dir = os.path.dirname(output_image)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_image, dpi=300)
    plt.close()
    print(
        f"Matriz generada: {output_image} | precision={precision:.4f} | recall={recall:.4f}"
    )


def main(gt_file, dt_file, output_image, score_thresh, iou_thresh):
    with open(gt_file, "r") as f:
        gt_data = json.load(f)

    with open(dt_file, "r") as f:
        dt_data = json.load(f)

    result = build_confusion_matrix(gt_data, dt_data, score_thresh, iou_thresh)
    _render_confusion_plot(result, output_image, score_thresh, iou_thresh)


def _parse_threshold_list(value):
    if value is None:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [float(p) for p in parts]


def _suffix_path(path, suffix):
    base, ext = os.path.splitext(path)
    return f"{base}_{suffix}{ext}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera matriz de confusión para detección de objetos"
    )
    parser.add_argument(
        "--gt_json",
        type=str,
        required=True,
        help="Ruta al JSON de anotaciones COCO original (Ground Truth)",
    )
    parser.add_argument(
        "--dt_json",
        type=str,
        required=True,
        help="Ruta al JSON de detecciones (Results)",
    )
    parser.add_argument(
        "--out_png", type=str, required=True, help="Ruta donde guardar la imagen PNG"
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.1,
        help="Umbral mínimo de confianza para la predicción",
    )
    parser.add_argument(
        "--score_thresholds",
        type=str,
        default="",
        help="Lista separada por comas para barrido de score thresholds, ej: 0.05,0.1,0.3,0.5",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="IoU mínimo para considerar que el bbox hace match",
    )
    args = parser.parse_args()

    score_thresholds = _parse_threshold_list(args.score_thresholds)
    if score_thresholds:
        with open(args.gt_json, "r") as f:
            gt_data = json.load(f)
        with open(args.dt_json, "r") as f:
            dt_data = json.load(f)

        for threshold in score_thresholds:
            out_path = _suffix_path(args.out_png, f"score_{threshold:.2f}")
            result = build_confusion_matrix(
                gt_data, dt_data, score_thresh=threshold, iou_thresh=args.iou_thresh
            )
            _render_confusion_plot(result, out_path, threshold, args.iou_thresh)
    else:
        main(
            args.gt_json, args.dt_json, args.out_png, args.score_thresh, args.iou_thresh
        )
