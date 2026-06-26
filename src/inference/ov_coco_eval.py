import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.utils.coco_utils import build_coco_max_dets, stats2dict


def _relabel_by_gt_matching(
    detections: list[dict],
    coco_gt: COCO,
    iou_thresh: float = 0.1,
) -> list[dict]:
    """Matching voraz independiente de clase de predicciones a cajas GT.

    Por imagen, las predicciones se ordenan por puntuación (descendente) y se
    emparejan vorazmente con cajas GT no emparejadas. Una predicción emparejada
    hereda el category_id de su caja GT. Las predicciones no emparejadas conservan
    su category_id original (permanecen como FP para la clase que predijeron).

    iou_thresh es intencionadamente bajo (0.1) para que las cajas bien ubicadas
    sean re-etiquetadas; COCOeval aplica el umbral de calidad real (0.5, 0.75…)
    al calcular AP.
    """
    from collections import defaultdict

    dets_by_img: dict[int, list] = defaultdict(list)
    for d in detections:
        dets_by_img[d["image_id"]].append(d)

    all_gt = coco_gt.loadAnns(coco_gt.getAnnIds())
    gt_by_img: dict[int, list] = defaultdict(list)
    for ann in all_gt:
        if not ann.get("iscrowd", 0):
            gt_by_img[ann["image_id"]].append(ann)

    relabeled: list[dict] = []
    for img_id, img_dets in dets_by_img.items():
        img_gts = gt_by_img[img_id]
        sorted_dets = sorted(img_dets, key=lambda x: x.get("score", 1.0), reverse=True)
        matched_gt_ids: set[int] = set()

        for det in sorted_dets:
            new_det = dict(det)
            dx1, dy1, dw, dh = det["bbox"]
            dx2, dy2 = dx1 + dw, dy1 + dh

            best_iou, best_gt = 0.0, None
            for gt in img_gts:
                if gt["id"] in matched_gt_ids:
                    continue
                gx1, gy1, gw, gh = gt["bbox"]
                gx2, gy2 = gx1 + gw, gy1 + gh
                ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
                ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = dw * dh + gw * gh - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is not None and best_iou >= iou_thresh:
                new_det["category_id"] = best_gt["category_id"]
                matched_gt_ids.add(best_gt["id"])

            relabeled.append(new_det)

    return relabeled


def evaluate_agnostic_per_class(
    coco_gt: COCO,
    detections: list[dict],
    max_dets: int,
    use_pr_metrics: bool = False,
    relabel_iou_thresh: float = 0.1,
) -> tuple[dict, dict]:
    """Evaluación independiente de clase mediante re-etiquetado con GT.

    Algoritmo (por imagen):
      1. Ordenar todas las predicciones por puntuación (descendente).
      2. Emparejar vorazmente cada predicción con la caja GT no emparejada más cercana
         (IoU ≥ relabel_iou_thresh, independiente de clase).
      3. Las predicciones emparejadas heredan el category_id del GT.
      4. Las predicciones no emparejadas conservan su category_id original (→ FP para
         su clase predicha).
      5. Ejecutar un COCOeval estándar por clase sobre las predicciones re-etiquetadas.

    Esto aísla la calidad de localización de la calidad de clasificación: una
    predicción bien ubicada pero mal etiquetada se convierte en TP; una predicción
    mal ubicada (IoU bajo) permanece como FP aunque el re-etiquetado la empareje.

    Devuelve (metrics_dict, metrics_per_class_dict).
    """
    cat_names_list = [cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())]

    relabeled = _relabel_by_gt_matching(
        detections, coco_gt, iou_thresh=relabel_iou_thresh
    )

    if not relabeled:
        if use_pr_metrics:
            return empty_pr_metrics(), {n: -1.0 for n in cat_names_list}
        return empty_metrics(max_dets), {n: -1.0 for n in cat_names_list}

    max_dets_triplet = build_coco_max_dets(max_dets)

    try:
        coco_dt = coco_gt.loadRes(relabeled)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.maxDets = max_dets_triplet
        coco_eval.evaluate()
        coco_eval.accumulate()

        cat_ids = coco_gt.getCatIds()
        cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

        if use_pr_metrics:
            metrics, metrics_per_class = _pr_summary_from_coco_eval(
                coco_eval, cat_ids, cat_names
            )
        else:
            coco_eval.summarize()
            metrics = stats2dict(coco_eval)
            metrics_per_class = compute_class_ap50_metrics(coco_eval, coco_gt)

    except Exception as exc:
        print(f"Warning: agnostic evaluation failed: {exc}")
        if use_pr_metrics:
            return empty_pr_metrics(), {n: -1.0 for n in cat_names_list}
        return empty_metrics(max_dets), {n: -1.0 for n in cat_names_list}

    return metrics, metrics_per_class


def _pr_from_eval_imgs(eval_imgs: list, iou_idx: int) -> tuple[float, float]:
    """Agrega TP/FP/FN de evalImgs en un índice de umbral IoU → (precision, recall)."""
    tp = fp = fn = 0
    for e in eval_imgs:
        if e is None:
            continue
        dtm = e["dtMatches"]  # [T, D]
        dtig = e["dtIgnore"]  # [T, D]
        gtm = e["gtMatches"]  # [T, G]
        gtig = e["gtIgnore"]  # [G]
        if dtm.shape[1] > 0:
            tp += int(np.sum((dtm[iou_idx] > 0) & ~dtig[iou_idx].astype(bool)))
            fp += int(np.sum((dtm[iou_idx] == 0) & ~dtig[iou_idx].astype(bool)))
        if gtm.shape[1] > 0:
            fn += int(np.sum((gtm[iou_idx] == 0) & ~gtig.astype(bool)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return p, r


def _pr_summary_from_coco_eval(
    coco_eval: COCOeval, cat_ids: list[int], cat_names: dict[int, str]
) -> tuple[dict, dict]:
    """Calcula P50, P75, P50:95, R50, R75, R50:95 (media sobre clases) y P50 por clase.

    Filtra evalImgs solo al rango de área 'all'.
    """
    area_all = coco_eval.params.areaRng[0]
    n_iou = len(coco_eval.params.iouThrs)  # 10

    # Agrupar evalImgs relevantes por categoría
    cat_eval_imgs: dict[int, list] = {cat_id: [] for cat_id in cat_ids}
    for e in coco_eval.evalImgs:
        if e is not None and e["aRng"] == area_all:
            cid = e["category_id"]
            if cid in cat_eval_imgs:
                cat_eval_imgs[cid].append(e)

    all_p50, all_p75, all_p5095 = [], [], []
    all_r50, all_r75, all_r5095 = [], [], []
    per_class_p50: dict[str, float] = {}

    for cat_id in cat_ids:
        imgs = cat_eval_imgs[cat_id]
        p50, r50 = _pr_from_eval_imgs(imgs, 0)  # IoU=0.50 → índice 0
        p75, r75 = _pr_from_eval_imgs(imgs, 5)  # IoU=0.75 → índice 5
        p5095 = float(np.mean([_pr_from_eval_imgs(imgs, t)[0] for t in range(n_iou)]))
        r5095 = float(np.mean([_pr_from_eval_imgs(imgs, t)[1] for t in range(n_iou)]))

        per_class_p50[cat_names[cat_id]] = round(p50, 4)
        all_p50.append(p50)
        all_p75.append(p75)
        all_p5095.append(p5095)
        all_r50.append(r50)
        all_r75.append(r75)
        all_r5095.append(r5095)

    metrics = {
        "P50:95": round(float(np.mean(all_p5095)), 4),
        "P50": round(float(np.mean(all_p50)), 4),
        "P75": round(float(np.mean(all_p75)), 4),
        "R50:95": round(float(np.mean(all_r5095)), 4),
        "R50": round(float(np.mean(all_r50)), 4),
        "R75": round(float(np.mean(all_r75)), 4),
    }
    return metrics, per_class_p50


def empty_pr_metrics() -> dict:
    return {
        "P50:95": -1.0,
        "P50": -1.0,
        "P75": -1.0,
        "R50:95": -1.0,
        "R50": -1.0,
        "R75": -1.0,
    }


def compute_class_ap50_metrics(coco_eval: COCOeval, coco_gt: COCO) -> dict[str, float]:
    precisions = coco_eval.eval.get("precision")
    if precisions is None:
        return {}

    cat_ids = coco_eval.params.catIds
    categories_info = coco_gt.loadCats(cat_ids)
    cat_dict = {cat["id"]: cat["name"] for cat in categories_info}

    class_metrics: dict[str, float] = {}

    for idx, cat_id in enumerate(cat_ids):
        precision_curve = precisions[0, :, idx, 0, -1]
        valid_values = precision_curve[precision_curve > -1]
        class_name = cat_dict.get(cat_id, str(cat_id))

        if len(valid_values) > 0:
            class_metrics[class_name] = round(float(np.mean(valid_values)), 4)
        else:
            class_metrics[class_name] = -1.0

    return class_metrics


def empty_metrics(max_dets: int) -> dict[str, float]:
    max_dets_triplet = build_coco_max_dets(max_dets)
    keys = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        f"AR_{max_dets_triplet[0]}",
        f"AR_{max_dets_triplet[1]}",
        f"AR_{max_dets_triplet[2]}",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {key: -1.0 for key in keys}


def materialize_artifact(source: Path, destination: Path, mode: str = "auto") -> None:
    """Crea un artefacto de salida sin copias completas innecesarias.

    Modos:
    - auto: intenta hardlink, luego symlink, luego copia
    - hardlink: requiere hardlink, copia como fallback si falla
    - symlink: requiere symlink, copia como fallback si falla
    - copy: siempre copia completa
    """

    source = source.resolve()
    destination = destination.resolve()

    if source == destination:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    mode = mode.lower().strip()
    if mode not in {"auto", "hardlink", "symlink", "copy"}:
        raise ValueError(
            "artifact mode must be one of {'auto','hardlink','symlink','copy'}"
        )

    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return
        except OSError:
            if mode == "hardlink":
                shutil.copy2(source, destination)
                return

    if mode in {"auto", "symlink"}:
        try:
            os.symlink(source, destination)
            return
        except OSError:
            if mode == "symlink":
                shutil.copy2(source, destination)
                return

    shutil.copy2(source, destination)


def evaluate_ov_predictions(
    ann_file: Path,
    detection_results: Path,
    output_folder: Path,
    mode: str,
    max_dets: int = 1000,
    raw_predictions: Path | None = None,
    artifact_mode: str = "auto",
    materialize_raw: bool = False,
    use_pr_metrics: bool = False,
) -> None:
    mode = mode.lower().strip()
    if mode not in {"aware", "agnostic"}:
        raise ValueError("mode must be one of {'aware', 'agnostic'}")

    output_folder.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting OV COCO evaluation ({mode}) | "
        f"gt={ann_file} detections={detection_results}"
    )

    output_detections_path = output_folder / "detection_results.json"
    materialize_artifact(
        source=detection_results,
        destination=output_detections_path,
        mode=artifact_mode,
    )

    if materialize_raw and raw_predictions is not None and raw_predictions.exists():
        materialize_artifact(
            source=raw_predictions,
            destination=output_folder / "raw_predictions.json",
            mode=artifact_mode,
        )

    print("Loading ground truth annotations...")
    coco_gt = COCO(str(ann_file))
    valid_img_ids = set(coco_gt.getImgIds())

    try:
        print("Loading detections and running COCOeval...")
        with open(output_detections_path, encoding="utf-8") as fh:
            all_detections = json.load(fh)
        detections = [d for d in all_detections if d["image_id"] in valid_img_ids]
        del all_detections

        coco_dt = coco_gt.loadRes(detections)

        if mode == "aware":
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.params.maxDets = build_coco_max_dets(max_dets)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            if use_pr_metrics:
                cat_ids = coco_eval.params.catIds
                cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
                metrics, metrics_per_class = _pr_summary_from_coco_eval(
                    coco_eval, cat_ids, cat_names
                )
            else:
                metrics = stats2dict(coco_eval)
                metrics_per_class = compute_class_ap50_metrics(coco_eval, coco_gt)
        else:
            print("Running agnostic-per-class evaluation...")
            metrics, metrics_per_class = evaluate_agnostic_per_class(
                coco_gt, detections, max_dets, use_pr_metrics=use_pr_metrics
            )
    except Exception as exc:
        print(f"Warning: evaluation failed, writing empty metrics. Reason: {exc}")
        category_names = [cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())]
        metrics = (
            empty_pr_metrics() if use_pr_metrics else empty_metrics(max_dets=max_dets)
        )
        metrics_per_class = {name: -1.0 for name in category_names}

    with (output_folder / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with (output_folder / "metrics_per_class.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics_per_class, handle, indent=2)

    print(f"OV COCO evaluation complete ({mode}) | output={output_folder}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Grounding DINO detections in class-aware or class-agnostic mode"
    )
    parser.add_argument("--ann_file", type=Path, required=True)
    parser.add_argument("--detection_results", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["aware", "agnostic"], required=True
    )
    parser.add_argument("--max_dets", type=int, default=1000)
    parser.add_argument("--raw_predictions", type=Path, default=None)
    parser.add_argument(
        "--artifact_mode",
        type=str,
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
    )
    parser.add_argument("--materialize_raw", action="store_true")
    parser.add_argument(
        "--use_pr_metrics",
        action="store_true",
        help="Output Precision/Recall instead of AP (for detectors without confidence scores)",
    )

    args = parser.parse_args()
    evaluate_ov_predictions(
        ann_file=args.ann_file,
        detection_results=args.detection_results,
        output_folder=args.output_folder,
        mode=args.mode,
        max_dets=args.max_dets,
        raw_predictions=args.raw_predictions,
        artifact_mode=args.artifact_mode,
        materialize_raw=args.materialize_raw,
        use_pr_metrics=args.use_pr_metrics,
    )


if __name__ == "__main__":
    main()
