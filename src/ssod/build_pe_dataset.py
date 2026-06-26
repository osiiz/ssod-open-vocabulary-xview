"""
Fusiona anotaciones COCO etiquetadas con detecciones de pseudo-etiquetas de uno o más
detectores en un único JSON COCO para entrenamiento semi-supervisado.

- Las imágenes etiquetadas conservan sus IDs originales.
- Los IDs de imágenes no etiquetadas se reasignan para evitar colisiones (offset = max ID etiquetado).
- Los IDs de anotaciones se reasignan de forma análoga.
- file_name en el COCO fusionado usa subcarpetas relativas a tile_images_root:
    "train_sampled/images/<nombre>.tif"
    "train_unlabeled_eval/images/<nombre>.tif"
  por lo que --train_img_folder para el entrenamiento debe apuntar a tile_images_root.

Uso:
    python -m src.ssod.build_pe_dataset \
        --labeled_ann_file   .../train_sampled/COCO_annotations.json \
        --unlabeled_ann_file .../train_unlabeled_eval/COCO_annotations.json \
        --tile_images_root   ./results/preprocess/tile_images \
        --detections         faster10.json dino.json \
        --thresholds         faster10_threshold.json dino_threshold.json \
        --output_ann_file    results/ssod/pe_datasets/ab/COCO_annotations.json \
        --output_stats_file  results/ssod/pe_datasets/ab/build_stats.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path) -> dict | list:
    with path.open() as fh:
        return json.load(fh)


def _load_threshold(path: Path | None) -> float | None:
    if path is None:
        return None
    data = _load_json(path)
    return float(data["threshold"])


def _filter_detections(detections: list, threshold: float | None) -> list:
    if threshold is None:
        return detections
    return [d for d in detections if d["score"] >= threshold]


def _iou_xywh(box: list[float], kept_boxes: list[list[float]]) -> float:
    """IoU máximo entre `box` (xywh) e calquera caixa de `kept_boxes` (xywh)."""
    if not kept_boxes:
        return 0.0
    bx, by, bw, bh = box
    bx2, by2 = bx + bw, by + bh
    b_area = bw * bh
    best = 0.0
    for kx, ky, kw, kh in kept_boxes:
        ix1 = bx if bx > kx else kx
        iy1 = by if by > ky else ky
        ix2 = bx2 if bx2 < kx + kw else kx + kw
        iy2 = by2 if by2 < ky + kh else ky + kh
        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw <= 0.0 or ih <= 0.0:
            continue
        inter = iw * ih
        union = b_area + kw * kh - inter
        if union <= 0.0:
            continue
        iou = inter / union
        if iou > best:
            best = iou
    return best


def _dedup_cross_source(
    pe_by_image: dict[int, list],
    iou_thresh: float,
    priority: list[int],
) -> tuple[dict[int, list], int]:
    """
    Dedup intra-clase con prioridade fixa entre fontes. Cando dúas caixas da
    mesma categoría solapan con IoU >= iou_thresh, queda a da fonte con maior
    prioridade (orde definido por `priority`, que lista índices `_source_idx`
    de máis a menos prioritario). Dentro da mesma fonte, gana a de maior score.

    Caixas de categorías distintas mantéñense aínda que solapen (asúmense como
    obxectos diferentes na proxección aérea).

    Devolve (pe_by_image_dedup, n_dropped).
    """
    rank = {idx: r for r, idx in enumerate(priority)}
    out: dict[int, list] = {}
    dropped = 0
    for img_id, dets in pe_by_image.items():
        if not dets:
            out[img_id] = []
            continue
        ordered = sorted(
            dets,
            key=lambda d: (
                rank.get(d.get("_source_idx", 0), len(rank)),
                -d.get("score", 0.0),
            ),
        )
        kept_by_cat: dict[int, list[list[float]]] = defaultdict(list)
        kept: list[dict] = []
        for det in ordered:
            cat = det["category_id"]
            if _iou_xywh(det["bbox"], kept_by_cat[cat]) >= iou_thresh:
                dropped += 1
                continue
            kept.append(det)
            kept_by_cat[cat].append(det["bbox"])
        out[img_id] = kept
    return out, dropped


def _detections_to_annotations(
    detections: list,
    ann_id_start: int,
) -> tuple[list, int]:
    annotations = []
    next_id = ann_id_start
    for det in detections:
        x, y, w, h = det["bbox"]
        annotations.append(
            {
                "id": next_id,
                "image_id": det["image_id"],
                "category_id": det["category_id"],
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )
        next_id += 1
    return annotations, next_id


EXCLUSION_CATEGORY_ID = -1


def build_merged_coco(
    labeled_coco: dict,
    unlabeled_coco: dict,
    pe_sources: list[list],
    thresholds: list[float | None],
    labeled_subfolder: str,
    unlabeled_subfolder: str,
    drop_empty_unlabeled: bool,
    dedup_iou_thresh: float | None = None,
    dedup_priority: list[int] | None = None,
    exclusion_sources: list[list] | None = None,
) -> tuple[dict, dict]:
    image_id_offset = max(img["id"] for img in labeled_coco["images"])
    ann_id_start = max(ann["id"] for ann in labeled_coco["annotations"]) + 1

    # Reasignar file_names etiquetados para incluir la subcarpeta
    labeled_images = []
    for img in labeled_coco["images"]:
        new_img = dict(img)
        new_img["file_name"] = f"{labeled_subfolder}/{img['file_name']}"
        labeled_images.append(new_img)

    # Construir mapeo de image_id original no etiquetado → image_id reasignado
    id_remap = {
        img["id"]: img["id"] + image_id_offset for img in unlabeled_coco["images"]
    }

    # Construir anotaciones PE por imagen de todas las fuentes, aplicando umbrales
    unlabeled_pe_by_image: dict[int, list] = defaultdict(list)
    source_counts: dict[str, int] = {}
    for source_idx, (source_dets, thresh) in enumerate(zip(pe_sources, thresholds)):
        filtered = _filter_detections(source_dets, thresh)
        source_counts[f"source_{source_idx}"] = len(filtered)
        for det in filtered:
            orig_img_id = det["image_id"]
            if orig_img_id not in id_remap:
                continue
            unlabeled_pe_by_image[orig_img_id].append(
                {
                    "image_id": id_remap[orig_img_id],
                    "category_id": det["category_id"],
                    "bbox": det["bbox"],
                    "score": det.get("score", 1.0),
                    "_source_idx": source_idx,
                }
            )

    # Deduplicación intra-clase con prioridade fixa entre fontes: cando dúas
    # caixas da mesma categoría solapan con IoU >= thresh, queda a da fonte de
    # maior prioridade (`dedup_priority` lista índices `_source_idx` de máis a
    # menos prioritario). Por defecto, a prioridade é a orde das `--detections`.
    dedup_dropped = 0
    if dedup_iou_thresh is not None:
        priority = (
            dedup_priority
            if dedup_priority is not None
            else list(range(len(pe_sources)))
        )
        unlabeled_pe_by_image, dedup_dropped = _dedup_cross_source(
            unlabeled_pe_by_image, dedup_iou_thresh, priority=priority
        )

    # Construir lista de imágenes no etiquetadas (opcionalmente saltar imágenes sin PEs).
    # Usamos .get() porque _dedup_cross_source devolve un dict normal e
    # KeyError'a para imaxes sen PE orixinal.
    unlabeled_images = []
    for img in unlabeled_coco["images"]:
        if drop_empty_unlabeled and not unlabeled_pe_by_image.get(img["id"]):
            continue
        new_img = dict(img)
        new_img["id"] = id_remap[img["id"]]
        new_img["file_name"] = f"{unlabeled_subfolder}/{img['file_name']}"
        unlabeled_images.append(new_img)

    included_unlabeled_ids = {img["id"] - image_id_offset for img in unlabeled_images}

    # Convertir anotaciones PE al formato COCO
    all_pe_anns: list[dict] = []
    for orig_id in included_unlabeled_ids:
        for det_ann in unlabeled_pe_by_image.get(orig_id, []):
            x, y, w, h = det_ann["bbox"]
            all_pe_anns.append(
                {
                    "id": ann_id_start,
                    "image_id": det_ann["image_id"],
                    "category_id": det_ann["category_id"],
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id_start += 1

    # Zonas de exclusión: anotacións con category_id = EXCLUSION_CATEGORY_ID.
    # Engadense só nas imaxes que xa están incluídas no dataset (i.e., que teñen
    # PE) para non resucitar imaxes baleiras. As caixas de zonas son
    # independentes do dedup PE: poden solapar con PE positivas e entre sí.
    exclusion_anns: list[dict] = []
    exclusion_per_source: dict[str, int] = {}
    if exclusion_sources:
        for source_idx, source_dets in enumerate(exclusion_sources):
            kept = 0
            for det in source_dets:
                orig_img_id = det["image_id"]
                if orig_img_id not in id_remap:
                    continue
                if orig_img_id not in included_unlabeled_ids:
                    continue
                x, y, w, h = det["bbox"]
                exclusion_anns.append(
                    {
                        "id": ann_id_start,
                        "image_id": id_remap[orig_img_id],
                        "category_id": EXCLUSION_CATEGORY_ID,
                        "bbox": [x, y, w, h],
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )
                ann_id_start += 1
                kept += 1
            exclusion_per_source[f"exclusion_source_{source_idx}"] = kept

    merged = {
        "images": labeled_images + unlabeled_images,
        "annotations": labeled_coco["annotations"] + all_pe_anns + exclusion_anns,
        "categories": labeled_coco["categories"],
    }

    stats = {
        "labeled_images": len(labeled_images),
        "labeled_annotations": len(labeled_coco["annotations"]),
        "unlabeled_images_with_pe": len(unlabeled_images),
        "pe_annotations_total": len(all_pe_anns),
        "pe_per_source": source_counts,
        "dedup_iou_thresh": dedup_iou_thresh,
        "dedup_priority": (
            dedup_priority
            if dedup_priority is not None
            else (list(range(len(pe_sources))) if dedup_iou_thresh is not None else None)
        ),
        "pe_annotations_dedup_dropped": dedup_dropped,
        "exclusion_annotations_total": len(exclusion_anns),
        "exclusion_per_source": exclusion_per_source,
        "total_images": len(merged["images"]),
        "total_annotations": len(merged["annotations"]),
    }

    return merged, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge labeled COCO GT with pseudo-label detections for SSOD training."
    )
    parser.add_argument("--labeled_ann_file", type=Path, required=True)
    parser.add_argument("--unlabeled_ann_file", type=Path, required=True)
    parser.add_argument(
        "--tile_images_root", type=str, default="./results/preprocess/tile_images"
    )
    parser.add_argument("--labeled_subfolder", type=str, default="train_sampled/images")
    parser.add_argument(
        "--unlabeled_subfolder", type=str, default="train_unlabeled_eval/images"
    )
    parser.add_argument("--detections", type=Path, nargs="+", required=True)
    parser.add_argument("--thresholds", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--exclusion_zones",
        type=Path,
        nargs="*",
        default=[],
        help="JSONs (estilo COCO results) con caixas a marcar como zonas de "
             "exclusión (category_id = -1 no COCO de saída). A orde NON ten que "
             "coincidir con --detections; cada arquivo é independente. Pase "
             "cadea baleira por fonte para sinalar 'sen zonas'.",
    )
    parser.add_argument("--output_ann_file", type=Path, required=True)
    parser.add_argument("--output_stats_file", type=Path, default=None)
    parser.add_argument(
        "--drop_empty_unlabeled",
        action="store_true",
        default=True,
        help="Skip unlabeled images with no PE annotations after filtering.",
    )
    parser.add_argument(
        "--dedup_iou_thresh",
        type=float,
        default=None,
        help="If set, run intra-class dedup with fixed source priority across "
        "sources per image: when two boxes of the same category overlap with "
        "IoU >= this threshold, keep the one from the highest-priority source.",
    )
    parser.add_argument(
        "--dedup_priority",
        type=int,
        nargs="+",
        default=None,
        help="List of --detections indices ordered from highest to lowest "
        "priority. Defaults to the order in which --detections were passed.",
    )
    args = parser.parse_args()

    if args.dedup_priority is not None:
        if sorted(args.dedup_priority) != list(range(len(args.detections))):
            raise ValueError(
                f"--dedup_priority must be a permutation of "
                f"[0..{len(args.detections) - 1}], got {args.dedup_priority}."
            )

    if len(args.thresholds) > len(args.detections):
        raise ValueError(
            f"--thresholds length ({len(args.thresholds)}) cannot exceed "
            f"--detections length ({len(args.detections)}). "
            "Sources without a threshold entry are treated as unfiltered."
        )

    print(f"Loading labeled COCO from {args.labeled_ann_file} ...")
    labeled_coco = _load_json(args.labeled_ann_file)

    print(f"Loading unlabeled image list from {args.unlabeled_ann_file} ...")
    unlabeled_coco = _load_json(args.unlabeled_ann_file)

    pe_sources = []
    thresholds: list[float | None] = []
    for idx, det_path in enumerate(args.detections):
        print(f"Loading detections [{idx}] from {det_path} ...")
        pe_sources.append(_load_json(det_path))
        thresh_path = args.thresholds[idx] if idx < len(args.thresholds) else None
        thresh_val = _load_threshold(thresh_path)
        thresholds.append(thresh_val)
        if thresh_val is not None:
            print(f"  Threshold: {thresh_val:.4f} (from {thresh_path})")
        else:
            print("  No threshold applied (all detections included).")

    exclusion_sources: list[list] = []
    for idx, ez_path in enumerate(args.exclusion_zones):
        print(f"Loading exclusion zones [{idx}] from {ez_path} ...")
        ez = _load_json(ez_path)
        exclusion_sources.append(ez)
        print(f"  {len(ez):,} caixas de zona de exclusión cargadas.")

    print("Building merged COCO dataset ...")
    merged, stats = build_merged_coco(
        labeled_coco=labeled_coco,
        unlabeled_coco=unlabeled_coco,
        pe_sources=pe_sources,
        thresholds=thresholds,
        labeled_subfolder=args.labeled_subfolder,
        unlabeled_subfolder=args.unlabeled_subfolder,
        drop_empty_unlabeled=args.drop_empty_unlabeled,
        dedup_iou_thresh=args.dedup_iou_thresh,
        dedup_priority=args.dedup_priority,
        exclusion_sources=exclusion_sources or None,
    )

    if args.dedup_iou_thresh is not None:
        used_priority = (
            args.dedup_priority
            if args.dedup_priority is not None
            else list(range(len(args.detections)))
        )
        print(
            f"Dedup intra-clase IoU>={args.dedup_iou_thresh} con prioridade "
            f"{used_priority}: {stats['pe_annotations_dedup_dropped']:,} "
            f"detecciones eliminadas."
        )

    print(
        f"Merged dataset: {stats['total_images']} images "
        f"({stats['labeled_images']} labeled + {stats['unlabeled_images_with_pe']} unlabeled), "
        f"{stats['total_annotations']} annotations "
        f"({stats['labeled_annotations']} GT + {stats['pe_annotations_total']} PE"
        + (f" + {stats['exclusion_annotations_total']} exclusion zones"
           if stats.get('exclusion_annotations_total', 0) > 0 else "")
        + ")."
    )

    args.output_ann_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_ann_file.open("w") as fh:
        json.dump(merged, fh)
    print(f"Merged COCO saved to {args.output_ann_file}")

    if args.output_stats_file is not None:
        args.output_stats_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_stats_file.open("w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"Stats saved to {args.output_stats_file}")


if __name__ == "__main__":
    main()
