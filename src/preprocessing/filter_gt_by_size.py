#!/usr/bin/env python3
"""
Filtra un COCO JSON eliminando anotaciones cuya area (w*h) sea menor que min_area.

Para splits de entrenamiento (train, train_sampled) se eliminan tambien las imagenes
que quedan sin ninguna anotacion tras el filtro. Para val/test se conservan todas las
imagenes para no sesgar la evaluacion.

Las imagenes fisicas NO se copian: los paths de fichero en el COCO de salida siguen
apuntando al directorio de imagenes original.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


TRAIN_SPLITS = {"train", "train_sampled", "train_unlabeled_eval"}


def filter_gt(
    input_path: Path,
    output_path: Path,
    min_area: int,
    drop_empty_images: bool,
) -> None:
    with open(input_path) as f:
        gt = json.load(f)

    total_before = len(gt["annotations"])
    cats = {c["id"]: c["name"] for c in gt["categories"]}

    before_per_class: dict[int, int] = defaultdict(int)
    after_per_class: dict[int, int] = defaultdict(int)

    kept_anns = []
    for ann in gt["annotations"]:
        before_per_class[ann["category_id"]] += 1
        w, h = ann["bbox"][2], ann["bbox"][3]
        if w * h >= min_area:
            kept_anns.append(ann)
            after_per_class[ann["category_id"]] += 1

    if drop_empty_images:
        kept_img_ids = {a["image_id"] for a in kept_anns}
        kept_imgs = [img for img in gt["images"] if img["id"] in kept_img_ids]
    else:
        kept_imgs = gt["images"]

    total_after = len(kept_anns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "images": kept_imgs,
                "annotations": kept_anns,
                "categories": gt["categories"],
            },
            f,
        )

    print(f"min_area={min_area}px2  drop_empty={drop_empty_images}")
    print(
        f"  annotations: {total_before} → {total_after}  ({100*total_after/total_before:.1f}% retenido)"
    )
    print(f"  imagenes:    {len(gt['images'])} → {len(kept_imgs)}")
    print()
    print(f"  {'clase':<25}  {'antes':>7}  {'despues':>8}  {'% ret':>6}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*8}  {'-'*6}")
    for cat_id in sorted(cats):
        b = before_per_class.get(cat_id, 0)
        a = after_per_class.get(cat_id, 0)
        pct = 100 * a / b if b else 0.0
        print(f"  {cats[cat_id]:<25}  {b:>7}  {a:>8}  {pct:>5.1f}%")
    print()
    print(f"Escrito en: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min_area", required=True, type=int)
    parser.add_argument(
        "--drop_empty_images",
        action="store_true",
        default=None,
        help="Eliminar imagenes sin anotaciones tras el filtro (por defecto: True para splits de train)",
    )
    parser.add_argument(
        "--keep_empty_images",
        action="store_true",
        help="Conservar todas las imagenes aunque no tengan anotaciones",
    )
    args = parser.parse_args()

    if args.keep_empty_images:
        drop_empty = False
    elif args.drop_empty_images:
        drop_empty = True
    else:
        # Heuristica: si el path contiene un split de train, eliminar imagenes vacias
        split = args.input.parent.name
        drop_empty = split in TRAIN_SPLITS

    filter_gt(args.input, args.output, args.min_area, drop_empty)


if __name__ == "__main__":
    main()
