"""
Selecciona 100 imagenes estratificadas de train_unlabeled_eval para el benchmark
multi-detector. Garantiza que las 9 clases esten representadas y que la distribucion
por clase sea proporcional al dataset completo.

Reutiliza la logica de sample_coco_stratified.py (hill-climbing).

Uso
---
    python -m src.utils.select_benchmark_images \
        --ann_file results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \
        --output_dir results/benchmark \
        --n_images 100 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.preprocessing.sample_coco_stratified import (
    build_category_stats,
    build_coco_split,
    calculate_mean_deviation,
    calculate_sampled_counts,
    perform_optimized_sample_from_stats,
)


def select_benchmark_images(
    ann_file: Path,
    output_dir: Path,
    n_images: int = 100,
    seed: int = 42,
    max_iters: int = 10000,
) -> list[int]:
    with ann_file.open(encoding="utf-8") as fh:
        coco = json.load(fh)

    images = {m["id"]: m for m in coco["images"]}
    categories = coco["categories"]
    annotations = coco["annotations"]

    img_to_cats, global_cat_counts = build_category_stats(annotations)

    # Imagenes que tienen al menos una anotacion
    annotated_ids = sorted(img_to_cats.keys())
    total = len(annotated_ids)
    ratio = n_images / total

    print(f"Dataset: {total} imagenes con anotaciones, {len(annotations)} instancias")
    print(f"Clases: {len(categories)}")
    print(f"Objetivo: {n_images} imagenes (ratio={ratio:.4f})")

    # Garantizar cobertura de todas las clases: incluir al menos 1 imagen por clase
    cat_ids = [c["id"] for c in categories]
    fixed_ids: set[int] = set()
    img_to_cats_list = defaultdict(set)
    for ann in annotations:
        img_to_cats_list[ann["image_id"]].add(ann["category_id"])

    for cat_id in cat_ids:
        # Imagen con mas instancias de esta clase (para coverage maxima)
        best_img = max(
            (img_id for img_id in annotated_ids if cat_id in img_to_cats_list[img_id]),
            key=lambda iid: img_to_cats[iid].get(cat_id, 0),
            default=None,
        )
        if best_img is not None:
            fixed_ids.add(best_img)

    print(f"Imagenes fijas para cobertura de clases: {len(fixed_ids)}")
    assert (
        len(fixed_ids) <= n_images
    ), "No hay suficientes imagenes para cubrir todas las clases"

    selected_ids, _ = perform_optimized_sample_from_stats(
        image_ids=annotated_ids,
        img_to_cats=img_to_cats,
        global_cat_counts=global_cat_counts,
        ratio=ratio,
        target_size=n_images,
        seed=seed,
        max_iters=max_iters,
        fixed_ids=fixed_ids,
    )

    # Estadisticas de la muestra
    sampled_counts = calculate_sampled_counts(
        selected_ids, img_to_cats, global_cat_counts
    )
    deviation = calculate_mean_deviation(sampled_counts, global_cat_counts, ratio)
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    print(f"\nImagenes seleccionadas: {len(selected_ids)}")
    print(f"Desviacion media de proporcion: {deviation:.4f}")
    print(
        f"\n{'Clase':<25} {'Total':>8} {'Muestra':>8} {'% global':>10} {'% muestra':>10}"
    )
    print("-" * 65)
    for cat_id in sorted(global_cat_counts):
        total_c = global_cat_counts[cat_id]
        sample_c = sampled_counts.get(cat_id, 0)
        pct_g = 100 * total_c / sum(global_cat_counts.values())
        pct_s = (
            100 * sample_c / sum(sampled_counts.values())
            if sum(sampled_counts.values())
            else 0
        )
        print(
            f"{cat_id_to_name[cat_id]:<25} {total_c:>8} {sample_c:>8} {pct_g:>9.2f}% {pct_s:>9.2f}%"
        )

    # Verificar cobertura
    covered = set()
    for img_id in selected_ids:
        covered.update(img_to_cats_list[img_id])
    missing = set(cat_ids) - covered
    if missing:
        print(
            f"\nATENCION: clases sin cobertura: {[cat_id_to_name[c] for c in missing]}"
        )
    else:
        print(f"\nTodas las {len(cat_ids)} clases estan cubiertas.")

    # Guardar outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    ids_path = output_dir / "image_ids.json"
    with ids_path.open("w", encoding="utf-8") as fh:
        json.dump(selected_ids, fh)
    print(f"\nIDs guardados → {ids_path}")

    gt_path = output_dir / "benchmark_100_gt.json"
    gt_coco = build_coco_split(
        images=images,
        annotations=annotations,
        categories=categories,
        selected_image_ids=selected_ids,
        include_annotations=True,
    )
    with gt_path.open("w", encoding="utf-8") as fh:
        json.dump(gt_coco, fh)
    print(f"COCO GT guardado → {gt_path}")
    print(
        f"  {len(gt_coco['images'])} imagenes, {len(gt_coco['annotations'])} anotaciones"
    )

    return selected_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_file",
        type=Path,
        default=Path(
            "results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json"
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iters", type=int, default=10000)
    args = parser.parse_args()

    select_benchmark_images(
        ann_file=args.ann_file,
        output_dir=args.output_dir,
        n_images=args.n_images,
        seed=args.seed,
        max_iters=args.max_iters,
    )


if __name__ == "__main__":
    main()
