import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_nested_ratios(raw: str):
    ratios = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not ratios:
        raise ValueError("--nested_ratios no puede estar vacio")

    prev = -1.0
    for ratio in ratios:
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError("Cada nested ratio debe estar en el rango (0, 1)")
        if ratio <= prev:
            raise ValueError("Los nested ratios deben ser estrictamente crecientes")
        prev = ratio

    return ratios


def build_category_stats(annotations: list):
    img_to_cats = defaultdict(lambda: defaultdict(int))
    global_cat_counts = defaultdict(int)

    for ann in annotations:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        img_to_cats[img_id][cat_id] += 1
        global_cat_counts[cat_id] += 1

    return img_to_cats, global_cat_counts


def calculate_sampled_counts(sampled_image_ids, img_to_cats, global_cat_counts):
    sampled_counts = {cat_id: 0 for cat_id in global_cat_counts.keys()}
    for img_id in sampled_image_ids:
        for cat_id, count in img_to_cats.get(img_id, {}).items():
            sampled_counts[cat_id] += count
    return sampled_counts


def get_target_size(total_images: int, ratio: float) -> int:
    target_size = int(round(total_images * ratio))
    return max(1, min(total_images, target_size))


def apply_image_delta(sampled_counts, img_to_cats, img_id, sign):
    for cat_id, count in img_to_cats.get(img_id, {}).items():
        sampled_counts[cat_id] += sign * count


def calculate_mean_deviation(sampled_counts, global_cat_counts, ratio):
    deviations = []
    for cat_id, total in global_cat_counts.items():
        if total == 0:
            continue
        actual_pct = sampled_counts[cat_id] / total
        deviations.append(abs(actual_pct - ratio))
    return sum(deviations) / len(deviations) if deviations else 0


def perform_optimized_sample_from_stats(
    image_ids,
    img_to_cats,
    global_cat_counts,
    ratio: float,
    target_size: int,
    seed: int,
    max_iters: int,
    fixed_ids=None,
):
    rng = random.Random(seed)
    image_ids = list(image_ids)
    fixed_ids = set(fixed_ids or [])

    all_ids_set = set(image_ids)
    if not fixed_ids.issubset(all_ids_set):
        raise ValueError("fixed_ids contiene imagenes que no pertenecen al dataset")
    if len(fixed_ids) > target_size:
        raise ValueError("No se puede mantener el conjunto fijo para este ratio")

    remaining_candidates = [img_id for img_id in image_ids if img_id not in fixed_ids]
    remaining_needed = target_size - len(fixed_ids)

    sampled_set = set(fixed_ids)
    if remaining_needed > 0:
        sampled_set.update(rng.sample(remaining_candidates, remaining_needed))

    unsampled_set = all_ids_set - sampled_set

    movable_sampled = list(sampled_set - fixed_ids)
    unsampled_list = list(unsampled_set)

    sampled_counts = calculate_sampled_counts(
        sampled_set, img_to_cats, global_cat_counts
    )
    best_score = calculate_mean_deviation(sampled_counts, global_cat_counts, ratio)

    # Búsqueda local con swaps 1x1 conservando cardinalidad exacta del sample.
    if max_iters > 0 and movable_sampled and unsampled_list:
        for _ in range(max_iters):
            in_idx = rng.randrange(len(movable_sampled))
            out_idx = rng.randrange(len(unsampled_list))

            remove_id = movable_sampled[in_idx]
            add_id = unsampled_list[out_idx]

            apply_image_delta(sampled_counts, img_to_cats, remove_id, -1)
            apply_image_delta(sampled_counts, img_to_cats, add_id, +1)

            new_score = calculate_mean_deviation(
                sampled_counts, global_cat_counts, ratio
            )

            if new_score <= best_score:
                best_score = new_score
                sampled_set.remove(remove_id)
                sampled_set.add(add_id)
                unsampled_set.remove(add_id)
                unsampled_set.add(remove_id)

                movable_sampled[in_idx] = add_id
                unsampled_list[out_idx] = remove_id
            else:
                apply_image_delta(sampled_counts, img_to_cats, remove_id, +1)
                apply_image_delta(sampled_counts, img_to_cats, add_id, -1)

    sampled_images_ids = sorted(sampled_set)
    unlabeled_images_ids = sorted(unsampled_set)

    return sampled_images_ids, unlabeled_images_ids


def build_coco_split(
    images,
    annotations,
    categories,
    selected_image_ids,
    include_annotations,
):
    selected_ids_set = set(selected_image_ids)
    selected_images = [images[img_id] for img_id in selected_image_ids]
    selected_annotations = (
        [ann for ann in annotations if ann["image_id"] in selected_ids_set]
        if include_annotations
        else []
    )
    return {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": categories,
    }


def write_coco_file(path, coco_data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f)


def ratio_to_pct(ratio: float) -> int:
    return int(round(ratio * 100))


def stratified_sample(
    coco_path: Path,
    out_file_labeled: Path,
    out_file_unlabeled: Path,
    ratio=0.15,
    seed=42,
    max_iters=10000,
    nested_ratios=None,
):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    img_to_cats, global_cat_counts = build_category_stats(annotations)
    image_ids = list(images.keys())

    if nested_ratios:
        nested = sorted(nested_ratios)
        sampled_by_ratio = {}
        fixed_sampled_ids = set()

        for nested_ratio in nested:
            target_size = get_target_size(len(image_ids), nested_ratio)
            sampled_img_ids, unlabeled_img_ids = perform_optimized_sample_from_stats(
                image_ids,
                img_to_cats,
                global_cat_counts,
                nested_ratio,
                target_size,
                seed,
                max_iters,
                fixed_ids=fixed_sampled_ids,
            )
            fixed_sampled_ids = set(sampled_img_ids)
            sampled_by_ratio[nested_ratio] = (sampled_img_ids, unlabeled_img_ids)

            pct = ratio_to_pct(nested_ratio)
            labeled_nested = out_file_labeled.with_name(
                f"{out_file_labeled.stem}_{pct}{out_file_labeled.suffix}"
            )
            unlabeled_nested = out_file_unlabeled.with_name(
                f"{out_file_unlabeled.stem}_{pct}{out_file_unlabeled.suffix}"
            )

            write_coco_file(
                labeled_nested,
                build_coco_split(
                    images,
                    annotations,
                    categories,
                    sampled_img_ids,
                    include_annotations=True,
                ),
            )
            write_coco_file(
                unlabeled_nested,
                build_coco_split(
                    images,
                    annotations,
                    categories,
                    unlabeled_img_ids,
                    include_annotations=False,
                ),
            )

        # Salida principal: ratio solicitado (si no está en nested, usamos el último).
        active_ratio = nested[-1]
        for nested_ratio in nested:
            if abs(nested_ratio - ratio) < 1e-12:
                active_ratio = nested_ratio
                break

        sampled_img_ids, unlabeled_img_ids = sampled_by_ratio[active_ratio]

        write_coco_file(
            out_file_labeled,
            build_coco_split(
                images,
                annotations,
                categories,
                sampled_img_ids,
                include_annotations=True,
            ),
        )
        write_coco_file(
            out_file_unlabeled,
            build_coco_split(
                images,
                annotations,
                categories,
                unlabeled_img_ids,
                include_annotations=False,
            ),
        )

        print("Muestreo anidado completado:")
        for nested_ratio in nested:
            sampled_img_ids, unlabeled_img_ids = sampled_by_ratio[nested_ratio]
            pct = ratio_to_pct(nested_ratio)
            print(
                f"  ratio={nested_ratio:.2f}: labeled={len(sampled_img_ids)}/{len(image_ids)} "
                f"({out_file_labeled.stem}_{pct}{out_file_labeled.suffix}), "
                f"unlabeled={len(unlabeled_img_ids)} ({out_file_unlabeled.stem}_{pct}{out_file_unlabeled.suffix})"
            )
        return

    target_size = get_target_size(len(image_ids), ratio)
    sampled_img_ids, unlabeled_img_ids = perform_optimized_sample_from_stats(
        image_ids,
        img_to_cats,
        global_cat_counts,
        ratio,
        target_size,
        seed,
        max_iters,
    )

    write_coco_file(
        out_file_labeled,
        build_coco_split(
            images,
            annotations,
            categories,
            sampled_img_ids,
            include_annotations=True,
        ),
    )
    write_coco_file(
        out_file_unlabeled,
        build_coco_split(
            images,
            annotations,
            categories,
            unlabeled_img_ids,
            include_annotations=False,
        ),
    )

    print(
        "Muestreo estratificado completado. "
        f"Labeled: {len(sampled_img_ids)} imágenes, "
        f"Unlabeled: {len(unlabeled_img_ids)} imágenes."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Muestreo estratificado con búsqueda local para SSOD"
    )
    parser.add_argument(
        "input_json", type=str, help="Ruta al archivo COCO JSON original"
    )
    parser.add_argument(
        "output_labeled", type=str, help="Ruta destino para volcar el JSON muestreado"
    )
    parser.add_argument(
        "output_unlabeled",
        type=str,
        help="Ruta destino para volcar el JSON de imágenes no muestreadas",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.15,
        help="Proporción del muestreo",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10000,
        help="Iteraciones de optimización (0 para solo voraz)",
    )
    parser.add_argument(
        "--nested_ratios",
        type=str,
        default="",
        help="Ratios anidados separados por coma (ej: 0.10,0.20,0.30)",
    )
    args = parser.parse_args()

    if args.ratio <= 0 or args.ratio >= 1:
        raise ValueError("--ratio debe estar en el rango (0, 1)")

    nested_ratios = None
    if args.nested_ratios:
        nested_ratios = parse_nested_ratios(args.nested_ratios)

    stratified_sample(
        Path(args.input_json),
        Path(args.output_labeled),
        Path(args.output_unlabeled),
        ratio=args.ratio,
        seed=args.seed,
        max_iters=args.iters,
        nested_ratios=nested_ratios,
    )


if __name__ == "__main__":
    main()
