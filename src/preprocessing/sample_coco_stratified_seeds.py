import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def calculate_mean_deviation(sampled_counts, global_cat_counts, ratio):
    deviations = []
    for cat_id, total in global_cat_counts.items():
        if total == 0:
            continue
        actual_pct = sampled_counts[cat_id] / total
        deviations.append(abs(actual_pct - ratio))
    return sum(deviations) / len(deviations) if deviations else 0


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


def calculate_candidate_metrics(
    sampled_image_ids,
    sampled_counts,
    global_cat_counts,
    ratio,
    min_class_ratio,
    total_images,
):
    sampled_ratios = {}
    shortfalls = {}

    for cat_id, total in global_cat_counts.items():
        sampled_ratio = (sampled_counts[cat_id] / total) if total > 0 else 0.0
        sampled_ratios[cat_id] = sampled_ratio
        shortfalls[cat_id] = max(0.0, min_class_ratio - sampled_ratio)

    violating_classes = sum(1 for v in shortfalls.values() if v > 0)
    worst_shortfall = max(shortfalls.values()) if shortfalls else 0.0
    mean_abs_deviation = (
        sum(abs(v - ratio) for v in sampled_ratios.values()) / len(sampled_ratios)
        if sampled_ratios
        else 0.0
    )

    mean_deviation = calculate_mean_deviation(sampled_counts, global_cat_counts, ratio)

    sampled_image_ratio = (
        (len(sampled_image_ids) / total_images) if total_images > 0 else 0
    )
    image_ratio_gap = abs(sampled_image_ratio - ratio)

    return {
        "violating_classes": int(violating_classes),
        "worst_shortfall": float(worst_shortfall),
        "mean_abs_deviation": float(mean_abs_deviation),
        "mean_deviation": float(mean_deviation),
        "sampled_image_ratio": float(sampled_image_ratio),
        "image_ratio_gap": float(image_ratio_gap),
        "sampled_ratios": sampled_ratios,
        "shortfalls": shortfalls,
    }


def candidate_sort_key(candidate):
    metrics = candidate["metrics"]
    return (
        metrics["violating_classes"],
        metrics["worst_shortfall"],
        metrics["mean_abs_deviation"],
        metrics["mean_deviation"],
        metrics["image_ratio_gap"],
        candidate["seed"],
    )


def select_best_candidate(candidates):
    if not candidates:
        raise ValueError("No se generaron candidatos para seleccionar")
    return min(candidates, key=candidate_sort_key)


def parse_candidate_seeds(seed, num_candidates, candidate_seeds):
    if candidate_seeds:
        parsed = [int(s.strip()) for s in candidate_seeds.split(",") if s.strip()]
        # Conservamos el orden y eliminamos duplicados.
        return list(dict.fromkeys(parsed))

    n = max(1, int(num_candidates))
    return [seed + i for i in range(n)]


def build_coco_split(
    images, annotations, categories, selected_image_ids, include_annotations
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


def write_selection_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def perform_optimized_sample_from_stats(
    image_ids,
    img_to_cats,
    global_cat_counts,
    ratio: float,
    target_size: int,
    seed: int,
    max_iters: int,
):
    rng = random.Random(seed)
    image_ids = list(image_ids)

    if target_size > len(image_ids):
        raise ValueError("target_size no puede ser mayor que el número de imágenes")

    sampled_set = set(rng.sample(image_ids, target_size))
    unsampled_set = set(image_ids) - sampled_set

    sampled_list = list(sampled_set)
    unsampled_list = list(unsampled_set)

    sampled_counts = calculate_sampled_counts(
        sampled_set,
        img_to_cats,
        global_cat_counts,
    )
    best_score = calculate_mean_deviation(sampled_counts, global_cat_counts, ratio)

    # Búsqueda local con swaps 1x1, conservando cardinalidad exacta del sample.
    if max_iters > 0 and unsampled_list:
        for _ in range(max_iters):
            in_idx = rng.randrange(len(sampled_list))
            out_idx = rng.randrange(len(unsampled_list))

            remove_id = sampled_list[in_idx]
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

                sampled_list[in_idx] = add_id
                unsampled_list[out_idx] = remove_id
            else:
                apply_image_delta(sampled_counts, img_to_cats, remove_id, +1)
                apply_image_delta(sampled_counts, img_to_cats, add_id, -1)

    sampled_images_ids = sorted(sampled_set)
    unlabeled_images_ids = sorted(unsampled_set)

    return sampled_images_ids, unlabeled_images_ids


def perform_optimized_sample(
    images: dict, annotations: list, ratio: float, seed: int, max_iters: int
):
    img_to_cats, global_cat_counts = build_category_stats(annotations)
    image_ids = list(images.keys())
    target_size = get_target_size(len(image_ids), ratio)

    return perform_optimized_sample_from_stats(
        image_ids,
        img_to_cats,
        global_cat_counts,
        ratio,
        target_size,
        seed,
        max_iters,
    )


def stratified_sample(
    coco_path: Path,
    out_file_labeled: Path,
    out_file_unlabeled: Path,
    ratio=0.15,
    seed=42,
    max_iters=10000,
    num_candidates=1,
    candidate_seeds=None,
    min_class_ratio=None,
    class_ratio_tolerance=0.02,
    selection_report_path=None,
    candidates_dir=None,
):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    seeds = parse_candidate_seeds(seed, num_candidates, candidate_seeds)
    if min_class_ratio is None:
        min_class_ratio = max(0.0, ratio - class_ratio_tolerance)

    img_to_cats, global_cat_counts = build_category_stats(annotations)
    category_names = {cat["id"]: cat["name"] for cat in categories}

    candidates = []
    total_images = len(images)
    target_size = get_target_size(total_images, ratio)

    for candidate_seed in seeds:
        sampled_img_ids, unlabeled_img_ids = perform_optimized_sample_from_stats(
            images.keys(),
            img_to_cats,
            global_cat_counts,
            ratio,
            target_size,
            candidate_seed,
            max_iters,
        )

        sampled_counts = calculate_sampled_counts(
            sampled_img_ids,
            img_to_cats,
            global_cat_counts,
        )
        metrics = calculate_candidate_metrics(
            sampled_img_ids,
            sampled_counts,
            global_cat_counts,
            ratio,
            min_class_ratio,
            total_images=total_images,
        )

        candidate = {
            "seed": candidate_seed,
            "sampled_image_ids": sampled_img_ids,
            "unlabeled_image_ids": unlabeled_img_ids,
            "metrics": metrics,
        }
        candidates.append(candidate)

        if candidates_dir:
            candidates_path = Path(candidates_dir)
            labeled_candidate_path = (
                candidates_path
                / f"{out_file_labeled.stem}_seed{candidate_seed}{out_file_labeled.suffix}"
            )
            unlabeled_candidate_path = (
                candidates_path
                / f"{out_file_unlabeled.stem}_seed{candidate_seed}{out_file_unlabeled.suffix}"
            )

            labeled_candidate_coco = build_coco_split(
                images,
                annotations,
                categories,
                sampled_img_ids,
                include_annotations=True,
            )
            unlabeled_candidate_coco = build_coco_split(
                images,
                annotations,
                categories,
                unlabeled_img_ids,
                include_annotations=False,
            )
            write_coco_file(labeled_candidate_path, labeled_candidate_coco)
            write_coco_file(unlabeled_candidate_path, unlabeled_candidate_coco)

    best_candidate = select_best_candidate(candidates)

    sampled_img_ids = best_candidate["sampled_image_ids"]
    unlabeled_img_ids = best_candidate["unlabeled_image_ids"]

    split_coco = build_coco_split(
        images,
        annotations,
        categories,
        sampled_img_ids,
        include_annotations=True,
    )
    unlabeled_coco = build_coco_split(
        images,
        annotations,
        categories,
        unlabeled_img_ids,
        include_annotations=False,
    )

    write_coco_file(out_file_labeled, split_coco)
    write_coco_file(out_file_unlabeled, unlabeled_coco)

    if selection_report_path:
        report_candidates = []
        for candidate in sorted(candidates, key=candidate_sort_key):
            metrics = candidate["metrics"]
            retention_by_category = {
                category_names.get(cat_id, str(cat_id)): round(value, 5)
                for cat_id, value in metrics["sampled_ratios"].items()
            }
            shortfall_by_category = {
                category_names.get(cat_id, str(cat_id)): round(value, 5)
                for cat_id, value in metrics["shortfalls"].items()
            }
            report_candidates.append(
                {
                    "seed": candidate["seed"],
                    "violating_classes": metrics["violating_classes"],
                    "worst_shortfall": round(metrics["worst_shortfall"], 6),
                    "mean_abs_deviation": round(metrics["mean_abs_deviation"], 6),
                    "mean_deviation": round(metrics["mean_deviation"], 6),
                    "sampled_image_ratio": round(metrics["sampled_image_ratio"], 6),
                    "image_ratio_gap": round(metrics["image_ratio_gap"], 6),
                    "retention_by_category": retention_by_category,
                    "shortfall_by_category": shortfall_by_category,
                }
            )

        report = {
            "ratio": ratio,
            "min_class_ratio": min_class_ratio,
            "class_ratio_tolerance": class_ratio_tolerance,
            "candidate_seeds": seeds,
            "selected_seed": best_candidate["seed"],
            "selected_metrics": {
                "violating_classes": best_candidate["metrics"]["violating_classes"],
                "worst_shortfall": round(
                    best_candidate["metrics"]["worst_shortfall"], 6
                ),
                "mean_abs_deviation": round(
                    best_candidate["metrics"]["mean_abs_deviation"], 6
                ),
                "mean_deviation": round(best_candidate["metrics"]["mean_deviation"], 6),
            },
            "candidates": report_candidates,
        }
        write_selection_report(Path(selection_report_path), report)

    selected_metrics = best_candidate["metrics"]
    print(
        "Muestreo estratificado completado. "
        f"Semilla seleccionada: {best_candidate['seed']}. "
        f"Candidatos evaluados: {len(candidates)}. "
        f"Clases fuera del mínimo: {selected_metrics['violating_classes']}. "
        f"Labeled: {len(sampled_img_ids)} imágenes, "
        f"Unlabeled: {len(unlabeled_img_ids)} imágenes."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Muestreo estratificado con búsqueda local para SSOD (ratio estricto por número de imágenes)"
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
        help="Proporción objetivo estricta de imágenes en el conjunto labeled",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10000,
        help="Iteraciones de optimización (0 para solo voraz)",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="Número de candidatos a evaluar (usa seed, seed+1, ...)",
    )
    parser.add_argument(
        "--candidate_seeds",
        type=str,
        default="",
        help="Lista opcional de semillas separadas por coma (ej: 42,43,44)",
    )
    parser.add_argument(
        "--class_ratio_tolerance",
        type=float,
        default=0.02,
        help="Tolerancia absoluta respecto a ratio para el mínimo por clase",
    )
    parser.add_argument(
        "--min_class_ratio",
        type=float,
        default=None,
        help="Mínimo absoluto de retención por clase en sampled (si no, ratio - class_ratio_tolerance)",
    )
    parser.add_argument(
        "--selection_report",
        type=str,
        default="",
        help="Ruta opcional para guardar el ranking de candidatos en JSON",
    )
    parser.add_argument(
        "--candidates_dir",
        type=str,
        default="",
        help="Directorio opcional para exportar cada candidato con sufijo por semilla",
    )
    args = parser.parse_args()

    if args.ratio <= 0 or args.ratio >= 1:
        raise ValueError("--ratio debe estar en el rango (0, 1)")
    if args.class_ratio_tolerance < 0:
        raise ValueError("--class_ratio_tolerance no puede ser negativo")
    if args.min_class_ratio is not None and (
        args.min_class_ratio < 0 or args.min_class_ratio > 1
    ):
        raise ValueError("--min_class_ratio debe estar en [0, 1]")

    stratified_sample(
        Path(args.input_json),
        Path(args.output_labeled),
        Path(args.output_unlabeled),
        ratio=args.ratio,
        seed=args.seed,
        max_iters=args.iters,
        num_candidates=args.num_candidates,
        candidate_seeds=args.candidate_seeds if args.candidate_seeds else None,
        min_class_ratio=args.min_class_ratio,
        class_ratio_tolerance=args.class_ratio_tolerance,
        selection_report_path=(
            args.selection_report if args.selection_report else None
        ),
        candidates_dir=(args.candidates_dir if args.candidates_dir else None),
    )


if __name__ == "__main__":
    main()
