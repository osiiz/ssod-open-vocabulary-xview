import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def calculate_mean_deviation(current_counts, global_cat_counts, ratios):
    deviations = []
    for cat_id, total in global_cat_counts.items():
        if total == 0:
            continue
        for s in ["train", "val", "test"]:
            actual_pct = current_counts[s][cat_id] / total
            target_pct = ratios[["train", "val", "test"].index(s)]
            deviations.append(abs(actual_pct - target_pct))
    return sum(deviations) / len(deviations) if deviations else 0


def perform_optimized_split(
    images: dict, annotations: list, ratios: tuple, seed: int, max_iters: int
):
    random.seed(seed)

    img_to_cats = defaultdict(lambda: defaultdict(int))
    global_cat_counts = defaultdict(int)

    for ann in annotations:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        img_to_cats[img_id][cat_id] += 1
        global_cat_counts[cat_id] += 1

    targets = {
        "train": {cat: count * ratios[0] for cat, count in global_cat_counts.items()},
        "val": {cat: count * ratios[1] for cat, count in global_cat_counts.items()},
        "test": {cat: count * ratios[2] for cat, count in global_cat_counts.items()},
    }

    current_counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }
    assignments = {}
    assigned_images = set()

    sorted_cats = sorted(global_cat_counts.keys(), key=lambda k: global_cat_counts[k])

    # Fase inicial: Algoritmo Voraz
    for cat_id in sorted_cats:
        valid_imgs = [
            i
            for i, cats in img_to_cats.items()
            if cats[cat_id] > 0 and i not in assigned_images
        ]
        valid_imgs.sort(key=lambda i: img_to_cats[i][cat_id], reverse=True)

        for img_id in valid_imgs:
            deficits = {
                s: targets[s][cat_id] - current_counts[s][cat_id]
                for s in ["train", "val", "test"]
            }
            best_split = max(deficits, key=lambda s: deficits[s])

            assignments[img_id] = best_split
            assigned_images.add(img_id)
            for cid, count in img_to_cats[img_id].items():
                current_counts[best_split][cid] += count

    unassigned = [img_id for img_id in images.keys() if img_id not in assigned_images]
    for img_id in unassigned:
        r = random.random()
        if r < ratios[0]:
            target_split = "train"
        elif r < ratios[0] + ratios[1]:
            target_split = "val"
        else:
            target_split = "test"

        assignments[img_id] = target_split
        assigned_images.add(img_id)

    best_score = calculate_mean_deviation(current_counts, global_cat_counts, ratios)

    # Fase de optimizacion: Búsqueda Local (Hill Climbing)
    if max_iters > 0:
        all_image_ids = list(images.keys())
        splits = ["train", "val", "test"]

        for _ in range(max_iters):
            img_id = random.choice(all_image_ids)
            old_split = assignments[img_id]
            new_split = random.choice([s for s in splits if s != old_split])

            # Aplicar intercambio temporal
            assignments[img_id] = new_split
            for cid, count in img_to_cats[img_id].items():
                current_counts[old_split][cid] -= count
                current_counts[new_split][cid] += count

            new_score = calculate_mean_deviation(
                current_counts, global_cat_counts, ratios
            )

            if new_score < best_score:
                best_score = new_score
            else:
                # Deshacer intercambio si no mejora
                assignments[img_id] = old_split
                for cid, count in img_to_cats[img_id].items():
                    current_counts[new_split][cid] -= count
                    current_counts[old_split][cid] += count

    # Formatear salida
    splits_img_ids = {"train": [], "val": [], "test": []}
    for img_id, s in assignments.items():
        splits_img_ids[s].append(img_id)

    return splits_img_ids


def stratified_split(
    coco_path: Path, out_dir: Path, ratios=(0.7, 0.1, 0.2), seed=42, max_iters=50000
):

    with open(coco_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    splits_img_ids = perform_optimized_split(
        images, annotations, ratios, seed, max_iters
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    for s_name in ["train", "val", "test"]:
        split_imgs = [images[img_id] for img_id in splits_img_ids[s_name]]
        split_img_ids_set = set(splits_img_ids[s_name])
        split_anns = [
            ann for ann in annotations if ann["image_id"] in split_img_ids_set
        ]

        split_coco = {
            "images": split_imgs,
            "annotations": split_anns,
            "categories": categories,
        }

        out_file = out_dir / f"xview_{s_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(split_coco, f)


def main():
    parser = argparse.ArgumentParser(
        description="Particion estratificada con busqueda local"
    )
    parser.add_argument(
        "input_json", type=str, help="Ruta al archivo COCO JSON original"
    )
    parser.add_argument(
        "out_dir", type=str, help="Directorio destino para volcar los 3 JSONs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.1, 0.2),
        help="Proporciones de particion (train, val, test)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10000,
        help="Iteraciones de optimizacion (0 para solo voraz)",
    )
    args = parser.parse_args()

    stratified_split(
        Path(args.input_json),
        Path(args.out_dir),
        tuple(args.ratios),
        args.seed,
        args.iters,
    )


if __name__ == "__main__":
    main()
