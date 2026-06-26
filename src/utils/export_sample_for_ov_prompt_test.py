"""
Exporta una tile del dataset (como PNG) junto con las categorías GT presentes,
para probar prompts en la web de Rex-Omni / Grounding DINO.

Uso:
    python -m src.utils.export_sample_for_ov_prompt_test \
        --ann_file results/preprocess/tile_images/train_sampled/COCO_annotations.json \
        --img_dir  results/preprocess/tile_images/train_sampled/images \
        --output   /tmp/sample_tile.png \
        [--filename img_825_0_1400.tif]   # opcional: fija una imagen concreta
        [--min_categories 3]              # mínimo de categorías distintas presentes
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", type=Path, required=True)
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/sample_tile.png"))
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--min_categories", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with args.ann_file.open() as fh:
        coco = json.load(fh)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs_by_id = {i["id"]: i for i in coco["images"]}
    imgs_by_name = {i["file_name"]: i for i in coco["images"]}

    img_cats: dict[int, set[str]] = defaultdict(set)
    for ann in coco["annotations"]:
        img_cats[ann["image_id"]].add(cats[ann["category_id"]])

    if args.filename:
        img_meta = imgs_by_name.get(args.filename)
        if img_meta is None:
            raise ValueError(f"No se encontró '{args.filename}' en {args.ann_file}")
        img_id = img_meta["id"]
    else:
        candidates = [
            iid for iid, cs in img_cats.items() if len(cs) >= args.min_categories
        ]
        if not candidates:
            raise ValueError(
                f"No hay imágenes con al menos {args.min_categories} categorías distintas."
            )
        img_id = random.choice(candidates)
        img_meta = imgs_by_id[img_id]

    present_cats = sorted(img_cats[img_id])
    img_path = args.img_dir / img_meta["file_name"]
    img = Image.open(img_path).convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)

    print(f"\nImagen exportada: {args.output}")
    print(f"Archivo original: {img_meta['file_name']}")
    print(f"Tamaño: {img.width}×{img.height} px")
    print(f"\nCategorías GT presentes ({len(present_cats)}):")
    for cat in present_cats:
        n = sum(
            1
            for a in coco["annotations"]
            if a["image_id"] == img_id and cats[a["category_id"]] == cat
        )
        print(f"  - {cat}: {n} instancias")

    print("\n--- PROMPTS PARA PROBAR ---")
    print("\n[PROMPTS SIMPLES — sin contexto de dominio]")
    simple = " . ".join(present_cats)
    print(f"  {simple}")

    print("\n[PROMPTS CON CONTEXTO AÉREO — satélite/aerial view]")
    aerial_map = {
        "Aircraft": "aircraft seen from above . airplane on the ground seen from satellite",
        "Light Vehicle": "small car seen from above . light vehicle seen from satellite",
        "Heavy Vehicle": "truck seen from above . large vehicle seen from satellite",
        "Railway Vehicle": "train seen from above . railroad vehicle seen from satellite",
        "Maritime Vessel": "boat seen from above . ship seen from satellite",
        "Engineering Vehicle": "construction vehicle seen from above . bulldozer seen from satellite",
        "Building": "building rooftop seen from above . building seen from satellite",
        "Storage Tank": "circular storage tank seen from above . oil tank seen from satellite",
        "Tower & Pylon": "tower seen from above . pylon seen from satellite",
    }
    aerial = " . ".join(
        aerial_map.get(cat, cat + " seen from above") for cat in present_cats
    )
    print(f"  {aerial}")

    print("\n[PROMPTS CON CONTEXTO COMPACTO — una frase por clase]")
    compact_map = {
        "Aircraft": "aerial view of aircraft",
        "Light Vehicle": "aerial view of car",
        "Heavy Vehicle": "aerial view of truck",
        "Railway Vehicle": "aerial view of train",
        "Maritime Vessel": "aerial view of ship",
        "Engineering Vehicle": "aerial view of construction vehicle",
        "Building": "aerial view of building",
        "Storage Tank": "aerial view of storage tank",
        "Tower & Pylon": "aerial view of tower",
    }
    compact = " . ".join(
        compact_map.get(cat, "aerial view of " + cat.lower()) for cat in present_cats
    )
    print(f"  {compact}")


if __name__ == "__main__":
    main()
