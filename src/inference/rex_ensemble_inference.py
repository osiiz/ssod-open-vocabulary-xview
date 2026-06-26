"""
Inferencia por ensamble de Rex-Omni: ejecuta cada conjunto de prompts de
ensemble_prompts.yaml sobre las imágenes objetivo y produce un único
detection_results.json donde cada detección lleva la frase coincidente como
`rex_label` y el conjunto de prompts de origen como `prompt_set`. Este fichero
es la entrada directa para multi_prompt_ensemble.py.

Rex-Omni no tiene límite de tokens relevante para nuestros conjuntos de prompts,
así que cada conjunto se ejecuta en un único pase de inferencia sin división.

Uso
---
    python -m src.inference.rex_ensemble_inference \\
        --img_dir   results/preprocess/tile_images/train_unlabeled_eval/images \\
        --ann_file  results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \\
        --config    configs/prompts/ensemble_prompts.yaml \\
        --output    results/rexomni/ensemble_raw/detection_results.json \\
        --model_id  IDEA-Research/Rex-Omni
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

from src.inference.multi_prompt_ensemble import get_prompt_sets
from src.inference.rex_inference import parse_rexomni_detection_output


def run_rex_ensemble_inference(
    img_dir: Path,
    ann_file: Path,
    config_path: Path,
    output_path: Path,
    model_id: str = "IDEA-Research/Rex-Omni",
    batch_size: int = 4,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.05,
    top_k: int = 1,
    repetition_penalty: float = 1.05,
) -> None:
    # ---- cargar configuración ----
    prompt_sets = get_prompt_sets(config_path)

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    phrase_to_class: dict[str, str] = {}
    for set_classes in cfg.get("prompt_sets", {}).values():
        for class_name, phrases in (set_classes or {}).items():
            for phrase in phrases or []:
                phrase_to_class[str(phrase).lower().strip()] = class_name

    # ---- cargar imágenes ----
    with ann_file.open(encoding="utf-8") as fh:
        coco = json.load(fh)
    images_meta = [m for m in coco["images"] if (img_dir / m["file_name"]).exists()]
    print(f"Images to process: {len(images_meta)}")

    # ---- cargar Rex-Omni ----
    print(f"Loading Rex-Omni from {model_id} ...")
    sys.path.insert(0, str(Path("vendor/Rex-Omni")))
    from rex_omni import RexOmniWrapper  # noqa: PLC0415

    model = RexOmniWrapper(
        model_path=model_id,
        backend="transformers",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        attn_implementation="eager",
        torch_dtype="float16",
        device_map="auto",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint: saltar conjuntos ya guardados y reanudar desde donde se dejó
    completed_sets: set[str] = set()
    all_detections: list[dict] = []
    for set_name in prompt_sets:
        ckpt = output_path.parent / f"_ckpt_{set_name}.json"
        if ckpt.exists():
            with ckpt.open(encoding="utf-8") as fh:
                set_dets = json.load(fh)
            all_detections.extend(set_dets)
            completed_sets.add(set_name)
            print(
                f"  [checkpoint] '{set_name}' already done ({len(set_dets)} dets), skipping."
            )

    for set_name, set_classes in prompt_sets.items():
        if set_name in completed_sets:
            continue

        all_phrases = [phrase for phrases in set_classes.values() for phrase in phrases]
        print(f"\n=== Set '{set_name}': {len(all_phrases)} phrases ===")
        n_batch = (len(images_meta) + batch_size - 1) // batch_size

        set_detections: list[dict] = []
        for b_idx in range(0, len(images_meta), batch_size):
            batch_meta = images_meta[b_idx : b_idx + batch_size]
            batch_imgs = [
                Image.open(img_dir / m["file_name"]).convert("RGB") for m in batch_meta
            ]

            raw_batch = model.inference(
                images=batch_imgs,
                task="detection",
                categories=all_phrases,
                return_scores=True,
            )

            for img_meta, img_pil, raw in zip(batch_meta, batch_imgs, raw_batch):
                W, H = img_pil.size
                parsed = parse_rexomni_detection_output(raw, image_size=(W, H))
                for det in parsed:
                    x1, y1, x2, y2 = det["box_xyxy"]
                    w, h = x2 - x1, y2 - y1
                    raw_label = str(det.get("label", "")).lower().strip()
                    macro = phrase_to_class.get(raw_label, "")
                    set_detections.append(
                        {
                            "image_id": img_meta["id"],
                            "category_id": -1,
                            "bbox": [
                                round(x1, 2),
                                round(y1, 2),
                                round(w, 2),
                                round(h, 2),
                            ],
                            "score": round(float(det.get("score", 1.0)), 6),
                            "rex_label": raw_label,
                            "prompt_set": set_name,
                            "class_name": macro,
                        }
                    )

            bi = b_idx // batch_size + 1
            print(f"  batch {bi}/{n_batch}  set dets: {len(set_detections)}", end="\r")
        print()

        # Guardar checkpoint antes de pasar al siguiente conjunto
        ckpt = output_path.parent / f"_ckpt_{set_name}.json"
        with ckpt.open("w", encoding="utf-8") as fh:
            json.dump(set_detections, fh)
        print(f"  [checkpoint] '{set_name}' saved → {ckpt}")

        all_detections.extend(set_detections)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(all_detections, fh)
    print(f"\nSaved {len(all_detections)} raw detections → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument("--ann_file", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/prompts/ensemble_prompts.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_id", default="IDEA-Research/Rex-Omni")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_rex_ensemble_inference(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        config_path=args.config,
        output_path=args.output,
        model_id=args.model_id,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
