"""
Inferencia por ensamble de Grounding DINO: ejecuta cada conjunto de prompts de
ensemble_prompts.yaml sobre las imágenes objetivo y produce un único
detection_results.json donde cada detección lleva la frase coincidente como
`dino_label` y el conjunto de prompts de origen como `prompt_set`. Este fichero
es la entrada directa para multi_prompt_ensemble.py.

Gestión del límite de tokens
-----------------------------
El codificador de texto de Grounding DINO tiene un límite estricto (~256 tokens).
El bucle principal itera por clase (no por conjunto de prompts): cada pase agrupa
todas las frases de esa clase procedentes de todos los conjuntos. Si un grupo
supera el límite, se divide automáticamente en el mínimo número de subgrupos que
caben. Las detecciones de todos los subgrupos se fusionan; la división es
transparente para el código posterior.

Uso
---
    python -m src.inference.dino_ensemble_inference \\
        --img_dir   results/preprocess/tile_images/train_unlabeled_eval/images \\
        --ann_file  results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \\
        --config    configs/prompts/ensemble_prompts.yaml \\
        --output    results/dino/ensemble_raw/detection_results.json \\
        --model_id  IDEA-Research/grounding-dino-base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.inference.multi_prompt_ensemble import get_prompt_sets


# ---------------------------------------------------------------------------
# Helpers para el límite de tokens
# ---------------------------------------------------------------------------


def _phrases_to_prompt(phrases: list[str]) -> str:
    return " . ".join(p.strip().lower() for p in phrases) + " ."


def _split_for_token_limit(
    phrases: list[str],
    tokenizer,
    max_tokens: int = 240,
) -> list[list[str]]:
    """
    Divide phrases recursivamente en el mínimo número de grupos tal que el
    texto de prompt de cada grupo cabe en max_tokens.
    """
    if not phrases:
        return []
    prompt_text = _phrases_to_prompt(phrases)
    if len(tokenizer.encode(prompt_text)) <= max_tokens:
        return [phrases]
    mid = len(phrases) // 2
    return _split_for_token_limit(
        phrases[:mid], tokenizer, max_tokens
    ) + _split_for_token_limit(phrases[mid:], tokenizer, max_tokens)


# ---------------------------------------------------------------------------
# Inferencia DINO de un único pase (devuelve etiquetas + puntuaciones + cajas)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _run_dino_pass(
    processor,
    model,
    images: list[Image.Image],
    image_ids: list[int],
    phrases: list[str],
    score_thresh: float,
    text_thresh: float,
    device: torch.device,
    prompt_set_name: str,
    phrase_to_class: dict[str, str],
) -> list[dict]:
    """
    Ejecuta un pase hacia adelante de DINO con phrases sobre images.
    Devuelve lista plana de detecciones en formato COCO con campos extra:
      dino_label, prompt_set.
    """
    prompt_text = _phrases_to_prompt(phrases)
    target_sizes = [(img.height, img.width) for img in images]
    inputs = processor(
        images=images,
        text=[prompt_text] * len(images),
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=score_thresh,
        text_threshold=text_thresh,
        target_sizes=target_sizes,
    )

    detections: list[dict] = []
    for img_id, result in zip(image_ids, results):
        W, H = result.get("target_size", (1, 1)) if "target_size" in result else (1, 1)
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        # transformers >=4.51: "labels" devuelve ids enteros; "text_labels" tiene cadenas de texto
        labels = result.get("text_labels") or result.get("labels", [])

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (float(v) for v in box)
            w, h = x2 - x1, y2 - y1
            raw_label = str(label).lower().strip()
            macro = phrase_to_class.get(raw_label, "")
            detections.append(
                {
                    "image_id": img_id,
                    "category_id": -1,  # se resuelve tras el ensamble
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(float(score), 6),
                    "dino_label": raw_label,
                    "prompt_set": prompt_set_name,
                    "class_name": macro,
                }
            )

    return detections


# ---------------------------------------------------------------------------
# Bucle principal de inferencia
# ---------------------------------------------------------------------------


def run_dino_ensemble_inference(
    img_dir: Path,
    ann_file: Path,
    config_path: Path,
    output_path: Path,
    model_id: str = "IDEA-Research/grounding-dino-base",
    score_thresh: float = 0.05,
    text_thresh: float = 0.25,
    batch_size: int = 4,
    max_tokens: int = 240,
    device_str: str = "cuda",
) -> None:
    # ---- cargar configuración ----
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Pivotar config: {class_name: [todas las frases de todos los conjuntos]}
    # Procesar una clase por pase garantiza que cada caja detectada tenga una
    # etiqueta de clase inequívoca — sin confusión entre prompts multi-clase.
    class_phrases: dict[str, list[str]] = {}
    for set_classes in cfg.get("prompt_sets", {}).values():
        for class_name, phrases in (set_classes or {}).items():
            for phrase in phrases or []:
                class_phrases.setdefault(class_name, []).append(str(phrase))

    # ---- cargar imágenes ----
    with ann_file.open(encoding="utf-8") as fh:
        coco = json.load(fh)
    images_meta = [m for m in coco["images"] if (img_dir / m["file_name"]).exists()]
    print(f"Images to process: {len(images_meta)}")

    # ---- cargar modelo ----
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_id} on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint: saltar clases ya guardadas y reanudar desde donde se dejó
    completed_classes: set[str] = set()
    all_detections: list[dict] = []
    for class_name in class_phrases:
        ckpt_key = class_name.lower().replace(" ", "_").replace("&", "and")
        ckpt = output_path.parent / f"_ckpt_{ckpt_key}.json"
        if ckpt.exists():
            with ckpt.open(encoding="utf-8") as fh:
                cls_dets = json.load(fh)
            all_detections.extend(cls_dets)
            completed_classes.add(class_name)
            print(
                f"  [checkpoint] '{class_name}' already done ({len(cls_dets)} dets), skipping."
            )

    for class_name, phrases in sorted(class_phrases.items()):
        if class_name in completed_classes:
            continue

        # Cada pase usa solo las frases de esta clase → la etiqueta siempre apunta a class_name
        phrase_to_class = {p.lower().strip(): class_name for p in phrases}

        subgroups = _split_for_token_limit(phrases, processor.tokenizer, max_tokens)
        n_splits = len(subgroups)
        split_info = f" (auto-split into {n_splits} groups)" if n_splits > 1 else ""
        print(f"\n=== Class '{class_name}': {len(phrases)} phrases{split_info} ===")

        cls_detections: list[dict] = []
        for sg_idx, subgroup in enumerate(subgroups, start=1):
            if n_splits > 1:
                print(f"  Sub-group {sg_idx}/{n_splits}: {len(subgroup)} phrases")
            n_batch = (len(images_meta) + batch_size - 1) // batch_size
            for b_idx in range(0, len(images_meta), batch_size):
                batch_meta = images_meta[b_idx : b_idx + batch_size]
                batch_imgs = [
                    Image.open(img_dir / m["file_name"]).convert("RGB")
                    for m in batch_meta
                ]
                batch_ids = [m["id"] for m in batch_meta]

                dets = _run_dino_pass(
                    processor,
                    model,
                    images=batch_imgs,
                    image_ids=batch_ids,
                    phrases=subgroup,
                    score_thresh=score_thresh,
                    text_thresh=text_thresh,
                    device=device,
                    prompt_set_name=class_name,
                    phrase_to_class=phrase_to_class,
                )
                # Forzar class_name incondicionalmente — el pase es de una sola clase,
                # por lo que cada caja detectada pertenece a esta clase sin importar la etiqueta
                for det in dets:
                    det["class_name"] = class_name
                cls_detections.extend(dets)

                bi = b_idx // batch_size + 1
                print(
                    f"    batch {bi}/{n_batch}  +{len(dets)} dets"
                    f" (class total {len(cls_detections)})",
                    end="\r",
                )
            print()

        # Guardar checkpoint de esta clase antes de pasar a la siguiente
        ckpt_key = class_name.lower().replace(" ", "_").replace("&", "and")
        ckpt = output_path.parent / f"_ckpt_{ckpt_key}.json"
        with ckpt.open("w", encoding="utf-8") as fh:
            json.dump(cls_detections, fh)
        print(f"  [checkpoint] '{class_name}' saved → {ckpt}")

        all_detections.extend(cls_detections)

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
    parser.add_argument("--model_id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--score_thresh", type=float, default=0.05)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=240)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dino_ensemble_inference(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        config_path=args.config,
        output_path=args.output,
        model_id=args.model_id,
        score_thresh=args.score_thresh,
        text_thresh=args.text_thresh,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
