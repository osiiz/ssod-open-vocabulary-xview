"""
Inferencia DINO ensemble con asignacion de clase por argmax sobre pred_logits.

En lugar de lanzar un pase por clase (9 pases), lanza un pase por prompt set
(5 pases) con todas las clases en el prompt. La clase de cada caja detectada
se asigna por argmax sobre las puntuaciones por clase derivadas directamente de
outputs.logits [B, Q, 256], evitando el decodificado de etiquetas de texto que
colapsa con umbrales bajos.

Ventajas frente al enfoque de 9 pases por clase:
  - 5 pases en lugar de 9 (~45% menos computo).
  - Sin detecciones cross-class: cada caja recibe una unica clase por pase, lo
    que evita que falsos positivos de clases incorrectas corrompan el voto de
    agregacion downstream.

Uso
---
    python -m src.inference.dino_ensemble_inference_argmax \\
        --img_dir   results/preprocess/tile_images/train_unlabeled_eval/images \\
        --ann_file  results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \\
        --config    configs/prompts/ensemble_prompts.yaml \\
        --output    results/dino/ensemble_argmax_raw/detection_results.json \\
        --model_id  IDEA-Research/grounding-dino-base
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.inference.multi_prompt_ensemble import get_prompt_sets

# Token id del separador "." en el vocabulario BERT
_PERIOD_TOKEN_ID = 1012


# ---------------------------------------------------------------------------
# Construccion del mapa token → frase y frase → clase
# ---------------------------------------------------------------------------


def _build_phrase_masks(
    input_ids: list[int],
    phrases_ordered: list[tuple[str, str]],
    class_names: list[str],
    max_len: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construye mascaras de token a nivel de frase individual.

    Devuelve:
      phrase_masks    [n_phrases, max_len] bool  — tokens de cada frase
      phrase_cls_idx  [n_phrases]          int   — indice de clase de cada frase

    La agregacion posterior es: max dentro de cada frase (score de frase),
    luego media entre frases de la misma clase (score de clase). Esto evita
    que clases con mas frases o con frases mas largas dominen el argmax.

    El prompt tiene la forma "frase1 . frase2 . frase3 ." (BERT usa id=1012 para ".").
    Dividiendo input_ids por las posiciones del separador se obtienen segmentos
    consecutivos, uno por entrada en phrases_ordered.
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    n_phrases = len(phrases_ordered)
    phrase_masks = torch.zeros(n_phrases, max_len, dtype=torch.bool)
    phrase_cls_idx = torch.full((n_phrases,), -1, dtype=torch.long)

    period_positions = [i for i, tid in enumerate(input_ids) if tid == _PERIOD_TOKEN_ID]
    seg_starts = [1] + [p + 1 for p in period_positions]
    seg_ends = period_positions + [len(input_ids)]

    for phrase_idx, (seg_start, seg_end) in enumerate(zip(seg_starts, seg_ends)):
        if phrase_idx >= n_phrases:
            break
        _phrase, class_name = phrases_ordered[phrase_idx]
        cls_idx = class_to_idx.get(class_name)
        if cls_idx is None:
            continue
        phrase_cls_idx[phrase_idx] = cls_idx
        for t in range(seg_start, min(seg_end, max_len)):
            phrase_masks[phrase_idx, t] = True

    return phrase_masks, phrase_cls_idx


# ---------------------------------------------------------------------------
# Inferencia de un unico pase con argmax sobre logits
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _run_dino_argmax_pass(
    processor,
    model,
    images: list[Image.Image],
    image_ids: list[int],
    phrases_ordered: list[tuple[str, str]],
    class_names: list[str],
    score_thresh: float,
    device: torch.device,
    prompt_set_name: str,
    save_probs: bool = False,
) -> tuple[list[dict], "np.ndarray | None"]:
    """
    Ejecuta un pase de DINO con todas las clases en el prompt y asigna clase
    por argmax sobre las puntuaciones por clase derivadas de pred_logits.

    Parametros
    ----------
    phrases_ordered : lista de (frase, nombre_clase) en el mismo orden que
                      aparecen concatenadas en el texto del prompt.
    save_probs      : si True, devuelve tambien el array de vectores de
                      probabilidad [K, 256] float16 (uno por deteccion, mismo
                      orden que la lista de dicts devuelta).

    Devuelve (detections, probs_array | None).
    probs_array tiene shape [n_detecciones_del_pase, 256] float16 y es
    index-aligned con la lista de dicts: detections[i] ↔ probs_array[i].
    """
    phrases = [p for p, _ in phrases_ordered]
    prompt_text = " . ".join(p.strip().lower() for p in phrases) + " ."

    target_sizes = [(img.height, img.width) for img in images]
    inputs = processor(
        images=images,
        text=[prompt_text] * len(images),
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_ctx:
        outputs = model(**inputs)

    # logits: [B, Q, 256]  pred_boxes: [B, Q, 4] cxcywh normalizado
    logits = outputs.logits
    pred_boxes = outputs.pred_boxes

    # Construir mascaras por frase una sola vez por batch (el prompt es identico
    # para todas las imagenes del batch)
    input_ids_list = inputs["input_ids"][0].tolist()
    phrase_masks, phrase_cls_idx = _build_phrase_masks(
        input_ids_list, phrases_ordered, class_names, max_len=logits.shape[-1]
    )
    phrase_masks = phrase_masks.to(device)  # [n_phrases, 256]
    phrase_cls_idx = phrase_cls_idx.to(device)  # [n_phrases]

    detections: list[dict] = []
    collected_probs: list[np.ndarray] = []  # solo si save_probs=True
    n_classes = len(class_names)
    n_phrases = len(phrases_ordered)

    for b_idx, (img_id, (H, W)) in enumerate(zip(image_ids, target_sizes)):
        probs = torch.sigmoid(logits[b_idx])  # [Q, 256]
        box_scores = probs.max(dim=-1).values  # [Q]

        keep = box_scores > score_thresh
        if not keep.any():
            continue

        kept_probs = probs[keep]  # [K, 256]
        kept_boxes = pred_boxes[b_idx][keep]  # [K, 4] cxcywh norm
        kept_scores = box_scores[keep]  # [K]
        K = kept_probs.shape[0]

        # Paso 1: score por frase = max sobre los tokens de esa frase → [K, n_phrases]
        phrase_scores = torch.zeros(K, n_phrases, device=device)
        for p in range(n_phrases):
            mask_p = phrase_masks[p]  # [256] bool
            if mask_p.any():
                phrase_scores[:, p] = kept_probs[:, mask_p].max(dim=-1).values

        # Paso 2: score por clase = media de los scores de sus frases → [K, n_classes]
        # Evita que clases con mas frases o frases mas largas dominen el argmax.
        per_class_scores = torch.zeros(K, n_classes, device=device)
        per_class_counts = torch.zeros(n_classes, device=device)
        for p in range(n_phrases):
            c = phrase_cls_idx[p].item()
            if c >= 0:
                per_class_scores[:, c] += phrase_scores[:, p]
                per_class_counts[c] += 1
        valid = per_class_counts > 0
        per_class_scores[:, valid] /= per_class_counts[valid]

        best_class_idx = per_class_scores.argmax(dim=-1)  # [K]

        # Convertir cajas de cxcywh normalizado a xywh pixel
        cx = kept_boxes[:, 0] * W
        cy = kept_boxes[:, 1] * H
        bw = kept_boxes[:, 2] * W
        bh = kept_boxes[:, 3] * H
        x1 = cx - bw / 2
        y1 = cy - bh / 2

        for k in range(K):
            cls_name = class_names[best_class_idx[k].item()]
            detections.append(
                {
                    "image_id": img_id,
                    "category_id": -1,
                    "bbox": [
                        round(float(x1[k]), 2),
                        round(float(y1[k]), 2),
                        round(float(bw[k]), 2),
                        round(float(bh[k]), 2),
                    ],
                    "score": round(float(kept_scores[k]), 6),
                    "dino_label": cls_name.lower(),
                    "prompt_set": prompt_set_name,
                    "class_name": cls_name,
                }
            )

        if save_probs:
            collected_probs.append(kept_probs.cpu().to(torch.float16).numpy())

    probs_array: np.ndarray | None = None
    if save_probs:
        probs_array = (
            np.concatenate(collected_probs, axis=0)
            if collected_probs
            else np.empty((0, logits.shape[-1]), dtype=np.float16)
        )

    return detections, probs_array


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------


def run_dino_ensemble_argmax(
    img_dir: Path,
    ann_file: Path,
    config_path: Path,
    output_path: Path,
    model_id: str = "IDEA-Research/grounding-dino-base",
    score_thresh: float = 0.05,
    batch_size: int = 8,
    device_str: str = "cuda",
    save_probs: bool = False,
    image_ids: list[int] | None = None,
) -> None:
    """
    save_probs : si True, guarda un fichero ``probs.npz`` junto a
                 ``detection_results.json`` con:
                   probs      — float16 [n_dets, 256], index-aligned con el JSON
                   image_ids  — int32   [n_dets]
                 Permite experimentar con distintas agregaciones sin re-inferir.
                 Cada checkpoint de pase tambien guarda su propio ``_ckpt_<set>_probs.npy``.
    """
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    prompt_sets: dict[str, dict[str, list[str]]] = cfg.get("prompt_sets", {})

    with ann_file.open(encoding="utf-8") as fh:
        coco = json.load(fh)
    images_meta = [m for m in coco["images"] if (img_dir / m["file_name"]).exists()]
    if image_ids is not None:
        _ids = set(image_ids)
        images_meta = [m for m in images_meta if m["id"] in _ids]
    print(f"Images to process: {len(images_meta)}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_id} on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reanudar desde checkpoints de pases ya completados
    completed_sets: set[str] = set()
    all_detections: list[dict] = []
    all_probs: list[np.ndarray] = []  # solo si save_probs=True

    for set_name in prompt_sets:
        ckpt = output_path.parent / f"_ckpt_{set_name}.json"
        if ckpt.exists():
            with ckpt.open(encoding="utf-8") as fh:
                set_dets = json.load(fh)
            all_detections.extend(set_dets)
            completed_sets.add(set_name)
            print(
                f"  [checkpoint] '{set_name}' ya completado ({len(set_dets)} dets), saltando."
            )
            if save_probs:
                ckpt_probs = output_path.parent / f"_ckpt_{set_name}_probs.npy"
                if ckpt_probs.exists():
                    all_probs.append(np.load(ckpt_probs))
                else:
                    print(
                        f"  [aviso] '{set_name}': checkpoint de probs no encontrado, se omite."
                    )

    for set_name, set_classes in prompt_sets.items():
        if set_name in completed_sets:
            continue

        # Construir lista ordenada de (frase, clase) para este prompt set
        phrases_ordered: list[tuple[str, str]] = []
        for class_name, phrases in (set_classes or {}).items():
            for phrase in phrases or []:
                phrases_ordered.append((str(phrase), class_name))

        class_names = sorted({cls for _, cls in phrases_ordered})
        n_phrases = len(phrases_ordered)
        print(
            f"\n=== Prompt set '{set_name}': {n_phrases} frases, {len(class_names)} clases ==="
        )

        set_detections: list[dict] = []
        set_probs: list[np.ndarray] = []
        n_batch = (len(images_meta) + batch_size - 1) // batch_size

        for b_idx in range(0, len(images_meta), batch_size):
            batch_meta = images_meta[b_idx : b_idx + batch_size]
            batch_imgs = [
                Image.open(img_dir / m["file_name"]).convert("RGB") for m in batch_meta
            ]
            batch_ids = [m["id"] for m in batch_meta]

            dets, batch_probs = _run_dino_argmax_pass(
                processor,
                model,
                images=batch_imgs,
                image_ids=batch_ids,
                phrases_ordered=phrases_ordered,
                class_names=class_names,
                score_thresh=score_thresh,
                device=device,
                prompt_set_name=set_name,
                save_probs=save_probs,
            )
            set_detections.extend(dets)
            if save_probs and batch_probs is not None and len(batch_probs):
                set_probs.append(batch_probs)

            bi = b_idx // batch_size + 1
            print(
                f"  batch {bi}/{n_batch}  +{len(dets)} dets"
                f" (set total {len(set_detections)})",
                end="\r",
            )
        print()

        ckpt = output_path.parent / f"_ckpt_{set_name}.json"
        with ckpt.open("w", encoding="utf-8") as fh:
            json.dump(set_detections, fh)
        print(f"  [checkpoint] '{set_name}' guardado → {ckpt}")

        if save_probs and set_probs:
            set_probs_arr = np.concatenate(set_probs, axis=0)
            ckpt_probs = output_path.parent / f"_ckpt_{set_name}_probs.npy"
            np.save(ckpt_probs, set_probs_arr)
            all_probs.append(set_probs_arr)

        all_detections.extend(set_detections)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(all_detections, fh)
    print(f"\nGuardadas {len(all_detections)} detecciones brutas → {output_path}")

    if save_probs:
        probs_path = output_path.parent / "probs.npz"
        final_probs = (
            np.concatenate(all_probs, axis=0)
            if all_probs
            else np.empty((0, 256), dtype=np.float16)
        )
        image_ids_arr = np.array(
            [d["image_id"] for d in all_detections], dtype=np.int32
        )
        np.savez_compressed(probs_path, probs=final_probs, image_ids=image_ids_arr)
        print(
            f"Vectores de probabilidad → {probs_path}  ({final_probs.shape}, float16)"
        )


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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save_probs",
        action="store_true",
        help="Guardar vectores de probabilidad en probs.npz (float16, index-aligned con detection_results.json)",
    )
    parser.add_argument(
        "--image_ids_file",
        type=Path,
        default=None,
        help="JSON con lista de image_ids a procesar (para benchmark subset)",
    )
    args = parser.parse_args()

    image_ids = None
    if args.image_ids_file is not None:
        with args.image_ids_file.open(encoding="utf-8") as fh:
            image_ids = json.load(fh)

    run_dino_ensemble_argmax(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        config_path=args.config,
        output_path=args.output,
        model_id=args.model_id,
        score_thresh=args.score_thresh,
        batch_size=args.batch_size,
        device_str=args.device,
        save_probs=args.save_probs,
        image_ids=image_ids,
    )


if __name__ == "__main__":
    main()
