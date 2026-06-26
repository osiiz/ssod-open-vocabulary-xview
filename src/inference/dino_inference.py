"""Grounding DINO inference for open-vocabulary detection on xView tiles.

Outputs COCO-compatible detection_results.json (macro-class IDs), a raw
predictions dump (for error analysis), and a prompt_context.json summarising
the run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from src.inference.common import (
    accumulate_detection_records,
    batched,
    build_alias_index,
    build_prompt_context,
    load_json,
    parse_prompt_phrases,
    str2bool,
    write_outputs,
)

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except ImportError as exc:
    raise ImportError(
        "transformers is required for Grounding DINO inference. "
        "Install it in your environment first."
    ) from exc


def run_grounding_dino_inference(
    img_dir: Path,
    ann_file: Path,
    output_folder: Path,
    model_id: str,
    xview_classes_path: Path,
    xview_macro_classes_path: Path,
    prompt_file: Path,
    score_thresh: float = 0.001,
    text_thresh: float = 0.001,
    device: str = "cuda",
    batch_size: int = 1,
    max_images: int | None = None,
    use_amp: bool = True,
    save_raw_predictions: bool = True,
    detection_label_key: str = "dino_label",
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    detection_label_key = detection_label_key.strip()
    if not detection_label_key:
        raise ValueError("detection_label_key cannot be empty")

    tiled_coco = load_json(ann_file)
    images = tiled_coco.get("images", [])

    if max_images is not None:
        images = images[: max(0, int(max_images))]

    xview_classes = load_json(xview_classes_path)
    xview_macro_classes = load_json(xview_macro_classes_path)
    prompt_phrases = parse_prompt_phrases(prompt_file)

    # DINO benefits from seeing all canonical class names in the prompt
    # (text-image alignment), so append_canonical_names=True.
    (
        prompt_text,
        alias_to_macro_id,
        macro_id_to_name,
        merged_prompts,
    ) = build_prompt_context(
        xview_classes=xview_classes,
        xview_macro_classes=xview_macro_classes,
        prompt_phrases=prompt_phrases,
        append_canonical_names=True,
    )
    alias_index = build_alias_index(alias_to_macro_id)

    resolved_device = (
        torch.device(device)
        if str(device).startswith("cuda") and torch.cuda.is_available()
        else torch.device("cpu")
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(resolved_device)
    model.eval()
    if resolved_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    detection_results: list[dict] = []
    raw_predictions: list[dict] | None = [] if save_raw_predictions else None

    total_predictions = 0
    mapped_predictions = 0
    aggregate_unmapped: dict[str, int] = {}
    resolved_label_cache: dict[str, tuple] = {}
    processed_images = 0

    total_images = len(images)
    start_time = time.perf_counter()
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")

    print(
        f"Running Grounding DINO | images={total_images} "
        f"batch_size={max(1, int(batch_size))} device={resolved_device} "
        f"amp={amp_enabled} prompt_phrases={len(merged_prompts)}"
    )

    for batch_items in batched(images, batch_size=batch_size):
        batch_valid: list[dict] = []
        batch_rgb_images: list[Image.Image] = []
        batch_target_sizes: list[tuple[int, int]] = []

        for image_info in batch_items:
            image_id = int(image_info["id"])
            file_name = str(image_info["file_name"])
            image_path = img_dir / file_name

            if not image_path.exists():
                print(f"Skipping missing image: {image_path}")
                continue

            with Image.open(image_path) as pil_image:
                rgb_image = pil_image.convert("RGB")
                width, height = rgb_image.size

            batch_valid.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "width": int(width),
                    "height": int(height),
                }
            )
            batch_rgb_images.append(rgb_image)
            batch_target_sizes.append((height, width))

        if not batch_valid:
            continue

        batch_prompts = [prompt_text for _ in range(len(batch_valid))]
        inputs = processor(
            images=batch_rgb_images, text=batch_prompts, return_tensors="pt"
        )
        inputs = {
            key: tensor.to(resolved_device, non_blocking=True)
            if hasattr(tensor, "to")
            else tensor
            for key, tensor in inputs.items()
        }

        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        post_processed_batch = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=float(score_thresh),
            text_threshold=float(text_thresh),
            target_sizes=batch_target_sizes,
        )

        for image_meta, post_processed in zip(batch_valid, post_processed_batch):
            image_predictions = []
            for label, score, box in zip(
                post_processed.get("labels", []),
                post_processed.get("scores", []),
                post_processed.get("boxes", []),
            ):
                image_predictions.append(
                    {
                        "label": str(label),
                        "score": float(score),
                        "box_xyxy": box.tolist(),
                    }
                )

            processed_images += 1
            entries, raw_entry, unmapped, total, mapped = accumulate_detection_records(
                image_id=image_meta["image_id"],
                file_name=image_meta["file_name"],
                width=image_meta["width"],
                height=image_meta["height"],
                image_predictions=image_predictions,
                alias_to_macro_id=alias_to_macro_id,
                alias_index=alias_index,
                macro_id_to_name=macro_id_to_name,
                resolved_label_cache=resolved_label_cache,
                detection_label_key=detection_label_key,
                save_raw_predictions=save_raw_predictions,
            )

            detection_results.extend(entries)
            if raw_entry is not None and raw_predictions is not None:
                raw_predictions.append(raw_entry)
            total_predictions += total
            mapped_predictions += mapped
            for label, count in unmapped.items():
                aggregate_unmapped[label] = aggregate_unmapped.get(label, 0) + count

        if processed_images > 0 and processed_images % 50 == 0:
            elapsed = max(1e-6, time.perf_counter() - start_time)
            print(
                f"[DINO] images={processed_images}/{total_images} "
                f"({processed_images / elapsed:.2f} img/s) | "
                f"mapped={mapped_predictions}/{total_predictions} | "
                f"elapsed={elapsed:.1f}s"
            )

    prompt_context = {
        "detector_name": "Grounding DINO",
        "model_id": model_id,
        "device": str(resolved_device),
        "score_thresh": float(score_thresh),
        "text_thresh": float(text_thresh),
        "detection_label_key": detection_label_key,
        "prompt_file": str(prompt_file),
        "save_raw_predictions": bool(save_raw_predictions),
        "prompt_text": prompt_text,
        "prompt_phrases_count": len(merged_prompts),
        "categories": [
            {"id": int(macro_id), "name": name}
            for macro_id, name in sorted(
                macro_id_to_name.items(), key=lambda item: item[0]
            )
        ],
        "total_predictions": int(total_predictions),
        "mapped_predictions": int(mapped_predictions),
        "unmapped_predictions": int(total_predictions - mapped_predictions),
        "top_unmapped_labels": sorted(
            (
                {"label": label, "count": count}
                for label, count in aggregate_unmapped.items()
                if label
            ),
            key=lambda item: item["count"],
            reverse=True,
        )[:50],
    }

    write_outputs(
        output_folder=output_folder,
        detection_results=detection_results,
        raw_predictions=raw_predictions,
        prompt_context=prompt_context,
    )

    print(
        f"Grounding DINO inference complete | images={len(images)} "
        f"detections={len(detection_results)} "
        f"mapped={mapped_predictions}/{total_predictions}"
    )
    print(f"Detection results: {output_folder / 'detection_results.json'}")
    print(f"Raw predictions:   {output_folder / 'raw_predictions.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO inference and export class-aware COCO detections"
    )
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument("--ann_file", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--xview_classes_path", type=Path, required=True)
    parser.add_argument("--xview_macro_classes_path", type=Path, required=True)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--score_thresh", type=float, default=0.001)
    parser.add_argument("--text_thresh", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--save_raw_predictions", type=str2bool, default=True)
    parser.add_argument("--detection_label_key", type=str, default="dino_label")

    args = parser.parse_args()
    run_grounding_dino_inference(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        output_folder=args.output_folder,
        model_id=args.model_id,
        xview_classes_path=args.xview_classes_path,
        xview_macro_classes_path=args.xview_macro_classes_path,
        prompt_file=args.prompt_file,
        score_thresh=args.score_thresh,
        text_thresh=args.text_thresh,
        device=args.device,
        batch_size=args.batch_size,
        max_images=args.max_images,
        use_amp=args.use_amp,
        save_raw_predictions=args.save_raw_predictions,
        detection_label_key=args.detection_label_key,
    )


if __name__ == "__main__":
    main()
