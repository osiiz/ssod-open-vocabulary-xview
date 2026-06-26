"""Rex-Omni inference for open-vocabulary detection on xView tiles.

Differences vs the Grounding DINO pipeline:

* Rex-Omni is a 3B-param VLM (Qwen2.5-VL-3B) that generates detection tokens
  autoregressively. The prompt must be a short list of ``categories`` (ideally
  <= 15), not a DINO-style "a . b . c ." sentence packed with every canonical
  xView name.
* The wrapper accepts batched image lists natively; we actually batch here
  instead of looping image-by-image.
* The vLLM backend is strongly recommended for throughput; we expose all
  relevant knobs through the CLI.
* The raw output format is quirky (dict with ``extracted_predictions``,
  sometimes just a ``raw_output`` token string). The parser below handles both
  shapes and includes a ``--debug_raw_output`` flag that dumps the first N raw
  outputs to disk for inspection — invaluable when recall is suspicious.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.inference.common import (
    accumulate_detection_records,
    batched,
    build_alias_index,
    build_prompt_context,
    load_json,
    normalize_label,
    parse_prompt_phrases,
    str2bool,
    write_outputs,
)


# ---------------------------------------------------------------------------
# Rex-Omni output parsing
# ---------------------------------------------------------------------------


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_xyxy_sequence(values: Any, assume_xywh: bool = False) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None

    maybe_coords = [_coerce_float(item) for item in values]
    if any(item is None for item in maybe_coords):
        return None

    coords = [float(item) for item in maybe_coords if item is not None]
    if len(coords) != 4:
        return None

    x1, y1, x2, y2 = coords
    if assume_xywh:
        return [x1, y1, x1 + max(0.0, x2), y1 + max(0.0, y2)]
    return [x1, y1, x2, y2]


def extract_rexomni_xyxy_box(prediction: dict[str, Any]) -> list[float] | None:
    if not isinstance(prediction, dict):
        return None

    direct_xyxy_sets = (
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    )
    for key_set in direct_xyxy_sets:
        if all(key in prediction for key in key_set):
            values = [_coerce_float(prediction[key]) for key in key_set]
            if all(value is not None for value in values):
                coords = [float(value) for value in values if value is not None]
                if len(coords) != 4:
                    continue
                x1, y1, x2, y2 = coords
                return [x1, y1, x2, y2]

    direct_xywh_sets = (
        ("x", "y", "w", "h"),
        ("x", "y", "width", "height"),
    )
    for key_set in direct_xywh_sets:
        if all(key in prediction for key in key_set):
            values = [_coerce_float(prediction[key]) for key in key_set]
            if all(value is not None for value in values):
                coords = [float(value) for value in values if value is not None]
                if len(coords) != 4:
                    continue
                x, y, w, h = coords
                return [x, y, x + max(0.0, w), y + max(0.0, h)]

    bbox_format = (
        str(prediction.get("bbox_format", prediction.get("box_format", "")))
        .strip()
        .lower()
    )
    treat_ambiguous_as_xywh = bbox_format in {"xywh", "coco", "xywh_abs"}

    for key in ("bbox_xyxy", "xyxy", "box_xyxy"):
        if key in prediction:
            parsed = _coerce_xyxy_sequence(prediction.get(key), assume_xywh=False)
            if parsed is not None:
                return parsed

    for key in ("bbox_xywh", "xywh", "box_xywh"):
        if key in prediction:
            parsed = _coerce_xyxy_sequence(prediction.get(key), assume_xywh=True)
            if parsed is not None:
                return parsed

    for key in ("bbox", "box", "bounding_box", "coordinates"):
        if key not in prediction:
            continue

        value = prediction.get(key)
        if isinstance(value, dict):
            nested = extract_rexomni_xyxy_box(value)
            if nested is not None:
                return nested

        parsed = _coerce_xyxy_sequence(value, assume_xywh=treat_ambiguous_as_xywh)
        if parsed is not None:
            return parsed

    return None


def extract_rexomni_prediction_label(prediction: dict[str, Any]) -> str:
    label_keys = ("label", "category", "class", "class_name", "name", "text")
    for key in label_keys:
        if key not in prediction:
            continue

        value = prediction.get(key)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]

        if value is None:
            continue

        text = str(value).strip()
        if text:
            return text

    return ""


def extract_rexomni_prediction_score(prediction: dict[str, Any]) -> float:
    score_keys = ("score", "confidence", "conf", "probability", "prob")
    for key in score_keys:
        if key not in prediction:
            continue

        score = _coerce_float(prediction.get(key))
        if score is not None:
            return float(score)

    return 1.0


def extract_prediction_entries(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]

    if not isinstance(payload, dict):
        return []

    if any(
        key in payload
        for key in ("label", "category", "class", "class_name", "name", "text")
    ):
        return [payload]

    for key in (
        "detections",
        "predictions",
        "objects",
        "instances",
        "items",
        "results",
        "outputs",
    ):
        values = payload.get(key)
        if isinstance(values, list):
            return [entry for entry in values if isinstance(entry, dict)]

    return []


def extract_prediction_entries_from_category_map(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    normalized_entries: list[dict[str, Any]] = []
    for category, annotations in payload.items():
        if not isinstance(category, str):
            continue

        label = category.strip()
        if not label:
            continue

        if annotations is None:
            continue

        if isinstance(annotations, dict):
            annotations = [annotations]
        elif not isinstance(annotations, list):
            continue

        for annotation in annotations:
            score_value = 1.0
            box_xyxy = None

            if isinstance(annotation, dict):
                ann_type = str(annotation.get("type", "")).strip().lower()
                if ann_type == "box" and "coords" in annotation:
                    box_xyxy = _coerce_xyxy_sequence(
                        annotation.get("coords"), assume_xywh=False
                    )

                if box_xyxy is None:
                    box_xyxy = extract_rexomni_xyxy_box(annotation)

                score_value = extract_rexomni_prediction_score(annotation)
            elif isinstance(annotation, (list, tuple)):
                box_xyxy = _coerce_xyxy_sequence(annotation, assume_xywh=False)

            if box_xyxy is None:
                continue

            normalized_entries.append(
                {
                    "label": label,
                    "score": score_value,
                    "box_xyxy": box_xyxy,
                }
            )

    return normalized_entries


def _parse_rexomni_raw_output_boxes(
    raw_output: str,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """Parse Rex-Omni's native tokenised output (binned 0-999)."""
    text = str(raw_output)
    if not text.strip():
        return []

    text = text.split("<|im_end|>")[0]
    if not text.endswith("<|box_end|>"):
        text = f"{text}<|box_end|>"

    pattern = (
        r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*"
        r"<\|box_start\|>(.*?)<\|box_end\|>"
    )
    coord_pattern = r"<(\d+)>"

    parsed: list[dict[str, Any]] = []
    for label, coords_text in re.findall(pattern, text):
        label_text = str(label).strip()
        if not label_text:
            continue

        for coord_str in str(coords_text).split(","):
            token_strs = re.findall(coord_pattern, coord_str)
            if len(token_strs) != 4:
                continue

            try:
                x0_bin, y0_bin, x1_bin, y1_bin = [int(token) for token in token_strs]
            except ValueError:
                continue

            box_xyxy = [
                (x0_bin / 999.0) * float(image_width),
                (y0_bin / 999.0) * float(image_height),
                (x1_bin / 999.0) * float(image_width),
                (y1_bin / 999.0) * float(image_height),
            ]
            parsed.append(
                {
                    "label": label_text,
                    "score": 1.0,
                    "box_xyxy": box_xyxy,
                }
            )

    return parsed


def parse_rexomni_detection_output(
    output: Any,
    image_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Normalise a Rex-Omni inference result for a single image."""
    root = output
    raw_output = None

    if isinstance(root, list):
        if not root:
            return []
        root = root[0]

    if isinstance(root, dict):
        raw_output = root.get("raw_output")

    if isinstance(root, dict) and "extracted_predictions" in root:
        root = root.get("extracted_predictions")

    prediction_entries = extract_prediction_entries(root)
    if not prediction_entries:
        prediction_entries = extract_prediction_entries_from_category_map(root)

    normalized: list[dict[str, Any]] = []

    for prediction in prediction_entries:
        label = extract_rexomni_prediction_label(prediction)
        if not label:
            continue

        box_xyxy = prediction.get("box_xyxy") if isinstance(prediction, dict) else None
        if box_xyxy is None:
            box_xyxy = extract_rexomni_xyxy_box(prediction)

        if box_xyxy is None:
            continue

        score_value = prediction.get("score") if isinstance(prediction, dict) else None
        if score_value is None:
            score_value = extract_rexomni_prediction_score(prediction)

        normalized.append(
            {
                "label": label,
                "score": float(score_value),
                "box_xyxy": box_xyxy,
            }
        )

    if not normalized and raw_output and image_size is not None:
        width, height = image_size
        if int(width) > 0 and int(height) > 0:
            normalized = _parse_rexomni_raw_output_boxes(
                raw_output=str(raw_output),
                image_width=int(width),
                image_height=int(height),
            )

    return normalized


# ---------------------------------------------------------------------------
# Rex-Omni categories (prompt -> short list)
# ---------------------------------------------------------------------------


def build_rexomni_categories(
    prompt_phrases: list[str], max_categories: int
) -> list[str]:
    """Keep prompt phrases as a short, dedup'd, flat list for Rex-Omni."""
    categories: list[str] = []
    seen: set[str] = set()

    for phrase in prompt_phrases:
        key = normalize_label(phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        categories.append(phrase.strip())

    if max_categories > 0 and len(categories) > max_categories:
        print(
            f"[rex-omni] Truncating categories from {len(categories)} to "
            f"{max_categories} (--max_categories)."
        )
        categories = categories[:max_categories]

    if not categories:
        raise ValueError("No valid categories available for Rex-Omni inference")

    return categories


# ---------------------------------------------------------------------------
# Torch dtype parsing
# ---------------------------------------------------------------------------


def parse_torch_dtype(dtype_name: str):
    import torch

    key = str(dtype_name).strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(
            "rex_omni_torch_dtype must be one of {'float16','bfloat16','float32'}"
        )
    return mapping[key]


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------


def run_rex_omni_inference(
    img_dir: Path,
    ann_file: Path,
    output_folder: Path,
    model_id: str,
    xview_classes_path: Path,
    xview_macro_classes_path: Path,
    prompt_file: Path,
    device: str = "cuda",
    batch_size: int = 4,
    max_images: int | None = None,
    save_raw_predictions: bool = True,
    detection_label_key: str = "rex_label",
    max_categories: int = 15,
    rex_omni_backend: str = "transformers",
    rex_omni_attn_implementation: str = "eager",
    rex_omni_torch_dtype: str = "float16",
    rex_omni_device_map: str = "auto",
    rex_omni_max_tokens: int = 2048,
    rex_omni_temperature: float = 0.0,
    rex_omni_top_p: float = 0.05,
    rex_omni_top_k: int = 1,
    rex_omni_repetition_penalty: float = 1.05,
    rex_omni_quantization: str | None = None,
    debug_raw_output: int = 0,
    fail_fast_threshold: float = 0.0,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    detection_label_key = detection_label_key.strip()
    if not detection_label_key:
        raise ValueError("detection_label_key cannot be empty")

    try:
        from rex_omni import RexOmniWrapper
    except ImportError as exc:
        raise ImportError(
            "rex_omni is not installed. Install it from the official repository:\n"
            "  git clone https://github.com/IDEA-Research/Rex-Omni\n"
            "  pip install -e Rex-Omni\n"
            "If optional heavy dependencies fail, try `pip install -e Rex-Omni --no-deps` "
            "and then install qwen_vl_utils separately. For the vLLM backend you also need "
            "`pip install vllm` and optionally `flash_attn`."
        ) from exc

    tiled_coco = load_json(ann_file)
    images = tiled_coco.get("images", [])

    if max_images is not None:
        images = images[: max(0, int(max_images))]

    xview_classes = load_json(xview_classes_path)
    xview_macro_classes = load_json(xview_macro_classes_path)
    prompt_phrases = parse_prompt_phrases(prompt_file)

    # Rex-Omni uses its own short category list; we still build the alias map
    # (with canonical names included) so fine-grained labels that Rex-Omni
    # might emit still resolve back to macro-class IDs.
    _, alias_to_macro_id, macro_id_to_name, _ = build_prompt_context(
        xview_classes=xview_classes,
        xview_macro_classes=xview_macro_classes,
        prompt_phrases=prompt_phrases,
        append_canonical_names=True,
    )
    alias_index = build_alias_index(alias_to_macro_id)

    rexomni_categories = build_rexomni_categories(prompt_phrases, max_categories)

    rexomni_init_kwargs: dict[str, Any] = {
        "max_tokens": int(rex_omni_max_tokens),
        "temperature": float(rex_omni_temperature),
        "top_p": float(rex_omni_top_p),
        "top_k": int(rex_omni_top_k),
        "repetition_penalty": float(rex_omni_repetition_penalty),
    }

    backend_normalized = str(rex_omni_backend).strip().lower()
    if backend_normalized == "transformers":
        rexomni_init_kwargs["attn_implementation"] = str(
            rex_omni_attn_implementation
        ).strip()
        rexomni_init_kwargs["torch_dtype"] = parse_torch_dtype(rex_omni_torch_dtype)
        rexomni_init_kwargs["device_map"] = rex_omni_device_map
    elif backend_normalized == "vllm":
        if rex_omni_quantization:
            rexomni_init_kwargs["quantization"] = str(rex_omni_quantization).strip()
    else:
        raise ValueError(
            f"rex_omni_backend must be 'transformers' or 'vllm', got {rex_omni_backend!r}"
        )

    print(
        f"Loading Rex-Omni | model={model_id} backend={backend_normalized} "
        f"categories={len(rexomni_categories)} batch_size={max(1, int(batch_size))}"
    )

    rexomni_model = RexOmniWrapper(
        model_path=model_id,
        backend=backend_normalized,
        **rexomni_init_kwargs,
    )

    detection_results: list[dict] = []
    raw_predictions: list[dict] | None = [] if save_raw_predictions else None

    total_predictions = 0
    mapped_predictions = 0
    aggregate_unmapped: dict[str, int] = {}
    resolved_label_cache: dict[str, tuple] = {}
    processed_images = 0
    failed_images = 0

    debug_dumps: list[dict] = []
    debug_budget = max(0, int(debug_raw_output))

    total_images = len(images)
    start_time = time.perf_counter()

    print(f"Rex-Omni categories used: {rexomni_categories}")

    for batch_items in batched(images, batch_size=batch_size):
        batch_valid: list[dict] = []
        batch_rgb_images: list[Image.Image] = []

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

        if not batch_valid:
            continue

        # Rex-Omni accepts a list for batched inference. If vLLM backend is
        # available this yields a real throughput win; with transformers the
        # wrapper still loops internally but keeps overhead lower.
        try:
            batch_outputs = rexomni_model.inference(
                images=batch_rgb_images
                if len(batch_rgb_images) > 1
                else batch_rgb_images[0],
                task="detection",
                categories=rexomni_categories,
            )
        except Exception as exc:
            failed_images += len(batch_valid)
            print(
                f"[rex-omni] Batch inference failed "
                f"(files={[m['file_name'] for m in batch_valid]}): {exc}"
            )
            batch_outputs = [None] * len(batch_valid)

        # Normalise wrapper return to a per-image list.
        if not isinstance(batch_outputs, list):
            batch_outputs = [batch_outputs]
        if len(batch_outputs) != len(batch_valid):
            # Some wrapper versions return nested lists; flatten conservatively.
            if len(batch_outputs) == 1 and isinstance(batch_outputs[0], list):
                batch_outputs = batch_outputs[0]

        expected_outputs = len(batch_valid)
        if len(batch_outputs) < expected_outputs:
            missing_outputs = expected_outputs - len(batch_outputs)
            failed_images += missing_outputs
            print(
                f"[rex-omni] Warning: expected {expected_outputs} outputs but got "
                f"{len(batch_outputs)}. Marking {missing_outputs} images as failed."
            )
            batch_outputs = batch_outputs + [None] * missing_outputs
        elif len(batch_outputs) > expected_outputs:
            extra_outputs = len(batch_outputs) - expected_outputs
            print(
                f"[rex-omni] Warning: got {len(batch_outputs)} outputs for "
                f"{expected_outputs} images. Ignoring {extra_outputs} extra outputs."
            )
            batch_outputs = batch_outputs[:expected_outputs]

        for image_meta, single_output in zip(batch_valid, batch_outputs):
            if single_output is None:
                failed_images += 1

            if debug_budget > 0:
                debug_dumps.append(
                    {
                        "file_name": image_meta["file_name"],
                        "image_id": image_meta["image_id"],
                        "raw": _safe_debug_repr(single_output),
                    }
                )
                debug_budget -= 1

            try:
                image_predictions = parse_rexomni_detection_output(
                    single_output,
                    image_size=(int(image_meta["width"]), int(image_meta["height"])),
                )
            except Exception as exc:
                print(f"[rex-omni] Parsing failed for {image_meta['file_name']}: {exc}")
                image_predictions = []

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

        # Early abort if the run is clearly broken (e.g. every image fails).
        if (
            fail_fast_threshold > 0.0
            and processed_images >= 20
            and failed_images / max(1, processed_images) >= fail_fast_threshold
        ):
            print(
                f"[rex-omni] Aborting: {failed_images}/{processed_images} "
                f"images failed (>= {fail_fast_threshold:.0%} threshold)."
            )
            break

        if processed_images > 0 and processed_images % 50 == 0:
            elapsed = max(1e-6, time.perf_counter() - start_time)
            print(
                f"[rex-omni] images={processed_images}/{total_images} "
                f"({processed_images / elapsed:.2f} img/s) | "
                f"mapped={mapped_predictions}/{total_predictions} | "
                f"failed={failed_images} | elapsed={elapsed:.1f}s"
            )

    if debug_dumps:
        debug_path = output_folder / "debug_raw_outputs.json"
        debug_path.write_text(
            json.dumps(debug_dumps, indent=2, default=str), encoding="utf-8"
        )
        print(f"[rex-omni] Wrote {len(debug_dumps)} debug raw outputs to {debug_path}")

    prompt_context = {
        "detector_name": "Rex-Omni",
        "model_id": model_id,
        "rex_omni_backend": backend_normalized,
        "rex_omni_attn_implementation": (
            rex_omni_attn_implementation
            if backend_normalized == "transformers"
            else None
        ),
        "rex_omni_torch_dtype": (
            rex_omni_torch_dtype if backend_normalized == "transformers" else None
        ),
        "rex_omni_device_map": (
            rex_omni_device_map if backend_normalized == "transformers" else None
        ),
        "rex_omni_quantization": rex_omni_quantization,
        "rex_omni_max_tokens": int(rex_omni_max_tokens),
        "rex_omni_temperature": float(rex_omni_temperature),
        "rex_omni_top_p": float(rex_omni_top_p),
        "rex_omni_top_k": int(rex_omni_top_k),
        "rex_omni_repetition_penalty": float(rex_omni_repetition_penalty),
        "device_hint": str(device),
        "detection_label_key": detection_label_key,
        "prompt_file": str(prompt_file),
        "save_raw_predictions": bool(save_raw_predictions),
        "categories_used": rexomni_categories,
        "categories_count": len(rexomni_categories),
        "max_categories": int(max_categories),
        "categories": [
            {"id": int(macro_id), "name": name}
            for macro_id, name in sorted(
                macro_id_to_name.items(), key=lambda item: item[0]
            )
        ],
        "total_predictions": int(total_predictions),
        "mapped_predictions": int(mapped_predictions),
        "unmapped_predictions": int(total_predictions - mapped_predictions),
        "failed_images": int(failed_images),
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
        f"Rex-Omni inference complete | images={len(images)} "
        f"detections={len(detection_results)} "
        f"mapped={mapped_predictions}/{total_predictions} failed={failed_images}"
    )
    print(f"Detection results: {output_folder / 'detection_results.json'}")
    print(f"Raw predictions:   {output_folder / 'raw_predictions.json'}")


def _safe_debug_repr(obj: Any) -> Any:
    """Best-effort JSON-serialisable snapshot of a Rex-Omni output."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Rex-Omni inference and export class-aware COCO detections"
    )
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument("--ann_file", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--xview_classes_path", type=Path, required=True)
    parser.add_argument("--xview_macro_classes_path", type=Path, required=True)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--save_raw_predictions", type=str2bool, default=True)
    parser.add_argument("--detection_label_key", type=str, default="rex_label")
    parser.add_argument(
        "--max_categories",
        type=int,
        default=15,
        help="Truncate the category list passed to Rex-Omni. Set to 0 to disable.",
    )
    parser.add_argument(
        "--rex_omni_backend",
        type=str,
        choices=["transformers", "vllm"],
        default="transformers",
    )
    parser.add_argument(
        "--rex_omni_attn_implementation",
        type=str,
        choices=["sdpa", "eager", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument(
        "--rex_omni_torch_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--rex_omni_device_map", type=str, default="auto")
    parser.add_argument("--rex_omni_max_tokens", type=int, default=2048)
    parser.add_argument("--rex_omni_temperature", type=float, default=0.0)
    parser.add_argument("--rex_omni_top_p", type=float, default=0.05)
    parser.add_argument("--rex_omni_top_k", type=int, default=1)
    parser.add_argument("--rex_omni_repetition_penalty", type=float, default=1.05)
    parser.add_argument(
        "--rex_omni_quantization",
        type=str,
        default=None,
        help="Only used with vLLM backend, e.g. 'awq' for Rex-Omni-AWQ.",
    )
    parser.add_argument(
        "--debug_raw_output",
        type=int,
        default=0,
        help="Dump the first N raw Rex-Omni outputs to debug_raw_outputs.json.",
    )
    parser.add_argument(
        "--fail_fast_threshold",
        type=float,
        default=0.0,
        help=(
            "Abort the run if the fraction of failed images exceeds this value "
            "after at least 20 images. 0 disables the check."
        ),
    )

    args = parser.parse_args()
    run_rex_omni_inference(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        output_folder=args.output_folder,
        model_id=args.model_id,
        xview_classes_path=args.xview_classes_path,
        xview_macro_classes_path=args.xview_macro_classes_path,
        prompt_file=args.prompt_file,
        device=args.device,
        batch_size=args.batch_size,
        max_images=args.max_images,
        save_raw_predictions=args.save_raw_predictions,
        detection_label_key=args.detection_label_key,
        max_categories=args.max_categories,
        rex_omni_backend=args.rex_omni_backend,
        rex_omni_attn_implementation=args.rex_omni_attn_implementation,
        rex_omni_torch_dtype=args.rex_omni_torch_dtype,
        rex_omni_device_map=args.rex_omni_device_map,
        rex_omni_max_tokens=args.rex_omni_max_tokens,
        rex_omni_temperature=args.rex_omni_temperature,
        rex_omni_top_p=args.rex_omni_top_p,
        rex_omni_top_k=args.rex_omni_top_k,
        rex_omni_repetition_penalty=args.rex_omni_repetition_penalty,
        rex_omni_quantization=args.rex_omni_quantization,
        debug_raw_output=args.debug_raw_output,
        fail_fast_threshold=args.fail_fast_threshold,
    )


if __name__ == "__main__":
    main()
