import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

try:
    import yaml
except ImportError:
    yaml = None

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except ImportError as exc:
    raise ImportError(
        "transformers is required for open-vocabulary inference. "
        "Install it in your environment first."
    ) from exc


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    effective_batch_size = max(1, int(batch_size))
    for start in range(0, len(items), effective_batch_size):
        yield items[start : start + effective_batch_size]


def normalize_label(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_label(text: str) -> set[str]:
    normalized = normalize_label(text)
    return {token for token in normalized.split(" ") if token}


def _dedupe_phrases(raw_phrases: list[str]) -> list[str]:
    phrases = []
    seen_normalized = set()

    for raw_phrase in raw_phrases:
        if not isinstance(raw_phrase, str):
            continue

        phrase = raw_phrase.strip()
        if not phrase:
            continue

        key = normalize_label(phrase)
        if not key or key in seen_normalized:
            continue

        seen_normalized.add(key)
        phrases.append(phrase)

    return phrases


def _split_prompt_payload(payload: str) -> list[str]:
    return [part.strip() for part in payload.split(".") if part.strip()]


def _parse_legacy_prompt_file(prompt_file: Path) -> list[str]:
    raw_phrases: list[str] = []

    with prompt_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            payload = ""
            if ":" in line and line.lower().startswith("prompt"):
                payload = line.split(":", maxsplit=1)[1]
            elif line.startswith("-"):
                payload = line[1:].strip()

            if not payload:
                continue

            raw_phrases.extend(_split_prompt_payload(payload))

    return _dedupe_phrases(raw_phrases)


def _parse_structured_prompt_file(prompt_file: Path) -> list[str]:
    suffix = prompt_file.suffix.lower()
    with prompt_file.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        else:
            if yaml is None:
                raise ImportError(
                    "PyYAML is required to parse YAML prompt files. "
                    "Install pyyaml or provide a JSON/TXT prompt file."
                )
            payload = yaml.safe_load(handle)

    raw_phrases: list[str] = []
    if payload is None:
        return raw_phrases

    if isinstance(payload, list):
        raw_phrases.extend(payload)
    elif isinstance(payload, dict):
        direct_phrases = payload.get("phrases")
        if isinstance(direct_phrases, list):
            raw_phrases.extend(direct_phrases)

        groups = payload.get("prompt_groups") or payload.get("groups")
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, str):
                    raw_phrases.append(group)
                    continue

                if not isinstance(group, dict):
                    continue

                group_phrases = group.get("phrases") or group.get("items")
                if isinstance(group_phrases, list):
                    raw_phrases.extend(group_phrases)
    else:
        raise ValueError(
            "Structured prompt file must be a list or object with phrases/prompt_groups"
        )

    return _dedupe_phrases(raw_phrases)


def parse_prompt_phrases(prompt_file: Path) -> list[str]:
    suffix = prompt_file.suffix.lower()
    if suffix in {".yaml", ".yml", ".json"}:
        return _parse_structured_prompt_file(prompt_file)
    return _parse_legacy_prompt_file(prompt_file)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_prompt_context(
    xview_classes: dict,
    xview_macro_classes: dict,
    prompt_phrases: list[str],
) -> tuple[str, dict[str, int], dict[int, str]]:
    original_classes = {int(k): v for k, v in xview_classes.items()}
    macro_map = {int(k): v for k, v in xview_macro_classes.items()}

    valid_original_ids = sorted(set(original_classes.keys()) & set(macro_map.keys()))

    alias_to_macro_id: dict[str, int] = {}
    macro_id_to_name: dict[int, str] = {}
    canonical_original_names = []

    for original_id in valid_original_ids:
        original_name = str(original_classes[original_id]).strip()
        macro_info = macro_map[original_id]
        macro_id = int(macro_info["id"])
        macro_name = str(macro_info["name"]).strip()

        macro_id_to_name[macro_id] = macro_name
        canonical_original_names.append(original_name)
        alias_to_macro_id[normalize_label(original_name)] = macro_id

    for macro_id, macro_name in macro_id_to_name.items():
        alias_to_macro_id[normalize_label(macro_name)] = macro_id

    macro_name_to_id = {normalize_label(v): k for k, v in macro_id_to_name.items()}
    fallback_aliases = {
        "aircraft": macro_name_to_id.get("aircraft"),
        "plane": macro_name_to_id.get("aircraft"),
        "helicopter": macro_name_to_id.get("aircraft"),
        "car": macro_name_to_id.get("light vehicle"),
        "truck": macro_name_to_id.get("heavy vehicle"),
        "train": macro_name_to_id.get("railway vehicle"),
        "ship": macro_name_to_id.get("maritime vessel"),
        "boat": macro_name_to_id.get("maritime vessel"),
        "vessel": macro_name_to_id.get("maritime vessel"),
        "bulldozer": macro_name_to_id.get("engineering vehicle"),
        "excavator": macro_name_to_id.get("engineering vehicle"),
        "loader": macro_name_to_id.get("engineering vehicle"),
        "building": macro_name_to_id.get("building"),
        "facility": macro_name_to_id.get("building"),
        "hut": macro_name_to_id.get("building"),
        "hangar": macro_name_to_id.get("building"),
        "storage tank": macro_name_to_id.get("storage tank"),
        "tower": macro_name_to_id.get("tower pylon"),
        "pylon": macro_name_to_id.get("tower pylon"),
    }
    for alias, macro_id in fallback_aliases.items():
        if macro_id is not None:
            alias_to_macro_id[normalize_label(alias)] = int(macro_id)

    merged_prompts = []
    seen_prompt_keys = set()

    for phrase in prompt_phrases + canonical_original_names:
        key = normalize_label(phrase)
        if not key or key in seen_prompt_keys:
            continue
        seen_prompt_keys.add(key)
        merged_prompts.append(phrase)

    if not merged_prompts:
        raise ValueError("No valid prompt phrases found for open-vocabulary inference")

    prompt_text = " . ".join(merged_prompts) + " ."
    return prompt_text, alias_to_macro_id, macro_id_to_name


def build_alias_index(
    alias_to_macro_id: dict[str, int]
) -> list[tuple[str, int, set[str]]]:
    index = []
    for alias, macro_id in alias_to_macro_id.items():
        token_set = tokenize_label(alias)
        if not token_set:
            continue
        index.append((alias, int(macro_id), token_set))
    return index


def resolve_label_to_macro_id(
    label: str,
    alias_to_macro_id: dict[str, int],
    alias_index: list[tuple[str, int, set[str]]],
    min_score: float = 0.45,
) -> tuple[int | None, float, str | None]:
    normalized = normalize_label(label)
    if not normalized:
        return None, 0.0, None

    direct_hit = alias_to_macro_id.get(normalized)
    if direct_hit is not None:
        return int(direct_hit), 1.0, normalized

    label_tokens = tokenize_label(normalized)
    if not label_tokens:
        return None, 0.0, None

    best_score = 0.0
    best_macro_id = None
    best_alias = None

    for alias, macro_id, alias_tokens in alias_index:
        intersection = len(label_tokens & alias_tokens)
        if intersection == 0:
            continue

        union = len(label_tokens | alias_tokens)
        jaccard = intersection / union if union else 0.0
        containment = intersection / min(len(label_tokens), len(alias_tokens))
        score = max(jaccard, containment * 0.95)

        if score > best_score:
            best_score = score
            best_macro_id = macro_id
            best_alias = alias

    if best_macro_id is None or best_score < min_score:
        return None, best_score, best_alias

    return int(best_macro_id), float(best_score), best_alias


def clip_xyxy_to_image(
    box_xyxy: Iterable[float], width: int, height: int
) -> list[float]:
    x_min, y_min, x_max, y_max = [float(value) for value in box_xyxy]
    x_min = max(0.0, min(x_min, float(width)))
    y_min = max(0.0, min(y_min, float(height)))
    x_max = max(0.0, min(x_max, float(width)))
    y_max = max(0.0, min(y_max, float(height)))

    return [x_min, y_min, x_max, y_max]


def xyxy_to_coco_bbox(box_xyxy: list[float]) -> list[float]:
    x_min, y_min, x_max, y_max = box_xyxy
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    return [x_min, y_min, width, height]


def choose_inference_backend(
    model_id: str,
    detector_name: str,
    inference_backend: str,
) -> str:
    backend = str(inference_backend).strip().lower()
    valid_backends = {"auto", "transformers", "rex_omni"}
    if backend not in valid_backends:
        raise ValueError(
            f"inference_backend must be one of {sorted(valid_backends)}, got: {inference_backend}"
        )

    if backend != "auto":
        return backend

    detector_hint = normalize_label(detector_name)
    model_hint = normalize_label(model_id)
    combined_hint = f"{detector_hint} {model_hint}".strip()
    if "rex omni" in combined_hint or "rexomni" in combined_hint:
        return "rex_omni"

    return "transformers"


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_xyxy_sequence(values: Any, assume_xywh: bool = False) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None

    coords = [_coerce_float(item) for item in values]
    if any(item is None for item in coords):
        return None

    x1, y1, x2, y2 = [float(item) for item in coords]
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
                x1, y1, x2, y2 = [float(value) for value in values]
                return [x1, y1, x2, y2]

    direct_xywh_sets = (
        ("x", "y", "w", "h"),
        ("x", "y", "width", "height"),
    )
    for key_set in direct_xywh_sets:
        if all(key in prediction for key in key_set):
            values = [_coerce_float(prediction[key]) for key in key_set]
            if all(value is not None for value in values):
                x, y, w, h = [float(value) for value in values]
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

    normalized = []

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


def prompt_text_to_categories(prompt_text: str) -> list[str]:
    categories = [
        chunk.strip() for chunk in str(prompt_text).split(".") if chunk.strip()
    ]

    deduped = []
    seen = set()
    for category in categories:
        key = normalize_label(category)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(category)

    return deduped


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
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


def run_open_vocabulary_inference(
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
    detector_name: str = "Grounding DINO",
    detection_label_key: str = "dino_label",
    inference_backend: str = "auto",
    rex_omni_backend: str = "transformers",
    rex_omni_attn_implementation: str = "eager",
    rex_omni_torch_dtype: str = "float16",
    rex_omni_device_map: str = "auto",
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    detection_label_key = str(detection_label_key).strip()
    if not detection_label_key:
        raise ValueError("detection_label_key cannot be empty")

    tiled_coco = load_json(ann_file)
    images = tiled_coco.get("images", [])

    if max_images is not None:
        images = images[: max(0, int(max_images))]

    xview_classes = load_json(xview_classes_path)
    xview_macro_classes = load_json(xview_macro_classes_path)
    prompt_phrases = parse_prompt_phrases(prompt_file)

    prompt_text, alias_to_macro_id, macro_id_to_name = build_prompt_context(
        xview_classes=xview_classes,
        xview_macro_classes=xview_macro_classes,
        prompt_phrases=prompt_phrases,
    )
    alias_index = build_alias_index(alias_to_macro_id)

    selected_backend = choose_inference_backend(
        model_id=model_id,
        detector_name=detector_name,
        inference_backend=inference_backend,
    )

    resolved_device = (
        torch.device(device)
        if str(device).startswith("cuda") and torch.cuda.is_available()
        else torch.device("cpu")
    )

    processor = None
    model = None
    rexomni_model = None
    rexomni_categories: list[str] = []

    if selected_backend == "transformers":
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        model.to(resolved_device)
        model.eval()
        if resolved_device.type == "cuda":
            torch.backends.cudnn.benchmark = True
    else:
        try:
            from rex_omni import RexOmniWrapper
        except ImportError as exc:
            raise ImportError(
                "RexOmni backend selected but rex_omni is not installed. "
                "Install it from the official repository (for example: "
                "git clone https://github.com/IDEA-Research/Rex-Omni && pip install -e Rex-Omni). "
                "If optional heavy dependencies fail in your environment, try installing with --no-deps "
                "and then install qwen_vl_utils separately."
            ) from exc

        rexomni_categories = prompt_text_to_categories(prompt_text)
        if not rexomni_categories:
            raise ValueError("No valid categories available for RexOmni inference")

        rexomni_init_kwargs = {}
        if str(rex_omni_backend).strip().lower() == "transformers":
            rexomni_init_kwargs = {
                "attn_implementation": str(rex_omni_attn_implementation).strip(),
                "torch_dtype": parse_torch_dtype(rex_omni_torch_dtype),
                "device_map": rex_omni_device_map,
            }

        rexomni_model = RexOmniWrapper(
            model_path=model_id,
            backend=rex_omni_backend,
            **rexomni_init_kwargs,
        )

    detection_results = []
    raw_predictions = [] if save_raw_predictions else None

    total_predictions = 0
    mapped_predictions = 0
    unmapped_counts: dict[str, int] = {}
    resolved_label_cache: dict[str, tuple[int | None, float, str | None]] = {}
    processed_images = 0

    total_images = len(images)
    start_time = time.perf_counter()
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")

    print(
        f"Running {detector_name} | images={total_images} "
        f"backend={selected_backend} batch_size={max(1, int(batch_size))} "
        f"device={resolved_device} amp={amp_enabled}"
    )

    for batch_items in batched(images, batch_size=batch_size):
        batch_valid: list[dict] = []
        batch_rgb_images = []
        batch_target_sizes = []

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

        batch_predictions: list[list[dict[str, Any]]] = []

        if selected_backend == "transformers":
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

            for post_processed in post_processed_batch:
                labels = post_processed.get("labels", [])
                scores = post_processed.get("scores", [])
                boxes = post_processed.get("boxes", [])
                image_predictions = []

                for label, score, box in zip(labels, scores, boxes):
                    image_predictions.append(
                        {
                            "label": str(label),
                            "score": float(score),
                            "box_xyxy": box.tolist(),
                        }
                    )

                batch_predictions.append(image_predictions)
        else:
            for image_meta, rgb_image in zip(batch_valid, batch_rgb_images):
                try:
                    rex_output = rexomni_model.inference(
                        images=rgb_image,
                        task="detection",
                        categories=rexomni_categories,
                    )
                    image_predictions = parse_rexomni_detection_output(
                        rex_output,
                        image_size=(
                            int(image_meta["width"]),
                            int(image_meta["height"]),
                        ),
                    )
                except Exception as exc:
                    print(
                        "Warning: RexOmni inference failed for image "
                        f"{image_meta['file_name']}: {exc}"
                    )
                    image_predictions = []

                batch_predictions.append(image_predictions)

        for image_meta, image_predictions in zip(batch_valid, batch_predictions):
            processed_images += 1
            image_id = image_meta["image_id"]
            file_name = image_meta["file_name"]
            width = image_meta["width"]
            height = image_meta["height"]

            image_raw_items = [] if raw_predictions is not None else None

            for prediction in image_predictions:
                label_text = str(prediction["label"])
                normalized_label = normalize_label(label_text)
                score_value = float(prediction["score"])
                box_xyxy = clip_xyxy_to_image(
                    prediction["box_xyxy"],
                    width=width,
                    height=height,
                )
                coco_box = xyxy_to_coco_bbox(box_xyxy)

                cached_hit = resolved_label_cache.get(normalized_label)
                if cached_hit is None:
                    macro_id, match_score, matched_alias = resolve_label_to_macro_id(
                        label=normalized_label,
                        alias_to_macro_id=alias_to_macro_id,
                        alias_index=alias_index,
                    )
                    cached_hit = (macro_id, match_score, matched_alias)
                    resolved_label_cache[normalized_label] = cached_hit
                else:
                    macro_id, match_score, matched_alias = cached_hit

                total_predictions += 1
                if macro_id is not None:
                    mapped_predictions += 1
                    detection_results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(macro_id),
                            "bbox": [round(value, 3) for value in coco_box],
                            "score": round(score_value, 6),
                            detection_label_key: label_text,
                        }
                    )
                else:
                    unmapped_counts[normalized_label] = (
                        unmapped_counts.get(normalized_label, 0) + 1
                    )

                if image_raw_items is not None:
                    image_raw_items.append(
                        {
                            "label": label_text,
                            "normalized_label": normalized_label,
                            "score": round(score_value, 6),
                            "bbox_xyxy": [round(value, 3) for value in box_xyxy],
                            "bbox": [round(value, 3) for value in coco_box],
                            "mapped_category_id": int(macro_id)
                            if macro_id is not None
                            else None,
                            "mapped_category_name": macro_id_to_name.get(int(macro_id))
                            if macro_id is not None
                            else None,
                            "label_match_score": round(float(match_score), 4),
                            "matched_alias": matched_alias,
                        }
                    )

            if raw_predictions is not None:
                raw_predictions.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "width": int(width),
                        "height": int(height),
                        "detections": image_raw_items,
                    }
                )

        if processed_images > 0 and processed_images % 50 == 0:
            elapsed = max(1e-6, time.perf_counter() - start_time)
            print(
                f"Processed images={processed_images}/{total_images} "
                f"({processed_images / elapsed:.2f} img/s) | "
                f"mapped={mapped_predictions}/{total_predictions} | "
                f"elapsed={elapsed:.1f}s"
            )

    with (output_folder / "detection_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(detection_results, handle, separators=(",", ":"))

    raw_payload = raw_predictions if raw_predictions is not None else []
    with (output_folder / "raw_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(raw_payload, handle, separators=(",", ":"))

    prompt_context = {
        "detector_name": detector_name,
        "model_id": model_id,
        "inference_backend": selected_backend,
        "rex_omni_backend": rex_omni_backend
        if selected_backend == "rex_omni"
        else None,
        "rex_omni_attn_implementation": (
            rex_omni_attn_implementation if selected_backend == "rex_omni" else None
        ),
        "rex_omni_torch_dtype": (
            rex_omni_torch_dtype if selected_backend == "rex_omni" else None
        ),
        "rex_omni_device_map": (
            rex_omni_device_map if selected_backend == "rex_omni" else None
        ),
        "device": str(resolved_device),
        "score_thresh": float(score_thresh),
        "text_thresh": float(text_thresh),
        "detection_label_key": detection_label_key,
        "prompt_file": str(prompt_file),
        "save_raw_predictions": bool(save_raw_predictions),
        "prompt_text": prompt_text,
        "prompt_phrases_count": len(prompt_phrases),
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
                for label, count in unmapped_counts.items()
                if label
            ),
            key=lambda item: item["count"],
            reverse=True,
        )[:50],
    }
    (output_folder / "prompt_context.json").write_text(
        json.dumps(prompt_context, indent=2), encoding="utf-8"
    )

    print(
        f"{detector_name} inference complete | backend={selected_backend} images={len(images)} "
        f"detections={len(detection_results)} mapped={mapped_predictions}/{total_predictions}"
    )
    print(f"Detection results: {output_folder / 'detection_results.json'}")
    print(f"Raw predictions: {output_folder / 'raw_predictions.json'}")


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
    detector_name: str = "Grounding DINO",
    detection_label_key: str = "dino_label",
    inference_backend: str = "transformers",
    rex_omni_backend: str = "transformers",
    rex_omni_attn_implementation: str = "eager",
    rex_omni_torch_dtype: str = "float16",
    rex_omni_device_map: str = "auto",
) -> None:
    run_open_vocabulary_inference(
        img_dir=img_dir,
        ann_file=ann_file,
        output_folder=output_folder,
        model_id=model_id,
        xview_classes_path=xview_classes_path,
        xview_macro_classes_path=xview_macro_classes_path,
        prompt_file=prompt_file,
        score_thresh=score_thresh,
        text_thresh=text_thresh,
        device=device,
        batch_size=batch_size,
        max_images=max_images,
        use_amp=use_amp,
        save_raw_predictions=save_raw_predictions,
        detector_name=detector_name,
        detection_label_key=detection_label_key,
        inference_backend=inference_backend,
        rex_omni_backend=rex_omni_backend,
        rex_omni_attn_implementation=rex_omni_attn_implementation,
        rex_omni_torch_dtype=rex_omni_torch_dtype,
        rex_omni_device_map=rex_omni_device_map,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run open-vocabulary inference and export class-aware COCO detections"
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
    parser.add_argument("--detector_name", type=str, default="Grounding DINO")
    parser.add_argument("--detection_label_key", type=str, default="dino_label")
    parser.add_argument(
        "--inference_backend",
        type=str,
        choices=["auto", "transformers", "rex_omni"],
        default="auto",
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

    args = parser.parse_args()
    run_open_vocabulary_inference(
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
        detector_name=args.detector_name,
        detection_label_key=args.detection_label_key,
        inference_backend=args.inference_backend,
        rex_omni_backend=args.rex_omni_backend,
        rex_omni_attn_implementation=args.rex_omni_attn_implementation,
        rex_omni_torch_dtype=args.rex_omni_torch_dtype,
        rex_omni_device_map=args.rex_omni_device_map,
    )


if __name__ == "__main__":
    main()
