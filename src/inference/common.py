"""Shared utilities for open-vocabulary inference pipelines.

This module is the common layer behind `dino_inference.py` and
`rex_inference.py`. It owns everything that is detector-agnostic:

* Prompt file parsing (YAML / JSON / legacy TXT).
* Label normalization and alias indexing.
* Mapping raw model labels to xView macro-class IDs.
* COCO-style bounding box helpers.
* Batch iteration.
* JSON I/O.

Detector-specific logic (model loading, forward pass, output post-processing)
lives in the dedicated scripts so this module has no heavy ML imports.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def str2bool(value: str | bool) -> bool:
    """argparse-compatible boolean parser."""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    """Yield successive batches of `batch_size` from `items`."""
    effective_batch_size = max(1, int(batch_size))
    for start in range(0, len(items), effective_batch_size):
        yield items[start : start + effective_batch_size]


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


def normalize_label(text: str) -> str:
    """Lowercase, strip diacritics, collapse separators to single spaces."""
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


# ---------------------------------------------------------------------------
# Prompt file parsing
# ---------------------------------------------------------------------------


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
    """Parse a plain-text prompt file (one phrase per line, or dot-separated)."""
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
    """Parse a YAML or JSON prompt file with `phrases` and/or `prompt_groups`."""
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
    """Load and deduplicate prompt phrases from a YAML/JSON/TXT file."""
    suffix = prompt_file.suffix.lower()
    if suffix in {".yaml", ".yml", ".json"}:
        return _parse_structured_prompt_file(prompt_file)
    return _parse_legacy_prompt_file(prompt_file)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Prompt context (alias map + canonical prompt string)
# ---------------------------------------------------------------------------


def build_prompt_context(
    xview_classes: dict,
    xview_macro_classes: dict,
    prompt_phrases: list[str],
    append_canonical_names: bool = True,
) -> tuple[str, dict[str, int], dict[int, str], list[str]]:
    """Assemble the alias->macro_id table and the final prompt string.

    Parameters
    ----------
    xview_classes : dict
        Mapping original_id -> original_name (fine-grained xView classes).
    xview_macro_classes : dict
        Mapping original_id -> {"id": macro_id, "name": macro_name}.
    prompt_phrases : list[str]
        Phrases loaded from the prompt YAML.
    append_canonical_names : bool
        Grounding DINO benefits from seeing every fine-grained canonical name
        in the prompt because it matches via text-image alignment. Rex-Omni
        should keep the prompt short to avoid token-generation blow-up, so it
        passes False here.

    Returns
    -------
    (prompt_text, alias_to_macro_id, macro_id_to_name, merged_prompts)
        `merged_prompts` is the list of phrases actually used (for Rex-Omni's
        categories API and for logging).
    """
    original_classes = {int(k): v for k, v in xview_classes.items()}
    macro_map = {int(k): v for k, v in xview_macro_classes.items()}

    valid_original_ids = sorted(set(original_classes.keys()) & set(macro_map.keys()))

    alias_to_macro_id: dict[str, int] = {}
    macro_id_to_name: dict[int, str] = {}
    canonical_original_names: list[str] = []

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

    merged_prompts: list[str] = []
    seen_prompt_keys: set[str] = set()

    sources = list(prompt_phrases)
    if append_canonical_names:
        sources += canonical_original_names

    for phrase in sources:
        key = normalize_label(phrase)
        if not key or key in seen_prompt_keys:
            continue
        seen_prompt_keys.add(key)
        merged_prompts.append(phrase)

    if not merged_prompts:
        raise ValueError("No valid prompt phrases found for open-vocabulary inference")

    prompt_text = " . ".join(merged_prompts) + " ."
    return prompt_text, alias_to_macro_id, macro_id_to_name, merged_prompts


# ---------------------------------------------------------------------------
# Alias matching (raw label -> macro_id)
# ---------------------------------------------------------------------------


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
    """Map a raw detector label to an xView macro-class id.

    Uses exact match first, then token-level Jaccard / containment similarity.
    Returns (macro_id | None, score, matched_alias | None).
    """
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


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared post-processing: raw predictions -> COCO detection_results
# ---------------------------------------------------------------------------


def accumulate_detection_records(
    *,
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    image_predictions: list[dict],
    alias_to_macro_id: dict[str, int],
    alias_index: list[tuple[str, int, set[str]]],
    macro_id_to_name: dict[int, str],
    resolved_label_cache: dict[str, tuple[int | None, float, str | None]],
    detection_label_key: str,
    save_raw_predictions: bool,
) -> tuple[list[dict], dict | None, dict[str, int], int, int]:
    """Convert a batch of raw predictions into COCO detection records.

    Returns
    -------
    (detection_entries, raw_entry_or_none, unmapped_counts, total, mapped)
    """
    detection_entries: list[dict] = []
    image_raw_items: list[dict] | None = [] if save_raw_predictions else None
    unmapped_counts: dict[str, int] = {}
    total = 0
    mapped = 0

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

        total += 1
        if macro_id is not None:
            mapped += 1
            detection_entries.append(
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
                    "mapped_category_name": (
                        macro_id_to_name.get(int(macro_id))
                        if macro_id is not None
                        else None
                    ),
                    "label_match_score": round(float(match_score), 4),
                    "matched_alias": matched_alias,
                }
            )

    raw_entry = None
    if image_raw_items is not None:
        raw_entry = {
            "image_id": image_id,
            "file_name": file_name,
            "width": int(width),
            "height": int(height),
            "detections": image_raw_items,
        }

    return detection_entries, raw_entry, unmapped_counts, total, mapped


def write_outputs(
    *,
    output_folder: Path,
    detection_results: list[dict],
    raw_predictions: list[dict] | None,
    prompt_context: dict,
) -> None:
    """Persist detection_results.json, raw_predictions.json, prompt_context.json."""
    output_folder.mkdir(parents=True, exist_ok=True)

    with (output_folder / "detection_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(detection_results, handle, separators=(",", ":"))

    raw_payload = raw_predictions if raw_predictions is not None else []
    with (output_folder / "raw_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(raw_payload, handle, separators=(",", ":"))

    (output_folder / "prompt_context.json").write_text(
        json.dumps(prompt_context, indent=2), encoding="utf-8"
    )
