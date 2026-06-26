import tempfile
from pathlib import Path

import torch

from src.inference.open_vocabulary_inference import (
    build_alias_index,
    build_prompt_context,
    choose_inference_backend,
    parse_torch_dtype,
    parse_rexomni_detection_output,
    parse_prompt_phrases,
    resolve_label_to_macro_id,
)


def test_parse_prompt_phrases_reads_prompt_lines_and_splits_items():
    content = """
Prompt 1: fixed-wing aircraft . cargo truck .
Prompt 2: hut . tower .
Other line to ignore
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
        handle.write(content)
        prompt_path = Path(handle.name)

    try:
        phrases = parse_prompt_phrases(prompt_path)
    finally:
        prompt_path.unlink()

    assert phrases == [
        "fixed-wing aircraft",
        "cargo truck",
        "hut",
        "tower",
    ]


def test_parse_prompt_phrases_reads_structured_yaml_prompt_file():
    content = """
version: 1
language: en
prompt_groups:
  - name: group_1
    phrases:
      - fixed-wing aircraft
      - cargo truck
  - name: group_2
    phrases:
      - hut
      - tower
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        handle.write(content)
        prompt_path = Path(handle.name)

    try:
        phrases = parse_prompt_phrases(prompt_path)
    finally:
        prompt_path.unlink()

    assert phrases == [
        "fixed-wing aircraft",
        "cargo truck",
        "hut",
        "tower",
    ]


def test_resolve_label_to_macro_id_handles_exact_and_fuzzy_matches():
    xview_classes = {
        "11": "Fixed-wing Aircraft",
        "13": "Passenger/Cargo Plane",
        "24": "Cargo Truck",
        "73": "Building",
        "93": "Pylon",
    }
    xview_macro_classes = {
        "11": {"id": 1, "name": "Aircraft"},
        "13": {"id": 1, "name": "Aircraft"},
        "24": {"id": 3, "name": "Heavy Vehicle"},
        "73": {"id": 7, "name": "Building"},
        "93": {"id": 9, "name": "Tower & Pylon"},
    }

    prompt_text, alias_to_macro_id, _ = build_prompt_context(
        xview_classes=xview_classes,
        xview_macro_classes=xview_macro_classes,
        prompt_phrases=["passenger plane", "cargo truck", "hut", "tower"],
    )

    assert "passenger plane" in prompt_text

    alias_index = build_alias_index(alias_to_macro_id)

    macro_id, score, _ = resolve_label_to_macro_id(
        "passenger plane", alias_to_macro_id, alias_index
    )
    assert macro_id == 1
    assert score >= 0.45

    macro_id, score, _ = resolve_label_to_macro_id(
        "cargo truck", alias_to_macro_id, alias_index
    )
    assert macro_id == 3
    assert score >= 0.45

    macro_id, score, _ = resolve_label_to_macro_id(
        "hut", alias_to_macro_id, alias_index
    )
    assert macro_id == 7
    assert score >= 0.45

    macro_id, score, _ = resolve_label_to_macro_id(
        "tower", alias_to_macro_id, alias_index
    )
    assert macro_id == 9
    assert score >= 0.45


def test_choose_inference_backend_auto_detects_rexomni_from_model_or_detector():
    assert (
        choose_inference_backend(
            model_id="IDEA-Research/Rex-Omni",
            detector_name="Anything",
            inference_backend="auto",
        )
        == "rex_omni"
    )

    assert (
        choose_inference_backend(
            model_id="IDEA-Research/grounding-dino-base",
            detector_name="RexOmni",
            inference_backend="auto",
        )
        == "rex_omni"
    )

    assert (
        choose_inference_backend(
            model_id="IDEA-Research/grounding-dino-base",
            detector_name="Grounding DINO",
            inference_backend="auto",
        )
        == "transformers"
    )


def test_choose_inference_backend_honors_explicit_override():
    assert (
        choose_inference_backend(
            model_id="IDEA-Research/Rex-Omni",
            detector_name="RexOmni",
            inference_backend="transformers",
        )
        == "transformers"
    )


def test_parse_rexomni_detection_output_supports_list_and_nested_shapes():
    output = [
        {
            "extracted_predictions": [
                {
                    "label": "small car",
                    "score": 0.81,
                    "bbox": [10, 20, 30, 40],
                },
                {
                    "category": "tower",
                    "confidence": 0.55,
                    "box": {"x1": 1, "y1": 2, "x2": 11, "y2": 12},
                },
                {
                    "label": "missing box",
                    "score": 0.1,
                },
            ]
        }
    ]

    normalized = parse_rexomni_detection_output(output)

    assert len(normalized) == 2
    assert normalized[0]["label"] == "small car"
    assert normalized[0]["score"] == 0.81
    assert normalized[0]["box_xyxy"] == [10.0, 20.0, 30.0, 40.0]

    assert normalized[1]["label"] == "tower"
    assert normalized[1]["score"] == 0.55
    assert normalized[1]["box_xyxy"] == [1.0, 2.0, 11.0, 12.0]


def test_parse_rexomni_detection_output_converts_xywh_when_flagged():
    output = {
        "extracted_predictions": {
            "detections": [
                {
                    "class_name": "hut",
                    "probability": 0.4,
                    "bbox": [5, 6, 7, 8],
                    "bbox_format": "xywh",
                }
            ]
        }
    }

    normalized = parse_rexomni_detection_output(output)
    assert len(normalized) == 1
    assert normalized[0]["label"] == "hut"
    assert normalized[0]["score"] == 0.4
    assert normalized[0]["box_xyxy"] == [5.0, 6.0, 12.0, 14.0]


def test_parse_rexomni_detection_output_supports_category_map_with_box_type():
    output = {
        "extracted_predictions": {
            "small aircraft": [
                {"type": "box", "coords": [10, 20, 30, 40]},
                {"type": "point", "coords": [5, 6]},
            ],
            "tower": [
                {"bbox": [1, 2, 11, 12], "score": 0.7},
            ],
            "empty": [],
        }
    }

    normalized = parse_rexomni_detection_output(output)

    assert len(normalized) == 2
    assert normalized[0]["label"] == "small aircraft"
    assert normalized[0]["score"] == 1.0
    assert normalized[0]["box_xyxy"] == [10.0, 20.0, 30.0, 40.0]

    assert normalized[1]["label"] == "tower"
    assert normalized[1]["score"] == 0.7
    assert normalized[1]["box_xyxy"] == [1.0, 2.0, 11.0, 12.0]


def test_parse_rexomni_detection_output_falls_back_to_raw_output_when_needed():
    output = [
        {
            "raw_output": (
                "<|object_ref_start|>small car<|object_ref_end|><|box_start|>"
                "<100><200><300><400>,<10><20><30><40><|box_end|>,"
                " <|object_ref_start|>building<|object_ref_end|><|box_start|>"
                "None<|box_end|><|im_end|>"
            ),
            "extracted_predictions": {
                "small car": [],
                "building": [],
            },
        }
    ]

    normalized = parse_rexomni_detection_output(output, image_size=(1000, 500))

    assert len(normalized) == 2
    assert normalized[0]["label"] == "small car"
    assert normalized[0]["score"] == 1.0
    assert normalized[0]["box_xyxy"] == [
        100.10010010010011,
        100.10010010010011,
        300.3003003003003,
        200.20020020020021,
    ]

    assert normalized[1]["label"] == "small car"
    assert normalized[1]["score"] == 1.0
    assert normalized[1]["box_xyxy"] == [
        10.01001001001001,
        10.01001001001001,
        30.03003003003003,
        20.02002002002002,
    ]


def test_parse_torch_dtype_supports_expected_aliases():
    assert parse_torch_dtype("float16") == torch.float16
    assert parse_torch_dtype("fp16") == torch.float16
    assert parse_torch_dtype("bfloat16") == torch.bfloat16
    assert parse_torch_dtype("bf16") == torch.bfloat16
    assert parse_torch_dtype("float32") == torch.float32
