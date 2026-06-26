import json
import os
from pathlib import Path

from src.inference.ov_coco_eval import evaluate_ov_predictions, materialize_artifact


def build_tiny_coco_gt(path: Path) -> None:
    gt = {
        "images": [
            {"id": 1, "file_name": "img_1.tif", "width": 100, "height": 100},
            {"id": 2, "file_name": "img_2.tif", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "bbox": [30, 30, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "Aircraft"},
            {"id": 2, "name": "Light Vehicle"},
        ],
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(gt, handle)


def build_tiny_detections(path: Path) -> None:
    detections = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 10, 20, 20],
            "score": 0.9,
            "dino_label": "small aircraft",
        },
        {
            "image_id": 2,
            "category_id": 2,
            "bbox": [30, 30, 20, 20],
            "score": 0.9,
            "dino_label": "small car",
        },
    ]

    with path.open("w", encoding="utf-8") as handle:
        json.dump(detections, handle)


def test_evaluate_ov_predictions_aware_and_agnostic(tmp_path):
    ann_file = tmp_path / "gt.json"
    detections_file = tmp_path / "detections.json"

    build_tiny_coco_gt(ann_file)
    build_tiny_detections(detections_file)

    aware_output = tmp_path / "aware"
    evaluate_ov_predictions(
        ann_file=ann_file,
        detection_results=detections_file,
        output_folder=aware_output,
        mode="aware",
        max_dets=100,
    )

    assert (aware_output / "detection_results.json").exists()
    assert (aware_output / "metrics.json").exists()
    assert (aware_output / "metrics_per_class.json").exists()

    aware_metrics = json.loads(
        (aware_output / "metrics.json").read_text(encoding="utf-8")
    )
    aware_per_class = json.loads(
        (aware_output / "metrics_per_class.json").read_text(encoding="utf-8")
    )

    assert "AP50" in aware_metrics
    assert "Aircraft" in aware_per_class

    agnostic_output = tmp_path / "agnostic"
    evaluate_ov_predictions(
        ann_file=ann_file,
        detection_results=detections_file,
        output_folder=agnostic_output,
        mode="agnostic",
        max_dets=100,
    )

    agnostic_metrics = json.loads(
        (agnostic_output / "metrics.json").read_text(encoding="utf-8")
    )
    agnostic_per_class = json.loads(
        (agnostic_output / "metrics_per_class.json").read_text(encoding="utf-8")
    )

    assert "AP50" in agnostic_metrics
    # metrics_per_class uses category names as keys (same format as aware mode)
    assert "Aircraft" in agnostic_per_class
    assert "Light Vehicle" in agnostic_per_class


def test_materialize_artifact_hardlink_mode(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}', encoding="utf-8")

    destination = tmp_path / "dest.json"
    materialize_artifact(source, destination, mode="hardlink")

    assert destination.exists()
    assert destination.read_text(encoding="utf-8") == '{"ok": true}'
    assert os.stat(source).st_ino == os.stat(destination).st_ino


def test_evaluate_dino_materializes_raw_when_requested(tmp_path):
    ann_file = tmp_path / "gt.json"
    detections_file = tmp_path / "detections.json"
    raw_file = tmp_path / "raw_predictions.json"

    build_tiny_coco_gt(ann_file)
    build_tiny_detections(detections_file)
    raw_file.write_text('[{"image_id": 1, "detections": []}]', encoding="utf-8")

    aware_output = tmp_path / "aware_with_raw"
    evaluate_ov_predictions(
        ann_file=ann_file,
        detection_results=detections_file,
        output_folder=aware_output,
        mode="aware",
        max_dets=100,
        raw_predictions=raw_file,
        artifact_mode="auto",
        materialize_raw=True,
    )

    assert (aware_output / "raw_predictions.json").exists()
