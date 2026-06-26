import json
from pathlib import Path

import cv2
import numpy as np

from pycocotools.coco import COCO

from src.utils.evaluate_per_image_performance import (
    compute_ap50_per_image,
    save_worst_visualizations,
)


def _write_tiny_coco_gt(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    img1 = np.zeros((64, 64, 3), dtype=np.uint8)
    img2 = np.zeros((64, 64, 3), dtype=np.uint8)
    img3 = np.zeros((64, 64, 3), dtype=np.uint8)

    cv2.imwrite(str(images_dir / "img1.png"), img1)
    cv2.imwrite(str(images_dir / "img2.png"), img2)
    cv2.imwrite(str(images_dir / "img3.png"), img3)

    gt = {
        "images": [
            {"id": 1, "file_name": "img1.png", "width": 64, "height": 64},
            {"id": 2, "file_name": "img2.png", "width": 64, "height": 64},
            {"id": 3, "file_name": "img3.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [30.0, 30.0, 16.0, 16.0],
                "area": 256.0,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "Building"}],
    }

    gt_path = tmp_path / "gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f)

    return gt_path, images_dir


def test_compute_ap50_per_image_handles_images_with_and_without_predictions(tmp_path):
    gt_path, _ = _write_tiny_coco_gt(tmp_path)
    coco_gt = COCO(str(gt_path))

    detections = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "score": 0.99,
        }
    ]

    dt_path = tmp_path / "detections.json"
    with open(dt_path, "w") as f:
        json.dump(detections, f)

    coco_dt = coco_gt.loadRes(str(dt_path))

    metrics = compute_ap50_per_image(
        coco_gt,
        coco_dt,
        eval_img_ids=[1, 2, 3],
        include_empty_images=False,
    )

    assert len(metrics) == 2

    by_id = {m["image_id"]: m for m in metrics}
    assert set(by_id.keys()) == {1, 2}

    assert by_id[1]["num_gt"] == 1
    assert by_id[1]["num_pred"] == 1
    assert 0.0 <= by_id[1]["ap50"] <= 1.0

    assert by_id[2]["num_gt"] == 1
    assert by_id[2]["num_pred"] == 0
    assert by_id[2]["ap50"] == 0.0


def test_save_worst_visualizations_creates_ranked_outputs(tmp_path):
    gt_path, images_dir = _write_tiny_coco_gt(tmp_path)
    coco_gt = COCO(str(gt_path))

    detections = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "score": 0.95,
        },
        {
            "image_id": 2,
            "category_id": 1,
            "bbox": [5.0, 5.0, 10.0, 10.0],
            "score": 0.20,
        },
    ]

    worst_images = [
        {
            "image_id": 2,
            "file_name": "img2.png",
            "ap50": 0.0,
            "num_gt": 1,
            "num_pred": 1,
        },
        {
            "image_id": 1,
            "file_name": "img1.png",
            "ap50": 0.5,
            "num_gt": 1,
            "num_pred": 1,
        },
    ]

    output_dir = tmp_path / "out"
    save_worst_visualizations(
        coco_gt=coco_gt,
        detections=detections,
        worst_images=worst_images,
        img_dir=str(images_dir),
        output_dir=output_dir,
        vis_score_thresh=0.0,
    )

    vis_dir = output_dir / "worst_images"
    assert vis_dir.exists()

    files = sorted(p.name for p in vis_dir.glob("*.png"))
    assert len(files) == 2
    assert files[0].startswith("rank_01_ap50_0.0000_img2")
    assert files[1].startswith("rank_02_ap50_0.5000_img1")
