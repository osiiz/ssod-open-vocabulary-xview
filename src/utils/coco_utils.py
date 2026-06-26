"""Utility functions for COCO dataset evaluation and result formatting.
"""
from pycocotools.cocoeval import COCOeval


def build_coco_max_dets(max_detections: int) -> list[int]:
    """Build a valid COCO maxDets triple aligned to a detection cap.

    The returned value is monotonic and keeps the conventional structure
    [small, medium, large], where large equals the configured detection cap.
    """

    max_detections = max(1, int(max_detections))
    medium = min(500, max_detections)
    small = min(100, medium)
    return [small, medium, max_detections]


def stats2dict(coco_eval: COCOeval) -> dict:
    """Convert COCOeval stats to a dictionary with readable keys."""

    maxDets = coco_eval.params.maxDets

    dynamic_keys = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        f"AR_{maxDets[0]}",
        f"AR_{maxDets[1]}",
        f"AR_{maxDets[2]}",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]

    metrics = {}
    for k, s in zip(dynamic_keys, coco_eval.stats):
        metrics[k] = round(float(s), 4)

    return metrics


def torch2coco_results(
    results: list, coco_img_id: int, categories: list, score_thresh: float = 0.5
) -> list:
    """Convert torchvision model outputs to COCO results format."""

    # Score thresholding
    keep = results["scores"] > score_thresh
    boxes = results["boxes"][keep].cpu().numpy()
    labels = results["labels"][keep].cpu().numpy()
    scores = results["scores"][keep].cpu().numpy()

    print(f"Image ID: {coco_img_id}, Detections kept: {len(boxes)}")

    detections = []

    for box, label, score in zip(boxes, labels, scores):

        class_name = (
            categories[label]
            if categories is not None and label < len(categories)
            else str(label)
        )
        x_min, y_min, x_max, y_max = box
        coco_box = [x_min, y_min, x_max - x_min, y_max - y_min]
        coco_box = [round(float(x), 1) for x in coco_box]

        detection = {
            "image_id": int(coco_img_id),
            "category_id": int(label),
            "bbox": coco_box,
            "score": round(float(score), 4),
        }

        detections.append(detection)

    return detections
