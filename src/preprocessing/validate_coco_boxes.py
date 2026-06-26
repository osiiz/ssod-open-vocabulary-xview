import argparse
import json
from collections import defaultdict
from pathlib import Path


def _clip_box_to_image(x, y, w, h, img_w, img_h):
    right = x + w
    bottom = y + h

    clipped_x = max(0.0, min(float(x), float(img_w)))
    clipped_y = max(0.0, min(float(y), float(img_h)))
    clipped_right = max(0.0, min(float(right), float(img_w)))
    clipped_bottom = max(0.0, min(float(bottom), float(img_h)))

    clipped_w = clipped_right - clipped_x
    clipped_h = clipped_bottom - clipped_y
    return clipped_x, clipped_y, clipped_w, clipped_h


def validate_and_fix_boxes(
    input_coco_json: Path, output_coco_json: Path, report_file: Path
):
    with open(input_coco_json, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])

    stats = {
        "total_annotations": int(len(annotations)),
        "invalid_outside_frame": 0,
        "invalid_non_positive_area": 0,
        "clipped_annotations": 0,
        "removed_annotations": 0,
        "missing_image_reference": 0,
    }
    invalid_by_class = defaultdict(
        lambda: {
            "outside_frame": 0,
            "non_positive_area": 0,
            "removed": 0,
        }
    )

    fixed_annotations = []
    for ann in annotations:
        image_id = ann["image_id"]
        category_id = int(ann.get("category_id", -1))
        image_info = images.get(image_id)
        if image_info is None:
            stats["missing_image_reference"] += 1
            stats["removed_annotations"] += 1
            invalid_by_class[category_id]["removed"] += 1
            continue

        img_w = float(image_info["width"])
        img_h = float(image_info["height"])
        x, y, w, h = [float(v) for v in ann["bbox"]]

        outside_frame = x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h
        non_positive_area = w <= 0 or h <= 0

        if outside_frame:
            stats["invalid_outside_frame"] += 1
            invalid_by_class[category_id]["outside_frame"] += 1

        if non_positive_area:
            stats["invalid_non_positive_area"] += 1
            invalid_by_class[category_id]["non_positive_area"] += 1

        clipped_x, clipped_y, clipped_w, clipped_h = _clip_box_to_image(
            x, y, w, h, img_w, img_h
        )

        if (clipped_w <= 0) or (clipped_h <= 0):
            stats["removed_annotations"] += 1
            invalid_by_class[category_id]["removed"] += 1
            continue

        if outside_frame:
            stats["clipped_annotations"] += 1

        fixed_ann = dict(ann)
        fixed_ann["bbox"] = [clipped_x, clipped_y, clipped_w, clipped_h]
        fixed_ann["area"] = clipped_w * clipped_h
        fixed_annotations.append(fixed_ann)

    # Reasignar IDs de anotación para dejar consistencia interna
    for new_id, ann in enumerate(fixed_annotations):
        ann["id"] = int(new_id)

    fixed_coco = {
        "images": coco.get("images", []),
        "annotations": fixed_annotations,
        "categories": coco.get("categories", []),
    }

    output_coco_json.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_coco_json, "w") as f:
        json.dump(fixed_coco, f, indent=2)

    report = {
        "input_file": str(input_coco_json),
        "output_file": str(output_coco_json),
        "stats": stats,
        "remaining_annotations": int(len(fixed_annotations)),
        "invalid_by_class": {str(k): v for k, v in sorted(invalid_by_class.items())},
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved cleaned COCO annotations: {output_coco_json}")
    print(f"Saved invalid box report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Checks COCO bounding boxes for invalid geometry (outside frame or non-positive area), "
            "clips out-of-frame boxes, and removes invalid/degenerate boxes."
        )
    )
    parser.add_argument("input_coco_json", type=str, help="Input COCO annotations JSON")
    parser.add_argument("output_coco_json", type=str, help="Output cleaned COCO JSON")
    parser.add_argument(
        "--report_file",
        type=str,
        required=True,
        help="Path to JSON report with invalid-box statistics",
    )
    args = parser.parse_args()

    validate_and_fix_boxes(
        Path(args.input_coco_json),
        Path(args.output_coco_json),
        Path(args.report_file),
    )


if __name__ == "__main__":
    main()
