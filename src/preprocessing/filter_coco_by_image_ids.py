import argparse
import json
from pathlib import Path


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}")

    return data


def extract_image_ids(coco: dict) -> set[int]:
    image_ids = set()
    for image in coco.get("images", []):
        image_id = image.get("id")
        if image_id is None:
            continue
        image_ids.add(int(image_id))
    return image_ids


def filter_coco_by_image_ids(
    source_coco: dict,
    selected_image_ids: set[int],
    include_annotations: bool = True,
) -> tuple[dict, list[int]]:
    selected_ids = {int(image_id) for image_id in selected_image_ids}

    filtered_images = [
        image
        for image in source_coco.get("images", [])
        if int(image.get("id", -1)) in selected_ids
    ]
    kept_image_ids = {int(image["id"]) for image in filtered_images if "id" in image}

    filtered_annotations = []
    if include_annotations:
        filtered_annotations = [
            annotation
            for annotation in source_coco.get("annotations", [])
            if int(annotation.get("image_id", -1)) in kept_image_ids
        ]

    missing_ids = sorted(selected_ids - kept_image_ids)

    filtered_coco = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": source_coco.get("categories", []),
    }

    return filtered_coco, missing_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a source COCO file by image IDs from another COCO file. "
            "Useful to create an evaluable unlabeled subset with GT annotations."
        )
    )
    parser.add_argument(
        "source_coco",
        type=Path,
        help="COCO source file containing GT annotations (e.g. xview_train.json).",
    )
    parser.add_argument(
        "selection_coco",
        type=Path,
        help="COCO file whose image IDs define the subset to keep.",
    )
    parser.add_argument(
        "output_coco",
        type=Path,
        help="Path where filtered COCO JSON will be written.",
    )
    parser.add_argument(
        "--strip_annotations",
        action="store_true",
        help="If set, output annotations will be empty.",
    )
    args = parser.parse_args()

    source_coco = load_coco(args.source_coco)
    selection_coco = load_coco(args.selection_coco)
    selected_image_ids = extract_image_ids(selection_coco)

    filtered_coco, missing_ids = filter_coco_by_image_ids(
        source_coco,
        selected_image_ids,
        include_annotations=not args.strip_annotations,
    )

    args.output_coco.parent.mkdir(parents=True, exist_ok=True)
    with args.output_coco.open("w", encoding="utf-8") as handle:
        json.dump(filtered_coco, handle, indent=2)

    print(
        "Filtered COCO written to ",
        args.output_coco,
        f"| images={len(filtered_coco['images'])}",
        f"annotations={len(filtered_coco['annotations'])}",
        f"missing_image_ids={len(missing_ids)}",
        sep="",
    )

    if missing_ids:
        preview = ", ".join(str(image_id) for image_id in missing_ids[:10])
        suffix = " ..." if len(missing_ids) > 10 else ""
        print(f"Missing image IDs not found in source: {preview}{suffix}")


if __name__ == "__main__":
    main()
