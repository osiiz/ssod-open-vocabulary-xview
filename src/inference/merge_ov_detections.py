"""
Merge open-vocabulary detection results from a base (r80) inference run and a
delta inference run into a combined output (r90).

The base and delta runs were made on different tile COCO annotations, so their
image_ids differ from the final merged annotation. This script remaps all
image_ids using tile filenames (which are deterministic: img_{src_id}_{r}_{c}.tif)
as the common key between annotation files.

Usage:
    python -m src.inference.merge_ov_detections \
        --base_ann_file  results/preprocess/tile_images/r80_unlabeled_eval_coco_backup.json \
        --base_dir       results/dino/r80_dino_raw \
        --delta_ann_file results/preprocess/tile_images/delta_unlabeled_eval/COCO_annotations.json \
        --delta_dir      results/dino/r10delta_dino_raw \
        --merged_ann_file results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \
        --output_dir     results/dino/r90_dino_raw
"""

import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def build_id_to_filename(coco_images):
    return {img["id"]: img["file_name"] for img in coco_images}


def build_filename_to_id(coco_images):
    return {img["file_name"]: img["id"] for img in coco_images}


def remap_detections(detections, old_id_to_fn, new_fn_to_id):
    remapped = []
    skipped = 0
    for det in detections:
        fn = old_id_to_fn.get(det["image_id"])
        if fn is None:
            skipped += 1
            continue
        new_id = new_fn_to_id.get(fn)
        if new_id is None:
            skipped += 1
            continue
        remapped.append({**det, "image_id": new_id})
    if skipped:
        print(
            f"  Warning: {skipped} detections had no match in merged annotations (skipped)"
        )
    return remapped


def merge_prompt_context(base_pc, delta_pc):
    merged = dict(base_pc)
    numeric_sum_keys = [
        "total_predictions",
        "mapped_predictions",
        "unmapped_predictions",
        "failed_images",
    ]
    for key in numeric_sum_keys:
        if key in base_pc and key in delta_pc:
            merged[key] = base_pc[key] + delta_pc[key]

    # Merge top_unmapped_labels (list of {label, count} dicts)
    if "top_unmapped_labels" in base_pc and "top_unmapped_labels" in delta_pc:
        combined: dict = {}
        for entry in base_pc["top_unmapped_labels"] or []:
            combined[entry["label"]] = combined.get(entry["label"], 0) + entry["count"]
        for entry in delta_pc["top_unmapped_labels"] or []:
            combined[entry["label"]] = combined.get(entry["label"], 0) + entry["count"]
        merged["top_unmapped_labels"] = sorted(
            [{"label": k, "count": v} for k, v in combined.items()],
            key=lambda x: x["count"],
            reverse=True,
        )

    merged["_merge_note"] = "Merged from base (r80) + delta (r10delta) runs"
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_ann_file",
        required=True,
        help="Tile COCO annotations for the base (r80) run",
    )
    parser.add_argument(
        "--base_dir", required=True, help="Directory with base (r80) inference outputs"
    )
    parser.add_argument(
        "--delta_ann_file",
        required=True,
        help="Tile COCO annotations for the delta run",
    )
    parser.add_argument(
        "--delta_dir", required=True, help="Directory with delta inference outputs"
    )
    parser.add_argument(
        "--merged_ann_file",
        required=True,
        help="Tile COCO annotations for the new merged (r90) dataset",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for merged results"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    delta_dir = Path(args.delta_dir)
    out_dir = Path(args.output_dir)

    print("Loading annotation files...")
    base_coco = load_json(args.base_ann_file)
    delta_coco = load_json(args.delta_ann_file)
    merged_coco = load_json(args.merged_ann_file)

    base_id_to_fn = build_id_to_filename(base_coco["images"])
    delta_id_to_fn = build_id_to_filename(delta_coco["images"])
    merged_fn_to_id = build_filename_to_id(merged_coco["images"])

    print(
        f"Base tiles: {len(base_id_to_fn)}, Delta tiles: {len(delta_id_to_fn)}, Merged tiles: {len(merged_fn_to_id)}"
    )

    # --- detection_results.json ---
    print("Remapping base detections...")
    base_detections = load_json(base_dir / "detection_results.json")
    remapped_base = remap_detections(base_detections, base_id_to_fn, merged_fn_to_id)
    print(f"  Base: {len(base_detections)} → {len(remapped_base)} remapped")

    print("Remapping delta detections...")
    delta_detections = load_json(delta_dir / "detection_results.json")
    remapped_delta = remap_detections(delta_detections, delta_id_to_fn, merged_fn_to_id)
    print(f"  Delta: {len(delta_detections)} → {len(remapped_delta)} remapped")

    merged_detections = remapped_base + remapped_delta
    print(f"Total merged detections: {len(merged_detections)}")
    write_json(out_dir / "detection_results.json", merged_detections)

    # --- prompt_context.json ---
    base_pc_path = base_dir / "prompt_context.json"
    delta_pc_path = delta_dir / "prompt_context.json"
    if base_pc_path.exists() and delta_pc_path.exists():
        base_pc = load_json(base_pc_path)
        delta_pc = load_json(delta_pc_path)
        merged_pc = merge_prompt_context(base_pc, delta_pc)
        write_json(out_dir / "prompt_context.json", merged_pc)
        print("Merged prompt_context.json")
    elif base_pc_path.exists():
        write_json(out_dir / "prompt_context.json", load_json(base_pc_path))

    # --- raw_predictions.json ---
    write_json(out_dir / "raw_predictions.json", [])

    print(f"\nDone. Merged outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
