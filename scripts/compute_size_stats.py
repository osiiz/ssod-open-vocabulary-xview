"""Estadísticas del dataset xView por tamaño de objeto (definiciones COCO).

COCO size thresholds:
  small  : area <  1024 px²  (bbox < 32×32)
  medium : area in [1024, 9216) px²  (32×32 – 96×96)
  large  : area >= 9216 px²  (bbox >= 96×96)

Uso:
    python scripts/compute_size_stats.py
    python scripts/compute_size_stats.py --splits train val test --output docs/dataset_reports/size_distribution.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SMALL_MAX = 1024
MEDIUM_MAX = 9216

SPLITS = {
    "train": "results/preprocess/tile_images/train/COCO_annotations.json",
    "train_sampled": "results/preprocess/tile_images/train_sampled/COCO_annotations.json",
    "val": "results/preprocess/tile_images/val/COCO_annotations.json",
    "test": "results/preprocess/tile_images/test/COCO_annotations.json",
    "train_unlabeled_eval": "results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json",
}


def size_label(area: float) -> str:
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def compute_stats(ann_file: Path) -> dict:
    data = json.loads(ann_file.read_text())
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    annotations = data["annotations"]

    total = len(annotations)
    global_counts: dict[str, int] = defaultdict(int)
    per_class: dict[str, dict[str, int]] = {
        name: defaultdict(int) for name in cat_map.values()
    }

    for ann in annotations:
        area = ann.get("area", ann["bbox"][2] * ann["bbox"][3])
        label = size_label(area)
        global_counts[label] += 1
        cat_name = cat_map[ann["category_id"]]
        per_class[cat_name][label] += 1

    def pct(n, d):
        return round(100 * n / d, 1) if d > 0 else 0.0

    result = {
        "total_annotations": total,
        "global": {
            sz: {"count": global_counts[sz], "pct_total": pct(global_counts[sz], total)}
            for sz in ("small", "medium", "large")
        },
        "per_class": {},
    }
    for cat_name, counts in per_class.items():
        cat_total = sum(counts.values())
        result["per_class"][cat_name] = {
            "total": cat_total,
            "pct_of_dataset": pct(cat_total, total),
        }
        for sz in ("small", "medium", "large"):
            result["per_class"][cat_name][sz] = {
                "count": counts[sz],
                "pct_of_class": pct(counts[sz], cat_total),
                "pct_of_dataset": pct(counts[sz], total),
            }
    return result


def print_global_table(split: str, stats: dict) -> None:
    total = stats["total_annotations"]
    print(f"\n{'='*60}")
    print(f"Split: {split}  |  Total anotaciones: {total:,}")
    print(f"{'='*60}")
    print(f"{'Tamaño':<10} {'Count':>10} {'% total':>8}")
    print(f"{'-'*30}")
    for sz in ("small", "medium", "large"):
        s = stats["global"][sz]
        print(f"{sz:<10} {s['count']:>10,} {s['pct_total']:>7.1f}%")


def print_per_class_table(split: str, stats: dict) -> None:
    print(f"\n--- Por clase × tamaño ({split}) ---")
    header = f"{'Clase':<22} {'Total':>7} {'small%':>7} {'medium%':>8} {'large%':>7}"
    print(header)
    print("-" * len(header))
    for cat_name, d in stats["per_class"].items():
        s = d.get("small", {}).get("pct_of_class", 0)
        m = d.get("medium", {}).get("pct_of_class", 0)
        lg = d.get("large", {}).get("pct_of_class", 0)
        print(f"{cat_name:<22} {d['total']:>7,} {s:>6.1f}% {m:>7.1f}% {lg:>6.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=list(SPLITS.keys()),
    )
    parser.add_argument(
        "--output", default="docs/dataset_reports/size_distribution.json"
    )
    args = parser.parse_args()

    output: dict[str, dict] = {}
    for split in args.splits:
        ann_file = Path(SPLITS[split])
        if not ann_file.exists():
            print(f"[SKIP] {split}: {ann_file} no existe")
            continue
        print(f"[{split}] Cargando {ann_file}...")
        stats = compute_stats(ann_file)
        output[split] = stats
        print_global_table(split, stats)
        print_per_class_table(split, stats)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
