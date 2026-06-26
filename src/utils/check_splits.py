import argparse
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_tiled_counts(json_path: Path):
    if not json_path.exists():
        return {}, {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = defaultdict(int)
    for ann in data.get("annotations", []):
        counts[ann["category_id"]] += 1

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    return counts, categories


def evaluate_tiled_splits(tiled_dir: Path, ratios: tuple):
    train_path = tiled_dir / "train" / "COCO_annotations.json"
    val_path = tiled_dir / "val" / "COCO_annotations.json"
    test_path = tiled_dir / "test" / "COCO_annotations.json"

    train_counts, cats_train = load_tiled_counts(train_path)
    val_counts, cats_val = load_tiled_counts(val_path)
    test_counts, cats_test = load_tiled_counts(test_path)

    categories = {**cats_train, **cats_val, **cats_test}
    all_class_ids = (
        set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())
    )

    targets = {
        "Train": ratios[0] * 100,
        "Val": ratios[1] * 100,
        "Test": ratios[2] * 100,
    }
    records = []
    deviations = []

    for cid in all_class_ids:
        cls_name = categories.get(cid, str(cid))
        tr_c = train_counts.get(cid, 0)
        v_c = val_counts.get(cid, 0)
        te_c = test_counts.get(cid, 0)

        total = tr_c + v_c + te_c
        if total == 0:
            continue

        tr_pct = (tr_c / total) * 100
        v_pct = (v_c / total) * 100
        te_pct = (te_c / total) * 100

        dev_tr = abs(tr_pct - targets["Train"])
        dev_v = abs(v_pct - targets["Val"])
        dev_te = abs(te_pct - targets["Test"])

        deviations.extend([dev_tr, dev_v, dev_te])

        records.append(
            {
                "clase": cls_name,
                "total_obj": total,
                "train_pct": round(tr_pct, 2),
                "val_pct": round(v_pct, 2),
                "test_pct": round(te_pct, 2),
                "dev_train": round(dev_tr, 2),
                "dev_val": round(dev_v, 2),
                "dev_test": round(dev_te, 2),
            }
        )

    mean_deviation = sum(deviations) / len(deviations) if deviations else 0.0
    records.sort(key=lambda x: x["total_obj"], reverse=True)

    return records, mean_deviation


def main():
    parser = argparse.ArgumentParser(
        description="Audita las proporciones de los splits tras el tiling"
    )
    parser.add_argument("tiled_dir", type=str, help="Carpeta base del tiling")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.1, 0.2],
        help="Proporciones objetivo",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Ruta para guardar el reporte en JSON",
    )
    args = parser.parse_args()

    records, mean_dev = evaluate_tiled_splits(Path(args.tiled_dir), tuple(args.ratios))

    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "mean_deviation_pct": round(mean_dev, 3),
            "class_distribution": records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=4, ensure_ascii=False)

    df = pd.DataFrame(records)
    if not df.empty:
        df.columns = [
            "Clase",
            "Total Obj",
            "Train %",
            "Val %",
            "Test %",
            "Dev Train",
            "Dev Val",
            "Dev Test",
        ]
        pd.set_option("display.max_rows", None)
        pd.set_option("display.float_format", "{:.2f}".format)


if __name__ == "__main__":
    main()
