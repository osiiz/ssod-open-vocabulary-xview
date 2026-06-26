import json
import argparse
from collections import defaultdict
from pathlib import Path


def audit_sampling(full_json_path, sampled_json_path, out_report_path):
    with open(full_json_path, "r") as f:
        full_coco = json.load(f)

    with open(sampled_json_path, "r") as f:
        sampled_coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in full_coco["categories"]}

    # Contar instancias
    full_counts = defaultdict(int)
    for ann in full_coco["annotations"]:
        full_counts[ann["category_id"]] += 1

    sampled_counts = defaultdict(int)
    for ann in sampled_coco["annotations"]:
        sampled_counts[ann["category_id"]] += 1

    report = {}
    for cat_id, name in categories.items():
        original = full_counts[cat_id]
        sampled = sampled_counts[cat_id]
        pct = (sampled / original * 100) if original > 0 else 0

        report[name] = {
            "original_instances": original,
            "sampled_instances": sampled,
            "retention_percentage": round(pct, 2),
        }

    Path(out_report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nReporte guardado en: {out_report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Audita la cantidad de instancias retenidas en el muestreo."
    )
    parser.add_argument(
        "full_json", type=str, help="Ruta al JSON de train completo (100%)"
    )
    parser.add_argument(
        "sampled_json", type=str, help="Ruta al JSON de train muestreado (ej. 15%)"
    )
    parser.add_argument(
        "--out_report",
        type=str,
        default="docs/dataset_reports/sample_audit.json",
        help="Ruta de salida del reporte",
    )
    args = parser.parse_args()

    audit_sampling(args.full_json, args.sampled_json, args.out_report)


if __name__ == "__main__":
    main()
