import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json_dict(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return payload


def coerce_numeric_dict(raw: dict) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float))
    }


def plot_global_metrics(
    aware_metrics: dict[str, float],
    agnostic_metrics: dict[str, float],
    selected_metrics: list[str],
    output_path: Path,
    detector_label: str,
    unlabeled_tag: str,
) -> None:
    values_aware = [aware_metrics[metric] for metric in selected_metrics]
    values_agnostic = [agnostic_metrics[metric] for metric in selected_metrics]

    x_positions = list(range(len(selected_metrics)))
    bar_width = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(
        [x - bar_width / 2 for x in x_positions],
        values_aware,
        width=bar_width,
        label="Class-aware",
        alpha=0.9,
    )
    plt.bar(
        [x + bar_width / 2 for x in x_positions],
        values_agnostic,
        width=bar_width,
        label="Class-agnostic",
        alpha=0.9,
    )

    for idx, metric in enumerate(selected_metrics):
        plt.text(
            x_positions[idx] - bar_width / 2,
            values_aware[idx] + 0.004,
            f"{values_aware[idx]:.3f}",
            ha="center",
            fontsize=8,
        )
        plt.text(
            x_positions[idx] + bar_width / 2,
            values_agnostic[idx] + 0.004,
            f"{values_agnostic[idx]:.3f}",
            ha="center",
            fontsize=8,
        )

    title_suffix = f" {unlabeled_tag}" if unlabeled_tag else ""
    plt.title(f"{detector_label}{title_suffix} | Class-aware vs Class-agnostic")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(x_positions, selected_metrics)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_aware_per_class(
    aware_per_class: dict[str, float],
    output_path: Path,
    detector_label: str,
    unlabeled_tag: str,
) -> None:
    filtered = {
        class_name: value
        for class_name, value in aware_per_class.items()
        if isinstance(value, (int, float))
    }

    if not filtered:
        return

    ordered_items = sorted(filtered.items(), key=lambda item: item[1], reverse=True)
    class_names = [item[0] for item in ordered_items]
    scores = [item[1] for item in ordered_items]

    plt.figure(figsize=(max(9, len(class_names) * 0.8), 5.5))
    plt.bar(range(len(class_names)), scores)
    title_suffix = f" {unlabeled_tag}" if unlabeled_tag else ""
    plt.title(f"{detector_label}{title_suffix} | Class-aware score per class")
    plt.xlabel("Class")
    plt.ylabel("AP50")
    plt.xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_comparison_csv(
    aware_metrics: dict[str, float],
    agnostic_metrics: dict[str, float],
    selected_metrics: list[str],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "aware", "agnostic", "delta_aware_minus_agnostic"])
        for metric in selected_metrics:
            aware_value = aware_metrics[metric]
            agnostic_value = agnostic_metrics[metric]
            writer.writerow(
                [
                    metric,
                    f"{aware_value:.6f}",
                    f"{agnostic_value:.6f}",
                    f"{(aware_value - agnostic_value):.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot global and per-class comparisons for DINO class-aware vs class-agnostic metrics"
    )
    parser.add_argument("aware_dir", type=Path)
    parser.add_argument("agnostic_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("docs/charts_ov"))
    parser.add_argument(
        "--detector_label",
        type=str,
        default="Grounding DINO",
        help="Label used in plot titles.",
    )
    parser.add_argument(
        "--unlabeled_tag",
        type=str,
        default="",
        help="Optional split/tag suffix used in plot titles.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["AP", "AP50", "AP75"],
        help="Global metric keys to compare from metrics.json",
    )

    args = parser.parse_args()

    aware_metrics = coerce_numeric_dict(load_json_dict(args.aware_dir / "metrics.json"))
    agnostic_metrics = coerce_numeric_dict(
        load_json_dict(args.agnostic_dir / "metrics.json")
    )

    common_keys = [
        k for k in args.metrics if k in aware_metrics and k in agnostic_metrics
    ]
    if not common_keys:
        # Fallback: use all numeric keys present in both files, preserving order
        common_keys = [k for k in aware_metrics if k in agnostic_metrics]
    selected_metrics = common_keys

    plot_global_metrics(
        aware_metrics=aware_metrics,
        agnostic_metrics=agnostic_metrics,
        selected_metrics=selected_metrics,
        output_path=args.output_dir / "aware_vs_agnostic_global.png",
        detector_label=args.detector_label,
        unlabeled_tag=args.unlabeled_tag,
    )

    aware_per_class = load_json_dict(args.aware_dir / "metrics_per_class.json")
    plot_aware_per_class(
        aware_per_class=aware_per_class,
        output_path=args.output_dir / "aware_ap50_per_class.png",
        detector_label=args.detector_label,
        unlabeled_tag=args.unlabeled_tag,
    )

    write_comparison_csv(
        aware_metrics=aware_metrics,
        agnostic_metrics=agnostic_metrics,
        selected_metrics=selected_metrics,
        output_csv=args.output_dir / "aware_vs_agnostic_metrics.csv",
    )

    print(f"{args.detector_label} comparison charts generated at: {args.output_dir}")


if __name__ == "__main__":
    main()
