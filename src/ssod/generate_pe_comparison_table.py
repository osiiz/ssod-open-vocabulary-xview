"""
Carga las métricas de todos los experimentos PE y el baseline, produce una tabla
comparativa en CSV y un gráfico de barras.

Uso:
    python -m src.ssod.generate_pe_comparison_table \
        --baseline_metrics results/inference_test_ssod_baseline/metrics.json \
        --pe_metrics \
            a:results/inference_test_ssod_pe/a/metrics.json \
            b:results/inference_test_ssod_pe/b/metrics.json \
            c:results/inference_test_ssod_pe/c/metrics.json \
            ab:results/inference_test_ssod_pe/ab/metrics.json \
            ac:results/inference_test_ssod_pe/ac/metrics.json \
            abc:results/inference_test_ssod_pe/abc/metrics.json \
        --output_csv results/ssod/comparison_table.csv \
        --output_chart docs/charts_ssod_pe/comparison_bar_chart.png
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


_COLUMNS = ["AP", "AP50", "AP75", "AR_1500"]
_DISPLAY_NAMES = {
    "baseline": "Baseline (10% labeled)",
    "a": "A — Faster10 PE",
    "b": "B — DINO PE",
    "c": "C — Rex-Omni PE",
    "ab": "AB — Faster10 + DINO",
    "ac": "AC — Faster10 + Rex-Omni",
    "abc": "ABC — All sources",
}


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception:
        return None


def _parse_pe_metrics_arg(entries: list[str]) -> list[tuple[str, Path]]:
    result = []
    for entry in entries:
        label, _, path_str = entry.partition(":")
        result.append((label.strip(), Path(path_str.strip())))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison table and chart for SSOD PE experiments."
    )
    parser.add_argument("--baseline_metrics", type=Path, required=True)
    parser.add_argument(
        "--pe_metrics",
        nargs="+",
        required=True,
        help="label:path pairs, e.g. a:results/.../a/metrics.json",
    )
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_chart", type=Path, required=True)
    parser.add_argument(
        "--metric_key",
        type=str,
        default="AP50",
        help="Primary metric for sorting / chart.",
    )
    # Textos do gráfico (opcionais; por defecto, en inglés). Permiten xerar
    # versións localizadas sen tocar a saída por defecto do pipeline.
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--xlabel", type=str, default=None)
    parser.add_argument("--legend_baseline", type=str, default="Baseline")
    parser.add_argument("--legend_pe", type=str, default="PE experiment")
    parser.add_argument("--baseline_display", type=str, default=None)
    args = parser.parse_args()

    if args.baseline_display:
        _DISPLAY_NAMES["baseline"] = args.baseline_display

    rows: list[tuple[str, dict | None]] = [
        ("baseline", _load_metrics(args.baseline_metrics))
    ]
    for label, path in _parse_pe_metrics_arg(args.pe_metrics):
        rows.append((label, _load_metrics(path)))

    # Escribir CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment"] + _COLUMNS)
        for label, metrics in rows:
            if metrics is None:
                writer.writerow([label] + ["N/A"] * len(_COLUMNS))
            else:
                writer.writerow(
                    [label]
                    + [f"{metrics.get(col, float('nan')):.4f}" for col in _COLUMNS]
                )
    print(f"CSV saved to {args.output_csv}")

    # Gráfico: barras horizontales ordenadas por métrica principal, baseline resaltado
    valid_rows = [(label, m) for label, m in rows if m is not None]
    valid_rows.sort(key=lambda x: x[1].get(args.metric_key, 0.0))

    labels_display = [_DISPLAY_NAMES.get(label, label) for label, _ in valid_rows]
    # El chart represéntase en base 100 (ex.: 0.3498 -> 34.98), coherente coa memoria.
    values = [m.get(args.metric_key, 0.0) * 100 for _, m in valid_rows]
    colors = [
        "#d62728" if label == "baseline" else "#1f77b4" for label, _ in valid_rows
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, len(valid_rows) * 0.7)))
    bars = ax.barh(labels_display, values, color=colors, alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
        )

    if values:
        ax.set_xlim(0, max(values) * 1.12)  # marxe á dereita para as etiquetas
    ax.set_xlabel(args.xlabel or f"{args.metric_key} (base 100)")
    ax.set_title(args.title or f"SSOD PE experiments — {args.metric_key} on test set")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#d62728", alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color="#1f77b4", alpha=0.85),
        ],
        labels=[args.legend_baseline, args.legend_pe],
        loc="lower right",
    )
    plt.tight_layout()

    args.output_chart.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_chart, dpi=300)
    plt.close()
    print(f"Chart saved to {args.output_chart}")


if __name__ == "__main__":
    main()
