import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class RatioResult:
    name: str
    percent: float
    global_metrics: dict
    per_class_metrics: dict


def _parse_ratio_percent(label: str) -> float:
    """Extract a numeric sampling percentage from names like r10, 10, or 0.10."""
    match = re.search(r"(\d+(?:\.\d+)?)", label)
    if not match:
        raise ValueError(
            f"No se pudo extraer porcentaje desde '{label}'. "
            "Usa nombres tipo r10/r20/r30 o incluye un numero en el nombre."
        )

    value = float(match.group(1))
    if value <= 1.0:
        value *= 100.0
    return value


def _load_json_dict(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {file_path}")
    return data


def _coerce_numeric_dict(raw: dict, source_name: str) -> dict:
    coerced = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            coerced[str(key)] = float(value)
    if not coerced:
        raise ValueError(f"No hay valores numericos validos en {source_name}")
    return coerced


def _discover_ratio_dirs(
    ratios_root: Path, ratio_names: list, inference_subdir: str
) -> list:
    if ratio_names:
        candidates = [ratios_root / name for name in ratio_names]
    else:
        candidates = sorted([p for p in ratios_root.iterdir() if p.is_dir()])

    selected = []
    for ratio_dir in candidates:
        metrics_file = ratio_dir / inference_subdir / "metrics.json"
        per_class_file = ratio_dir / inference_subdir / "metrics_per_class.json"
        if metrics_file.exists() and per_class_file.exists():
            selected.append(ratio_dir)

    if not selected:
        raise ValueError(
            "No se encontraron ratios validos. Cada ratio debe contener "
            f"{inference_subdir}/metrics.json y {inference_subdir}/metrics_per_class.json"
        )

    return selected


def load_ratio_results(
    ratios_root: Path,
    ratio_names: list,
    inference_subdir: str,
) -> list:
    ratio_dirs = _discover_ratio_dirs(ratios_root, ratio_names, inference_subdir)

    loaded = []
    for ratio_dir in ratio_dirs:
        percent = _parse_ratio_percent(ratio_dir.name)
        metrics_file = ratio_dir / inference_subdir / "metrics.json"
        per_class_file = ratio_dir / inference_subdir / "metrics_per_class.json"

        raw_global = _load_json_dict(metrics_file)
        raw_per_class = _load_json_dict(per_class_file)

        loaded.append(
            RatioResult(
                name=ratio_dir.name,
                percent=percent,
                global_metrics=_coerce_numeric_dict(raw_global, str(metrics_file)),
                per_class_metrics=_coerce_numeric_dict(
                    raw_per_class, str(per_class_file)
                ),
            )
        )

    loaded.sort(key=lambda item: (item.percent, item.name))
    return loaded


def _validate_requested_metrics(results: list, requested_metrics: list) -> list:
    available = set(results[0].global_metrics.keys())
    missing = [metric for metric in requested_metrics if metric not in available]
    if missing:
        raise ValueError(
            "Metricas no disponibles en metrics.json: "
            f"{', '.join(missing)}. Disponibles: {', '.join(sorted(available))}"
        )
    return requested_metrics


def _build_per_class_matrix(results: list, top_n_classes: int) -> tuple:
    class_names = sorted(
        {
            class_name
            for result in results
            for class_name in result.per_class_metrics.keys()
        }
    )

    # Use the highest sampling ratio as reference ordering for readability.
    reference_result = max(results, key=lambda item: item.percent)
    class_names.sort(
        key=lambda class_name: reference_result.per_class_metrics.get(class_name, 0.0),
        reverse=True,
    )

    if top_n_classes and top_n_classes > 0:
        class_names = class_names[:top_n_classes]

    matrix = np.array(
        [
            [result.per_class_metrics.get(class_name, 0.0) for result in results]
            for class_name in class_names
        ],
        dtype=float,
    )
    return class_names, matrix


def plot_global_metrics(
    results: list,
    metrics_to_plot: list,
    output_image: Path,
    title_prefix: str,
) -> None:
    x_values = [result.percent for result in results]
    x_labels = [f"{value:g}%" for value in x_values]

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]

    plt.figure(figsize=(11, 6))
    for idx, metric_name in enumerate(metrics_to_plot):
        y_values = [result.global_metrics[metric_name] for result in results]
        plt.plot(
            x_values,
            y_values,
            marker=marker_cycle[idx % len(marker_cycle)],
            linewidth=2.2,
            markersize=7,
            label=metric_name,
        )

        for x_value, y_value in zip(x_values, y_values):
            plt.text(
                x_value, y_value + 0.004, f"{y_value:.3f}", ha="center", fontsize=8
            )

    plt.title(f"{title_prefix} | Metricas globales vs % muestreo")
    plt.xlabel("Porcentaje de muestreo")
    plt.ylabel("Valor de metrica")
    plt.xticks(x_values, x_labels)
    plt.grid(axis="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Grafica global guardada en: {output_image}")


def plot_per_class_grouped_bars(
    results: list,
    class_names: list,
    matrix: np.ndarray,
    output_image: Path,
    title_prefix: str,
) -> None:
    plt.figure(figsize=(max(12, len(class_names) * 0.8), 7))

    x_pos = np.arange(len(class_names))
    ratio_count = len(results)
    bar_width = min(0.8 / max(ratio_count, 1), 0.28)

    for idx, result in enumerate(results):
        offsets = x_pos + (idx - (ratio_count - 1) / 2.0) * bar_width
        plt.bar(
            offsets,
            matrix[:, idx],
            width=bar_width,
            label=f"{result.name} ({result.percent:g}%)",
            alpha=0.9,
        )

    plt.title(f"{title_prefix} | AP por clase vs % muestreo")
    plt.xlabel("Clase")
    plt.ylabel("AP por clase")
    plt.xticks(x_pos, class_names, rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Grafica por clase (barras) guardada en: {output_image}")


def plot_per_class_heatmap(
    results: list,
    class_names: list,
    matrix: np.ndarray,
    output_image: Path,
    title_prefix: str,
) -> None:
    plt.figure(figsize=(8, max(6, len(class_names) * 0.4)))
    column_labels = [f"{result.name}\n({result.percent:g}%)" for result in results]
    annotate = len(class_names) <= 20

    sns.heatmap(
        matrix,
        annot=annotate,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=column_labels,
        yticklabels=class_names,
        cbar_kws={"label": "AP por clase"},
    )

    plt.title(f"{title_prefix} | AP por clase (heatmap)")
    plt.xlabel("Ratio")
    plt.ylabel("Clase")
    plt.tight_layout()

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Grafica por clase (heatmap) guardada en: {output_image}")


def write_global_csv(results: list, output_csv: Path) -> None:
    metric_keys = sorted(
        {key for result in results for key in result.global_metrics.keys()}
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ratio_name", "sampling_percent", *metric_keys])
        for result in results:
            writer.writerow(
                [
                    result.name,
                    f"{result.percent:g}",
                    *[
                        f"{result.global_metrics.get(key, float('nan')):.6f}"
                        for key in metric_keys
                    ],
                ]
            )
    print(f"CSV global guardado en: {output_csv}")


def write_per_class_csv(
    results: list, class_names: list, matrix: np.ndarray, output_csv: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    ratio_headers = [f"{result.name}_{result.percent:g}pct" for result in results]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", *ratio_headers])
        for class_idx, class_name in enumerate(class_names):
            row = [class_name, *[f"{value:.6f}" for value in matrix[class_idx, :]]]
            writer.writerow(row)

    print(f"CSV por clase guardado en: {output_csv}")


def _select_per_class_mode(mode: str, class_count: int) -> str:
    if mode != "auto":
        return mode
    return "heatmap" if class_count > 15 else "grouped-bar"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara resultados de inferencia en funcion del porcentaje de muestreo"
    )
    parser.add_argument(
        "ratios_root",
        type=Path,
        help="Directorio raiz con subdirectorios por ratio (ej. r10/r20/r30)",
    )
    parser.add_argument(
        "--ratio-names",
        nargs="+",
        default=None,
        help="Lista opcional de nombres de ratio a incluir (si se omite, autodeteccion)",
    )
    parser.add_argument(
        "--inference-subdir",
        default="inference_test",
        help="Subdirectorio donde estan metrics.json y metrics_per_class.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de salida para graficas y CSV (default: docs/charts_ssod_baseline)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["AP", "AP50", "AP75"],
        help="Metricas globales a representar en la grafica principal",
    )
    parser.add_argument(
        "--top-n-classes",
        type=int,
        default=0,
        help="Limitar numero de clases para grafica por clase (0 = todas)",
    )
    parser.add_argument(
        "--per-class-mode",
        choices=["auto", "grouped-bar", "heatmap"],
        default="auto",
        help="Tipo de grafica por clase",
    )
    parser.add_argument(
        "--no-per-class",
        action="store_true",
        help="No generar grafica por clase",
    )
    parser.add_argument(
        "--title-prefix",
        default="Comparacion SSOD",
        help="Prefijo de titulo para las graficas",
    )

    args = parser.parse_args()

    ratios_root = args.ratios_root
    if not ratios_root.exists() or not ratios_root.is_dir():
        raise ValueError(f"El directorio raiz no existe o no es valido: {ratios_root}")

    output_dir = args.output_dir or Path("docs/charts_ssod_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_ratio_results(
        ratios_root=ratios_root,
        ratio_names=args.ratio_names,
        inference_subdir=args.inference_subdir,
    )

    metrics_to_plot = _validate_requested_metrics(results, args.metrics)
    class_names, class_matrix = _build_per_class_matrix(results, args.top_n_classes)

    plot_global_metrics(
        results=results,
        metrics_to_plot=metrics_to_plot,
        output_image=output_dir / "metrics_vs_sampling_ratio.png",
        title_prefix=args.title_prefix,
    )

    if not args.no_per_class:
        mode = _select_per_class_mode(args.per_class_mode, len(class_names))
        if mode == "grouped-bar":
            plot_per_class_grouped_bars(
                results=results,
                class_names=class_names,
                matrix=class_matrix,
                output_image=output_dir / "per_class_vs_sampling_ratio_grouped.png",
                title_prefix=args.title_prefix,
            )
        else:
            plot_per_class_heatmap(
                results=results,
                class_names=class_names,
                matrix=class_matrix,
                output_image=output_dir / "per_class_vs_sampling_ratio_heatmap.png",
                title_prefix=args.title_prefix,
            )

    write_global_csv(results=results, output_csv=output_dir / "metrics_by_ratio.csv")
    write_per_class_csv(
        results=results,
        class_names=class_names,
        matrix=class_matrix,
        output_csv=output_dir / "per_class_by_ratio.csv",
    )


if __name__ == "__main__":
    main()
