import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_sample_deviation(json_file, output_image, target_ratio):
    with open(json_file, "r") as f:
        data = json.load(f)

    classes = []
    retentions = []
    originals = []
    sampleds = []

    # Extraer datos ignorando clases vacías
    for cls_name, stats in data.items():
        if stats["original_instances"] > 0:
            classes.append(cls_name)
            retentions.append(stats["retention_percentage"])
            originals.append(stats["original_instances"])
            sampleds.append(stats["sampled_instances"])

    # Ordenamos de peor retención a mejor retención
    sorted_indices = np.argsort(retentions)
    classes = [classes[i] for i in sorted_indices]
    retentions = [retentions[i] for i in sorted_indices]
    originals = [originals[i] for i in sorted_indices]
    sampleds = [sampleds[i] for i in sorted_indices]

    y_pos = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colorimetría: Azul si está a +/- 2% de la meta, Rojo si se desvía peligrosamente
    target_pct = target_ratio * 100
    colors = ["skyblue" if abs(r - target_pct) <= 2.0 else "tomato" for r in retentions]

    # Gráfico de barras horizontales
    ax.barh(y_pos, retentions, color=colors, alpha=0.9)

    # Línea de meta ideal
    ax.axvline(
        x=target_pct,
        color="gold",
        linestyle="--",
        linewidth=2,
        label=f"Objetivo de Muestreo ({target_pct}%)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=10)

    # Etiquetas y título
    ax.set_xlabel("Porcentaje Retenido (%)", fontsize=12)
    ax.set_title("Auditoría de Submuestreo SSOD: Retención por Clase", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # Ponemos el porcentaje total y el conteo bruto al lado de las barras
    max_retention = max(retentions) if retentions else target_pct
    for i, (retention, orig, samp) in enumerate(zip(retentions, originals, sampleds)):
        ax.text(
            retention + (max_retention * 0.01),  # Un pequeño margen dinámico
            i,
            f"{retention:.1f}%  ({samp}/{orig})",
            va="center",
            fontsize=10,
            fontweight="bold" if colors[i] == "tomato" else "normal",
        )

    plt.tight_layout()
    Path(output_image).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=300)
    print(f"Gráfico guardado en: {output_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grafica la desviación del muestreo SSOD"
    )
    parser.add_argument(
        "json_file", help="Path al JSON del reporte generado por audit_sample.py"
    )
    parser.add_argument("output_image", help="Path al archivo .png de salida")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.15,
        help="El ratio de muestreo objetivo (ej. 0.15)",
    )
    args = parser.parse_args()

    plot_sample_deviation(args.json_file, args.output_image, args.ratio)
