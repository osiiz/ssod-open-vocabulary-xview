import json
import matplotlib.pyplot as plt
import numpy as np


def plot_ap_per_class(json_file, output_image):
    with open(json_file, "r") as f:
        metrics = json.load(f)

    # Filtramos para tener solo valores válidos (AP >= 0)
    valid_metrics = {k: v for k, v in metrics.items() if v >= 0}

    # Ordenar por AP de menor a mayor
    sorted_metrics = dict(sorted(valid_metrics.items(), key=lambda item: item[1]))

    # Si hay más de 30 clases, mostrar solo las 15 con peor AP y las 15 con mejor AP
    classes = list(sorted_metrics.keys())
    aps = list(sorted_metrics.values())

    if len(classes) > 30:
        classes = classes[:15] + ["..."] + classes[-15:]
        aps = aps[:15] + [0] + aps[-15:]
        colors = ["tomato"] * 15 + ["white"] + ["mediumseagreen"] * 15
    else:
        colors = ["skyblue"] * len(classes)

    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(classes))

    plt.barh(y_pos, aps, color=colors)
    plt.yticks(y_pos, classes)
    plt.xlabel("Average Precision (AP50 @ IoU=0.50)")
    plt.title("Average Precision per Class (Top and Bottom)")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Agregar etiquetas de AP al final de cada barra
    for i, v in enumerate(aps):
        if classes[i] != "...":
            plt.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Gráfico guardado en: {output_image}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to metrics per class JSON file")
    parser.add_argument("output_image", help="Path to output image file")
    args = parser.parse_args()

    plot_ap_per_class(args.json_file, args.output_image)
