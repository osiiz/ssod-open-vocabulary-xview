import json
import matplotlib.pyplot as plt


def plot_ap_vs_count(stats_json, ap_json, output_image):
    with open(stats_json, "r") as f:
        train_data = json.load(f)["class_distribution"]
    with open(ap_json, "r") as f:
        ap_data = json.load(f)

    classes, counts, aps = [], [], []

    for c in ap_data.keys():
        if c in train_data and ap_data[c] >= 0:
            classes.append(c)
            counts.append(train_data[c]["count"])
            aps.append(ap_data[c])

    plt.figure(figsize=(12, 8))
    # Creamos un scatter plot donde el color depende del rendimiento
    plt.scatter(
        counts,
        aps,
        alpha=0.7,
        c=aps,
        cmap="viridis",
        s=120,
        edgecolors="w",
        linewidth=1,
    )

    # Ponemos el nombre solo a las clases interesantes para no saturar
    for i, c in enumerate(classes):
        plt.annotate(
            c,
            (counts[i], aps[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.xscale("log")
    plt.xlabel("Nº Instancias en Entrenamiento (Escala Logarítmica)", fontsize=12)
    plt.ylabel("Precisión Media (AP50 @ IoU=0.50)", fontsize=12)
    plt.title(
        "Correlación: Frecuencia de la Clase vs Rendimiento del Modelo", fontsize=14
    )
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Guardado: {output_image}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stats_json", help="Path to training stats JSON file")
    parser.add_argument("ap_json", help="Path to AP metrics JSON file")
    parser.add_argument("output_image", help="Path to output image file")
    args = parser.parse_args()

    plot_ap_vs_count(args.stats_json, args.ap_json, args.output_image)
