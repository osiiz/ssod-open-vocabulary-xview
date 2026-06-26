import json
import matplotlib.pyplot as plt
import numpy as np


def plot_split_deviation(json_file, output_image):
    with open(json_file, "r") as f:
        data = json.load(f)

    dist = data["class_distribution"]

    # Calculamos la desviación total (suma de las 3 desviaciones absolutas)
    for item in dist:
        item["total_dev"] = (
            item.get("dev_train", 0) + item.get("dev_val", 0) + item.get("dev_test", 0)
        )

    # Ordenamos de peor distribuida a mejor distribuida
    dist_sorted = sorted(dist, key=lambda x: x["total_dev"], reverse=True)

    # Filtramos para no mostrar clases con 0 objetos
    dist_sorted = [item for item in dist_sorted if item.get("total_obj", 0) > 0]

    classes = [item["clase"] for item in dist_sorted]
    dev_train = [item["dev_train"] for item in dist_sorted]
    dev_val = [item["dev_val"] for item in dist_sorted]
    dev_test = [item["dev_test"] for item in dist_sorted]

    y_pos = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(12, 10))

    # Gráfico de barras apiladas (Stacked Bar Chart)
    ax.barh(y_pos, dev_train, label="Desviación en Train", color="tomato", alpha=0.9)
    ax.barh(
        y_pos,
        dev_val,
        left=dev_train,
        label="Desviación en Val",
        color="gold",
        alpha=0.9,
    )
    ax.barh(
        y_pos,
        dev_test,
        left=np.array(dev_train) + np.array(dev_val),
        label="Desviación en Test",
        color="skyblue",
        alpha=0.9,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=10)
    ax.invert_yaxis()  # Para que el Top 1 aparezca arriba del todo

    # Etiquetas y título
    ax.set_xlabel("Desviación Porcentual Absoluta Sumada (%)", fontsize=12)
    ax.set_title("Auditoría de Partición: Distribución de Clases", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # Ponemos el porcentaje total al lado de las barras para que sea legible
    for i, item in enumerate(dist_sorted):
        if item["clase"] != "...":
            ax.text(
                item["total_dev"] + 0.5,
                i,
                f"{item['total_dev']:.1f}%",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Gráfico guardado en: {output_image}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file", help="Path to split distribution metrics JSON file"
    )
    parser.add_argument("output_image", help="Path to output image file")
    args = parser.parse_args()

    plot_split_deviation(args.json_file, args.output_image)
