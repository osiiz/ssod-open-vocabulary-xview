import json
import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(train_json, val_json, output_image):
    with open(train_json, "r") as f:
        train_data = json.load(f)["class_distribution"]
    with open(val_json, "r") as f:
        val_data = json.load(f)["class_distribution"]

    common_classes = list(set(train_data.keys()).union(set(val_data.keys())))

    # Extraemos el count del diccionario anidado
    class_counts_train = {
        c: train_data[c]["count"] if c in train_data else 0 for c in common_classes
    }
    class_counts_val = {
        c: val_data[c]["count"] if c in val_data else 0 for c in common_classes
    }

    # Ordenamos y cogemos las 20 clases mayoritarias
    sorted_classes = sorted(
        class_counts_train.keys(), key=lambda k: class_counts_train[k], reverse=True
    )[:20]

    train_counts = [class_counts_train[c] for c in sorted_classes]
    val_counts = [class_counts_val[c] for c in sorted_classes]

    x = np.arange(len(sorted_classes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, train_counts, width, label="Train", color="teal")
    ax.bar(x + width / 2, val_counts, width, label="Validation", color="coral")

    ax.set_ylabel("Nº de Instancias (Escala Logarítmica)", fontsize=12)
    ax.set_title("Top 20 Clases en xView (Train vs Val)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_classes, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.set_yscale("log")

    fig.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Guardado: {output_image}")

    # Hacer el mismo gráfico pero sin escala logarítmica
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, train_counts, width, label="Train", color="teal")
    ax.bar(x + width / 2, val_counts, width, label="Validation", color="coral")
    ax.set_ylabel("Nº de Instancias", fontsize=12)
    ax.set_title("Top 20 Clases en xView (Train vs Val)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_classes, rotation=45, ha="right", fontsize=10)
    ax.legend()
    fig.tight_layout()
    plt.savefig(output_image.replace(".png", "_linear.png"), dpi=300)
    print(f"Guardado: {output_image.replace('.png', '_linear.png')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_json", help="Path to training stats JSON file")
    parser.add_argument("val_json", help="Path to validation stats JSON file")
    parser.add_argument("output_image", help="Path to output image file")
    args = parser.parse_args()

    plot_class_distribution(args.train_json, args.val_json, args.output_image)
