import json
import random
import cv2
from pathlib import Path
import argparse


def visualize_random_samples(coco_json_path, images_dir, output_dir, num_samples=10):
    """
    Carga anotaciones en formato COCO, dibuja las cajas sobre imágenes aleatorias
    y las guarda en un directorio de salida.
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Crear mapeo de IDs a nombres de categorías
    category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Agrupar anotaciones por ID de imagen para un acceso rápido
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Seleccionar imágenes al azar que tengan al menos una anotación
    images_with_annotations = [
        img for img in coco_data["images"] if img["id"] in annotations_by_image
    ]

    if not images_with_annotations:
        print("Error: No se encontraron imágenes con anotaciones.")
        return

    samples = random.sample(
        images_with_annotations, min(num_samples, len(images_with_annotations))
    )

    for img_info in samples:
        img_path = images_dir / img_info["file_name"]

        # Cargar imagen con OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Advertencia: No se pudo cargar la imagen {img_path}")
            continue

        # Dibujar cada caja
        for ann in annotations_by_image[img_info["id"]]:
            x, y, w, h = ann["bbox"]

            # Convertir a coordenadas absolutas para OpenCV (x_min, y_min, x_max, y_max)
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            cat_name = category_map.get(ann["category_id"], "Desconocido")

            # Dibujar rectángulo (Verde, grosor 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Poner fondo negro al texto para que sea legible
            (text_w, text_h), _ = cv2.getTextSize(
                cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 0, 0), -1)

            # Escribir el nombre de la clase
            cv2.putText(
                img,
                cat_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Guardar la imagen procesada
        output_path = output_dir / f"debug_{img_info['file_name']}"
        cv2.imwrite(str(output_path), img)
        print(f"  Guardada: {output_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Herramienta de depuración visual para dataset COCO"
    )
    parser.add_argument(
        "--json",
        type=str,
        default="./results/xview_preprocessed/COCO_annotations.json",
        help="Ruta al JSON de COCO",
    )
    parser.add_argument(
        "--imgs",
        type=str,
        default="./results/xview_preprocessed/images",
        help="Ruta a las imágenes recortadas",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./results/debug",
        help="Carpeta de salida para las visualizaciones",
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="Número de imágenes a visualizar"
    )

    args = parser.parse_args()

    visualize_random_samples(args.json, args.imgs, args.out, args.samples)
