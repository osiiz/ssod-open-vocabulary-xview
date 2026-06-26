import json
import cv2
import random
import argparse
from pathlib import Path


def draw_ground_truth(
    json_path: str,
    img_dir: str,
    output_dir: str,
    num_images: int = 5,
    specific_image: str = None,
):
    """
    Lee un JSON de COCO y dibuja las cajas reales sobre las imágenes.
    """
    print(f"Cargando anotaciones desde: {json_path}...")
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    #  Mapeo de categorías y generación de colores consistentes
    category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Fijamos una semilla para que la clase 'Building' siempre tenga el mismo color, etc.
    random.seed(42)
    colors = {
        cat_id: (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
        for cat_id in category_map.keys()
    }

    # Agrupar las anotaciones por ID de imagen para acceso rápido
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    images = coco_data["images"]

    # Seleccionar las imágenes a procesar
    if specific_image:
        images_to_process = [
            img for img in images if img["file_name"] == specific_image
        ]
        if not images_to_process:
            print(f"Error: No se encontró la imagen '{specific_image}' en el JSON.")
            return
    else:
        # Cogemos N imágenes al azar que sepamos que tienen anotaciones
        images_with_anns = [img for img in images if img["id"] in annotations_by_image]
        images_to_process = random.sample(
            images_with_anns, min(num_images, len(images_with_anns))
        )

    # Preparar el directorio de salida
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dibujar las cajas
    for img_info in images_to_process:
        img_path = Path(img_dir) / img_info["file_name"]

        if not img_path.exists():
            print(f"⚠️ Advertencia: No existe el archivo físico {img_path}")
            continue

        # Leer la imagen con OpenCV
        img = cv2.imread(str(img_path))
        anns = annotations_by_image.get(img_info["id"], [])

        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = category_map.get(cat_id, f"ID_{cat_id}")
            color = colors.get(cat_id, (0, 255, 0))  # BGR format in OpenCV

            # Formato COCO: [x_min, y_min, width, height]
            x_min, y_min, w, h = [int(v) for v in ann["bbox"]]
            x_max, y_max = x_min + w, y_min + h

            # Dibujar el rectángulo principal (grosor 2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Crear un fondo para el texto para que sea legible
            label = cat_name
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cv2.rectangle(
                img, (x_min, y_min - text_h - 4), (x_min + text_w, y_min), color, -1
            )

            # Dibujar el texto en blanco o negro dependiendo de si el color es oscuro/claro
            # (Simplificado a blanco con grosor 1)
            cv2.putText(
                img,
                label,
                (x_min, y_min - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        out_file = out_dir / f"gt_{img_info['file_name']}"
        cv2.imwrite(str(out_file), img)
        print(f"Imagen generada ({len(anns)} objetos): {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dibuja las cajas del Ground Truth de COCO sobre las imágenes."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Ruta al archivo COCO annotations JSON",
    )
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Carpeta donde están las imágenes"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/debug_groundtruth",
        help="Carpeta de salida",
    )
    parser.add_argument(
        "--num", type=int, default=5, help="Número de imágenes aleatorias a generar"
    )
    parser.add_argument(
        "--img_name",
        type=str,
        default=None,
        help="Nombre de un archivo específico (ej: 1046.tif)",
    )

    args = parser.parse_args()
    draw_ground_truth(
        args.json_path, args.img_dir, args.out_dir, args.num, args.img_name
    )
