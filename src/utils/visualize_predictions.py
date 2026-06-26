import json
import random
import cv2
from pathlib import Path
import argparse


def visualize_predictions(
    gt_json_path,
    pred_json_path,
    images_dir,
    output_dir,
    num_samples=10,
    score_thresh=0.5,
):
    """
    Carga las predicciones del modelo y las dibuja sobre las imágenes originales.
    """
    # Cargar el Ground Truth para obtener los nombres de archivos y categorías
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)

    category_map = {cat["id"]: cat["name"] for cat in gt_data["categories"]}
    image_map = {img["id"]: img["file_name"] for img in gt_data["images"]}

    # Cargar las predicciones (detection_results.json)
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    # Agrupar predicciones por imagen, filtrando las de baja confianza
    preds_by_image = {}
    for p in preds:
        if p["score"] < score_thresh:
            continue

        img_id = p["image_id"]
        if img_id not in preds_by_image:
            preds_by_image[img_id] = []
        preds_by_image[img_id].append(p)

    if not preds_by_image:
        print(
            f"Error: Ninguna predicción supera el umbral de confianza de {score_thresh}."
        )
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)

    # Seleccionar imágenes al azar que tengan predicciones válidas
    sample_ids = random.sample(
        list(preds_by_image.keys()), min(num_samples, len(preds_by_image))
    )

    for img_id in sample_ids:
        file_name = image_map[img_id]
        img_path = images_dir / file_name

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Advertencia: No se pudo cargar la imagen {img_path}")
            continue

        # Dibujar las predicciones
        for ann in preds_by_image[img_id]:
            x, y, w, h = ann["bbox"]
            score = ann["score"]
            cat_name = category_map.get(ann["category_id"], "Desconocido")

            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            # Dibujar rectángulo (Rojo para predicciones, grosor 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Etiqueta de texto con el nombre y la confianza (ej: "Small Car 0.85")
            label = f"{cat_name} {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 0, 0), -1)
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        out_path = output_dir / f"pred_{file_name}"
        cv2.imwrite(str(out_path), img)
        print(f"Guardada visualización en: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualizador de predicciones de inferencia"
    )
    parser.add_argument(
        "--gt_json", type=str, required=True, help="Ruta al JSON original (val o test)"
    )
    parser.add_argument(
        "--pred_json", type=str, required=True, help="Ruta al detection_results.json"
    )
    parser.add_argument(
        "--imgs", type=str, required=True, help="Carpeta de imágenes reales"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./results/debug_predictions",
        help="Carpeta de salida",
    )
    parser.add_argument(
        "--samples", type=int, default=15, help="Número de imágenes a visualizar"
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.3,
        help="Umbral de confianza para pintar la caja",
    )

    args = parser.parse_args()

    visualize_predictions(
        args.gt_json, args.pred_json, args.imgs, args.out, args.samples, args.thresh
    )
