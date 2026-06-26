import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def load_xview_class_map(class_map_path: Path) -> dict[int, dict]:
    """Carga el mapeo de clases original de xView desde un archivo JSON."""
    with open(class_map_path, "r") as f:
        raw_map = json.load(f)
    return {int(k): v for k, v in raw_map.items()}


def create_class_mapping(
    xview_class_map: dict[int, dict]
) -> tuple[dict[int, str], dict[int, int]]:
    """
    Crea un mapeo usando el diccionario de macro-categorías.
    Devuelve:
    - new_class_map: dict {nueva_id_macro: "Nombre_Macro"}
    - old_to_new: dict {id_original_xview: nueva_id_macro}
    """
    new_class_map = {}
    old_to_new = {}

    for old_id_str, macro_info in xview_class_map.items():
        old_id = int(old_id_str)
        macro_id = macro_info["id"]
        macro_name = macro_info["name"]

        # Para el diccionario final COCO (id -> nombre)
        if macro_id not in new_class_map:
            new_class_map[macro_id] = macro_name

        # Para saber qué ID antigua de xView mapea a qué macro_id nueva
        old_to_new[old_id] = macro_id

    return new_class_map, old_to_new


def find_extracted_data(raw_data_path: Path) -> tuple[Path, Path]:
    """Busca las rutas de la carpeta de imágenes y el archivo geoJSON en el directorio proporcionado."""
    labels_json_path = raw_data_path / "xView_train.geojson"
    img_folder_path = raw_data_path / "train_images"

    if not labels_json_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de anotaciones: {labels_json_path}"
        )
    if not img_folder_path.is_dir():
        raise FileNotFoundError(
            f"No se encontró la carpeta de imágenes: {img_folder_path}"
        )

    return img_folder_path, labels_json_path


def load_and_clean_annotations(
    labels_json_path: Path,
    old_to_new: dict[int, int],
    erroneous_type_ids: list[int],
    missing_image_ids: list[str],
) -> pd.DataFrame:
    """Carga las anotaciones desde el archivo geoJSON, limpia los errores y mapea los IDs de clase antiguos a nuevos."""
    with open(labels_json_path, "r") as f:
        data = json.load(f)

    # Convertimos las anotaciones a un DataFrame para facilitar su manipulación
    annotations = []
    for feature in data["features"]:
        properties = feature["properties"]
        image_id = properties["image_id"]
        type_id = properties["type_id"]
        bbox = properties["bounds_imcoords"].split(",")

        annotations.append(
            [image_id, type_id, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        )

    df = pd.DataFrame(
        annotations, columns=["image_id", "type_id", "x_min", "y_min", "x_max", "y_max"]
    )

    # Eliminamos las anotaciones con IDs de clase erróneos y las anotaciones para las imágenes que faltan
    df = df[~df["type_id"].isin(erroneous_type_ids)]
    df = df[~df["image_id"].isin(missing_image_ids)]

    # Mapeamos los IDs de clase antiguos a nuevos
    df["type_id"] = df["type_id"].map(old_to_new)

    # Eliminamos filas con type_id que no se pudieron mapear
    df = df.dropna(subset=["type_id"])

    # Normalizamos tipo y descartamos cajas degeneradas
    df["type_id"] = df["type_id"].astype(int)
    bbox_width = df["x_max"] - df["x_min"]
    bbox_height = df["y_max"] - df["y_min"]
    df = df[(bbox_width > 0) & (bbox_height > 0)]

    return df


def create_full_coco_annotations(
    df: pd.DataFrame, img_folder_path: Path, class_map: dict[int, str]
) -> dict[str, Any]:
    """Crea las anotaciones en formato COCO para las imágenes completas."""
    coco_data = {"images": [], "annotations": [], "categories": []}

    for class_id, class_name in class_map.items():
        coco_data["categories"].append({"id": class_id, "name": class_name})

    annotation_id = 0
    grouped_annotations = {
        image_id: group for image_id, group in df.groupby("image_id", sort=False)
    }
    img_files = sorted(file_path.name for file_path in img_folder_path.glob("*.tif"))

    for file_name in img_files:
        img_path = str(img_folder_path / file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_height, img_width, _ = img.shape
        image_id = int(
            file_name.replace("img_", "").replace(".tif", "").replace("_", "")
        )

        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": img_width,
                "height": img_height,
            }
        )

        for _, row in grouped_annotations.get(file_name, pd.DataFrame()).iterrows():
            type_id = row["type_id"]

            # Clip de seguridad al frame original para evitar cajas fuera de imagen.
            x_min = max(0.0, min(float(row["x_min"]), float(img_width)))
            y_min = max(0.0, min(float(row["y_min"]), float(img_height)))
            x_max = max(0.0, min(float(row["x_max"]), float(img_width)))
            y_max = max(0.0, min(float(row["y_max"]), float(img_height)))

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            if bbox_width <= 0 or bbox_height <= 0:
                continue

            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(type_id),
                    "bbox": [
                        float(x_min),
                        float(y_min),
                        float(bbox_width),
                        float(bbox_height),
                    ],
                    "area": float(bbox_width * bbox_height),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return coco_data


def main():
    parser = argparse.ArgumentParser(
        description="Clean xView dataset and convert to COCO format."
    )
    parser.add_argument(
        "raw_data_path", type=str, help="Path to the raw xView dataset (unzipped)."
    )
    parser.add_argument(
        "output_data_path", type=str, help="Path to save the cleaned JSON."
    )
    parser.add_argument(
        "--class_map_path",
        type=str,
        default=None,
        help="Path to xView_classes.json. If not provided, looks for it in raw_data_path.",
    )
    parser.add_argument(
        "--erroneous_type_ids",
        type=int,
        nargs="+",
        default=[75, 82],
        help="Type IDs to exclude (known erroneous annotations). Default: [75, 82]",
    )
    parser.add_argument(
        "--missing_image_ids",
        type=str,
        nargs="+",
        default=["1395.tif"],
        help="Image IDs to exclude (missing from dataset). Default: ['1395.tif']",
    )
    args = parser.parse_args()

    # Convertimos las rutas a objetos Path
    raw_data_path = Path(args.raw_data_path)
    output_data_path = Path(args.output_data_path)

    # Creamos el directorio de salida si no existe
    output_data_path.mkdir(parents=True, exist_ok=True)

    # Cargamos el mapeo de clases original de xView
    class_map_path = (
        Path(args.class_map_path)
        if args.class_map_path
        else raw_data_path / "xView_classes.json"
    )
    xview_class_map = load_xview_class_map(class_map_path)

    # Creamos el nuevo mapeo de clases y el mapeo de IDs antiguos a nuevos
    class_map, old_to_new = create_class_mapping(xview_class_map)

    # Buscamos las rutas de la carpeta de imágenes y el archivo geoJSON
    img_folder_path, labels_json_path = find_extracted_data(raw_data_path)

    # Cargamos y limpiamos las anotaciones, mapeando los IDs de clase antiguos a nuevos
    df = load_and_clean_annotations(
        labels_json_path, old_to_new, args.erroneous_type_ids, args.missing_image_ids
    )

    # Creamos las anotaciones en formato COCO para las imágenes completas
    coco_data = create_full_coco_annotations(df, img_folder_path, class_map)

    # Guardamos las anotaciones en formato COCO y el nuevo mapeo de clases
    with open(output_data_path / "COCO_annotations.json", "w") as f:
        json.dump(coco_data, f, indent=2)

    with open(output_data_path / "xView_class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)


if __name__ == "__main__":
    main()
