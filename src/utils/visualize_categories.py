import json
import cv2
import random
import argparse
import ast
from pathlib import Path


def get_random_color(seed_val):
    random.seed(seed_val)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def load_xview_classes(classes_json_path):
    """Carga el diccionario de clases de xView desde un archivo JSON."""
    path = Path(classes_json_path)
    if not path.exists():
        print(f"Error: No se encontró el archivo de clases en {path}")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return {int(k): v for k, v in data.items()}


def process_geojson(json_path, target_categories, xview_classes):
    """Parsea el GeoJSON de xView y devuelve una estructura compatible para dibujar."""
    print(f"Cargando GeoJSON original desde: {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    name_to_id = {v: k for k, v in xview_classes.items()}

    valid_targets = []
    for cat in target_categories:
        if cat in name_to_id:
            valid_targets.append(cat)
        else:
            print(
                f"Categoría '{cat}' no encontrada en el diccionario xView. Ignorando."
            )

    if not valid_targets:
        return None, None, None

    # Agrupar por nombre de archivo (image_id en xView es el nombre del .tif)
    anns_by_image = {}

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        img_name = props.get("image_id")
        type_id = props.get("type_id")
        bounds = props.get("bounds_imcoords")

        if not img_name or not type_id or not bounds:
            continue

        # xView bounds vienen como string "xmin,ymin,xmax,ymax"
        if isinstance(bounds, str):
            try:
                if bounds.startswith("["):
                    b = ast.literal_eval(bounds)
                else:
                    b = [int(float(x)) for x in bounds.split(",")]
            except:
                continue
        else:
            b = bounds

        if len(b) != 4:
            continue

        x_min, y_min, x_max, y_max = b

        if img_name not in anns_by_image:
            anns_by_image[img_name] = []

        anns_by_image[img_name].append(
            {
                "category_id": int(type_id),
                "bbox": [
                    x_min,
                    y_min,
                    x_max - x_min,
                    y_max - y_min,
                ],  # Convertir a COCO format (x,y,w,h)
            }
        )

    return anns_by_image, xview_classes, valid_targets


def process_coco(json_path, target_categories):
    """Parsea formato COCO estándar."""
    print(f"Cargando COCO JSON desde: {json_path}...")
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    name_to_id = {cat["name"]: cat["id"] for cat in coco_data["categories"]}

    valid_targets = []
    for cat in target_categories:
        if cat in name_to_id:
            valid_targets.append(cat)
        else:
            print(f"Categoría '{cat}' no encontrada en el JSON. Ignorando.")

    if not valid_targets:
        return None, None, None

    anns_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        # Buscar el nombre real del archivo
        img_name = next(
            (img["file_name"] for img in coco_data["images"] if img["id"] == img_id),
            str(img_id),
        )

        if img_name not in anns_by_image:
            anns_by_image[img_name] = []
        anns_by_image[img_name].append(ann)

    return anns_by_image, category_map, valid_targets


def visualize_categories(
    json_path,
    img_dir,
    out_dir,
    target_categories,
    num_per_cat,
    format_type,
    classes_json,
):

    if format_type == "geojson":
        # Intentamos cargar el diccionario externo
        xview_classes = load_xview_classes(classes_json)
        if not xview_classes:
            return
        anns_by_image, category_map, valid_targets = process_geojson(
            json_path, target_categories, xview_classes
        )
    else:
        anns_by_image, category_map, valid_targets = process_coco(
            json_path, target_categories
        )

    if not valid_targets:
        print("Operación abortada. Revisa los nombres de las categorías.")
        return

    name_to_id = {v: k for k, v in category_map.items()}
    colors = {cat_id: get_random_color(cat_id) for cat_id in category_map.keys()}

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for target_cat in valid_targets:
        target_id = name_to_id[target_cat]

        eligible_images = []
        for img_name, anns in anns_by_image.items():
            if any(ann["category_id"] == target_id for ann in anns):
                eligible_images.append(img_name)

        if not eligible_images:
            print(f"No hay imágenes con la categoría '{target_cat}' en este JSON.")
            continue

        sampled_imgs = random.sample(
            eligible_images, min(num_per_cat, len(eligible_images))
        )
        print(
            f"\nGenerando {len(sampled_imgs)} imágenes de ejemplo para: '{target_cat}'..."
        )

        for img_name in sampled_imgs:
            img_path = Path(img_dir) / img_name

            if not img_path.exists():
                print(f"No existe el archivo físico: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            anns = anns_by_image.get(img_name, [])

            for ann in anns:
                cat_id = ann["category_id"]
                cat_name = category_map.get(cat_id, "Unknown")
                color = colors.get(cat_id, (0, 255, 0))

                x_min, y_min, w, h = [int(v) for v in ann["bbox"]]
                x_max, y_max = x_min + w, y_min + h

                # Hacemos la caja más gruesa si es de la clase que estamos buscando
                thickness = 3 if cat_id == target_id else 1

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

                label = cat_name
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.rectangle(
                    img, (x_min, y_min - text_h - 4), (x_min + text_w, y_min), color, -1
                )
                cv2.putText(
                    img,
                    label,
                    (x_min, y_min - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

            safe_cat_name = target_cat.replace("/", "_").replace(" ", "_")
            out_file = out_path / f"{safe_cat_name}__{img_name}"

            cv2.imwrite(str(out_file), img)
            print(f"Guardada: {out_path}/{out_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dibuja ejemplos específicos de ciertas categorías (soporta xView GeoJSON y COCO)."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Ruta al archivo de anotaciones (JSON/GeoJSON)",
    )
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Carpeta de imágenes físicas"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./debug/debug_categories",
        help="Carpeta de salida",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["coco", "geojson"],
        default="coco",
        help="Formato de las anotaciones",
    )
    parser.add_argument(
        "--num_per_cat", type=int, default=3, help="Imágenes a sacar por cada categoría"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        required=True,
        help="Lista de categorías a buscar",
    )
    parser.add_argument(
        "--classes_json",
        type=str,
        default=None,
        help="Ruta al xView_classes.json. Si no se indica, lo busca en img_dir.",
    )

    args = parser.parse_args()

    # Si no se proporciona la ruta al diccionario, asume que está junto a las imágenes
    if args.classes_json is None:
        args.classes_json = str(Path(args.img_dir) / "xView_classes.json")

    visualize_categories(
        args.json_path,
        args.img_dir,
        args.out_dir,
        args.categories,
        args.num_per_cat,
        args.format,
        args.classes_json,
    )
