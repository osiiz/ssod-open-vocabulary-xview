import argparse
import json
from pathlib import Path
import pandas as pd


def format_size(size_bytes: float) -> str:
    """Convierte bytes a un formato legible."""
    for unit in ["B", "KB", "MB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} GB"


def get_dir_size(path: Path) -> int:
    """Calcula el tamaño total de un directorio recursivamente."""
    return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())


def analyze_coco(
    json_path: Path, img_dir: Path | None = None, out_file: Path | None = None
):
    """Analiza un dataset en formato COCO y genera un reporte con estadísticas relevantes."""
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

    num_images = len(images)
    num_anns = len(annotations)

    # Extraer estadísticas de las dimensiones de las imágenes
    if num_images > 0:
        avg_w = sum(img.get("width", 0) for img in images) / num_images
        avg_h = sum(img.get("height", 0) for img in images) / num_images
    else:
        avg_w, avg_h = 0, 0

    dir_size_str = format_size(
        get_dir_size(img_dir) if img_dir and img_dir.exists() else 0
    )

    # Extraer estadísticas de las anotaciones usando Pandas
    df_anns = pd.DataFrame(annotations)
    if not df_anns.empty:
        imgs_with_anns = int(df_anns["image_id"].nunique())
        class_counts = (
            df_anns["category_id"].fillna(0).astype(int).value_counts().to_dict()
        )
        max_objs_per_img = int(df_anns["image_id"].value_counts().max())

    else:
        imgs_with_anns = 0
        class_counts = {}
        max_objs_per_img = 0

    # Construir el diccionario de estadísticas
    stats = {
        "dataset_info": {
            "total_images": num_images,
            "images_with_objects": imgs_with_anns,
            "images_with_objects_pct": round((imgs_with_anns / num_images * 100), 2)
            if num_images
            else 0,
            "avg_resolution_width": round(avg_w, 2),
            "avg_resolution_height": round(avg_h, 2),
            "dir_size_str": dir_size_str,
        },
        "objects_info": {
            "total_objects": num_anns,
            "avg_objects_per_valid_image": round((num_anns / imgs_with_anns), 2)
            if imgs_with_anns
            else 0,
            "max_objects_in_single_image": max_objs_per_img,
        },
        "class_distribution": {},
    }

    # Ordenar las clases de mayor a menor frecuencia
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cat_id, count in sorted_classes:
        cat_name = categories.get(cat_id, f"Unknown_{cat_id}")
        stats["class_distribution"][cat_name] = {
            "id": int(str(cat_id)),
            "count": int(count),
            "pct": round((count / num_anns) * 100, 2) if num_anns else 0,
        }

    print(
        f"Estadísticas calculadas para {json_path.name}: {num_images} imágenes, {num_anns} objetos."
    )

    # Guardar en JSON si se especifica la ruta
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"JSON guardado en: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extrae estadísticas de un dataset en formato COCO"
    )
    parser.add_argument("json_path", type=str, help="Ruta al archivo COCO JSON")
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="Ruta a la carpeta de imágenes (para calcular tamaño)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Ruta donde guardar el reporte en formato JSON",
    )

    args = parser.parse_args()

    analyze_coco(
        Path(args.json_path),
        Path(args.img_dir) if args.img_dir else None,
        Path(args.out_file) if args.out_file else None,
    )


if __name__ == "__main__":
    main()
