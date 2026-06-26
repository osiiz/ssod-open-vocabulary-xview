import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import concurrent.futures
import multiprocessing as mp


def adjust_boxes_to_tile(
    coco_anns: list[dict], tile_limits: list, min_visibility: float
) -> tuple[list, dict]:
    """Ajusta las coordenadas de las cajas a los límnites del tile."""
    c, r, tile_width, tile_height = tile_limits
    adjusted_boxes = []
    stats = {
        "total_annotations_seen": 0,
        "intersecting_annotations": 0,
        "clipped_annotations": 0,
        "kept_annotations": 0,
        "dropped_by_visibility": 0,
        "dropped_non_positive_after_clip": 0,
    }

    for ann in coco_anns:
        stats["total_annotations_seen"] += 1
        cat_id = ann["category_id"]
        x, y, w, h = ann["bbox"]
        original_area = float(w * h)

        if original_area <= 0:
            continue

        # Convertir COCO a coordenadas absolutas globales
        o_left = x
        o_top = y
        o_right = x + w
        o_bottom = y + h

        # Desplazar las coordenadas relativas al tile
        left = o_left - c
        top = o_top - r
        right = o_right - c
        bottom = o_bottom - r

        # Comprobar si hay intersección
        h_match = (
            (0 <= left < tile_width)
            or (0 < right <= tile_width)
            or (left <= 0 and right >= tile_width)
        )
        v_match = (
            (0 <= top < tile_height)
            or (0 < bottom <= tile_height)
            or (top <= 0 and bottom >= tile_height)
        )

        if h_match and v_match:
            stats["intersecting_annotations"] += 1
            # Recortar a los límites del tile
            clipped_left = max(left, 0)
            clipped_top = max(top, 0)
            clipped_right = min(right, tile_width)
            clipped_bottom = min(bottom, tile_height)

            if (
                clipped_left != left
                or clipped_top != top
                or clipped_right != right
                or clipped_bottom != bottom
            ):
                stats["clipped_annotations"] += 1

            left = clipped_left
            top = clipped_top
            right = clipped_right
            bottom = clipped_bottom

            bbox_width = right - left
            bbox_height = bottom - top
            clipped_area = bbox_width * bbox_height
            visibility = clipped_area / original_area

            # Devolver en formato [cat_id, x, y, w, h]
            if bbox_width <= 0 or bbox_height <= 0:
                stats["dropped_non_positive_after_clip"] += 1
            elif visibility >= min_visibility:
                stats["kept_annotations"] += 1
                adjusted_boxes.append(
                    [
                        int(cat_id),
                        float(left),
                        float(top),
                        float(bbox_width),
                        float(bbox_height),
                    ]
                )
            else:
                stats["dropped_by_visibility"] += 1

    return adjusted_boxes, stats


def _merge_stats(dst: dict, src: dict):
    for key, value in src.items():
        dst[key] += int(value)


def process_single_image(
    args: tuple,
) -> tuple[list[str], list[int], list[int], dict[str, list], dict]:
    """Procesa una sola imagen: realiza el tiling y devuelve las anotaciones ajustadas."""
    (
        file_name,
        img_folder_path,
        images_output,
        labels_list,
        tile_width,
        tile_height,
        min_tile_width,
        min_tile_height,
        min_visibility,
    ) = args

    img_path = str(img_folder_path / file_name)
    local_stats = defaultdict(int)

    # Cargamos la imagen. Si cv2.imread devuelve None (p.ej. fallo de E/S
    # intermitente sobre disco compartido) lanzamos excepción en lugar de
    # silenciar el fallo, para que dvc repro pare antes de generar un
    # tileado incompleto.
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"cv2.imread devolveu None para {img_path}")

    img_height, img_width, _ = img.shape
    tile_boxes = {}
    file_names, widths, heights = [], [], []

    for r in range(0, img_height, tile_height):
        for c in range(0, img_width, tile_width):
            stem = file_name.split(".")[0]
            img_file_name = f"img_{stem}_{r}_{c}.tif"
            output_path = str(images_output / img_file_name)

            current_tile_width = min(tile_width, img_width - c)
            current_tile_height = min(tile_height, img_height - r)

            if (
                current_tile_height >= min_tile_height
                and current_tile_width >= min_tile_width
            ):
                tile = img[r : r + current_tile_height, c : c + current_tile_width, :]
                cv2.imwrite(output_path, tile, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

                file_names.append(img_file_name)
                widths.append(current_tile_width)
                heights.append(current_tile_height)

                tile_limits = [c, r, current_tile_width, current_tile_height]
                tile_anns, tile_stats = adjust_boxes_to_tile(
                    labels_list, tile_limits, min_visibility
                )
                tile_boxes[img_file_name] = tile_anns
                _merge_stats(local_stats, tile_stats)

    return file_names, widths, heights, tile_boxes, dict(local_stats)


def process_all_images(
    boxes_dict: dict[str, list],
    img_folder_path: Path,
    images_output: Path,
    tile_width: int,
    tile_height: int,
    min_tile_width: int,
    min_tile_height: int,
    min_visibility: float,
) -> tuple[list[str], list[int], list[int], dict[str, list], dict]:
    """Procesa todas las imágenes"""
    img_files = list(boxes_dict.keys())

    args_list = [
        (
            file_name,
            img_folder_path,
            images_output,
            boxes_dict[file_name],
            tile_width,
            tile_height,
            min_tile_width,
            min_tile_height,
            min_visibility,
        )
        for file_name in img_files
    ]
    file_names, widths, heights, tile_boxes = [], [], [], {}
    global_stats = defaultdict(int)

    num_workers = max(1, mp.cpu_count() - 1)
    with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
        results = list(executor.map(process_single_image, args_list))

    for (
        result_file_names,
        result_widths,
        result_heights,
        result_tile_boxes,
        result_stats,
    ) in results:
        file_names.extend(result_file_names)
        widths.extend(result_widths)
        heights.extend(result_heights)
        tile_boxes.update(result_tile_boxes)
        _merge_stats(global_stats, result_stats)

    return file_names, widths, heights, tile_boxes, dict(global_stats)


def create_coco_annotations(
    file_names: list, widths: list, heights: list, tile_boxes: dict, categories: list
) -> dict:
    """Crea las anotaciones en formato COCO para los tiles."""
    coco_data = {"images": [], "annotations": [], "categories": categories}
    annotation_id = 0
    img_id_counter = 1

    for file_name, width, height in zip(file_names, widths, heights):
        boxes_for_this_tile = tile_boxes.get(file_name, [])

        image_id = img_id_counter
        img_id_counter += 1

        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for box in boxes_for_this_tile:
            type_id, x, y, bbox_width, bbox_height = box
            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(type_id),
                    "bbox": [x, y, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return coco_data


def main():
    parser = argparse.ArgumentParser(
        description="Tile a dataset already in COCO format."
    )
    parser.add_argument("raw_images_path", type=str, help="Path to raw images.")
    parser.add_argument("input_coco_json", type=str, help="Path to the COCO json.")
    parser.add_argument(
        "output_data_path", type=str, help="Path to save the tiled dataset."
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        default=800,
        help="Tile width in pixels (default: 800).",
    )
    parser.add_argument(
        "--tile_height",
        type=int,
        default=800,
        help="Tile height in pixels (default: 800).",
    )
    parser.add_argument(
        "--min_tile_width",
        type=int,
        default=320,
        help="Minimum tile width in pixels (default: 320).",
    )
    parser.add_argument(
        "--min_tile_height",
        type=int,
        default=320,
        help="Minimum tile height in pixels (default: 320).",
    )
    parser.add_argument(
        "--min_visibility",
        type=float,
        default=0.3,
        help=(
            "Minimum visible area fraction (clipped_area/original_area) to keep "
            "a clipped bbox (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default=None,
        help="Optional JSON path to save post-tiling clipping/visibility stats.",
    )
    args = parser.parse_args()

    # Convertimos las rutas a objetos Path
    raw_images_path = Path(args.raw_images_path)
    output_data_path = Path(args.output_data_path)

    # Creamos los directorios de salida
    images_output = output_data_path / "images"
    images_output.mkdir(parents=True, exist_ok=True)

    # Cargamos el JSON de anotaciones COCO
    with open(args.input_coco_json, "r") as f:
        full_coco = json.load(f)

    # Convertimos las anotaciones al formato que espera adjust_boxes_to_tile
    boxes_dict = {img["file_name"]: [] for img in full_coco["images"]}
    img_id_to_filename = {img["id"]: img["file_name"] for img in full_coco["images"]}

    for ann in full_coco["annotations"]:
        file_name = img_id_to_filename[ann["image_id"]]
        boxes_dict[file_name].append(ann)

    # Procesamos todas las imágenes y obtenemos los resultados
    file_names, widths, heights, tile_boxes, tiling_stats = process_all_images(
        boxes_dict,
        raw_images_path,
        images_output,
        args.tile_width,
        args.tile_height,
        args.min_tile_width,
        args.min_tile_height,
        args.min_visibility,
    )

    # Creamos las anotaciones en formato COCO para los tiles
    coco_data = create_coco_annotations(
        file_names, widths, heights, tile_boxes, full_coco["categories"]
    )

    # Guardamos las anotaciones en formato COCO
    with open(output_data_path / "COCO_annotations.json", "w") as f:
        json.dump(coco_data, f, indent=2)

    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        intersecting = int(tiling_stats.get("intersecting_annotations", 0))
        kept = int(tiling_stats.get("kept_annotations", 0))
        dropped_visibility = int(tiling_stats.get("dropped_by_visibility", 0))

        report = {
            "input_coco_json": str(args.input_coco_json),
            "output_coco_json": str(output_data_path / "COCO_annotations.json"),
            "params": {
                "tile_width": int(args.tile_width),
                "tile_height": int(args.tile_height),
                "min_tile_width": int(args.min_tile_width),
                "min_tile_height": int(args.min_tile_height),
                "min_visibility": float(args.min_visibility),
            },
            "stats": {k: int(v) for k, v in sorted(tiling_stats.items())},
            "ratios": {
                "keep_ratio_on_intersections": round(
                    (kept / intersecting) if intersecting > 0 else 0.0,
                    6,
                ),
                "dropped_visibility_ratio_on_intersections": round(
                    (dropped_visibility / intersecting) if intersecting > 0 else 0.0,
                    6,
                ),
            },
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
