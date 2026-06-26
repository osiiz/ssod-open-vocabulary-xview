import argparse
import random
from pathlib import Path

import torch
from torchvision.utils import draw_bounding_boxes, save_image

from src.training.train import XViewDataset, get_configured_preprocesses
from src.utils.import_config import import_py_config


def _label_names(labels, categories):
    names = []
    for label in labels.tolist():
        label = int(label)
        if categories is not None and 0 <= label < len(categories):
            names.append(categories[label])
        else:
            names.append(str(label))
    return names


def visualize_train_augmentation(
    config_file: str,
    train_img_folder: str,
    train_ann_file: str,
    output_dir: str,
    num_samples: int = 20,
    seed: int = 42,
):
    """Visualize random training samples after train-time augmentations."""

    train_preprocess, _ = get_configured_preprocesses(config_file)
    cfg = import_py_config(config_file)
    categories = getattr(cfg, "categories", None)

    dataset = XViewDataset(
        train_img_folder,
        train_ann_file,
        custom_transforms=train_preprocess,
        filter_empty=True,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    sample_indices = indices[: min(num_samples, len(indices))]
    if not sample_indices:
        print("No hay muestras disponibles para visualizar.")
        return

    for i, idx in enumerate(sample_indices):
        image, target = dataset[idx]
        image = image.clamp(0, 1)
        image_uint8 = (image * 255).to(torch.uint8)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            labels = _label_names(target["labels"], categories)
            image_drawn = draw_bounding_boxes(
                image_uint8,
                boxes,
                labels=labels,
                colors="red",
                width=2,
            )
        else:
            image_drawn = image_uint8

        image_id = int(target["image_id"][0].item())
        file_name = dataset.coco.imgs.get(image_id, {}).get(
            "file_name", f"image_{image_id}"
        )
        source_stem = Path(file_name).stem if file_name else f"image_{image_id}"
        out_file = output_path / f"aug_{i:03d}_{source_stem}.png"
        save_image(image_drawn.float() / 255.0, str(out_file))

    print(f"Guardadas {len(sample_indices)} visualizaciones en: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualiza imágenes de entrenamiento tras data augmentation"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Archivo de configuración del modelo",
    )
    parser.add_argument(
        "--train_img_folder",
        type=str,
        required=True,
        help="Carpeta de imágenes de entrenamiento",
    )
    parser.add_argument(
        "--train_ann_file",
        type=str,
        required=True,
        help="Archivo COCO de anotaciones de entrenamiento",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./debug/debug_aug",
        help="Directorio de salida de visualizaciones",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Número de imágenes a visualizar",
    )
    parser.add_argument("--seed", type=int, default=1772110096)
    args = parser.parse_args()

    visualize_train_augmentation(
        config_file=args.config_file,
        train_img_folder=args.train_img_folder,
        train_ann_file=args.train_ann_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )
