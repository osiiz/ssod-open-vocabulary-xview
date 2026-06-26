import torch
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np

# En principio todas son .tif, pero si en el futuro se agregan otras, no habría que modificar el código
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class ImageFolderDataset(Dataset):
    """Dataset personalizado para cargar imágenes desde una carpeta."""

    def __init__(self, folder_path: str):
        folder = Path(folder_path)
        self.image_paths = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No se encontraron imágenes en: {folder_path}\n"
                f"Extensiones soportadas: {SUPPORTED_EXTENSIONS}"
            )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_np = np.array(image)
        return self.to_tensor(image_np)


def collate_list(batch):
    """Evita que DataLoader apile imágenes de distintas dimensiones."""
    return batch


def compute_mean_and_std(image_folder: str, batch_size: int = 64, num_workers: int = 4):
    """
    Calcula la media y desviación estándar por canal sobre todas las imágenes.
    Usa la fórmula E[X²] - E[X]² para evitar dos pasadas por el dataset.
    """
    dataset = ImageFolderDataset(image_folder)
    print(f"Imágenes encontradas: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_list,
    )

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_pixels = 0
    n_checked = 0

    for batch_idx, batch_images in enumerate(loader):
        for img in batch_images:
            # Verificar dimensiones consistentes
            h, w = img.shape[1], img.shape[2]
            channel_sum += img.sum(dim=[1, 2])
            channel_squared_sum += (img**2).sum(dim=[1, 2])
            num_pixels += h * w
            n_checked += 1

        # Log de progreso cada 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"  Procesados {n_checked}/{len(dataset)} tiles...", flush=True)

    mean = channel_sum / num_pixels
    std = torch.sqrt((channel_squared_sum / num_pixels) - mean**2)

    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcula media y desviación estándar de un conjunto de imágenes."
    )
    parser.add_argument(
        "image_folder",
        type=str,
        help="Ruta a la carpeta con las imágenes " "(.tif, .tiff, .png, .jpg, .jpeg)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    mean, std = compute_mean_and_std(
        args.image_folder, args.batch_size, args.num_workers
    )
    print(f"\nMedia (R, G, B): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"Std   (R, G, B): [{std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}]")
