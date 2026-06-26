import logging, torch, os, argparse, random, numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from src.utils.torvis_utils import (
    load_model,
    reduced_focal_loss,
    make_focal_fastrcnn_loss,
    make_weighted_fastrcnn_loss,
    compute_class_weights,
)
from src.utils.import_config import import_py_config
from src.utils.coco_utils import build_coco_max_dets
import torchvision.models.detection.roi_heads as roi_heads
import torchvision.transforms.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler


def set_deterministic_environment(seed: int = 1772110096):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Obliga a CuDNN a usar algoritmos deterministas y apaga el auto-tuning
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Reserva memoria de vídeo para operaciones de reducción deterministas
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Fuerza a PyTorch a usar operaciones deterministas (o lanzar warning si no puede)
    torch.use_deterministic_algorithms(True, warn_only=True)

    logging.info(f"Entorno determinista establecido con semilla={seed}")


def _default_pair_preprocess(image, target):
    """Fallback preprocess if the model config does not define train/val transforms."""
    return F.to_tensor(image), target


def select_train_preprocess(train_preprocess, val_preprocess, augment):
    """Selects train transform. If augment is disabled, train uses val preprocessing."""
    return train_preprocess if augment else val_preprocess


def remove_empty_images(
    coco_dataset, keep_empty_fraction=0.0, empty_sampling_seed=None
):
    original_size = len(coco_dataset.ids)
    valid_ids = []
    empty_ids = []

    for img_id in coco_dataset.ids:
        ann_ids = coco_dataset.coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) > 0:
            valid_ids.append(img_id)
        else:
            empty_ids.append(img_id)

    keep_empty_fraction = float(keep_empty_fraction)
    keep_empty_fraction = max(0.0, min(0.9, keep_empty_fraction))
    kept_empty = 0

    if keep_empty_fraction > 0.0 and len(valid_ids) > 0 and len(empty_ids) > 0:
        target_empty = int(
            (keep_empty_fraction / (1.0 - keep_empty_fraction)) * len(valid_ids)
        )
        target_empty = min(target_empty, len(empty_ids))
        if target_empty > 0:
            rng = random.Random(empty_sampling_seed)
            sampled_empty = rng.sample(empty_ids, target_empty)
            valid_ids.extend(sampled_empty)
            kept_empty = target_empty

    return valid_ids, kept_empty, original_size


EXCLUSION_CATEGORY_ID = -1


def build_target_dict(image_id, target):
    boxes = []
    labels = []
    areas = []
    iscrowd = []
    exclusion_boxes = []

    for annotation in target:
        x, y, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            continue

        if annotation["category_id"] == EXCLUSION_CATEGORY_ID:
            exclusion_boxes.append([x, y, x + width, y + height])
            continue

        boxes.append([x, y, x + width, y + height])
        labels.append(annotation["category_id"])
        areas.append(annotation["area"])
        iscrowd.append(annotation["iscrowd"])

    if len(exclusion_boxes) > 0:
        exclusion_tensor = torch.tensor(exclusion_boxes, dtype=torch.float32)
    else:
        exclusion_tensor = torch.zeros((0, 4), dtype=torch.float32)

    if len(boxes) > 0:
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            "exclusion_zones": exclusion_tensor,
        }

    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "image_id": torch.tensor([image_id]),
        "area": torch.zeros((0,), dtype=torch.float32),
        "iscrowd": torch.zeros((0,), dtype=torch.int64),
        "exclusion_zones": exclusion_tensor,
    }


def sanitize_target_dict(target_dict):
    if target_dict["boxes"].numel() > 0:
        box_widths = target_dict["boxes"][:, 2] - target_dict["boxes"][:, 0]
        box_heights = target_dict["boxes"][:, 3] - target_dict["boxes"][:, 1]
        valid_boxes = (box_widths > 0) & (box_heights > 0)

        if not torch.all(valid_boxes):
            target_dict["boxes"] = target_dict["boxes"][valid_boxes]
            target_dict["labels"] = target_dict["labels"][valid_boxes]
            target_dict["iscrowd"] = target_dict["iscrowd"][valid_boxes]

    if "exclusion_zones" in target_dict and target_dict["exclusion_zones"].numel() > 0:
        ez_widths = target_dict["exclusion_zones"][:, 2] - target_dict["exclusion_zones"][:, 0]
        ez_heights = target_dict["exclusion_zones"][:, 3] - target_dict["exclusion_zones"][:, 1]
        valid_ez = (ez_widths > 0) & (ez_heights > 0)
        if not torch.all(valid_ez):
            target_dict["exclusion_zones"] = target_dict["exclusion_zones"][valid_ez]

    if target_dict["boxes"].numel() == 0:
        target_dict["area"] = torch.zeros((0,), dtype=torch.float32)
    else:
        box_widths = (target_dict["boxes"][:, 2] - target_dict["boxes"][:, 0]).clamp(
            min=0
        )
        box_heights = (target_dict["boxes"][:, 3] - target_dict["boxes"][:, 1]).clamp(
            min=0
        )
        target_dict["area"] = box_widths * box_heights

    return target_dict


def get_configured_preprocesses(model_config_file: str):
    """Reads train/val preprocessing functions from model config with safe defaults."""
    cfg = import_py_config(model_config_file)
    train_preprocess = getattr(cfg, "train_preprocess", None)
    val_preprocess = getattr(cfg, "val_preprocess", None)

    if train_preprocess is None and val_preprocess is None:
        return _default_pair_preprocess, _default_pair_preprocess
    if train_preprocess is None:
        train_preprocess = val_preprocess
    if val_preprocess is None:
        val_preprocess = _default_pair_preprocess

    return train_preprocess, val_preprocess


class XViewDataset(CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        custom_transforms=None,
        filter_empty=True,
        keep_empty_fraction=0.0,
        empty_sampling_seed=None,
    ):
        super().__init__(img_folder, ann_file)
        self.custom_transforms = custom_transforms

        if filter_empty:
            valid_ids, kept_empty, original_size = remove_empty_images(
                self,
                keep_empty_fraction=keep_empty_fraction,
                empty_sampling_seed=empty_sampling_seed,
            )
            self.ids = valid_ids
            removed_empty = original_size - len(valid_ids)
            kept_ratio = (kept_empty / len(self.ids)) if len(self.ids) > 0 else 0.0
            logging.info(
                "Filtro aplicado: "
                f"eliminadas vacías={removed_empty}, "
                f"con objetos={len(valid_ids) - kept_empty}, "
                f"vacías conservadas={kept_empty}, "
                f"total={len(valid_ids)}, "
                f"fracción_vacías={kept_ratio:.3f}"
            )

    def __getitem__(self, idx):
        # Obtiene la imagen y las anotaciones usando el método original de CocoDetection
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target_dict = build_target_dict(image_id, target)

        if self.custom_transforms is not None:
            img, target_dict = self.custom_transforms(img, target_dict)
        else:
            img = F.to_tensor(img)

        img = torch.as_tensor(img, dtype=torch.float32)
        target_dict["boxes"] = torch.as_tensor(
            target_dict["boxes"], dtype=torch.float32
        )
        target_dict["labels"] = torch.as_tensor(
            target_dict["labels"], dtype=torch.int64
        )
        target_dict["image_id"] = torch.as_tensor(
            target_dict["image_id"], dtype=torch.int64
        )
        target_dict["iscrowd"] = torch.as_tensor(
            target_dict["iscrowd"], dtype=torch.int64
        )

        return img, sanitize_target_dict(target_dict)


def collate_fn(batch):
    """Función de colación personalizada para manejar lotes de imágenes con diferentes números de objetos."""
    return tuple(zip(*batch))


def seed_worker(worker_id):
    """Función para establecer la semilla en cada trabajador del DataLoader para reproducibilidad."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    train_img_folder,
    train_ann_file,
    val_img_folder,
    val_ann_file,
    batch_size,
    train_preprocess,
    val_preprocess,
    num_workers,
    seed=None,
    augment=False,
    train_empty_fraction=0.0,
):
    """Crea DataLoaders para los conjuntos de entrenamiento y validación."""
    selected_train_preprocess = select_train_preprocess(
        train_preprocess, val_preprocess, augment
    )

    train_dataset = XViewDataset(
        train_img_folder,
        train_ann_file,
        custom_transforms=selected_train_preprocess,
        filter_empty=True,
        keep_empty_fraction=train_empty_fraction,
        empty_sampling_seed=seed,
    )
    val_dataset = XViewDataset(
        val_img_folder,
        val_ann_file,
        custom_transforms=val_preprocess,
        filter_empty=True,
    )

    # g = torch.Generator()
    # g.manual_seed(seed)  # Semilla fija para reproducibilidad en el DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        # worker_init_fn=seed_worker,
        # generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        # worker_init_fn=seed_worker,
        # generator=g
    )

    return train_loader, val_loader


def get_custom_fasterrcnn(
    model_config_file: str,
    num_classes: int,
    exclusion_iou_thresh: float | None = None,
):
    """Carga el modelo Faster R-CNN preentrenado y reemplaza la cabeza de predicción para adaptarse al número de clases deseado.

    Si ``exclusion_iou_thresh`` no es None, se inyectan las subclases de RPN
    y RoIHeads que soportan zonas de exclusión.
    """
    model, _, categories = load_model(model_config_file)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if exclusion_iou_thresh is not None:
        from src.utils.exclusion_zones import add_exclusion_zones_support
        add_exclusion_zones_support(model, exclusion_iou_thresh=exclusion_iou_thresh)

    train_preprocess, val_preprocess = get_configured_preprocesses(model_config_file)

    return model, train_preprocess, val_preprocess, categories


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    accumulation_steps,
    epoch,
    scaler,
    log_every=0,
):
    """Entrena el modelo por una época completa con acumulación de gradientes."""
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(data_loader)
    inner_loop = tqdm(
        enumerate(data_loader),
        desc=f"Epoch {epoch} [Train]",
        leave=False,
        total=n_batches,
    )

    for i, (images, targets) in inner_loop:
        # Movemos las imágenes y anotaciones a la GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(images, targets)
            losses = torch.stack(list(loss_dict.values())).sum()

        total_loss += losses.item()
        running_loss += losses.item()

        losses = losses / accumulation_steps
        scaler.scale(losses).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == n_batches:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        inner_loop.set_postfix(loss=f"{losses.item() * accumulation_steps:.4f}")

        if log_every > 0 and (i + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            logging.info(
                f"Epoch {epoch} | iter {i+1}/{n_batches} | train_loss={avg_loss:.4f}"
            )
            running_loss = 0.0

    return total_loss / n_batches


def validate_one_epoch(model, data_loader, device, epoch, scaler, log_every=0):
    """Evalúa el modelo en validación y devuelve la pérdida promedio.

    Nota: Faster R-CNN no devuelve losses en eval(), por eso validamos en train()
    con BatchNorm congelado. Las curvas train/val loss son orientativas y no
    estrictamente comparables.
    """
    model.train()
    freeze_batchnorm(model)
    total_loss = 0.0
    running_loss = 0.0
    n_batches = len(data_loader)
    inner_loop = tqdm(data_loader, desc=f"Epoch {epoch} [Val Loss]", leave=False)

    with torch.no_grad():  # Para no actualizar los pesos
        for i, (images, targets) in enumerate(inner_loop):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast():
                loss_dict = model(images, targets)
                losses = torch.stack(list(loss_dict.values())).sum()
            total_loss += losses.item()
            running_loss += losses.item()

            inner_loop.set_postfix(loss=f"{losses.item():.4f}")

            if log_every > 0 and (i + 1) % log_every == 0:
                avg_loss = running_loss / log_every
                logging.info(
                    f"Epoch {epoch} | iter {i+1}/{n_batches} | val_loss={avg_loss:.4f}"
                )
                running_loss = 0.0

    return total_loss / len(data_loader)


def freeze_batchnorm(model):
    """Mantener en modo train para obtener losses, pero congelar los batchnorms"""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            module.eval()


def evaluate_map(model, data_loader, device, coco_gt):
    """Evalúa el modelo usando pycocotools y devuelve AP50."""
    model.eval()  # Cambiamos a evaluación para obtener detecciones, no losses
    results = []

    inner_loop = tqdm(data_loader, desc="[Eval AP50]", leave=False)

    with torch.no_grad():
        for images, targets in inner_loop:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    # COCO usa [x_min, y_min, width, height]
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min

                    if width <= 0 or height <= 0:
                        continue

                    results.append(
                        {
                            "image_id": int(image_id),
                            "category_id": int(label),
                            "bbox": [
                                float(x_min),
                                float(y_min),
                                float(width),
                                float(height),
                            ],
                            "score": float(score),
                        }
                    )

    if not results:
        logging.warning("No se generó ninguna predicción en la validación.")
        return 0.0

    try:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

        # Alineamos maxDets con la capacidad final de detecciones del modelo.
        detections_per_img = int(getattr(model.roi_heads, "detections_per_img", 1500))
        coco_eval.params.maxDets = build_coco_max_dets(detections_per_img)

        # Modo AP50-only: evitamos computar la batería completa AP@[0.50:0.95].
        coco_eval.params.iouThrs = np.array([0.5], dtype=np.float64)
        coco_eval.params.areaRng = [[0**2, 1e5**2]]
        coco_eval.params.areaRngLbl = ["all"]

        coco_eval.evaluate()
        coco_eval.accumulate()

        precision = coco_eval.eval.get("precision", None)
        if precision is None or precision.size == 0:
            return 0.0

        # precision dims: [IoU, Recall, Category, Area, MaxDets]
        p50 = precision[0, :, :, 0, -1]
        p50 = p50[p50 > -1]
        if p50.size == 0:
            return 0.0

        return float(np.mean(p50))
    except Exception as e:
        logging.error(f"Error al calcular COCO AP50: {e}")
        return 0.0


def cleanup_old_checkpoints(output_dir: Path, keep_last_n: int):
    checkpoints = sorted(output_dir.glob("fasterrcnn_epoch_*.pth"))
    for old_ckpt in checkpoints[:-keep_last_n]:
        old_ckpt.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de Faster R-CNN en el conjunto de datos xView"
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
        help="Archivo de anotaciones de entrenamiento",
    )
    parser.add_argument(
        "--val_img_folder",
        type=str,
        required=True,
        help="Carpeta de imágenes de validación",
    )
    parser.add_argument(
        "--val_ann_file",
        type=str,
        required=True,
        help="Archivo de anotaciones de validación",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directorio para guardar los modelos entrenados",
    )
    parser.add_argument(
        "--seed", type=int, default=1772110096, help="Semilla para reproducibilidad"
    )

    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Número de épocas de entrenamiento"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Tamaño del lote para entrenamiento"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        help="Tasa de aprendizaje para el optimizador",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Hilos de CPU para cargar los datos"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Número de clases (para xView con macro clases, 9 + fondo)",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=2,
        help="Número de pasos para acumulación de gradientes",
    )
    parser.add_argument(
        "--lr_milestones",
        type=str,
        nargs="*",
        default=[],
        help="Épocas exactas donde reducir el LR. Ej: --lr_milestones 20 40. Déjalo vacío para LR constante.",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.1,
        help="Factor de reducción del LR (MultiStepLR)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Ruta a un checkpoint para reanudar el entrenamiento",
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="Guardar checkpoint cada N épocas"
    )
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Calcular el AP50 cada N épocas"
    )
    parser.add_argument(
        "--keep_last_n",
        type=int,
        default=3,
        help="Número de checkpoints recientes a conservar",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo: cuda, cuda:0, cuda:1, cpu",
    )
    parser.add_argument(
        "--focal_loss",
        type=int,
        default=0,
        help="Usar focal loss (1) en lugar de cross-entropy (0) para el ROI head",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="α da focal loss (default 0.25, estilo RetinaNet).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="γ da focal loss (default 2.0, estilo RetinaNet).",
    )
    parser.add_argument(
        "--focal_threshold",
        type=float,
        default=None,
        help="Se se indica, focal weighting só se aplica para p_t < threshold "
        "(variante reducida estilo paper de xView). Sen indicar, focal puro.",
    )
    parser.add_argument(
        "--class_weights_mode",
        type=str,
        default="none",
        choices=["none", "inverse", "sqrt_inverse", "capped"],
        help="Pesos por clase na cross-entropy do ROI head. 'none' (defecto) = "
        "CE estándar. 'inverse' = 1/freq normalizado. 'sqrt_inverse' = 1/sqrt(freq) "
        "normalizado. 'capped' = inverse limitado por --class_weights_cap.",
    )
    parser.add_argument(
        "--class_weights_cap",
        type=float,
        default=20.0,
        help="Tope para o modo 'capped' (peso máximo por clase). Default 20.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        help="Épocas de warmup lineal del LR (0 para desactivar)",
    )
    parser.add_argument(
        "--log_every_train",
        type=int,
        default=100,
        help="Loguear la train loss media cada N iteraciones (0 para desactivar)",
    )
    parser.add_argument(
        "--log_every_eval",
        type=int,
        default=100,
        help="Loguear la val loss media cada N iteraciones (0 para desactivar)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Usar train_preprocess del config. Si se omite, train usa val_preprocess. Activa también el multi-res del paper de xView.",
    )
    parser.add_argument(
        "--train_empty_fraction",
        type=float,
        default=0.0,
        help="Fracción objetivo de imágenes vacías en el dataset final de entrenamiento (ej: 0.05).",
    )
    parser.add_argument(
        "--exclusion_iou_thresh",
        type=float,
        default=None,
        help="Se se indica, activa o soporte de zonas de exclusión no modelo. "
             "Anchors/proposals cuxo IoU con calquera zona de exclusión >= "
             "este limiar márcanse como ignorados (label=-1). As zonas léense "
             "directamente das anotacións con category_id = -1 do COCO de "
             "adestramento. Valor típico: 0.5.",
    )

    args = parser.parse_args()

    args.lr_milestones = [int(m) for m in args.lr_milestones if m.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Argumentos: {vars(args)}")

    # Desactivamos el entorno determinista para acelerar el entrenamiento
    # set_deterministic_environment(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    if args.focal_loss and args.class_weights_mode != "none":
        raise ValueError("--focal_loss e --class_weights_mode son mutuamente exclusivos.")

    if args.focal_loss:
        roi_heads.fastrcnn_loss = make_focal_fastrcnn_loss(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            threshold=args.focal_threshold,
        )
        logging.info(
            "Focal loss activada (alpha=%s, gamma=%s, threshold=%s)",
            args.focal_alpha,
            args.focal_gamma,
            args.focal_threshold,
        )
    elif args.class_weights_mode != "none":
        class_weights = compute_class_weights(
            args.train_ann_file,
            num_classes=args.num_classes,
            mode=args.class_weights_mode,
            cap=args.class_weights_cap,
        )
        roi_heads.fastrcnn_loss = make_weighted_fastrcnn_loss(class_weights)
        logging.info(
            "Cross-entropy pesada activada (mode=%s, cap=%s). Pesos (incluindo "
            "background no índice 0): %s",
            args.class_weights_mode,
            args.class_weights_cap if args.class_weights_mode == "capped" else "n/a",
            [round(float(w), 4) for w in class_weights],
        )
    if args.augment:
        logging.info("Usando train_preprocess definido en el config del modelo")

    logging.info("Cargando anotaciones de validación para evaluación AP50...")
    coco_gt = COCO(args.val_ann_file)

    model, train_preprocess, val_preprocess, categories = get_custom_fasterrcnn(
        args.config_file,
        args.num_classes,
        exclusion_iou_thresh=args.exclusion_iou_thresh,
    )
    if args.exclusion_iou_thresh is not None:
        logging.info(
            "Zonas de exclusión activadas (IoU thresh = %s).",
            args.exclusion_iou_thresh,
        )
    model.to(device)

    train_loader, val_loader = get_dataloaders(
        args.train_img_folder,
        args.train_ann_file,
        args.val_img_folder,
        args.val_ann_file,
        args.batch_size,
        train_preprocess,
        val_preprocess,
        args.num_workers,
        seed=args.seed,
        augment=args.augment,
        train_empty_fraction=args.train_empty_fraction,
    )

    # Loader sin filtro para evaluación AP50 (incluye imágenes vacías, protocolo COCO estándar)
    eval_dataset = XViewDataset(
        args.val_img_folder,
        args.val_ann_file,
        custom_transforms=val_preprocess,
        filter_empty=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005
    )

    # Scheduler: warmup lineal + MultiStepLR
    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    )
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, total_iters=args.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = main_scheduler

    best_ap50 = -1.0
    start_epoch = 0

    if args.resume:
        logging.info(f"Reanudando desde checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_ap50 = checkpoint.get("ap50", checkpoint.get("mAP", -1.0))

        if args.lr_milestones or (
            args.learning_rate != optimizer.param_groups[0]["lr"]
        ):
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate
                param_group["initial_lr"] = args.learning_rate
            main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
            )

            if args.warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.001, total_iters=args.warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[args.warmup_epochs],
                )
            else:
                scheduler = main_scheduler

            for _ in range(start_epoch):
                scheduler.step()
        elif "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler.step()

        logging.info(
            f"Reanudando desde época {start_epoch} con AP50={best_ap50:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}"
        )

    scaler = GradScaler()

    epoch_loop = tqdm(range(start_epoch, args.num_epochs), desc="Epochs")

    for epoch in epoch_loop:
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            args.accumulation_steps,
            epoch + 1,
            scaler,
            log_every=args.log_every_train,
        )
        val_loss = validate_one_epoch(
            model, val_loader, device, epoch + 1, scaler, log_every=args.log_every_eval
        )

        ap50 = 0.0
        if (epoch + 1) % args.eval_every == 0:
            ap50 = evaluate_map(model, eval_loader, device, coco_gt)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_loop.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            AP50=f"{ap50:.4f}",
            lr=f"{current_lr:.6f}",
        )
        logging.info(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | AP50={ap50:.4f} | lr={current_lr:.6f}"
        )

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"fasterrcnn_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss,
                    "val_loss": val_loss,
                    "mAP": ap50,
                    "ap50": ap50,
                },
                checkpoint_path,
            )
            cleanup_old_checkpoints(output_dir, args.keep_last_n)

        if ap50 > best_ap50:
            best_ap50 = ap50
            best_checkpoint_path = output_dir / "fasterrcnn_best.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss,
                    "val_loss": val_loss,
                    "mAP": ap50,
                    "ap50": ap50,
                },
                best_checkpoint_path,
            )
            logging.info(
                f"Nuevo mejor modelo guardado con AP50={best_ap50:.4f} (val_loss={val_loss:.4f})"
            )

    logging.info("Entrenamiento completo.")


if __name__ == "__main__":
    main()
