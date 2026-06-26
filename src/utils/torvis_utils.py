"""Utilities to deal with torchvision models and checkpoints."""

from pathlib import Path
import torch
from torchvision.models import get_model
import torchvision.models as torvis_models
import torchvision.models.detection as torvis_detection
from src.utils.import_config import import_py_config
import torchvision.models.detection.roi_heads as roi_heads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
import torch.nn.functional as F


def reduced_focal_loss(class_logits, box_regression, labels, regression_targets):
    """Computes the focal loss for object detection."""
    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)

    alpha = 0.25
    gamma = 2.0  # Focusing parameter, higher values focus more on hard examples
    th = 0.25  # Threshold for positive class

    ce_loss = F.cross_entropy(class_logits, labels_cat, reduction="none")
    pt = torch.exp(-ce_loss)  # Probability of the true class

    # Alpha weighting: alpha for foreground, (1 - alpha) for background
    alpha_t = torch.where(labels_cat > 0, alpha, 1 - alpha)

    focal_weight = torch.where(
        pt < th,
        torch.ones_like(pt),  # No focal weighting for easy examples
        alpha_t * (1 - pt) ** gamma,  # Focal weighting for hard examples
    )

    classification_loss = (alpha_t * focal_weight * ce_loss).mean()

    sampled_pos_ind_subset = torch.where(labels_cat > 0)[0]
    labels_pos = labels_cat[sampled_pos_ind_subset]

    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_ind_subset, labels_pos],
        regression_targets_cat[sampled_pos_ind_subset],
        beta=1.0 / 9.0,
        reduction="sum",
    )

    num_elements = max(1, labels_cat.numel())
    box_loss = box_loss / num_elements

    return classification_loss, box_loss


def make_focal_fastrcnn_loss(alpha=0.25, gamma=2.0, threshold=None):
    """Devolve unha función fastrcnn_loss con focal loss parametrizable.

    A clasificación segue a fórmula RetinaNet:
        FL(p_t) = -alpha_t * (1 - p_t) ** gamma * log(p_t)
    co peso de balance foreground/background:
        alpha_t = alpha   se a clase é foreground (label > 0)
                = 1-alpha se é background

    Se `threshold` non é None, replícase a variante reducida do paper de xView:
    a focal weight só se aplica cando p_t < threshold; en caso contrario, peso
    constante 1.0. Por defecto (`threshold=None`) emprégase o focal estándar
    sen descontinuidades, máis axeitado para casos con poucas clases.

    A perda de regresión de bbox queda igual que en fastrcnn_loss orixinal.
    """

    def loss(class_logits, box_regression, labels, regression_targets):
        labels_cat = torch.cat(labels, dim=0)
        regression_targets_cat = torch.cat(regression_targets, dim=0)

        ce_loss = F.cross_entropy(class_logits, labels_cat, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(labels_cat > 0, alpha, 1 - alpha)

        focal_w = (1 - pt) ** gamma
        if threshold is not None:
            focal_w = torch.where(pt < threshold, torch.ones_like(pt), alpha_t * focal_w)
            classification_loss = (alpha_t * focal_w * ce_loss).mean()
        else:
            classification_loss = (alpha_t * focal_w * ce_loss).mean()

        sampled_pos_ind_subset = torch.where(labels_cat > 0)[0]
        labels_pos = labels_cat[sampled_pos_ind_subset]

        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_ind_subset, labels_pos],
            regression_targets_cat[sampled_pos_ind_subset],
            beta=1.0 / 9.0,
            reduction="sum",
        )

        num_elements = max(1, labels_cat.numel())
        box_loss = box_loss / num_elements

        return classification_loss, box_loss

    return loss


def make_weighted_fastrcnn_loss(class_weights):
    """Devolve unha función fastrcnn_loss que usa cross-entropy pesada por clase.

    `class_weights` é un tensor 1-D de tamaño num_classes+1 (incluído background
    no índice 0). Os pesos aplícanse só á perda de clasificación; a perda de
    regresión de bbox queda igual que no fastrcnn_loss orixinal de torchvision.
    """

    def loss(class_logits, box_regression, labels, regression_targets):
        labels_cat = torch.cat(labels, dim=0)
        regression_targets_cat = torch.cat(regression_targets, dim=0)

        weight = class_weights.to(class_logits.device)
        classification_loss = F.cross_entropy(class_logits, labels_cat, weight=weight)

        sampled_pos_ind_subset = torch.where(labels_cat > 0)[0]
        labels_pos = labels_cat[sampled_pos_ind_subset]

        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_ind_subset, labels_pos],
            regression_targets_cat[sampled_pos_ind_subset],
            beta=1.0 / 9.0,
            reduction="sum",
        )

        num_elements = max(1, labels_cat.numel())
        box_loss = box_loss / num_elements

        return classification_loss, box_loss

    return loss


def compute_class_weights(coco_ann_file, num_classes, mode="sqrt_inverse", cap=None):
    """Calcula pesos por clase a partir dun ficheiro COCO de anotacións.

    O peso da clase 0 (background) sempre se fixa a 1.0. Para as clases
    en [1..num_classes-1] usa:
      - "inverse":      w_c = (total / freq_c) normalizado a media 1.
      - "sqrt_inverse": w_c = sqrt(total / freq_c) normalizado a media 1.
      - "capped":       igual a 'inverse' pero limitado por `cap` (ex. cap=20).

    Devolve un torch.Tensor de tamaño num_classes (background incluído).
    """
    import json
    from collections import Counter

    with open(coco_ann_file) as fh:
        coco = json.load(fh)
    counts = Counter(a["category_id"] for a in coco["annotations"])
    total = sum(counts.values())
    weights = torch.ones(num_classes, dtype=torch.float32)
    raw = []
    for cid in range(1, num_classes):
        freq = counts.get(cid, 0)
        if freq <= 0:
            raw.append(1.0)
            continue
        if mode == "inverse":
            w = total / freq
        elif mode == "sqrt_inverse":
            w = (total / freq) ** 0.5
        elif mode == "capped":
            w = min(cap if cap is not None else 20.0, total / freq)
        else:
            raise ValueError(f"class_weights_mode descoñecido: {mode}")
        raw.append(w)
    raw_t = torch.tensor(raw, dtype=torch.float32)
    # Normalizar para que a media dos pesos das clases foreground sexa 1
    raw_t = raw_t / raw_t.mean()
    weights[1:] = raw_t
    return weights


def load_checkpoint(
    model_path: Path,
    model_name: str,
    num_classes: int = 10,
    box_detections_per_img: int = 1600,
    anchor_sizes=None,
    aspect_ratios=None,
    min_size=800,
    max_size=1333,
    image_mean=None,
    image_std=None,
    rpn_pre_nms_top_n_train=2000,
    rpn_post_nms_top_n_train=1000,
    rpn_pre_nms_top_n_test=1000,
    rpn_post_nms_top_n_test=1000,
    rpn_batch_size_per_image=256,
):
    """Loads a model checkpoint from a .pt or .pth file."""

    model_path = Path(model_path)
    data = torch.load(model_path, map_location="cpu")
    if isinstance(data, torch.nn.Module):
        model = data
    elif isinstance(data, dict):
        state_dict = None
        if all(isinstance(v, torch.Tensor) for v in data.values()):
            state_dict = data
        else:
            for key in ["state_dict", "model", "model_state_dict", "net", "module"]:
                if key in data and isinstance(data[key], dict):
                    state_dict = data[key]
                    break

        if state_dict is not None:
            kwargs = {"weights": None}
            if num_classes is not None:
                kwargs["num_classes"] = num_classes

            is_detection = hasattr(torvis_detection, model_name)

            if is_detection:
                # We always reconstruct checkpoints from saved weights,
                # so backbone pretrained weights are not needed here.
                kwargs["weights_backbone"] = None
                kwargs["min_size"] = min_size
                kwargs["max_size"] = max_size
                if "rcnn" in model_name.lower():
                    kwargs["box_detections_per_img"] = box_detections_per_img
                    kwargs["rpn_pre_nms_top_n_train"] = rpn_pre_nms_top_n_train
                    kwargs["rpn_post_nms_top_n_train"] = rpn_post_nms_top_n_train
                    kwargs["rpn_pre_nms_top_n_test"] = rpn_pre_nms_top_n_test
                    kwargs["rpn_post_nms_top_n_test"] = rpn_post_nms_top_n_test
                    kwargs["rpn_batch_size_per_image"] = rpn_batch_size_per_image
                elif "retinanet" in model_name.lower() or "fcos" in model_name.lower():
                    kwargs["detections_per_img"] = box_detections_per_img

            if image_mean is not None and image_std is not None:
                kwargs["image_mean"] = image_mean
                kwargs["image_std"] = image_std

            model = get_model(model_name, **kwargs)

            if (
                is_detection
                and anchor_sizes is not None
                and aspect_ratios is not None
                and hasattr(model, "rpn")
                and hasattr(model.rpn, "anchor_generator")
            ):
                model.rpn.anchor_generator = AnchorGenerator(
                    sizes=anchor_sizes, aspect_ratios=aspect_ratios
                )
                num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
                in_channels = model.rpn.head.cls_logits.in_channels
                model.rpn.head = RPNHead(in_channels, num_anchors)

            # Handle DataParallel checkpoints
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Could not find state_dict in checkpoint: {model_path}")
    else:
        raise ValueError(f"Unrecognized checkpoint format: {model_path}")

    return model


def load_pretrained(
    model_name: str,
    weights_name: str,
    box_detections_per_img: int = 1600,
    image_mean=None,
    image_std=None,
    anchor_sizes=None,
    aspect_ratios=None,
    min_size=800,
    max_size=1333,
    rpn_pre_nms_top_n_train=2000,
    rpn_post_nms_top_n_train=1000,
    rpn_pre_nms_top_n_test=1000,
    rpn_post_nms_top_n_test=1000,
    rpn_batch_size_per_image=256,
):
    """Loads a pretrained torchvision model and extract its
    preprocessing transforms and categories."""

    if hasattr(torvis_detection, model_name):
        model_fn = getattr(torvis_detection, model_name)
        model_submodule = torvis_detection
        is_detection = True
    elif hasattr(torvis_models, model_name):
        model_fn = getattr(torvis_models, model_name)
        model_submodule = torvis_models
        is_detection = False
    else:
        raise ValueError(
            f"Not a torchvision classification/detection model: {model_name}"
        )

    try:
        weights_enum_name, weights_member = weights_name.split(".")
        weights_enum = getattr(model_submodule, weights_enum_name)
        weights = getattr(weights_enum, weights_member)
    except AttributeError:
        raise ValueError(f"Unknown weights: {weights_name}")

    kwargs = {"weights": weights}

    if is_detection:
        kwargs["min_size"] = min_size
        kwargs["max_size"] = max_size
        if "rcnn" in model_name.lower():
            kwargs["box_detections_per_img"] = box_detections_per_img
            kwargs["rpn_pre_nms_top_n_train"] = rpn_pre_nms_top_n_train
            kwargs["rpn_post_nms_top_n_train"] = rpn_post_nms_top_n_train
            kwargs["rpn_pre_nms_top_n_test"] = rpn_pre_nms_top_n_test
            kwargs["rpn_post_nms_top_n_test"] = rpn_post_nms_top_n_test
            kwargs["rpn_batch_size_per_image"] = rpn_batch_size_per_image
        elif "retinanet" in model_name.lower() or "fcos" in model_name.lower():
            kwargs["detections_per_img"] = box_detections_per_img

        if image_mean is not None and image_std is not None:
            kwargs["image_mean"] = image_mean
            kwargs["image_std"] = image_std

    model = model_fn(**kwargs)

    if (
        is_detection
        and anchor_sizes is not None
        and aspect_ratios is not None
        and hasattr(model, "rpn")
        and hasattr(model.rpn, "anchor_generator")
    ):
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios
        )
        model.rpn.anchor_generator = anchor_generator
        # Rebuild RPNHead to match the new number of anchors per location
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        in_channels = model.rpn.head.cls_logits.in_channels
        model.rpn.head = RPNHead(in_channels, num_anchors)

    preprocess = weights.transforms()
    categories = weights.meta.get("categories", weights.meta.get("classes", None))

    return model, preprocess, categories


def load_model(model_config_file, model_checkpoint_override=None):
    """Loads a torchvision model, either a pretrained model or
    a fine-tuned checkpoint."""

    cfg = import_py_config(model_config_file)

    model_path = (
        model_checkpoint_override if model_checkpoint_override is not None else cfg.path
    )
    model_name = cfg.name
    weights_name = cfg.weights_name
    categories = getattr(cfg, "categories", None)
    preprocess = getattr(cfg, "preprocess", None)
    num_classes = getattr(cfg, "num_classes", None)
    image_mean = getattr(cfg, "image_mean", None)
    image_std = getattr(cfg, "image_std", None)
    anchor_sizes = getattr(cfg, "anchor_sizes", None)
    aspect_ratios = getattr(cfg, "aspect_ratios", None)
    min_size = getattr(cfg, "min_size", 800)
    max_size = getattr(cfg, "max_size", 1333)
    rpn_pre = getattr(cfg, "rpn_pre_nms_top_n_train", 2000)
    rpn_post = getattr(cfg, "rpn_post_nms_top_n_train", 1000)
    rpn_pre_test = getattr(cfg, "rpn_pre_nms_top_n_test", 1000)
    rpn_post_test = getattr(cfg, "rpn_post_nms_top_n_test", 1000)
    rpn_batch = getattr(cfg, "rpn_batch_size_per_image", 256)
    max_detections = getattr(cfg, "box_detections_per_img", 1000)

    # roi_heads.fastrcnn_loss = focal_loss

    if Path(model_path).suffix in [".pt", ".pth"]:
        model = load_checkpoint(
            model_path,
            model_name,
            num_classes=num_classes,
            box_detections_per_img=max_detections,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            rpn_pre_nms_top_n_train=rpn_pre,
            rpn_post_nms_top_n_train=rpn_post,
            rpn_pre_nms_top_n_test=rpn_pre_test,
            rpn_post_nms_top_n_test=rpn_post_test,
            rpn_batch_size_per_image=rpn_batch,
        )
    elif model_path == "pytorch/vision":
        model, preprocess, categories = load_pretrained(
            model_name,
            weights_name,
            box_detections_per_img=max_detections,
            image_mean=image_mean,
            image_std=image_std,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            min_size=min_size,
            max_size=max_size,
            rpn_pre_nms_top_n_train=rpn_pre,
            rpn_post_nms_top_n_train=rpn_post,
            rpn_pre_nms_top_n_test=rpn_pre_test,
            rpn_post_nms_top_n_test=rpn_post_test,
            rpn_batch_size_per_image=rpn_batch,
        )
    else:
        raise ValueError(f"Unrecognized model path: {model_path}")

    return model, preprocess, categories
