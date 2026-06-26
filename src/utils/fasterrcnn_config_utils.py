import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as T


PYODI_SCALES = [
    0.8449820949735835,
    1.2090051389607381,
    2.158346849091415,
]
PYODI_RATIOS = [0.4662751642340896, 0.8841624698384491, 1.6894897595503928]
PYODI_BASE_SIZES = [4, 8, 16, 32, 64]


def build_anchor_sizes(base_sizes=None, scales=None):
    base_sizes = PYODI_BASE_SIZES if base_sizes is None else base_sizes
    scales = PYODI_SCALES if scales is None else scales
    return tuple(
        tuple(int(base_size * scale) for scale in scales) for base_size in base_sizes
    )


def build_aspect_ratios(base_sizes=None, ratios=None):
    base_sizes = PYODI_BASE_SIZES if base_sizes is None else base_sizes
    ratios = PYODI_RATIOS if ratios is None else ratios
    return (tuple(ratios),) * len(base_sizes)


def _prepare_inputs(image, target):
    image = T.ToImage()(image)
    boxes_tensor = torch.as_tensor(target["boxes"], dtype=torch.float32)
    canvas_size = (int(image.shape[-2]), int(image.shape[-1]))
    boxes = tv_tensors.BoundingBoxes(
        boxes_tensor,
        format="XYXY",
        canvas_size=canvas_size,
    )  # type: ignore[call-overload]
    exclusion = target.get("exclusion_zones")
    if exclusion is not None and (
        not torch.is_tensor(exclusion) or exclusion.numel() > 0
    ):
        ez_tensor = torch.as_tensor(exclusion, dtype=torch.float32)
        exclusion_bbox = tv_tensors.BoundingBoxes(
            ez_tensor,
            format="XYXY",
            canvas_size=canvas_size,
        )  # type: ignore[call-overload]
    else:
        exclusion_bbox = None
    return image, boxes, exclusion_bbox


def _finalize_target(target, boxes, exclusion):
    target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
    if exclusion is not None:
        target["exclusion_zones"] = torch.as_tensor(exclusion, dtype=torch.float32)
    elif "exclusion_zones" in target:
        # Preserva un campo baleiro coherente coa canvas redimensionada
        target["exclusion_zones"] = torch.zeros((0, 4), dtype=torch.float32)
    return target


def build_pair_preprocess(pair_transforms=None):
    """Build a (image, target) preprocess callable from pair transforms.

    pair_transforms must accept (image, boxes[, exclusion_zones]) and return
    the corresponding number of outputs. Os transforms de torchvision v2
    propagan as transformacións xeométricas a calquera BoundingBoxes que se
    pase como argumento, polo que basta con pasar tamén o tensor de zonas
    cando exista.
    """

    def preprocess(image, target):
        image, boxes, exclusion = _prepare_inputs(image, target)
        if pair_transforms is not None:
            if exclusion is not None:
                image, boxes, exclusion = pair_transforms(image, boxes, exclusion)
            else:
                image, boxes = pair_transforms(image, boxes)
        target = _finalize_target(target, boxes, exclusion)
        return image, target

    return preprocess


def build_train_preprocess():
    d4_transforms = T.RandomChoice(
        [
            T.Identity(),
            T.RandomRotation((90, 90)),
            T.RandomRotation((180, 180)),
            T.RandomRotation((270, 270)),
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0),
            T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomRotation((90, 90))]),
            T.Compose([T.RandomVerticalFlip(p=1.0), T.RandomRotation((90, 90))]),
        ]
    )

    train_transforms = T.Compose(
        [
            d4_transforms,
            T.ClampBoundingBoxes(),
            T.ToDtype(torch.float32, scale=True),
        ]
    )
    return build_pair_preprocess(train_transforms)


def build_val_preprocess():
    val_transforms = T.Compose([T.ToDtype(torch.float32, scale=True)])
    return build_pair_preprocess(val_transforms)
