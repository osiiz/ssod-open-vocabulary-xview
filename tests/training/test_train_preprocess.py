import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

from src.training.train import get_configured_preprocesses, select_train_preprocess


def _clone_target(target):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in target.items()}


def _dummy_target():
    boxes = torch.tensor([[16.0, 16.0, 64.0, 96.0]], dtype=torch.float32)
    return {
        "boxes": boxes,
        "labels": torch.tensor([1], dtype=torch.int64),
        "image_id": torch.tensor([1], dtype=torch.int64),
        "area": torch.tensor([(64.0 - 16.0) * (96.0 - 16.0)], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }


def _bbox_from_mask(mask: torch.Tensor):
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0:
        return None
    return torch.tensor(
        [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        ],
        dtype=torch.float32,
    )


def test_select_train_preprocess_switches_with_augment_flag():
    train_preprocess = lambda x, y: (x, y)
    val_preprocess = lambda x, y: (x, y)

    assert (
        select_train_preprocess(train_preprocess, val_preprocess, True)
        is train_preprocess
    )
    assert (
        select_train_preprocess(train_preprocess, val_preprocess, False)
        is val_preprocess
    )


def test_get_configured_preprocesses_exposes_distinct_train_and_val_transforms():
    config_file = str(Path("configs/models/fasterrcnn_pretrained.py"))
    train_preprocess, val_preprocess = get_configured_preprocesses(config_file)

    assert train_preprocess is not None
    assert val_preprocess is not None
    assert callable(train_preprocess)
    assert callable(val_preprocess)
    assert train_preprocess is not val_preprocess


def test_config_preprocesses_return_valid_detection_tensors():
    config_file = str(Path("configs/models/fasterrcnn_pretrained.py"))
    train_preprocess, val_preprocess = get_configured_preprocesses(config_file)

    assert train_preprocess is not None
    assert val_preprocess is not None

    image = Image.fromarray(np.full((128, 128, 3), 127, dtype=np.uint8))
    base_target = _dummy_target()

    random.seed(123)
    torch.manual_seed(123)

    train_image, train_target = train_preprocess(image, _clone_target(base_target))
    val_image, val_target = val_preprocess(image, _clone_target(base_target))

    assert isinstance(train_image, torch.Tensor)
    assert isinstance(val_image, torch.Tensor)
    assert train_image.dtype == torch.float32
    assert val_image.dtype == torch.float32
    assert tuple(train_image.shape) == (3, 128, 128)
    assert tuple(val_image.shape) == (3, 128, 128)

    assert train_target["boxes"].shape[1] == 4
    assert torch.all(train_target["boxes"][:, 0] >= 0)
    assert torch.all(train_target["boxes"][:, 1] >= 0)
    assert torch.all(train_target["boxes"][:, 2] <= 128)
    assert torch.all(train_target["boxes"][:, 3] <= 128)

    assert torch.allclose(val_target["boxes"], base_target["boxes"])


def test_d4_transforms_keep_boxes_aligned_with_image_content():
    h = w = 128
    image = tv_tensors.Image(torch.zeros((3, h, w), dtype=torch.uint8))
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[20.0, 30.0, 70.0, 90.0]], dtype=torch.float32),
        format="XYXY",
        canvas_size=(h, w),
    )
    mask = torch.zeros((h, w), dtype=torch.uint8)
    mask[30:90, 20:70] = 1
    mask = tv_tensors.Mask(mask)

    d4_ops = [
        ("identity", T.Identity()),
        ("rot90", T.RandomRotation((90, 90))),
        ("rot180", T.RandomRotation((180, 180))),
        ("rot270", T.RandomRotation((270, 270))),
        ("hflip", T.RandomHorizontalFlip(p=1.0)),
        ("vflip", T.RandomVerticalFlip(p=1.0)),
        (
            "hflip_rot90",
            T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomRotation((90, 90))]),
        ),
        (
            "vflip_rot90",
            T.Compose([T.RandomVerticalFlip(p=1.0), T.RandomRotation((90, 90))]),
        ),
    ]

    for name, op in d4_ops:
        transform = T.Compose([op, T.ClampBoundingBoxes()])
        _, out_boxes, out_mask = transform(image.clone(), boxes.clone(), mask.clone())

        expected = _bbox_from_mask(torch.as_tensor(out_mask))
        assert expected is not None, f"Mask vacia para op={name}"

        actual = torch.as_tensor(out_boxes[0], dtype=torch.float32)
        assert torch.all(
            torch.abs(actual - expected) <= 1.0
        ), f"Caja desalineada en op={name}: actual={actual.tolist()} expected={expected.tolist()}"
