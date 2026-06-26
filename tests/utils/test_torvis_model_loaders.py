import pytest

from pathlib import Path
import torch

from src.utils.torvis_utils import *


def _assert_state_dict_equal(model_a, model_b):
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    missing_keys = set(state_a.keys()) - set(state_b.keys())
    assert not missing_keys, (
        "Faltan claves del checkpoint original en el modelo cargado: "
        f"{sorted(missing_keys)[:5]}"
    )

    for name in state_a:
        assert torch.allclose(
            state_a[name], state_b[name]
        ), f"El tensor '{name}' del state_dict no coincide entre el modelo original y el cargado"


@pytest.mark.parametrize(
    "model_name,weights_name, is_faster_rcnn, expected_classes",
    [
        ("resnet50", "ResNet50_Weights.IMAGENET1K_V1", False, 1000),
        (
            "fasterrcnn_resnet50_fpn",
            "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
            True,
            91,
        ),
        ("retinanet_resnet50_fpn", "RetinaNet_ResNet50_FPN_Weights.COCO_V1", False, 91),
    ],
)
def test_load_tv_model(
    model_name, weights_name, is_faster_rcnn, expected_classes, tmp_path
):

    custom_max_dets = 1600
    custom_min_size = 800

    custom_anchor_sizes = ((4,), (8,), (16,), (32,), (64,)) if is_faster_rcnn else None
    custom_aspect_ratios = (
        ((0.5, 1.0, 2.0),) * len(custom_anchor_sizes) if is_faster_rcnn else None
    )

    # Test load of pretrained torchvision model
    model, preprocess, categories = load_pretrained(
        model_name=model_name,
        weights_name=weights_name,
        box_detections_per_img=custom_max_dets,
        min_size=custom_min_size,
        anchor_sizes=custom_anchor_sizes,
        aspect_ratios=custom_aspect_ratios,
    )
    assert isinstance(model, torch.nn.Module)

    if is_faster_rcnn:
        assert (
            model.roi_heads.detections_per_img == custom_max_dets
        ), "El límite de detecciones no se aplicó al RoI Head"
        assert (
            model.rpn.anchor_generator.sizes == custom_anchor_sizes
        ), "Las anclas personalizadas no se inyectaron en el RPN"
        expected_num_anchors = len(custom_anchor_sizes[0]) * len(
            custom_aspect_ratios[0]
        )
        assert (
            model.rpn.head.cls_logits.out_channels == expected_num_anchors
        ), "El RPN Head no se reconstruyó para las nuevas anclas"

    checkpoint_path = tmp_path / "model.pth"
    models = [
        model,
        model.state_dict(),
        {"state_dict": model.state_dict()},
        {"model_state_dict": model.state_dict()},
    ]

    for m in models:
        torch.save(m, checkpoint_path)
        assert checkpoint_path.exists()
        loaded_model = load_checkpoint(
            checkpoint_path,
            model_name,
            num_classes=expected_classes,
            box_detections_per_img=custom_max_dets,
            min_size=custom_min_size,
            anchor_sizes=custom_anchor_sizes,
            aspect_ratios=custom_aspect_ratios,
        )
        assert loaded_model is not None
        assert isinstance(loaded_model, torch.nn.Module)
        if is_faster_rcnn:
            assert loaded_model.roi_heads.detections_per_img == custom_max_dets
            assert loaded_model.rpn.anchor_generator.sizes == custom_anchor_sizes
        _assert_state_dict_equal(model, loaded_model)


def test_load_model_with_checkpoint_override(tmp_path):
    model_name = "resnet50"
    model = get_model(model_name, weights=None)

    checkpoint_path = tmp_path / "resnet50_ckpt.pth"
    torch.save(model.state_dict(), checkpoint_path)

    config_path = tmp_path / "model_config.py"
    config_path.write_text(
        "\n".join(
            [
                'path = "pytorch/vision"',
                'name = "resnet50"',
                'weights_name = "ResNet50_Weights.IMAGENET1K_V1"',
                "preprocess = None",
            ]
        )
    )

    loaded_model, preprocess, categories = load_model(
        str(config_path), model_checkpoint_override=str(checkpoint_path)
    )

    assert isinstance(loaded_model, torch.nn.Module)
    assert preprocess is None
    assert categories is None

    _assert_state_dict_equal(model, loaded_model)
