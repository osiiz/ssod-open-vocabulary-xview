from pathlib import Path

from src.utils.import_config import import_py_config


def test_pretrained_config_keeps_custom_anchors_and_core_fields():
    cfg = import_py_config(str(Path("configs/models/fasterrcnn_pretrained.py")))

    assert cfg.path == "pytorch/vision"
    assert cfg.name == "fasterrcnn_resnet50_fpn_v2"
    assert cfg.weights_name == "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"
    assert cfg.num_classes == 10
    assert isinstance(cfg.categories, list)
    assert len(cfg.categories) == 10

    assert hasattr(cfg, "anchor_sizes")
    assert hasattr(cfg, "aspect_ratios")
    assert len(cfg.anchor_sizes) == 5
    assert len(cfg.aspect_ratios) == 5


def test_pretrained_config_overrides_detector_runtime_params():
    cfg = import_py_config(str(Path("configs/models/fasterrcnn_pretrained.py")))

    assert cfg.min_size == 700
    assert cfg.max_size == 700
    assert cfg.rpn_pre_nms_top_n_train == 3000
    assert cfg.rpn_post_nms_top_n_train == 1500
    assert cfg.rpn_pre_nms_top_n_test == 3000
    assert cfg.rpn_post_nms_top_n_test == 1500
    assert cfg.rpn_batch_size_per_image == 256
    assert cfg.box_detections_per_img == 1500
