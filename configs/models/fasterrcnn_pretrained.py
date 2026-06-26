import torch
from torchvision import transforms
import torchvision.transforms.v2 as T
from src.preprocessing.custom_transforms import RandomChipping
from src.utils.fasterrcnn_config_utils import (
    PYODI_BASE_SIZES,
    PYODI_RATIOS,
    PYODI_SCALES,
    build_anchor_sizes,
    build_aspect_ratios,
    build_pair_preprocess,
)

path = "pytorch/vision"
name = "fasterrcnn_resnet50_fpn_v2"
weights_name = "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"
num_classes = 10
min_size = 700
max_size = 700
categories = [
    "__background__",  # 0
    "Aircraft",  # 1
    "Light Vehicle",  # 2
    "Heavy Vehicle",  # 3
    "Railway Vehicle",  # 4
    "Maritime Vessel",  # 5
    "Engineering Vehicle",  # 6
    "Building",  # 7
    "Storage Tank",  # 8
    "Tower & Pylon",  # 9
]
preprocess = transforms.ToTensor()

multires_crop_sizes = [300, 400, 500, 600, 700]
multires_min_visibility = 0.0

_D4_TRANSFORMS = T.RandomChoice(
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

_TRAIN_TRANSFORMS = T.Compose(
    [
        RandomChipping(
            crop_sizes=multires_crop_sizes,
            min_visibility=multires_min_visibility,
        ),
        _D4_TRANSFORMS,
        T.ClampBoundingBoxes(),
        T.ToDtype(torch.float32, scale=True),
    ]
)

_VAL_TRANSFORMS = T.Compose([T.ToDtype(torch.float32, scale=True)])

train_preprocess = build_pair_preprocess(_TRAIN_TRANSFORMS)
val_preprocess = build_pair_preprocess(_VAL_TRANSFORMS)

anchor_sizes = build_anchor_sizes(PYODI_BASE_SIZES, PYODI_SCALES)
aspect_ratios = build_aspect_ratios(PYODI_BASE_SIZES, PYODI_RATIOS)

# Parámetros de la RPN ajustados para xView
rpn_pre_nms_top_n_train = 3000
rpn_post_nms_top_n_train = 1500
rpn_pre_nms_top_n_test = 3000
rpn_post_nms_top_n_test = 1500
rpn_batch_size_per_image = 256
box_detections_per_img = 1500
