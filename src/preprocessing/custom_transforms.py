import random

import torch
import torchvision.transforms.functional as F
from torchvision import tv_tensors


class RandomChipping:
    """Apply random square crops while updating XYXY bounding boxes.

    This transform receives and returns ``(image, boxes, *additional_boxes)``.
    ``boxes`` and each tensor in ``additional_boxes`` must be a
    ``tv_tensors.BoundingBoxes`` instance in XYXY format. The same crop is
    applied to all box sets; each is independently clipped and filtered by
    ``min_visibility``.
    """

    def __init__(self, crop_sizes=None, min_visibility=0.0):
        self.crop_sizes = crop_sizes or [300, 400, 500, 600, 700]
        self.min_visibility = float(min_visibility)

    def _crop_boxes(self, boxes, top: int, left: int, size: int):
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes_tensor.numel() == 0:
            return tv_tensors.BoundingBoxes(
                boxes_tensor.reshape(0, 4),
                format="XYXY",
                canvas_size=(size, size),
            )

        original_widths = (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp(min=0)
        original_heights = (boxes_tensor[:, 3] - boxes_tensor[:, 1]).clamp(min=0)
        original_areas = original_widths * original_heights

        new_boxes = boxes_tensor.clone()
        new_boxes[:, 0] -= left
        new_boxes[:, 1] -= top
        new_boxes[:, 2] -= left
        new_boxes[:, 3] -= top

        new_boxes[:, 0] = new_boxes[:, 0].clamp(min=0, max=size)
        new_boxes[:, 1] = new_boxes[:, 1].clamp(min=0, max=size)
        new_boxes[:, 2] = new_boxes[:, 2].clamp(min=0, max=size)
        new_boxes[:, 3] = new_boxes[:, 3].clamp(min=0, max=size)

        clipped_widths = (new_boxes[:, 2] - new_boxes[:, 0]).clamp(min=0)
        clipped_heights = (new_boxes[:, 3] - new_boxes[:, 1]).clamp(min=0)
        clipped_areas = clipped_widths * clipped_heights

        keep = (clipped_widths > 0) & (clipped_heights > 0)
        if self.min_visibility > 0:
            visibility = torch.zeros_like(clipped_areas)
            positive_original = original_areas > 0
            visibility[positive_original] = (
                clipped_areas[positive_original] / original_areas[positive_original]
            )
            keep = keep & (visibility >= self.min_visibility)

        new_boxes = new_boxes[keep]
        return tv_tensors.BoundingBoxes(
            new_boxes,
            format="XYXY",
            canvas_size=(size, size),
        )

    def __call__(self, image, boxes, *additional_boxes):
        if isinstance(image, torch.Tensor):
            h, w = int(image.shape[-2]), int(image.shape[-1])
        else:
            w, h = image.size

        valid_sizes = [size for size in self.crop_sizes if size <= h and size <= w]
        if not valid_sizes:
            if additional_boxes:
                return (image, boxes, *additional_boxes)
            return image, boxes

        size = int(random.choice(valid_sizes))
        if size == h and size == w:
            if additional_boxes:
                return (image, boxes, *additional_boxes)
            return image, boxes

        top = random.randint(0, h - size) if size < h else 0
        left = random.randint(0, w - size) if size < w else 0

        image = F.crop(image, top=top, left=left, height=size, width=size)

        cropped_boxes = self._crop_boxes(boxes, top, left, size)
        if additional_boxes:
            cropped_additional = tuple(
                self._crop_boxes(b, top, left, size) for b in additional_boxes
            )
            return (image, cropped_boxes, *cropped_additional)
        return image, cropped_boxes
