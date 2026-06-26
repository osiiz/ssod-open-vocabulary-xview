import pytest
import torch

from src.utils.coco_utils import build_coco_max_dets, torch2coco_results


# --------------------------------------------------------------------------
# build_coco_max_dets
# --------------------------------------------------------------------------
def test_build_coco_max_dets_default_cap():
    assert build_coco_max_dets(1500) == [100, 500, 1500]


def test_build_coco_max_dets_clamps_monotonic_for_small_caps():
    assert build_coco_max_dets(300) == [100, 300, 300]
    assert build_coco_max_dets(100) == [100, 100, 100]
    assert build_coco_max_dets(50) == [50, 50, 50]


def test_build_coco_max_dets_floors_at_one():
    assert build_coco_max_dets(0) == [1, 1, 1]
    assert build_coco_max_dets(-5) == [1, 1, 1]


# --------------------------------------------------------------------------
# torch2coco_results
# --------------------------------------------------------------------------
def test_torch2coco_results_thresholds_and_converts_to_xywh():
    results = {
        "scores": torch.tensor([0.9, 0.3, 0.7]),
        "boxes": torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 2.0, 2.0], [5.0, 5.0, 15.0, 20.0]]
        ),
        "labels": torch.tensor([1, 2, 3]),
    }
    out = torch2coco_results(results, coco_img_id=7, categories=None, score_thresh=0.5)

    # O score 0.3 queda por debaixo do limiar (estrito) e descártase.
    assert len(out) == 2
    assert out[0]["image_id"] == 7
    assert out[0]["category_id"] == 1
    # xyxy [0,0,10,10] -> xywh [0,0,10,10]
    assert out[0]["bbox"] == [0.0, 0.0, 10.0, 10.0]
    # xyxy [5,5,15,20] -> xywh [5,5,10,15]
    assert out[1]["category_id"] == 3
    assert out[1]["bbox"] == [5.0, 5.0, 10.0, 15.0]
    assert out[0]["score"] == pytest.approx(0.9, abs=1e-4)


def test_torch2coco_results_empty_when_all_below_threshold():
    results = {
        "scores": torch.tensor([0.1, 0.2]),
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]),
        "labels": torch.tensor([1, 2]),
    }
    out = torch2coco_results(results, coco_img_id=1, categories=None, score_thresh=0.5)
    assert out == []
