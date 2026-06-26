import torch

from src.utils.exclusion_zones import _exclusion_mask


def test_exclusion_mask_none_zones_returns_none():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert _exclusion_mask(boxes, None, 0.5) is None


def test_exclusion_mask_empty_zones_returns_none():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    zones = torch.zeros((0, 4))
    assert _exclusion_mask(boxes, zones, 0.5) is None


def test_exclusion_mask_flags_box_inside_zone():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # idéntica á zona -> IoU 1.0
            [100.0, 100.0, 110.0, 110.0],  # lonxe -> False
        ]
    )
    zones = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    mask = _exclusion_mask(boxes, zones, 0.5)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, False]


def test_exclusion_mask_threshold_inclusive_at_exact_match():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    zones = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    # IoU = 1.0 >= 1.0 -> marcada
    assert _exclusion_mask(boxes, zones, 1.0).tolist() == [True]


def test_exclusion_mask_below_threshold_not_flagged():
    # A=[0,0,10,10], zona=[5,0,15,10]: inter 50, union 150 -> IoU 1/3 < 0.5
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    zones = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
    assert _exclusion_mask(boxes, zones, 0.5).tolist() == [False]


def test_exclusion_mask_any_zone_triggers():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    zones = torch.tensor(
        [
            [100.0, 100.0, 110.0, 110.0],  # non solapa
            [0.0, 0.0, 10.0, 10.0],  # solapa total
        ]
    )
    assert _exclusion_mask(boxes, zones, 0.5).tolist() == [True]


def test_exclusion_mask_shape_matches_candidates():
    boxes = torch.tensor([[float(i), float(i), float(i + 5), float(i + 5)] for i in range(7)])
    zones = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
    mask = _exclusion_mask(boxes, zones, 0.5)
    assert mask.shape == (7,)
