import json
from pathlib import Path

import pytest

from src.preprocessing.sample_coco_stratified import (
    parse_nested_ratios,
    stratified_sample,
)


def _build_dummy_coco():
    images = [{"id": i, "file_name": f"img_{i}.tif"} for i in range(1, 31)]
    annotations = []
    ann_id = 1

    for img in images:
        img_id = img["id"]
        for cat_id in (1, 2, 3):
            annotations.append(
                {"id": ann_id, "image_id": img_id, "category_id": cat_id}
            )
            ann_id += 1

    categories = [
        {"id": 1, "name": "ClassA"},
        {"id": 2, "name": "ClassB"},
        {"id": 3, "name": "ClassC"},
    ]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def test_parse_nested_ratios_validates_monotonic_growth():
    assert parse_nested_ratios("0.10,0.20,0.30") == [0.10, 0.20, 0.30]

    with pytest.raises(ValueError):
        parse_nested_ratios("0.20,0.10")

    with pytest.raises(ValueError):
        parse_nested_ratios("0.10,0.10")

    with pytest.raises(ValueError):
        parse_nested_ratios("1.00,0.20")


def test_stratified_sample_single_ratio_outputs_files(tmp_path):
    coco = _build_dummy_coco()

    input_json = tmp_path / "input.json"
    output_labeled = tmp_path / "sampled.json"
    output_unlabeled = tmp_path / "unlabeled.json"

    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    stratified_sample(
        coco_path=Path(input_json),
        out_file_labeled=Path(output_labeled),
        out_file_unlabeled=Path(output_unlabeled),
        ratio=0.3,
        seed=107,
        max_iters=0,
    )

    assert output_labeled.exists()
    assert output_unlabeled.exists()

    with open(output_labeled, "r", encoding="utf-8") as f:
        labeled = json.load(f)
    with open(output_unlabeled, "r", encoding="utf-8") as f:
        unlabeled = json.load(f)

    labeled_ids = {img["id"] for img in labeled["images"]}
    unlabeled_ids = {img["id"] for img in unlabeled["images"]}

    assert labeled_ids.isdisjoint(unlabeled_ids)
    assert len(labeled_ids | unlabeled_ids) == len(coco["images"])
    assert len(labeled_ids) == 9  # round(30 * 0.3)


def test_stratified_sample_nested_ratios_are_nested(tmp_path):
    coco = _build_dummy_coco()

    input_json = tmp_path / "input.json"
    output_labeled = tmp_path / "sampled.json"
    output_unlabeled = tmp_path / "unlabeled.json"

    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    stratified_sample(
        coco_path=Path(input_json),
        out_file_labeled=Path(output_labeled),
        out_file_unlabeled=Path(output_unlabeled),
        ratio=0.3,
        seed=107,
        max_iters=0,
        nested_ratios=[0.10, 0.20, 0.30],
    )

    sampled_10 = tmp_path / "sampled_10.json"
    sampled_20 = tmp_path / "sampled_20.json"
    sampled_30 = tmp_path / "sampled_30.json"
    unlabeled_10 = tmp_path / "unlabeled_10.json"
    unlabeled_20 = tmp_path / "unlabeled_20.json"
    unlabeled_30 = tmp_path / "unlabeled_30.json"

    assert sampled_10.exists()
    assert sampled_20.exists()
    assert sampled_30.exists()
    assert unlabeled_10.exists()
    assert unlabeled_20.exists()
    assert unlabeled_30.exists()

    with open(sampled_10, "r", encoding="utf-8") as f:
        s10 = {img["id"] for img in json.load(f)["images"]}
    with open(sampled_20, "r", encoding="utf-8") as f:
        s20 = {img["id"] for img in json.load(f)["images"]}
    with open(sampled_30, "r", encoding="utf-8") as f:
        s30 = {img["id"] for img in json.load(f)["images"]}

    assert s10.issubset(s20)
    assert s20.issubset(s30)
    assert len(s10) == 3  # round(30 * 0.10)
    assert len(s20) == 6  # round(30 * 0.20)
    assert len(s30) == 9  # round(30 * 0.30)


def test_nested_lowest_ratio_matches_single_ratio_with_same_seed(tmp_path):
    coco = _build_dummy_coco()

    input_json = tmp_path / "input.json"
    single_labeled = tmp_path / "single_sampled.json"
    single_unlabeled = tmp_path / "single_unlabeled.json"
    nested_labeled = tmp_path / "nested_sampled.json"
    nested_unlabeled = tmp_path / "nested_unlabeled.json"

    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    stratified_sample(
        coco_path=Path(input_json),
        out_file_labeled=Path(single_labeled),
        out_file_unlabeled=Path(single_unlabeled),
        ratio=0.10,
        seed=107,
        max_iters=50,
    )

    stratified_sample(
        coco_path=Path(input_json),
        out_file_labeled=Path(nested_labeled),
        out_file_unlabeled=Path(nested_unlabeled),
        ratio=0.30,
        seed=107,
        max_iters=50,
        nested_ratios=[0.10, 0.20, 0.30],
    )

    with open(single_labeled, "r", encoding="utf-8") as f:
        single_ids = {img["id"] for img in json.load(f)["images"]}
    with open(tmp_path / "nested_sampled_10.json", "r", encoding="utf-8") as f:
        nested_ids = {img["id"] for img in json.load(f)["images"]}

    assert nested_ids == single_ids
