import json
from pathlib import Path

from src.preprocessing.sample_coco_stratified_seeds import (
    parse_candidate_seeds,
    stratified_sample,
)


def _build_dummy_coco():
    images = [{"id": i, "file_name": f"img_{i}.tif"} for i in range(1, 31)]
    annotations = []
    ann_id = 1

    for img in images:
        img_id = img["id"]
        annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1})
        ann_id += 1

        if img_id % 2 == 0:
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 2})
            ann_id += 1

        if img_id % 3 == 0:
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 3})
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


def test_parse_candidate_seeds_supports_csv_and_sequence():
    assert parse_candidate_seeds(100, 3, "107,108,107") == [107, 108]
    assert parse_candidate_seeds(100, 3, None) == [100, 101, 102]


def test_candidate_sampling_uses_strict_image_count(tmp_path):
    coco = _build_dummy_coco()

    input_json = tmp_path / "input.json"
    output_labeled = tmp_path / "sampled.json"
    output_unlabeled = tmp_path / "unlabeled.json"
    candidates_dir = tmp_path / "candidates"
    selection_report = tmp_path / "selection_report.json"

    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    candidate_seeds = [101, 102, 103, 104]

    stratified_sample(
        coco_path=Path(input_json),
        out_file_labeled=Path(output_labeled),
        out_file_unlabeled=Path(output_unlabeled),
        ratio=0.10,
        seed=101,
        max_iters=250,
        num_candidates=4,
        candidate_seeds=",".join(str(s) for s in candidate_seeds),
        selection_report_path=str(selection_report),
        candidates_dir=str(candidates_dir),
    )

    expected_labeled = int(round(len(coco["images"]) * 0.10))

    with open(output_labeled, "r", encoding="utf-8") as f:
        selected_labeled = json.load(f)
    with open(output_unlabeled, "r", encoding="utf-8") as f:
        selected_unlabeled = json.load(f)

    assert len(selected_labeled["images"]) == expected_labeled
    assert len(selected_unlabeled["images"]) == len(coco["images"]) - expected_labeled

    for seed in candidate_seeds:
        labeled_path = candidates_dir / f"sampled_seed{seed}.json"
        unlabeled_path = candidates_dir / f"unlabeled_seed{seed}.json"

        assert labeled_path.exists()
        assert unlabeled_path.exists()

        with open(labeled_path, "r", encoding="utf-8") as f:
            labeled_candidate = json.load(f)
        with open(unlabeled_path, "r", encoding="utf-8") as f:
            unlabeled_candidate = json.load(f)

        assert len(labeled_candidate["images"]) == expected_labeled
        assert (
            len(unlabeled_candidate["images"]) == len(coco["images"]) - expected_labeled
        )

    assert selection_report.exists()

    with open(selection_report, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["selected_seed"] in candidate_seeds
    assert len(report["candidates"]) == len(candidate_seeds)
    for candidate in report["candidates"]:
        assert candidate["seed"] in candidate_seeds
        assert candidate["sampled_image_ratio"] == expected_labeled / len(
            coco["images"]
        )
