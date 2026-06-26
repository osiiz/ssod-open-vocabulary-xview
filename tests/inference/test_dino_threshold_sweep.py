import pytest

from src.inference.dino_threshold_sweep import (
    build_eval_subset_coco,
    materialize_eval_ann_file,
    parse_score_text_pairs,
)


def test_parse_score_text_pairs_supports_explicit_pairs_and_single_values():
    pairs = parse_score_text_pairs("0.02:0.03,0.05")
    assert pairs == [(0.02, 0.03), (0.05, 0.05)]


def test_parse_score_text_pairs_rejects_empty_values():
    with pytest.raises(ValueError):
        parse_score_text_pairs("   ")


def test_parse_score_text_pairs_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        parse_score_text_pairs("1.2:0.1")


def test_build_eval_subset_coco_filters_images_and_annotations():
    coco = {
        "images": [
            {"id": 1, "file_name": "a.tif"},
            {"id": 2, "file_name": "b.tif"},
            {"id": 3, "file_name": "c.tif"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 2, "category_id": 2},
            {"id": 3, "image_id": 3, "category_id": 3},
        ],
        "categories": [{"id": 1, "name": "A"}],
    }

    subset, sampled_images, total_images = build_eval_subset_coco(coco, sample_images=2)

    assert sampled_images == 2
    assert total_images == 3
    assert [image["id"] for image in subset["images"]] == [1, 2]
    assert {annotation["image_id"] for annotation in subset["annotations"]} == {1, 2}
    assert subset["categories"] == coco["categories"]


def test_materialize_eval_ann_file_writes_subset_when_sample_is_smaller(tmp_path):
    ann_file = tmp_path / "ann.json"
    ann_file.write_text(
        """
{
  "images": [
    {"id": 10, "file_name": "a.tif"},
    {"id": 20, "file_name": "b.tif"}
  ],
  "annotations": [
    {"id": 1, "image_id": 10, "category_id": 1},
    {"id": 2, "image_id": 20, "category_id": 1}
  ],
  "categories": [{"id": 1, "name": "obj"}]
}
""".strip(),
        encoding="utf-8",
    )

    eval_ann_file, eval_images, total_images = materialize_eval_ann_file(
        ann_file=ann_file,
        output_root=tmp_path,
        sample_images=1,
    )

    assert eval_ann_file != ann_file
    assert eval_ann_file.exists()
    assert eval_images == 1
    assert total_images == 2

    subset_payload = eval_ann_file.read_text(encoding="utf-8")
    assert '"id": 10' in subset_payload
    assert '"id": 20' not in subset_payload


def test_materialize_eval_ann_file_uses_full_annotations_when_sample_covers_all(
    tmp_path,
):
    ann_file = tmp_path / "ann.json"
    ann_file.write_text(
        """
{
  "images": [{"id": 1, "file_name": "a.tif"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
  "categories": [{"id": 1, "name": "obj"}]
}
""".strip(),
        encoding="utf-8",
    )

    eval_ann_file, eval_images, total_images = materialize_eval_ann_file(
        ann_file=ann_file,
        output_root=tmp_path,
        sample_images=5,
    )

    assert eval_ann_file == ann_file
    assert eval_images == 1
    assert total_images == 1
