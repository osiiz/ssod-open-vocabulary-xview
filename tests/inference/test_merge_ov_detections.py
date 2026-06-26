from src.inference.merge_ov_detections import (
    build_filename_to_id,
    build_id_to_filename,
    merge_prompt_context,
    remap_detections,
)

IMAGES = [{"id": 1, "file_name": "a.tif"}, {"id": 2, "file_name": "b.tif"}]


def test_build_id_to_filename():
    assert build_id_to_filename(IMAGES) == {1: "a.tif", 2: "b.tif"}


def test_build_filename_to_id():
    assert build_filename_to_id(IMAGES) == {"a.tif": 1, "b.tif": 2}


def test_remap_detections_translates_ids_via_filename():
    old_id_to_fn = {10: "a.tif", 20: "b.tif"}
    new_fn_to_id = {"a.tif": 1, "b.tif": 2}
    dets = [
        {"image_id": 10, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5},
        {"image_id": 20, "category_id": 2, "bbox": [1, 1, 2, 2], "score": 0.7},
    ]
    out = remap_detections(dets, old_id_to_fn, new_fn_to_id)
    assert [d["image_id"] for d in out] == [1, 2]
    # O resto de campos preservanse
    assert out[0]["category_id"] == 1
    assert out[0]["score"] == 0.5


def test_remap_detections_skips_id_absent_in_old_map():
    out = remap_detections(
        [{"image_id": 10}, {"image_id": 99}],
        old_id_to_fn={10: "a.tif"},
        new_fn_to_id={"a.tif": 1},
    )
    assert [d["image_id"] for d in out] == [1]


def test_remap_detections_skips_filename_absent_in_new_map():
    out = remap_detections(
        [{"image_id": 10}],
        old_id_to_fn={10: "a.tif"},
        new_fn_to_id={"b.tif": 2},
    )
    assert out == []


def test_merge_prompt_context_sums_counts_and_merges_labels():
    base = {
        "total_predictions": 100,
        "mapped_predictions": 80,
        "unmapped_predictions": 20,
        "failed_images": 1,
        "top_unmapped_labels": [
            {"label": "car", "count": 5},
            {"label": "boat", "count": 3},
        ],
    }
    delta = {
        "total_predictions": 10,
        "mapped_predictions": 7,
        "unmapped_predictions": 3,
        "failed_images": 0,
        "top_unmapped_labels": [
            {"label": "car", "count": 2},
            {"label": "tree", "count": 4},
        ],
    }
    merged = merge_prompt_context(base, delta)
    assert merged["total_predictions"] == 110
    assert merged["mapped_predictions"] == 87
    assert merged["unmapped_predictions"] == 23
    assert merged["failed_images"] == 1

    labels = {e["label"]: e["count"] for e in merged["top_unmapped_labels"]}
    assert labels == {"car": 7, "boat": 3, "tree": 4}
    counts = [e["count"] for e in merged["top_unmapped_labels"]]
    assert counts == sorted(counts, reverse=True)
