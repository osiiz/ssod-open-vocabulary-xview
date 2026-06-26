from src.preprocessing.filter_coco_by_image_ids import (
    extract_image_ids,
    filter_coco_by_image_ids,
)


def test_extract_image_ids_reads_ids_from_coco_images():
    selection_coco = {
        "images": [
            {"id": 10, "file_name": "a.tif"},
            {"id": 20, "file_name": "b.tif"},
        ]
    }

    ids = extract_image_ids(selection_coco)
    assert ids == {10, 20}


def test_filter_coco_keeps_selected_images_and_annotations():
    source_coco = {
        "images": [
            {"id": 1, "file_name": "img1.tif"},
            {"id": 2, "file_name": "img2.tif"},
            {"id": 3, "file_name": "img3.tif"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 2, "category_id": 2},
            {"id": 3, "image_id": 3, "category_id": 2},
        ],
        "categories": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
    }

    filtered, missing = filter_coco_by_image_ids(
        source_coco=source_coco,
        selected_image_ids={2, 3},
        include_annotations=True,
    )

    assert [image["id"] for image in filtered["images"]] == [2, 3]
    assert {ann["image_id"] for ann in filtered["annotations"]} == {2, 3}
    assert missing == []


def test_filter_coco_strip_annotations_and_reports_missing_ids():
    source_coco = {
        "images": [{"id": 1, "file_name": "img1.tif"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
        "categories": [{"id": 1, "name": "A"}],
    }

    filtered, missing = filter_coco_by_image_ids(
        source_coco=source_coco,
        selected_image_ids={1, 9},
        include_annotations=False,
    )

    assert [image["id"] for image in filtered["images"]] == [1]
    assert filtered["annotations"] == []
    assert missing == [9]
