import pytest

from src.ssod.build_pe_dataset import (
    EXCLUSION_CATEGORY_ID,
    _dedup_cross_source,
    _detections_to_annotations,
    _filter_detections,
    _iou_xywh,
    build_merged_coco,
)


# --------------------------------------------------------------------------
# _iou_xywh
# --------------------------------------------------------------------------
def test_iou_xywh_empty_kept_returns_zero():
    assert _iou_xywh([0, 0, 10, 10], []) == 0.0


def test_iou_xywh_identical_box_is_one():
    assert _iou_xywh([0, 0, 10, 10], [[0, 0, 10, 10]]) == 1.0


def test_iou_xywh_disjoint_is_zero():
    assert _iou_xywh([0, 0, 10, 10], [[100, 100, 10, 10]]) == 0.0


def test_iou_xywh_half_horizontal_overlap():
    # A=[0,0,10,10], B=[5,0,10,10] -> inter 5*10=50, union 100+100-50=150
    assert _iou_xywh([0, 0, 10, 10], [[5, 0, 10, 10]]) == pytest.approx(1 / 3)


def test_iou_xywh_returns_max_over_kept_boxes():
    val = _iou_xywh([0, 0, 10, 10], [[100, 100, 10, 10], [5, 0, 10, 10]])
    assert val == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# _filter_detections
# --------------------------------------------------------------------------
def test_filter_detections_none_threshold_keeps_all():
    dets = [{"score": 0.1}, {"score": 0.9}]
    assert _filter_detections(dets, None) == dets


def test_filter_detections_threshold_is_inclusive():
    dets = [{"score": 0.69}, {"score": 0.70}, {"score": 0.71}]
    out = _filter_detections(dets, 0.70)
    assert [d["score"] for d in out] == [0.70, 0.71]


# --------------------------------------------------------------------------
# _dedup_cross_source
# --------------------------------------------------------------------------
def _det(img, cat, bbox, score, source):
    return {
        "image_id": img,
        "category_id": cat,
        "bbox": bbox,
        "score": score,
        "_source_idx": source,
    }


def test_dedup_keeps_higher_priority_source_over_score():
    # Mesma categoría, caixas solapadas; fonte 0 (menor score) ten prioridade.
    dets = {
        1: [
            _det(1, 5, [0, 0, 10, 10], 0.3, 0),
            _det(1, 5, [1, 0, 10, 10], 0.9, 1),
        ]
    }
    out, dropped = _dedup_cross_source(dets, iou_thresh=0.5, priority=[0, 1])
    assert dropped == 1
    assert len(out[1]) == 1
    assert out[1][0]["_source_idx"] == 0


def test_dedup_different_categories_both_kept():
    dets = {
        1: [
            _det(1, 5, [0, 0, 10, 10], 0.3, 0),
            _det(1, 7, [0, 0, 10, 10], 0.9, 1),
        ]
    }
    out, dropped = _dedup_cross_source(dets, iou_thresh=0.5, priority=[0, 1])
    assert dropped == 0
    assert len(out[1]) == 2


def test_dedup_same_source_same_cat_keeps_higher_score():
    dets = {
        1: [
            _det(1, 5, [0, 0, 10, 10], 0.3, 0),
            _det(1, 5, [1, 0, 10, 10], 0.9, 0),
        ]
    }
    out, dropped = _dedup_cross_source(dets, iou_thresh=0.5, priority=[0])
    assert dropped == 1
    assert len(out[1]) == 1
    assert out[1][0]["score"] == 0.9


def test_dedup_non_overlapping_same_cat_both_kept():
    dets = {
        1: [
            _det(1, 5, [0, 0, 10, 10], 0.3, 0),
            _det(1, 5, [100, 100, 10, 10], 0.9, 1),
        ]
    }
    out, dropped = _dedup_cross_source(dets, iou_thresh=0.5, priority=[0, 1])
    assert dropped == 0
    assert len(out[1]) == 2


# --------------------------------------------------------------------------
# _detections_to_annotations
# --------------------------------------------------------------------------
def test_detections_to_annotations_ids_area_and_iscrowd():
    dets = [
        {"image_id": 3, "category_id": 2, "bbox": [1, 2, 4, 5]},
        {"image_id": 4, "category_id": 1, "bbox": [0, 0, 10, 10]},
    ]
    anns, next_id = _detections_to_annotations(dets, ann_id_start=100)
    assert next_id == 102
    assert [a["id"] for a in anns] == [100, 101]
    assert anns[0]["area"] == 20.0
    assert anns[0]["iscrowd"] == 0
    assert anns[0]["image_id"] == 3
    assert anns[0]["category_id"] == 2


# --------------------------------------------------------------------------
# build_merged_coco
# --------------------------------------------------------------------------
def _coco(images, annotations, categories):
    return {"images": images, "annotations": annotations, "categories": categories}


def _labeled():
    return _coco(
        images=[{"id": 1, "file_name": "L1.tif"}, {"id": 2, "file_name": "L2.tif"}],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 5, 5],
                "area": 25.0,
                "iscrowd": 0,
            }
        ],
        categories=[{"id": 1, "name": "A"}],
    )


def _unlabeled(n=2):
    return _coco(
        images=[{"id": i, "file_name": f"U{i}.tif"} for i in range(1, n + 1)],
        annotations=[],
        categories=[{"id": 1, "name": "A"}],
    )


def test_build_merged_coco_remaps_ids_and_prefixes_paths():
    pe = [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3], "score": 0.9}]
    merged, stats = build_merged_coco(
        labeled_coco=_labeled(),
        unlabeled_coco=_unlabeled(2),
        pe_sources=[pe],
        thresholds=[0.5],
        labeled_subfolder="train_sampled",
        unlabeled_subfolder="train_unlabeled",
        drop_empty_unlabeled=True,
    )

    labeled_imgs = [im for im in merged["images"] if im["file_name"].startswith("train_sampled/")]
    assert len(labeled_imgs) == 2

    # Só a imaxe non etiquetada 1 ten PE; image_id remapeado = 1 + offset(2) = 3.
    unlab_imgs = [im for im in merged["images"] if im["file_name"].startswith("train_unlabeled/")]
    assert len(unlab_imgs) == 1
    assert unlab_imgs[0]["id"] == 3
    assert unlab_imgs[0]["file_name"] == "train_unlabeled/U1.tif"

    pe_anns = [a for a in merged["annotations"] if a["image_id"] == 3]
    assert len(pe_anns) == 1
    assert pe_anns[0]["area"] == 9.0
    assert stats["unlabeled_images_with_pe"] == 1
    assert stats["pe_annotations_total"] == 1
    assert stats["labeled_images"] == 2


def test_build_merged_coco_applies_threshold_per_source():
    pe = [
        {"image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3], "score": 0.4},
        {"image_id": 1, "category_id": 1, "bbox": [20, 20, 3, 3], "score": 0.8},
    ]
    merged, stats = build_merged_coco(
        _labeled(), _unlabeled(1), [pe], [0.5], "ls", "us", True
    )
    assert stats["pe_per_source"]["source_0"] == 1
    assert stats["pe_annotations_total"] == 1


def test_build_merged_coco_drop_empty_false_keeps_all_unlabeled():
    pe = [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3], "score": 0.9}]
    merged, _ = build_merged_coco(
        _labeled(), _unlabeled(2), [pe], [0.5], "ls", "us", drop_empty_unlabeled=False
    )
    unlab_imgs = [im for im in merged["images"] if im["file_name"].startswith("us/")]
    assert len(unlab_imgs) == 2


def test_build_merged_coco_exclusion_only_on_included_images():
    pe = [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3], "score": 0.9}]
    excl = [
        {"image_id": 1, "bbox": [10, 10, 4, 4]},  # imaxe incluída (ten PE)
        {"image_id": 2, "bbox": [10, 10, 4, 4]},  # imaxe NON incluída -> ignorada
    ]
    merged, stats = build_merged_coco(
        _labeled(),
        _unlabeled(2),
        [pe],
        [0.5],
        "ls",
        "us",
        True,
        exclusion_sources=[excl],
    )
    excl_anns = [a for a in merged["annotations"] if a["category_id"] == EXCLUSION_CATEGORY_ID]
    assert len(excl_anns) == 1
    assert excl_anns[0]["image_id"] == 3
    assert stats["exclusion_annotations_total"] == 1


def test_build_merged_coco_cross_source_dedup():
    src0 = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}]
    src1 = [{"image_id": 1, "category_id": 1, "bbox": [1, 0, 10, 10], "score": 0.99}]
    merged, stats = build_merged_coco(
        _labeled(),
        _unlabeled(1),
        [src0, src1],
        [None, None],
        "ls",
        "us",
        True,
        dedup_iou_thresh=0.5,
        dedup_priority=[0, 1],
    )
    assert stats["pe_annotations_total"] == 1
    assert stats["pe_annotations_dedup_dropped"] == 1
