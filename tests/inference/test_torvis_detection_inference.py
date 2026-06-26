from pathlib import Path
import json

from pycocotools.coco import COCO

output_folder = "./results/inference_test/"
detections_filename = "detection_results.json"
metrics_filename = "metrics.json"
metrics_per_class_filename = "metrics_per_class.json"
test_ann_file = "./results/preprocess/tile_images/test/COCO_annotations.json"


def test_torvis_detection_inference():
    """Valida a saída de avaliación da cota superior.

    As métricas COCO (`metrics.json`, `metrics_per_class.json`) van incluídas no
    repositorio, polo que sempre se validan. As deteccións completas
    (`detection_results.json`) e as anotacións de test só están dispoñibles tras
    materializar o almacén DVC (`dvc pull`); cando o están, valídase tamén o seu
    formato COCO.
    """
    root_dir = Path(__file__).parent.parent.parent
    metrics_json = root_dir / output_folder / metrics_filename
    metrics_per_class_json = root_dir / output_folder / metrics_per_class_filename

    # --- Métricas: sempre presentes no repositorio ---
    assert metrics_json.exists(), "Falta results/inference_test/metrics.json"
    assert (
        metrics_per_class_json.exists()
    ), "Falta results/inference_test/metrics_per_class.json"

    with open(metrics_json, "r") as f:
        metrics = json.load(f)
    assert isinstance(metrics, dict) and metrics, "metrics.json debe ser un dict non baleiro"
    assert "AP50" in metrics, "metrics.json debe conter a clave AP50"
    assert all(
        isinstance(v, (int, float)) for v in metrics.values()
    ), "todas as métricas deben ser numéricas"

    with open(metrics_per_class_json, "r") as f:
        class_metrics = json.load(f)
    assert (
        isinstance(class_metrics, dict) and class_metrics
    ), "metrics_per_class.json debe ser un dict non baleiro"

    # --- Deteccións + anotacións de test: só se están materializadas (dvc pull) ---
    results_json = root_dir / output_folder / detections_filename
    ann_file = root_dir / test_ann_file
    if results_json.exists() and ann_file.exists():
        coco_gt = COCO(str(ann_file))
        valid_img_ids = set(coco_gt.getImgIds())

        with open(results_json, "r") as f:
            results = json.load(f)
        assert isinstance(results, list), "detection_results.json debe conter unha lista"
        assert len(results) > 0, "detection_results.json non contén deteccións"

        for det in results:
            assert set(["image_id", "category_id", "bbox", "score"]).issubset(det.keys())
            assert det["image_id"] in valid_img_ids
            assert isinstance(det["bbox"], list) and len(det["bbox"]) == 4
            assert det["bbox"][2] >= 0 and det["bbox"][3] >= 0
            assert 0.0 <= float(det["score"]) <= 1.0

        assert (
            coco_gt.loadRes(str(results_json)) is not None
        ), "pycocotools non puido cargar detection_results.json"
