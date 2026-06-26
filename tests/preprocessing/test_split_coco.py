import pytest
import json
from src.preprocessing.split_coco import (
    calculate_mean_deviation,
    perform_optimized_split,
    stratified_split,
)


@pytest.fixture
def sample_coco_data():
    """Genera un dataset COCO pequeño controlado para los tests."""
    images = {
        i: {"id": i, "file_name": f"img_{i}.tif", "width": 100, "height": 100}
        for i in range(1, 11)
    }

    # Distribuimos 3 clases entre las imágenes (la imagen 10 queda vacía para probar casos sin anotaciones)
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1},
        {"id": 2, "image_id": 2, "category_id": 1},
        {"id": 3, "image_id": 3, "category_id": 2},
        {"id": 4, "image_id": 4, "category_id": 2},
        {"id": 5, "image_id": 5, "category_id": 1},
        {"id": 6, "image_id": 6, "category_id": 3},
        {"id": 7, "image_id": 7, "category_id": 3},
        {"id": 8, "image_id": 8, "category_id": 1},
        {"id": 9, "image_id": 9, "category_id": 2},
    ]

    categories = [
        {"id": 1, "name": "Building"},
        {"id": 2, "name": "Car"},
        {"id": 3, "name": "Tree"},
    ]

    return images, annotations, categories


def test_calculate_mean_deviation():
    """Verifica que la desviación media se calcula correctamente."""
    ratios = (0.7, 0.1, 0.2)
    global_cat_counts = {1: 100, 2: 100}

    # Escenario 1: Reparto perfecto (Debe dar desviación 0)
    current_counts_perfect = {
        "train": {1: 70, 2: 70},
        "val": {1: 10, 2: 10},
        "test": {1: 20, 2: 20},
    }
    dev_perfect = calculate_mean_deviation(
        current_counts_perfect, global_cat_counts, ratios
    )
    assert dev_perfect == 0.0

    # Escenario 2: Desvío conocido (Clase 1 bien, Clase 2 desplazada un 10%)
    current_counts_bad = {
        "train": {1: 70, 2: 80},  # +10% en Clase 2
        "val": {1: 10, 2: 0},  # -10% en Clase 2
        "test": {1: 20, 2: 20},  # Perfecto
    }
    dev_bad = calculate_mean_deviation(current_counts_bad, global_cat_counts, ratios)

    # Cálculo manual: (0 + 0 + 0 + 0.10 + 0.10 + 0) / 6 ≈ 0.0333...
    assert pytest.approx(dev_bad, 0.01) == 0.0333


def test_perform_optimized_split_no_leakage(sample_coco_data):
    """Verifica que todas las imágenes se asignan sin repetirse."""
    images, annotations, _ = sample_coco_data
    ratios = (0.7, 0.1, 0.2)

    splits_img_ids = perform_optimized_split(
        images, annotations, ratios, seed=12345, max_iters=50
    )

    # 1. Comprobar que existen las tres cestas
    assert set(splits_img_ids.keys()) == {"train", "val", "test"}

    # 2. Comprobar que todas las imágenes están asignadas a exactamente una cesta
    total_assigned = sum(len(ids) for ids in splits_img_ids.values())
    assert total_assigned == len(images)

    # 3. Comprobar aislamientos estricto mediante intersección de conjuntos
    train_set = set(splits_img_ids["train"])
    val_set = set(splits_img_ids["val"])
    test_set = set(splits_img_ids["test"])

    assert train_set.isdisjoint(val_set), "Fuga de datos detectada entre Train y Val"
    assert train_set.isdisjoint(test_set), "Fuga de datos detectada entre Train y Test"
    assert val_set.isdisjoint(test_set), "Fuga de datos detectada entre Val y Test"


def test_stratified_split_integration_file_creation(tmp_path, sample_coco_data):
    """Simula el pipeline completo escribiendo y leyendo de un directorio temporal."""
    images, annotations, categories = sample_coco_data

    # Preparamos el entorno simulado
    dummy_coco = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": categories,
    }
    input_json = tmp_path / "dummy_coco.json"
    out_dir = tmp_path / "splits"

    with open(input_json, "w") as f:
        json.dump(dummy_coco, f)

    # Ejecutamos el script
    stratified_split(
        input_json, out_dir, ratios=(0.7, 0.1, 0.2), seed=12345, max_iters=50
    )

    # Verificamos que los 3 archivos físicos existen
    assert (out_dir / "xview_train.json").exists()
    assert (out_dir / "xview_val.json").exists()
    assert (out_dir / "xview_test.json").exists()

    # Verificamos que la estructura COCO es íntegra y arrastra las categorías
    with open(out_dir / "xview_train.json", "r") as f:
        train_data = json.load(f)

    assert "images" in train_data
    assert "annotations" in train_data
    assert "categories" in train_data
    assert (
        len(train_data["categories"]) == 3
    ), "El split debe arrastrar el catálogo entero de clases"
