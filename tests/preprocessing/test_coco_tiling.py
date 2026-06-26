import pytest
from src.preprocessing.coco_tiling import adjust_boxes_to_tile, create_coco_annotations


@pytest.mark.parametrize(
    "coco_ann, tile_limits, expected",
    [
        # Formato coco_ann: {"category_id": id, "bbox": [x, y, w, h]}
        # Formato tile_limits: [left, top, width, height] (Posición y tamaño del recorte)
        # Formato expected: [[cat_id, x, y, width, height]] (Formato COCO relativo al recorte)
        # Caso 1: Caja perfectamente contenida dentro del tile
        (
            {"category_id": 1, "bbox": [150, 150, 50, 50]},
            [100, 100, 800, 800],
            [[1, 50.0, 50.0, 50.0, 50.0]],
        ),
        # Caso 2: Caja completamente fuera del tile (no debe devolver nada)
        ({"category_id": 1, "bbox": [0, 0, 50, 50]}, [100, 100, 800, 800], []),
        # Caso 3: Caja cortada por el borde superior izquierdo (debe clipear a 0,0)
        (
            {
                "category_id": 1,
                "bbox": [80, 80, 70, 70],
            },  # x=80, y=80, w=70, h=70 -> xmax=150, ymax=150
            [100, 100, 800, 800],
            [[1, 0.0, 0.0, 50.0, 50.0]],
        ),
        # Caso 4: Caja en borde inferior derecho dentro de los límites del tile actual.
        # Con tile_width/tile_height=800, el tile va de X:100 a 900 e Y:100 a 900.
        (
            {
                "category_id": 1,
                "bbox": [700, 700, 100, 100],
            },  # x=700, y=700, w=100, h=100 -> xmax=800, ymax=800
            [100, 100, 800, 800],
            [[1, 600.0, 600.0, 100.0, 100.0]],
        ),
        # Caso 5: Caja que atraviesa todo el tile de lado a lado (ej. una carretera larga)
        (
            {"category_id": 1, "bbox": [50, 150, 750, 50]},  # x=50, y=150, w=750, h=50
            [100, 100, 800, 800],
            [[1, 0.0, 50.0, 700.0, 50.0]],
        ),
    ],
)
def test_adjust_boxes_to_tile(coco_ann, tile_limits, expected):
    """Valida la lógica de intersección y conversión a formato COCO de las bounding boxes."""
    # Le pasamos una lista con una sola anotación a la función
    result, _ = adjust_boxes_to_tile([coco_ann], tile_limits, min_visibility=0.0)

    # Comprobamos que el recorte coincide exactamente con la matemática esperada
    assert result == expected


def test_adjust_boxes_to_tile_multiple_annotations():
    """Verifica que múltiples anotaciones se procesan correctamente."""
    coco_anns = [
        {"category_id": 1, "bbox": [150, 150, 50, 50]},  # Dentro del tile
        {"category_id": 2, "bbox": [0, 0, 50, 50]},  # Fuera del tile
        {"category_id": 3, "bbox": [200, 200, 30, 30]},  # Dentro del tile
    ]
    tile_limits = [100, 100, 800, 800]

    result, _ = adjust_boxes_to_tile(coco_anns, tile_limits, min_visibility=0.0)

    # Solo 2 anotaciones deben estar dentro del tile
    assert len(result) == 2
    assert result[0] == [1, 50.0, 50.0, 50.0, 50.0]
    assert result[1] == [3, 100.0, 100.0, 30.0, 30.0]


def test_adjust_boxes_to_tile_empty_list():
    """Verifica que una lista vacía devuelve una lista vacía."""
    result, _ = adjust_boxes_to_tile([], [100, 100, 800, 800], min_visibility=0.0)
    assert result == []


def test_adjust_boxes_to_tile_box_touching_edge():
    """Verifica el comportamiento cuando una caja toca exactamente el borde del tile."""
    # Caja que termina exactamente donde empieza el tile (no debe incluirse)
    coco_ann = {"category_id": 1, "bbox": [0, 0, 100, 100]}  # termina en x=100, y=100
    tile_limits = [100, 100, 800, 800]  # empieza en x=100, y=100

    result, _ = adjust_boxes_to_tile([coco_ann], tile_limits, min_visibility=0.0)
    # La caja toca el borde pero no tiene área dentro del tile
    assert result == []


def test_adjust_boxes_to_tile_preserves_category_id():
    """Verifica que el category_id se preserva correctamente."""
    coco_anns = [
        {"category_id": 42, "bbox": [150, 150, 50, 50]},
        {"category_id": 99, "bbox": [200, 200, 30, 30]},
    ]
    tile_limits = [100, 100, 800, 800]

    result, _ = adjust_boxes_to_tile(coco_anns, tile_limits, min_visibility=0.0)

    assert result[0][0] == 42
    assert result[1][0] == 99


# Tests para create_coco_annotations
def test_create_coco_annotations_basic():
    """Verifica que create_coco_annotations genera el formato COCO correcto."""
    file_names = ["img_100_0_0.tiff", "img_100_0_800.tiff"]
    widths = [800, 800]
    heights = [800, 800]
    tile_boxes = {
        "img_100_0_0.tiff": [[1, 10.0, 20.0, 30.0, 40.0]],
        "img_100_0_800.tiff": [
            [2, 50.0, 60.0, 70.0, 80.0],
            [3, 100.0, 100.0, 50.0, 50.0],
        ],
    }
    categories = [
        {"id": 1, "name": "class1"},
        {"id": 2, "name": "class2"},
        {"id": 3, "name": "class3"},
    ]

    result = create_coco_annotations(
        file_names, widths, heights, tile_boxes, categories
    )

    # Verificar estructura básica
    assert "images" in result
    assert "annotations" in result
    assert "categories" in result

    # Verificar categorías
    assert result["categories"] == categories

    # Verificar imágenes
    assert len(result["images"]) == 2
    assert result["images"][0]["file_name"] == "img_100_0_0.tiff"
    assert result["images"][0]["width"] == 800
    assert result["images"][0]["height"] == 800

    # Verificar anotaciones
    assert len(result["annotations"]) == 3


def test_create_coco_annotations_annotation_format():
    """Verifica que las anotaciones tienen el formato COCO correcto."""
    file_names = ["img_100_0_0.tiff"]
    widths = [800]
    heights = [480]
    tile_boxes = {
        "img_100_0_0.tiff": [[5, 10.0, 20.0, 100.0, 50.0]],
    }
    categories = [{"id": 5, "name": "vehicle"}]

    result = create_coco_annotations(
        file_names, widths, heights, tile_boxes, categories
    )

    ann = result["annotations"][0]

    assert ann["id"] == 0
    assert ann["category_id"] == 5
    assert ann["bbox"] == [10.0, 20.0, 100.0, 50.0]
    assert ann["area"] == 100.0 * 50.0
    assert ann["iscrowd"] == 0


def test_create_coco_annotations_unique_annotation_ids():
    """Verifica que cada anotación tiene un ID único."""
    file_names = ["img_1_0_0.tiff", "img_2_0_0.tiff"]
    widths = [800, 800]
    heights = [800, 800]
    tile_boxes = {
        "img_1_0_0.tiff": [[1, 10.0, 10.0, 20.0, 20.0], [1, 30.0, 30.0, 20.0, 20.0]],
        "img_2_0_0.tiff": [[1, 50.0, 50.0, 20.0, 20.0]],
    }
    categories = [{"id": 1, "name": "class1"}]

    result = create_coco_annotations(
        file_names, widths, heights, tile_boxes, categories
    )

    annotation_ids = [ann["id"] for ann in result["annotations"]]
    assert len(annotation_ids) == len(
        set(annotation_ids)
    ), "Los IDs de anotación deben ser únicos"
    assert annotation_ids == [0, 1, 2]


def test_create_coco_annotations_empty_tile():
    """Verifica que se manejan correctamente los tiles sin anotaciones."""
    file_names = ["img_100_0_0.tiff", "img_100_800_0.tiff"]
    widths = [800, 800]
    heights = [800, 800]
    tile_boxes = {
        "img_100_0_0.tiff": [[1, 10.0, 20.0, 30.0, 40.0]],
        # img_100_800_0.tiff no tiene anotaciones
    }
    categories = [{"id": 1, "name": "class1"}]

    result = create_coco_annotations(
        file_names, widths, heights, tile_boxes, categories
    )

    # Debe haber 2 imágenes pero solo 1 anotación
    assert len(result["images"]) == 2
    assert len(result["annotations"]) == 1


def test_create_coco_annotations_image_id_generation():
    """Verifica que los image_id se generan correctamente."""
    file_names = ["img_1234_0_800.tiff"]
    widths = [800]
    heights = [800]
    tile_boxes = {}
    categories = []

    result = create_coco_annotations(
        file_names, widths, heights, tile_boxes, categories
    )

    # Comprobamos la asignación incremental (inicia en 1)
    expected_id = 1
    assert result["images"][0]["id"] == expected_id
