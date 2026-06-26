import pytest
import tempfile
import json
from pathlib import Path
from src.preprocessing.xView_to_coco import (
    create_class_mapping,
    load_and_clean_annotations,
    load_xview_class_map,
    find_extracted_data,
)


@pytest.fixture
def sample_xview_class_map():
    """Fixture con un subconjunto del mapeo de clases xView para pruebas."""
    return {
        11: {"id": 1, "name": "Aircraft"},
        12: {"id": 1, "name": "Aircraft"},
        17: {"id": 2, "name": "Light Vehicle"},
        18: {"id": 2, "name": "Light Vehicle"},
    }


@pytest.fixture
def full_xview_class_map():
    """Fixture con el mapeo completo de macro-clases xView (11 clases finales)."""
    raw_map = {
        "11": {"id": 1, "name": "Aircraft"},
        "12": {"id": 1, "name": "Aircraft"},
        "13": {"id": 1, "name": "Aircraft"},
        "15": {"id": 1, "name": "Aircraft"},
        "17": {"id": 2, "name": "Light Vehicle"},
        "18": {"id": 2, "name": "Light Vehicle"},
        "20": {"id": 2, "name": "Light Vehicle"},
        "21": {"id": 2, "name": "Light Vehicle"},
        "19": {"id": 3, "name": "Heavy Vehicle"},
        "23": {"id": 3, "name": "Heavy Vehicle"},
        "24": {"id": 3, "name": "Heavy Vehicle"},
        "25": {"id": 3, "name": "Heavy Vehicle"},
        "26": {"id": 3, "name": "Heavy Vehicle"},
        "27": {"id": 3, "name": "Heavy Vehicle"},
        "28": {"id": 3, "name": "Heavy Vehicle"},
        "29": {"id": 3, "name": "Heavy Vehicle"},
        "32": {"id": 3, "name": "Heavy Vehicle"},
        "33": {"id": 4, "name": "Railway Vehicle"},
        "34": {"id": 4, "name": "Railway Vehicle"},
        "35": {"id": 4, "name": "Railway Vehicle"},
        "36": {"id": 4, "name": "Railway Vehicle"},
        "37": {"id": 4, "name": "Railway Vehicle"},
        "38": {"id": 4, "name": "Railway Vehicle"},
        "40": {"id": 5, "name": "Maritime Vessel"},
        "41": {"id": 5, "name": "Maritime Vessel"},
        "42": {"id": 5, "name": "Maritime Vessel"},
        "44": {"id": 5, "name": "Maritime Vessel"},
        "45": {"id": 5, "name": "Maritime Vessel"},
        "47": {"id": 5, "name": "Maritime Vessel"},
        "49": {"id": 5, "name": "Maritime Vessel"},
        "50": {"id": 5, "name": "Maritime Vessel"},
        "51": {"id": 5, "name": "Maritime Vessel"},
        "52": {"id": 5, "name": "Maritime Vessel"},
        "53": {"id": 6, "name": "Engineering Vehicle"},
        "56": {"id": 6, "name": "Engineering Vehicle"},
        "57": {"id": 6, "name": "Engineering Vehicle"},
        "60": {"id": 6, "name": "Engineering Vehicle"},
        "61": {"id": 6, "name": "Engineering Vehicle"},
        "62": {"id": 6, "name": "Engineering Vehicle"},
        "63": {"id": 6, "name": "Engineering Vehicle"},
        "64": {"id": 6, "name": "Engineering Vehicle"},
        "65": {"id": 6, "name": "Engineering Vehicle"},
        "66": {"id": 6, "name": "Engineering Vehicle"},
        "71": {"id": 7, "name": "Building"},
        "72": {"id": 7, "name": "Building"},
        "73": {"id": 7, "name": "Building"},
        "74": {"id": 7, "name": "Building"},
        "76": {"id": 7, "name": "Building"},
        "77": {"id": 7, "name": "Building"},
        "86": {"id": 8, "name": "Storage Tank"},
        "93": {"id": 9, "name": "Tower & Pylon"},
        "94": {"id": 9, "name": "Tower & Pylon"},
    }
    return {int(k): v for k, v in raw_map.items()}


def test_load_xview_class_map():
    """Verifica que load_xview_class_map carga correctamente el JSON y convierte las claves a int."""
    test_map = {
        "11": {"id": 1, "name": "Aircraft"},
        "17": {"id": 2, "name": "Light Vehicle"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_map, f)
        temp_path = Path(f.name)

    try:
        result = load_xview_class_map(temp_path)
        assert result == {
            11: {"id": 1, "name": "Aircraft"},
            17: {"id": 2, "name": "Light Vehicle"},
        }
        assert all(isinstance(k, int) for k in result.keys())
    finally:
        temp_path.unlink()


def test_create_class_mapping(full_xview_class_map):
    """Validamos que el mapeo devuelva las 14 macro-clases de los 60 IDs originales."""
    new_class_map, old_to_new = create_class_mapping(full_xview_class_map)

    assert (
        len(new_class_map) == 9
    ), "El dataset limpio de xView debe tener exactamente 9 macro-clases"
    assert (
        len(old_to_new) == 52
    ), "Se deben mapear los 60 IDs originales menos construction site (13), vehicle lot (14), helipad (84), tower crane (54), container crane (55), mobile crane (58), shipping container (91) y shipping container lot (89)"
    assert (
        old_to_new[11] == 1
    ), "El ID original 11 debe mapearse al macro ID 1 (Aircraft)"


def test_create_class_mapping_sequential_ids(sample_xview_class_map):
    """Verifica la correcta asignación de IDs antiguos a macros."""
    new_class_map, old_to_new = create_class_mapping(sample_xview_class_map)

    assert set(new_class_map.keys()) == {1, 2}
    assert old_to_new == {11: 1, 12: 1, 17: 2, 18: 2}


def test_load_and_clean_annotations_removes_erroneous_classes(full_xview_class_map):
    """Verifica que las anotaciones erróneas y las imágenes faltantes son eliminadas."""
    test_geojson = {
        "features": [
            {
                "properties": {
                    "image_id": "100.tif",
                    "type_id": 11,  # Aircraft
                    "bounds_imcoords": "10,10,50,50",
                }
            },
            {
                "properties": {
                    "image_id": "100.tif",
                    "type_id": 75,  # Erróneo (no mapeado)
                    "bounds_imcoords": "60,60,100,100",
                }
            },
            {
                "properties": {
                    "image_id": "100.tif",
                    "type_id": 82,  # Erróneo (no mapeado)
                    "bounds_imcoords": "110,110,150,150",
                }
            },
            {
                "properties": {
                    "image_id": "1395.tif",  # Imagen faltante
                    "type_id": 17,
                    "bounds_imcoords": "10,10,50,50",
                }
            },
            {
                "properties": {
                    "image_id": "200.tif",
                    "type_id": 17,  # Light Vehicle
                    "bounds_imcoords": "20,20,60,60",
                }
            },
        ]
    }

    _, old_to_new = create_class_mapping(full_xview_class_map)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        json.dump(test_geojson, f)
        temp_path = Path(f.name)

    try:
        erroneous_type_ids = [75, 82]
        missing_image_ids = ["1395.tif"]

        df = load_and_clean_annotations(
            temp_path, old_to_new, erroneous_type_ids, missing_image_ids
        )

        assert len(df) == 2, "Deben quedar exactamente 2 anotaciones válidas"
        assert set(df["image_id"].unique()) == {"100.tif", "200.tif"}
    finally:
        temp_path.unlink()


def test_find_extracted_data_success():
    """Verifica que find_extracted_data encuentra correctamente los archivos cuando existen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_data_path = Path(tmpdir)

        # Crear estructura esperada
        (raw_data_path / "train_images").mkdir()
        (raw_data_path / "xView_train.geojson").touch()

        img_folder, labels_path = find_extracted_data(raw_data_path)

        assert img_folder == raw_data_path / "train_images"
        assert labels_path == raw_data_path / "xView_train.geojson"


def test_find_extracted_data_missing_geojson():
    """Verifica que find_extracted_data lanza error cuando falta el archivo geoJSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_data_path = Path(tmpdir)

        # Solo crear la carpeta de imágenes, no el geoJSON
        (raw_data_path / "train_images").mkdir()

        with pytest.raises(
            FileNotFoundError, match="No se encontró el archivo de anotaciones"
        ):
            find_extracted_data(raw_data_path)


def test_find_extracted_data_missing_images_folder():
    """Verifica que find_extracted_data lanza error cuando falta la carpeta de imágenes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_data_path = Path(tmpdir)

        # Solo crear el geoJSON, no la carpeta de imágenes
        (raw_data_path / "xView_train.geojson").touch()

        with pytest.raises(
            FileNotFoundError, match="No se encontró la carpeta de imágenes"
        ):
            find_extracted_data(raw_data_path)


def test_load_and_clean_annotations_maps_type_ids(full_xview_class_map):
    """Verifica que los type_ids se mapean correctamente a los nuevos IDs secuenciales."""
    test_geojson = {
        "features": [
            {
                "properties": {
                    "image_id": "100.tif",
                    "type_id": 11,
                    "bounds_imcoords": "10,10,50,50",
                }
            },
            {
                "properties": {
                    "image_id": "100.tif",
                    "type_id": 94,
                    "bounds_imcoords": "60,60,100,100",
                }
            },
        ]
    }

    _, old_to_new = create_class_mapping(full_xview_class_map)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        json.dump(test_geojson, f)
        temp_path = Path(f.name)

    try:
        df = load_and_clean_annotations(temp_path, old_to_new, [], [])

        # type_id=11 debe mapearse a 0, type_id=94 debe mapearse al último ID
        assert df.iloc[0]["type_id"] == old_to_new[11]
        assert df.iloc[1]["type_id"] == old_to_new[94]
    finally:
        temp_path.unlink()


def test_load_and_clean_annotations_parses_bbox_correctly(full_xview_class_map):
    """Verifica que las coordenadas del bbox se parsean correctamente desde bounds_imcoords."""
    test_geojson = {
        "features": [
            {
                "properties": {
                    "image_id": "100.tif",
                    "type_id": 11,
                    "bounds_imcoords": "100,200,300,400",
                }
            },
        ]
    }

    _, old_to_new = create_class_mapping(full_xview_class_map)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        json.dump(test_geojson, f)
        temp_path = Path(f.name)

    try:
        df = load_and_clean_annotations(temp_path, old_to_new, [], [])

        row = df.iloc[0]
        assert row["x_min"] == 100
        assert row["y_min"] == 200
        assert row["x_max"] == 300
        assert row["y_max"] == 400
    finally:
        temp_path.unlink()


def test_load_and_clean_annotations_removes_multiple_missing_images(
    full_xview_class_map,
):
    """Verifica que se pueden eliminar múltiples imágenes faltantes."""
    test_geojson = {
        "features": [
            {
                "properties": {
                    "image_id": "100.tif",
                    "type_id": 11,
                    "bounds_imcoords": "10,10,50,50",
                }
            },
            {
                "properties": {
                    "image_id": "missing1.tif",
                    "type_id": 11,
                    "bounds_imcoords": "10,10,50,50",
                }
            },
            {
                "properties": {
                    "image_id": "missing2.tif",
                    "type_id": 11,
                    "bounds_imcoords": "10,10,50,50",
                }
            },
            {
                "properties": {
                    "image_id": "200.tif",
                    "type_id": 11,
                    "bounds_imcoords": "10,10,50,50",
                }
            },
        ]
    }

    _, old_to_new = create_class_mapping(full_xview_class_map)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        json.dump(test_geojson, f)
        temp_path = Path(f.name)

    try:
        missing_image_ids = ["missing1.tif", "missing2.tif"]
        df = load_and_clean_annotations(temp_path, old_to_new, [], missing_image_ids)

        assert len(df) == 2
        assert set(df["image_id"].unique()) == {"100.tif", "200.tif"}
    finally:
        temp_path.unlink()
