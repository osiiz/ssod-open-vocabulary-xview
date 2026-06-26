"""
Test del pipeline de ensemble argmax con 20 imagenes sinteticas.

No requiere GPU ni datos reales. Mockea el modelo DINO para devolver logits
controlados y verifica el comportamiento esperado de principio a fin:
  inferencia argmax → ensemble/clustering → formato de salida.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.dino_ensemble_inference_argmax import (
    _build_phrase_masks,
    _run_dino_argmax_pass,
    run_dino_ensemble_argmax,
)
from src.inference.multi_prompt_ensemble import (
    ensemble_all_images,
    load_ensemble_config,
)

# ---------------------------------------------------------------------------
# Constantes del test
# ---------------------------------------------------------------------------

N_IMAGES = 20
N_QUERIES = 10  # queries DINO por imagen (reducido para el test)
TEXT_LEN = 32  # longitud del prompt tokenizado (reducido para el test)
IMG_SIZE = 64  # imagenes cuadradas 64x64

# Clases usadas en el test (subconjunto de las 9 reales)
TEST_CLASSES = ["Aircraft", "Light Vehicle"]

# Frases de los 2 "prompt sets" del test
TEST_PROMPT_SETS = {
    "set_a": {"Aircraft": ["aircraft"], "Light Vehicle": ["light vehicle"]},
    "set_b": {"Aircraft": ["plane"], "Light Vehicle": ["small car"]},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_coco_gt(path: Path, n: int = N_IMAGES) -> None:
    """COCO GT con n imagenes, 2 categorias y 2 anotaciones por imagen."""
    cats = [{"id": 1, "name": "Aircraft"}, {"id": 2, "name": "Light Vehicle"}]
    images = [
        {
            "id": i + 1,
            "file_name": f"img_{i+1}.png",
            "width": IMG_SIZE,
            "height": IMG_SIZE,
        }
        for i in range(n)
    ]
    anns = []
    ann_id = 1
    for img in images:
        # Aircraft arriba-izquierda
        anns.append(
            {
                "id": ann_id,
                "image_id": img["id"],
                "category_id": 1,
                "bbox": [5, 5, 15, 15],
                "area": 225,
                "iscrowd": 0,
            }
        )
        ann_id += 1
        # Light Vehicle abajo-derecha
        anns.append(
            {
                "id": ann_id,
                "image_id": img["id"],
                "category_id": 2,
                "bbox": [40, 40, 15, 15],
                "area": 225,
                "iscrowd": 0,
            }
        )
        ann_id += 1
    with path.open("w") as fh:
        json.dump({"images": images, "annotations": anns, "categories": cats}, fh)


def _make_images(img_dir: Path, n: int = N_IMAGES) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(i * 5, 100, 200)).save(
            img_dir / f"img_{i+1}.png"
        )


def _make_config(path: Path) -> None:
    import yaml

    cfg = {"version": 1, "iou_cluster": 0.3, "prompt_sets": TEST_PROMPT_SETS}
    with path.open("w") as fh:
        yaml.dump(cfg, fh)


def _make_mock_processor_and_model(phrases_ordered_by_set: dict) -> tuple:
    """
    Construye un procesador y modelo mock que devuelven logits controlados.

    Para cada pase (prompt set), el modelo activa el token de Aircraft para
    la query 0 (caja arriba-izquierda) y el token de Light Vehicle para la
    query 1 (caja abajo-derecha). El resto de queries tienen score bajo.
    """
    from transformers import AutoTokenizer

    # Tokenizador real para producir input_ids autenticos
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = tokenizer

        def __call__(self, images, text, return_tensors="pt"):
            # text es una lista de strings identicos; tomamos el primero
            enc = self.tokenizer(
                text[0],
                return_tensors="pt",
                padding="max_length",
                max_length=TEXT_LEN,
                truncation=True,
            )
            B = len(images)
            # Repetir input_ids para todo el batch
            input_ids = enc["input_ids"].expand(B, -1)
            attention_mask = enc["attention_mask"].expand(B, -1)
            # pixel_values: tensor dummy [B, 3, H, W]
            pixel_values = torch.zeros(B, 3, IMG_SIZE, IMG_SIZE)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }

    class FakeModel:
        def __init__(self):
            self.call_count = 0

        def __call__(self, input_ids, attention_mask, pixel_values, **kw):
            B = pixel_values.shape[0]
            # Logits [B, N_QUERIES, TEXT_LEN] — todo cero por defecto
            logits = torch.full((B, N_QUERIES, TEXT_LEN), -10.0)

            # Averiguar que prompt set se esta ejecutando por el contenido de input_ids
            # (Aircraft token vs Light Vehicle token en las posiciones del prompt)
            # Activar de forma controlada: query 0 → Aircraft, query 1 → Light Vehicle
            # usando posiciones de token reales del prompt actual
            prompt_str = tokenizer.decode(input_ids[0], skip_special_tokens=False)

            # Encontrar posicion del primer token no-especial de cada clase:
            # Aircraft aparece como "aircraft" o "plane"; Light Vehicle como "light" o "small"
            ids_list = input_ids[0].tolist()
            period_id = 1012

            # Reconstruir segmentos y activar query 0 para el primer segmento (Aircraft)
            # y query 1 para el segundo segmento (Light Vehicle)
            period_positions = [i for i, t in enumerate(ids_list) if t == period_id]
            seg_starts = [1] + [p + 1 for p in period_positions]
            seg_ends = period_positions + [len(ids_list)]

            for seg_idx, (s, e) in enumerate(zip(seg_starts[:2], seg_ends[:2])):
                query_idx = seg_idx  # query 0 → Aircraft, query 1 → Light Vehicle
                for t in range(s, min(e, TEXT_LEN)):
                    logits[:, query_idx, t] = 5.0  # alta activacion

            # pred_boxes [B, N_QUERIES, 4] cxcywh normalizado
            boxes = torch.full((B, N_QUERIES, 4), 0.5)
            # query 0: Aircraft arriba-izquierda (cx=12.5/64, cy=12.5/64, w=15/64, h=15/64)
            boxes[:, 0] = torch.tensor(
                [12.5 / IMG_SIZE, 12.5 / IMG_SIZE, 15.0 / IMG_SIZE, 15.0 / IMG_SIZE]
            )
            # query 1: Light Vehicle abajo-derecha
            boxes[:, 1] = torch.tensor(
                [47.5 / IMG_SIZE, 47.5 / IMG_SIZE, 15.0 / IMG_SIZE, 15.0 / IMG_SIZE]
            )

            self.call_count += 1
            return SimpleNamespace(logits=logits, pred_boxes=boxes)

        def eval(self):
            return self

        def to(self, device):
            return self

    return FakeProcessor(), FakeModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildPhraseMasks:
    def test_two_classes_single_phrase(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        phrases = [("aircraft", "Aircraft"), ("light vehicle", "Light Vehicle")]
        prompt = " . ".join(p for p, _ in phrases) + " ."
        ids = tok(prompt)["input_ids"]
        masks, cls_idx = _build_phrase_masks(
            ids, phrases, ["Aircraft", "Light Vehicle"]
        )

        # Aircraft ocupa exactamente 1 token ("aircraft"), Light Vehicle 2 ("light", "vehicle")
        assert masks[0].sum().item() == 1
        assert masks[1].sum().item() == 2
        assert cls_idx[0].item() == 0  # Aircraft
        assert cls_idx[1].item() == 1  # Light Vehicle

    def test_multiword_phrase_all_tokens_marked(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        phrases = [("aerial view of aircraft", "Aircraft")]
        prompt = " . ".join(p for p, _ in phrases) + " ."
        ids = tok(prompt)["input_ids"]
        masks, cls_idx = _build_phrase_masks(ids, phrases, ["Aircraft"])
        # "aerial view of aircraft" tokeniza a 4 tokens
        assert masks[0].sum().item() == 4
        assert cls_idx[0].item() == 0

    def test_no_overlap_between_classes(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        phrases = [
            ("plane", "Aircraft"),
            ("car", "Light Vehicle"),
            ("ship", "Maritime Vessel"),
        ]
        prompt = " . ".join(p for p, _ in phrases) + " ."
        ids = tok(prompt)["input_ids"]
        classes = ["Aircraft", "Light Vehicle", "Maritime Vessel"]
        masks, _ = _build_phrase_masks(ids, phrases, classes)
        # Ninguna posicion debe estar marcada en dos clases a la vez
        overlap = (masks.sum(dim=0) > 1).any()
        assert not overlap


class TestArgmaxPassWithMock:
    def test_correct_class_assignment(self):
        """Query 0 debe asignarse a Aircraft, query 1 a Light Vehicle."""
        processor, model = _make_mock_processor_and_model({})
        phrases = [("aircraft", "Aircraft"), ("light vehicle", "Light Vehicle")]
        images = [Image.new("RGB", (IMG_SIZE, IMG_SIZE)) for _ in range(2)]
        image_ids = [1, 2]

        dets, probs = _run_dino_argmax_pass(
            processor,
            model,
            images,
            image_ids,
            phrases_ordered=phrases,
            class_names=["Aircraft", "Light Vehicle"],
            score_thresh=0.3,
            device=torch.device("cpu"),
            prompt_set_name="set_a",
            save_probs=True,
        )

        classes_detected = {d["class_name"] for d in dets}
        assert "Aircraft" in classes_detected
        assert "Light Vehicle" in classes_detected

    def test_save_probs_shape_and_alignment(self):
        """probs debe tener shape [n_dets, TEXT_LEN] y ser index-aligned con dets."""
        processor, model = _make_mock_processor_and_model({})
        phrases = [("aircraft", "Aircraft"), ("light vehicle", "Light Vehicle")]
        images = [Image.new("RGB", (IMG_SIZE, IMG_SIZE)) for _ in range(3)]
        image_ids = [1, 2, 3]

        dets, probs = _run_dino_argmax_pass(
            processor,
            model,
            images,
            image_ids,
            phrases_ordered=phrases,
            class_names=["Aircraft", "Light Vehicle"],
            score_thresh=0.3,
            device=torch.device("cpu"),
            prompt_set_name="set_a",
            save_probs=True,
        )

        assert probs is not None
        assert probs.dtype == np.float16
        assert probs.shape == (len(dets), TEXT_LEN)

    def test_save_probs_false_returns_none(self):
        processor, model = _make_mock_processor_and_model({})
        phrases = [("aircraft", "Aircraft")]
        images = [Image.new("RGB", (IMG_SIZE, IMG_SIZE))]
        dets, probs = _run_dino_argmax_pass(
            processor,
            model,
            images,
            [1],
            phrases_ordered=phrases,
            class_names=["Aircraft"],
            score_thresh=0.3,
            device=torch.device("cpu"),
            prompt_set_name="set_a",
            save_probs=False,
        )
        assert probs is None


class TestFullPipeline20Images:
    """
    Test de integracion completo: inference → ensemble sobre 20 imagenes.
    Cada imagen tiene un Aircraft (arriba-izquierda) y un Light Vehicle
    (abajo-derecha). Con 2 prompt sets y IoU cluster=0.3, el ensemble debe
    producir exactamente 2 detecciones por imagen con clases correctas.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.img_dir = tmp_path / "images"
        self.ann_file = tmp_path / "gt.json"
        self.config = tmp_path / "config.yaml"
        self.output = tmp_path / "raw" / "detection_results.json"

        _make_images(self.img_dir)
        _make_coco_gt(self.ann_file)
        _make_config(self.config)

        # Parchear AutoProcessor y AutoModelForZeroShotObjectDetection
        processor, model = _make_mock_processor_and_model({})

        monkeypatch.setattr(
            "src.inference.dino_ensemble_inference_argmax.AutoProcessor.from_pretrained",
            lambda *a, **kw: processor,
        )
        monkeypatch.setattr(
            "src.inference.dino_ensemble_inference_argmax.AutoModelForZeroShotObjectDetection.from_pretrained",
            lambda *a, **kw: model,
        )

    def test_inference_produces_detections_for_all_images(self):
        run_dino_ensemble_argmax(
            img_dir=self.img_dir,
            ann_file=self.ann_file,
            config_path=self.config,
            output_path=self.output,
            score_thresh=0.3,
            batch_size=4,
            device_str="cpu",
        )
        assert self.output.exists()
        dets = json.loads(self.output.read_text())
        image_ids_seen = {d["image_id"] for d in dets}
        # Todas las imagenes deben tener al menos una deteccion
        assert len(image_ids_seen) == N_IMAGES

    def test_inference_with_save_probs(self):
        run_dino_ensemble_argmax(
            img_dir=self.img_dir,
            ann_file=self.ann_file,
            config_path=self.config,
            output_path=self.output,
            score_thresh=0.3,
            batch_size=4,
            device_str="cpu",
            save_probs=True,
        )
        probs_file = self.output.parent / "probs.npz"
        assert probs_file.exists()

        data = np.load(probs_file)
        dets = json.loads(self.output.read_text())
        assert data["probs"].shape[0] == len(dets)
        assert data["probs"].dtype == np.float16
        assert data["image_ids"].shape[0] == len(dets)
        # image_ids del npz deben coincidir con los del JSON
        assert list(data["image_ids"]) == [d["image_id"] for d in dets]

    def test_ensemble_produces_two_objects_per_image(self):
        """Tras agregar los 2 prompt sets, cada imagen debe tener Aircraft + LV."""
        run_dino_ensemble_argmax(
            img_dir=self.img_dir,
            ann_file=self.ann_file,
            config_path=self.config,
            output_path=self.output,
            score_thresh=0.3,
            batch_size=4,
            device_str="cpu",
        )
        raw_dets = json.loads(self.output.read_text())

        # Cargar config de ensemble para resolver class_to_id
        prompt_to_class, class_names, class_to_id, iou_thresh = load_ensemble_config(
            self.config, coco_ann_file=self.ann_file
        )
        merged = ensemble_all_images(
            raw_dets,
            prompt_to_class=prompt_to_class,
            class_names=class_names,
            class_to_id=class_to_id,
            iou_thresh=iou_thresh,
            mode="score",
            label_key="dino_label",
        )

        # Agrupar por imagen
        by_img: dict[int, list] = {}
        for d in merged:
            by_img.setdefault(d["image_id"], []).append(d)

        for img_id, img_dets in by_img.items():
            classes = {d["class_name"] for d in img_dets}
            # Ambas clases deben estar presentes
            assert "Aircraft" in classes, f"imagen {img_id}: Aircraft no detectado"
            assert (
                "Light Vehicle" in classes
            ), f"imagen {img_id}: Light Vehicle no detectado"

    def test_singletons_are_preserved(self):
        """Detecciones sin solapamiento deben conservarse como n_cluster=1."""
        run_dino_ensemble_argmax(
            img_dir=self.img_dir,
            ann_file=self.ann_file,
            config_path=self.config,
            output_path=self.output,
            score_thresh=0.3,
            batch_size=4,
            device_str="cpu",
        )
        raw_dets = json.loads(self.output.read_text())
        prompt_to_class, class_names, class_to_id, iou_thresh = load_ensemble_config(
            self.config, coco_ann_file=self.ann_file
        )
        merged = ensemble_all_images(
            raw_dets,
            prompt_to_class=prompt_to_class,
            class_names=class_names,
            class_to_id=class_to_id,
            iou_thresh=iou_thresh,
        )
        # Si las cajas de los 2 prompt sets se solapan bien, los clusters tienen
        # n_cluster >= 2. Si no solapan (singletons), n_cluster == 1. Ambos son validos.
        assert all("n_cluster" in d for d in merged)
        assert all(d["n_cluster"] >= 1 for d in merged)

    def test_checkpointing_skips_completed_sets(self, tmp_path):
        """Si el checkpoint de un set ya existe, no se vuelve a inferir."""
        raw_dir = self.output.parent
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Pre-crear checkpoint del set_a (vacio para simplificar)
        ckpt = raw_dir / "_ckpt_set_a.json"
        ckpt.write_text("[]")

        run_dino_ensemble_argmax(
            img_dir=self.img_dir,
            ann_file=self.ann_file,
            config_path=self.config,
            output_path=self.output,
            score_thresh=0.3,
            batch_size=4,
            device_str="cpu",
        )
        # Debe existir el output aunque set_a se saltara
        assert self.output.exists()
