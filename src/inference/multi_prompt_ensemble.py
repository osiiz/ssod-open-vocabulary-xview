"""
Agregación por ensamble multi-prompt para detectores de vocabulario abierto.

A partir de las detecciones brutas generadas al ejecutar múltiples prompts
(fino y a nivel macro) sobre Grounding DINO y/o Rex-Omni, este módulo:

  1. Mapea cada etiqueta de prompt cruda a su macro-clase usando la config del ensamble.
  2. Agrupa todas las detecciones por imagen mediante componentes conexas basadas en IoU
     (umbral configurable, por defecto 0.3). El agrupamiento es independiente de la clase
     para resolver juntas las cajas solapadas de prompts de distinta clase.
  3. Por cada cluster produce una detección fusionada:
       Decisión de localización  : caja delimitadora media sobre los miembros del cluster
       Incertidumbre de localiz. : desviación estándar componente a componente de las cajas
       Decisión de clasificación :
         modo "score" (Grounding DINO) – macro-clase con mayor puntuación media dentro del
           cluster; entropía del vector de puntuaciones normalizado por softmax como incert.
         modo "vote"  (Rex-Omni)       – macro-clase con más detecciones en el cluster;
           entropía de la distribución de frecuencias como incertidumbre.

Uso (librería)
--------------
    from src.inference.multi_prompt_ensemble import load_ensemble_config, ensemble_all_images

    prompt_to_class, class_names, class_to_id, iou_thresh = load_ensemble_config(
        "configs/prompts/ensemble_prompts.yaml",
        coco_ann_file="results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json",
    )
    merged = ensemble_all_images(
        raw_detections,          # list[dict] con campos bbox/score/dino_label
        prompt_to_class=prompt_to_class,
        class_names=class_names,
        class_to_id=class_to_id,
        iou_thresh=iou_thresh,
        mode="score",            # "score" | "vote" | "borda"
        label_key="dino_label",  # campo que contiene el texto de prompt crudo
    )

Uso (CLI)
---------
    python -m src.inference.multi_prompt_ensemble \\
        --detections  results/dino/r90_dino_raw/detection_results.json \\
        --ann_file    results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \\
        --config      configs/prompts/ensemble_prompts.yaml \\
        --mode        score \\
        --label_key   dino_label \\
        --output      results/dino/r90_dino_ensemble/detection_results.json
"""

from __future__ import annotations

import argparse
import json
import mmap
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson
import yaml


# ---------------------------------------------------------------------------
# Carga de configuración
# ---------------------------------------------------------------------------


def load_ensemble_config(
    config_path: str | Path,
    coco_ann_file: str | Path | None = None,
) -> tuple[dict[str, str], list[str], dict[str, int], float]:
    """
    Parsea ensemble_prompts.yaml y devuelve:

    prompt_to_class : dict  texto de prompt crudo (minúsculas) → nombre de macro-clase
    class_names     : list  nombres de macro-clase ordenados
    class_to_id     : dict  nombre de macro-clase → category_id COCO
                            (leído de coco_ann_file; usa índice base-1 si falta)
    iou_thresh      : float umbral de IoU para clustering de la config (por defecto 0.3)
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    iou_thresh: float = float(cfg.get("iou_cluster", 0.3))

    # Construir prompt_to_class a partir de todos los prompt_sets combinados
    prompt_to_class: dict[str, str] = {}
    class_names_set: set[str] = set()

    prompt_sets: dict = cfg.get("prompt_sets", {})
    for _set_name, set_classes in prompt_sets.items():
        for class_name, phrases in (set_classes or {}).items():
            class_names_set.add(class_name)
            for phrase in phrases or []:
                prompt_to_class[str(phrase).lower().strip()] = class_name

    class_names = sorted(class_names_set)

    # Construir class_to_id desde el fichero de anotaciones COCO cuando se proporciona
    class_to_id: dict[str, int] = {}
    if coco_ann_file is not None:
        with open(coco_ann_file, encoding="utf-8") as fh:
            ann = json.load(fh)
        for cat in ann.get("categories", []):
            class_to_id[cat["name"]] = cat["id"]

    # Fallback: asignar IDs secuenciales a las clases no encontradas en el fichero COCO
    for i, name in enumerate(class_names, start=1):
        if name not in class_to_id:
            class_to_id[name] = i

    return prompt_to_class, class_names, class_to_id, iou_thresh


def get_prompt_sets(config_path: str | Path) -> dict[str, dict[str, list[str]]]:
    """
    Devuelve los conjuntos de prompts de la config tal cual:
      {set_name: {class_name: [phrase, ...]}}

    Usado por los scripts de inferencia para saber qué frases ejecutar por conjunto.
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("prompt_sets", {})


# ---------------------------------------------------------------------------
# IoU y agrupamiento
# ---------------------------------------------------------------------------


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _cluster_indices(
    boxes_xyxy: list[list[float]],
    iou_thresh: float,
    use_centroid_criterion: bool = False,
    centroid_mode: str = "or",
) -> list[list[int]]:
    """
    Clustering por componentes conexas mediante union-find.

    Dos cajas pertenecen al mismo cluster segun centroid_mode:
      - 'or'  (default si use_centroid_criterion): IoU >= thresh O centroide_dentro.
              Mas permisivo: agrupa pares con IoU bajo si hay containment.
              Riesgo: en escenas densas (objetos pequenos dentro de cajas grandes)
              causa megaclusters absurdos.
      - 'and' : IoU >= thresh Y centroide_dentro.
              Semantica original de ATSS: el centroide actua como filtro ADICIONAL
              sobre pares ya solapados. Solo filtra casos raros donde IoU alto
              pero centroides desplazados (cajas alargadas y muy desalineadas).
              Conservador, suele dar resultados muy parecidos al baseline.

    Si use_centroid_criterion=False -> solo IoU (sin criterio centroide).
    """
    n = len(boxes_xyxy)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    boxes = np.asarray(boxes_xyxy, dtype=np.float32)  # (n, 4)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Calcular IoU solo para pares del triangulo superior (evita trabajo 2x redundante
    # y reduce a la mitad el pico de memoria frente a una matriz nxn completa).
    ri, ci = np.triu_indices(n, k=1)
    inter_w = np.maximum(0.0, np.minimum(x2[ri], x2[ci]) - np.maximum(x1[ri], x1[ci]))
    inter_h = np.maximum(0.0, np.minimum(y2[ri], y2[ci]) - np.maximum(y1[ri], y1[ci]))
    inter = inter_w * inter_h
    union_area = areas[ri] + areas[ci] - inter
    iou = np.divide(inter, union_area, out=np.zeros_like(inter), where=union_area > 0)

    iou_mask = iou >= iou_thresh

    if use_centroid_criterion:
        # Centroide de cada caja, dentro del bbox de la otra (en ambas direcciones).
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        i_in_j = (
            (cx[ri] >= x1[ci])
            & (cx[ri] <= x2[ci])
            & (cy[ri] >= y1[ci])
            & (cy[ri] <= y2[ci])
        )
        j_in_i = (
            (cx[ci] >= x1[ri])
            & (cx[ci] <= x2[ri])
            & (cy[ci] >= y1[ri])
            & (cy[ci] <= y2[ri])
        )
        centroid_mask = i_in_j | j_in_i

        if centroid_mode == "and":
            merge_mask = iou_mask & centroid_mask
        elif centroid_mode == "or":
            merge_mask = iou_mask | centroid_mask
        else:
            raise ValueError(
                f"centroid_mode debe ser 'or' o 'and', no {centroid_mode!r}"
            )
    else:
        merge_mask = iou_mask

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(ri[merge_mask].tolist(), ci[merge_mask].tolist()):
        parent[find(i)] = find(j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _cluster_by_wbf(
    detections: list[dict],
    image_size: tuple[int, int],
    iou_thr: float = 0.55,
    conf_type: str = "avg",
    use_centroid_criterion: bool = False,
) -> list[list[int]]:
    """
    Clustering via Weighted Boxes Fusion.

    Devuelve clusters como indices en `detections`, mismo formato que `_cluster_indices`.
    Las cajas originales se preservan en los checkpoints; aqui solo se reconstruye la
    membresia asignando cada original al bbox fusionado con mayor IoU.

    Cada prompt_set se trata como un "modelo" distinto para WBF. El clustering es
    class-agnostic (label=0); el voto de clase se hace despues sobre los miembros
    del cluster, manteniendo la semantica del pipeline actual.
    """
    from ensemble_boxes import weighted_boxes_fusion

    n = len(detections)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    W, H = image_size

    prompt_groups: dict[str, list[int]] = defaultdict(list)
    for i, det in enumerate(detections):
        prompt_groups[det.get("prompt_set", "default")].append(i)

    boxes_list: list[list[list[float]]] = []
    scores_list: list[list[float]] = []
    labels_list: list[list[int]] = []
    for ps in sorted(prompt_groups.keys()):
        boxes, scores, labels = [], [], []
        for i in prompt_groups[ps]:
            x, y, w, h = detections[i]["bbox"]
            # Clamp a [0,1]; valores fuera de rango harian fallar a WBF
            boxes.append(
                [
                    max(0.0, min(1.0, x / W)),
                    max(0.0, min(1.0, y / H)),
                    max(0.0, min(1.0, (x + w) / W)),
                    max(0.0, min(1.0, (y + h) / H)),
                ]
            )
            scores.append(float(detections[i].get("score", 1.0)))
            labels.append(0)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    fused_boxes, _, _ = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=0.0,
        conf_type=conf_type,
    )

    if len(fused_boxes) == 0:
        return [[i] for i in range(n)]

    # Asignacion post-hoc: cada original al bbox fusionado con mayor IoU.
    # El centroid criterion anade prioridad si el centro cae dentro del fusionado
    # (util cuando el IoU es bajo por diferencia de escala).
    cluster_assignment: dict[int, int] = {}
    for det_idx in range(n):
        x, y, w, h = detections[det_idx]["bbox"]
        det_box = [x / W, y / H, (x + w) / W, (y + h) / H]
        cx, cy = (det_box[0] + det_box[2]) * 0.5, (det_box[1] + det_box[3]) * 0.5

        best_priority = -1.0
        best_fused_idx = None
        for fused_idx, fb in enumerate(fused_boxes):
            iou = _iou_xyxy(det_box, fb.tolist())
            priority = iou
            if use_centroid_criterion:
                if fb[0] <= cx <= fb[2] and fb[1] <= cy <= fb[3]:
                    priority += 0.001  # desempate a favor de containment
            if priority > best_priority:
                best_priority = priority
                best_fused_idx = fused_idx

        if best_fused_idx is not None and best_priority > 0:
            cluster_assignment[det_idx] = best_fused_idx

    clusters_dict: dict[int, list[int]] = defaultdict(list)
    for det_idx, fused_idx in cluster_assignment.items():
        clusters_dict[fused_idx].append(det_idx)

    # Originales sin solapamiento con ningun fusionado: singletons
    unassigned = set(range(n)) - set(cluster_assignment.keys())
    next_id = len(fused_boxes)
    for det_idx in unassigned:
        clusters_dict[next_id] = [det_idx]
        next_id += 1

    return [clusters_dict[k] for k in sorted(clusters_dict.keys())]


# ---------------------------------------------------------------------------
# Agregación de localización
# ---------------------------------------------------------------------------


def _aggregate_localisation(
    bboxes_xywh: list[list[float]],
    scores: list[float] | None = None,
    fusion_mode: str = "mean",
) -> tuple[list[float], list[float]]:
    """
    Devuelve (fused_bbox_xywh, std_bbox_xywh).

    fusion_mode:
      - 'mean'           : media aritmetica simple (cada miembro contribuye igual)
      - 'score_weighted' : media ponderada por score, estilo WBF
                           bbox_fused = sum(s_i * bbox_i) / sum(s_i)
                           Si todos los scores son 0 (caso degenerado), recae en 'mean'.

    std siempre se calcula sin ponderar — refleja la dispersion espacial real
    entre miembros del cluster, util como medida de incertidumbre de localizacion
    independiente de la confianza de cada deteccion.

    Clusters de un solo elemento devuelven std = [0,0,0,0].
    """
    arr = np.array(bboxes_xywh, dtype=float)

    if fusion_mode == "score_weighted" and scores is not None:
        w = np.asarray(scores, dtype=float)
        total_w = float(w.sum())
        if total_w > 0:
            fused_bbox = ((arr * w[:, None]).sum(axis=0) / total_w).tolist()
        else:
            fused_bbox = arr.mean(axis=0).tolist()
    else:
        fused_bbox = arr.mean(axis=0).tolist()

    std_bbox = arr.std(axis=0).tolist() if len(arr) > 1 else [0.0, 0.0, 0.0, 0.0]
    return fused_bbox, std_bbox


# ---------------------------------------------------------------------------
# Agregación de clasificación
# ---------------------------------------------------------------------------


def _shannon_entropy(probs: np.ndarray) -> float:
    """Entropía de Shannon en nats; seguro ante ceros."""
    p = probs[probs > 0].astype(np.float64)
    return float(-np.sum(p * np.log(p)))


def _classify_by_precomputed_name(
    cluster_dets: list[dict],
    class_names: list[str],
) -> tuple[str, float, dict[str, int]]:
    """
    Se usa cuando cada detección del cluster ya tiene un class_name fiable
    (p.ej. de pases de inferencia por clase). Votación mayoritaria sobre class_name;
    entropía de la distribución de votos como incertidumbre.
    """
    from collections import Counter

    votes = Counter(det["class_name"] for det in cluster_dets if det.get("class_name"))
    total = sum(votes.values())
    probs = np.array(
        [
            votes.get(c, 0) / total if total else 1.0 / len(class_names)
            for c in class_names
        ],
        dtype=np.float64,
    )
    best_class = class_names[int(np.argmax(probs))]
    return best_class, _shannon_entropy(probs), dict(votes)


def _classify_score_mode(
    cluster_dets: list[dict],
    prompt_to_class: dict[str, str],
    class_names: list[str],
    label_key: str,
) -> tuple[str, float, dict[str, float]]:
    """
    Modo Grounding DINO.
    Para cada macro-clase: calcula la puntuación media de todas las detecciones del cluster
    cuyo prompt mapea a esa clase (0.0 si ninguna).
    Aplica softmax para obtener una distribución normalizada; elige el argmax.
    """
    class_score_lists: dict[str, list[float]] = {c: [] for c in class_names}
    for det in cluster_dets:
        # La clase difusa precalculada tiene prioridad; retrocede a coincidencia exacta de frase
        macro = det.get("class_name") or prompt_to_class.get(
            str(det.get(label_key, "")).lower().strip()
        )
        if macro and macro in class_score_lists:
            class_score_lists[macro].append(float(det.get("score", 0.0)))

    mean_scores = np.array(
        [np.mean(v) if v else 0.0 for v in class_score_lists.values()],
        dtype=float,
    )
    # Softmax (numéricamente estable)
    exp_s = np.exp(mean_scores - mean_scores.max())
    probs = exp_s / exp_s.sum()

    best_idx = int(np.argmax(probs))
    best_class = class_names[best_idx]
    entropy = _shannon_entropy(probs)
    score_dict = {c: float(p) for c, p in zip(class_names, probs)}
    return best_class, entropy, score_dict


def _classify_vote_mode(
    cluster_dets: list[dict],
    prompt_to_class: dict[str, str],
    class_names: list[str],
    label_key: str,
) -> tuple[str, float, dict[str, int]]:
    """
    Modo Rex-Omni.
    Cuenta votos (detecciones) por macro-clase; elige la mayoría.
    La entropía se calcula sobre la distribución de frecuencias normalizada.
    """
    votes: dict[str, int] = {c: 0 for c in class_names}
    for det in cluster_dets:
        macro = det.get("class_name") or prompt_to_class.get(
            str(det.get(label_key, "")).lower().strip()
        )
        if macro and macro in votes:
            votes[macro] += 1

    total = sum(votes.values())
    if total == 0:
        probs = np.ones(len(class_names), dtype=float) / len(class_names)
    else:
        probs = np.array([votes[c] / total for c in class_names], dtype=float)

    best_class = class_names[int(np.argmax(probs))]
    entropy = _shannon_entropy(probs)
    return best_class, entropy, votes


def _classify_borda_mode(
    cluster_dets: list[dict],
    prompt_to_class: dict[str, str],
    class_names: list[str],
    label_key: str,
) -> tuple[str, float, dict[str, float]]:
    """
    Voto de Borda con pesos 1/rank.

    Por cada prompt set que contribuye al cluster se ordenan las macro-clases
    detectadas por score descendente. La clase en posición k recibe 1/k puntos.
    Las clases no detectadas por ese set no contribuyen.

    La distribución de puntos Borda normalizada reemplaza a la distribución de
    softmax-de-scores: solo depende del orden relativo de los scores (robusto a
    scores no calibrados), no de sus valores absolutos.
    """
    from collections import defaultdict as _defaultdict

    # Agrupar detecciones del cluster por prompt_set
    by_set: dict[str, list[tuple[str, float]]] = _defaultdict(list)
    for det in cluster_dets:
        macro = det.get("class_name") or prompt_to_class.get(
            str(det.get(label_key, "")).lower().strip()
        )
        if macro and macro in set(class_names):
            by_set[det["prompt_set"]].append((macro, float(det.get("score", 0.0))))

    borda: dict[str, float] = {c: 0.0 for c in class_names}
    for cls_score_list in by_set.values():
        # Ordenar por score descendente; en caso de empate el orden es estable
        cls_score_list.sort(key=lambda x: x[1], reverse=True)
        seen: set[str] = set()
        rank = 1
        for cls, _ in cls_score_list:
            if cls not in seen:
                borda[cls] += 1.0 / rank
                seen.add(cls)
                rank += 1

    total = sum(borda.values())
    if total == 0:
        probs = np.ones(len(class_names), dtype=float) / len(class_names)
        borda_norm = {c: float(p) for c, p in zip(class_names, probs)}
    else:
        borda_norm = {c: borda[c] / total for c in class_names}
        probs = np.array([borda_norm[c] for c in class_names], dtype=float)

    best_class = class_names[int(np.argmax(probs))]
    entropy = _shannon_entropy(probs)
    return best_class, entropy, borda_norm


# ---------------------------------------------------------------------------
# Ensamble por imagen
# ---------------------------------------------------------------------------


def ensemble_detections(
    detections: list[dict],
    prompt_to_class: dict[str, str],
    class_names: list[str],
    class_to_id: dict[str, int],
    iou_thresh: float = 0.3,
    mode: str = "score",
    label_key: str = "dino_label",
    cluster_method: str = "union_find",
    use_centroid_criterion: bool = False,
    centroid_mode: str = "or",
    image_size: tuple[int, int] | None = None,
    wbf_conf_type: str = "avg",
    fusion_mode: str = "mean",
) -> list[dict]:
    """
    Agrupa y agrega detecciones de una **única imagen**.

    Parámetros
    ----------
    detections    : detecciones brutas de una imagen (bbox en formato COCO xywh)
    prompt_to_class: mapeo de texto de prompt en minúsculas → nombre de macro-clase
    class_names   : lista ordenada de todos los nombres de macro-clase
    class_to_id   : nombre de macro-clase → category_id COCO
    iou_thresh    : IoU para clustering (por defecto 0.3)
    mode          : "score" (media de scores + softmax) | "vote" (mayoría Rex-Omni) | "borda" (Borda 1/rank, robusto a scores no calibrados)
    label_key     : campo en cada dict de detección que contiene el texto de prompt crudo

    Devuelve
    --------
    Lista de dicts de detección fusionada, uno por cluster. Cada dict contiene:
      image_id, category_id, class_name,
      bbox (media de los miembros que votaron a la clase ganadora),
      score (media de scores de esos mismos miembros — NO de todo el cluster),
      bbox_std, loc_uncertainty (media escalar de los componentes de bbox_std,
        también restringido a los miembros de la clase ganadora),
      class_uncertainty (entropía de Shannon en nats — calculada sobre TODOS
        los miembros del cluster, refleja la incertidumbre de la clasificación),
      n_cluster (nº de miembros que votaron a la clase ganadora),
      n_cluster_total (nº total de miembros del cluster, sin filtrar por clase),
      contributing_sets (prompt_sets distintos entre los miembros de la clase
        ganadora — no entre todos los miembros),
      class_scores (modo score) | class_votes (modo vote) | borda_scores (modo borda)
    Las detecciones cuyo prompt_label no está en prompt_to_class se mantienen
    como singleton pero con class_uncertainty = NaN.
    """
    if not detections:
        return []

    if cluster_method == "wbf":
        if image_size is None:
            raise ValueError("cluster_method='wbf' requiere image_size=(W,H)")
        clusters = _cluster_by_wbf(
            detections,
            image_size,
            iou_thr=iou_thresh,
            conf_type=wbf_conf_type,
            use_centroid_criterion=use_centroid_criterion,
        )
    else:
        boxes_xyxy = [_xywh_to_xyxy(d["bbox"]) for d in detections]
        clusters = _cluster_indices(
            boxes_xyxy,
            iou_thresh,
            use_centroid_criterion=use_centroid_criterion,
            centroid_mode=centroid_mode,
        )

    merged: list[dict] = []
    image_id = detections[0]["image_id"]

    def _macro_of(det: dict) -> str | None:
        return det.get("class_name") or prompt_to_class.get(
            str(det.get(label_key, "")).lower().strip()
        )

    for cluster_idxs in clusters:
        cluster_dets = [detections[i] for i in cluster_idxs]

        if all(det.get("class_name") for det in cluster_dets) and mode != "borda":
            best_class, cls_unc, cls_extra = _classify_by_precomputed_name(
                cluster_dets, class_names
            )
            cls_field = "class_votes"
        elif mode == "vote":
            best_class, cls_unc, cls_extra = _classify_vote_mode(
                cluster_dets, prompt_to_class, class_names, label_key
            )
            cls_field = "class_votes"
        elif mode == "borda":
            best_class, cls_unc, cls_extra = _classify_borda_mode(
                cluster_dets, prompt_to_class, class_names, label_key
            )
            cls_field = "borda_scores"
        else:
            best_class, cls_unc, cls_extra = _classify_score_mode(
                cluster_dets, prompt_to_class, class_names, label_key
            )
            cls_field = "class_scores"

        category_id = class_to_id.get(best_class, -1)

        # Métricas restringidas a la clase ganadora: el bbox, el score, n_cluster
        # y contributing_sets se calculan sobre los miembros que votaron a best_class.
        # Esto evita que detecciones de clases minoritarias dentro de un cluster
        # contaminen la confianza/posición de la PE final.
        winning_dets = [d for d in cluster_dets if _macro_of(d) == best_class]
        if not winning_dets:
            winning_dets = cluster_dets  # fallback degenerado (no debería ocurrir en operación normal)

        bboxes = [d["bbox"] for d in winning_dets]
        scores = [float(d.get("score", 0.0)) for d in winning_dets]

        mean_bbox, std_bbox = _aggregate_localisation(
            bboxes, scores=scores, fusion_mode=fusion_mode
        )
        loc_unc = float(np.mean(std_bbox))
        mean_score = float(np.mean(scores))

        contributing_sets = sorted(
            {d["prompt_set"] for d in winning_dets if "prompt_set" in d}
        )

        out = {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [round(v, 3) for v in mean_bbox],
            "score": round(mean_score, 6),
            "bbox_std": [round(v, 3) for v in std_bbox],
            "loc_uncertainty": round(loc_unc, 4),
            "class_uncertainty": round(cls_unc, 6),
            "class_name": best_class,
            "n_cluster": len(winning_dets),
            "n_cluster_total": len(cluster_idxs),
            "contributing_sets": contributing_sets,
            cls_field: cls_extra,
        }
        merged.append(out)

    return merged


def ensemble_all_images(
    all_detections: list[dict],
    prompt_to_class: dict[str, str],
    class_names: list[str],
    class_to_id: dict[str, int],
    iou_thresh: float = 0.3,
    mode: str = "score",
    label_key: str = "dino_label",
    cluster_method: str = "union_find",
    use_centroid_criterion: bool = False,
    centroid_mode: str = "or",
    image_sizes: dict[int, tuple[int, int]] | None = None,
    wbf_conf_type: str = "avg",
    fusion_mode: str = "mean",
) -> list[dict]:
    """
    Aplica :func:`ensemble_detections` a cada imagen en *all_detections*.
    Agrupa por image_id antes de procesar.

    image_sizes: dict {image_id: (W, H)} requerido cuando cluster_method='wbf'.
    """
    by_image: dict[int, list[dict]] = defaultdict(list)
    for det in all_detections:
        by_image[det["image_id"]].append(det)

    results: list[dict] = []
    for img_id in sorted(by_image):
        img_size = image_sizes.get(img_id) if image_sizes is not None else None
        results.extend(
            ensemble_detections(
                by_image[img_id],
                prompt_to_class=prompt_to_class,
                class_names=class_names,
                class_to_id=class_to_id,
                iou_thresh=iou_thresh,
                mode=mode,
                label_key=label_key,
                cluster_method=cluster_method,
                use_centroid_criterion=use_centroid_criterion,
                centroid_mode=centroid_mode,
                image_size=img_size,
                wbf_conf_type=wbf_conf_type,
                fusion_mode=fusion_mode,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Resolver de clase difuso — maneja los spans de etiqueta multi-token de DINO
# ---------------------------------------------------------------------------


def _build_phrase_fuzzy_index(
    prompt_sets_cfg: dict[str, dict[str, list[str]]],
    class_names: list[str],
) -> dict[str, list[tuple[int, frozenset[str]]]]:
    """
    Construye por adelantado el índice de búsqueda por conjunto para el matching
    de etiqueta→clase basado en contención.

    Devuelve {set_name: [(class_idx, phrase_tokens), ...]} donde phrase_tokens
    es un frozenset de tokens de palabras en minúscula de una frase de esa clase.
    Cada frase de un conjunto se almacena como entrada separada para que las clases
    multi-frase (p.ej. conjunto "original") tengan su propia oportunidad de coincidir.
    """
    index: dict[str, list[tuple[int, frozenset[str]]]] = {}
    for set_name, classes in prompt_sets_cfg.items():
        entries: list[tuple[int, frozenset[str]]] = []
        for class_name, phrases in (classes or {}).items():
            ci = class_names.index(class_name) if class_name in class_names else -1
            if ci < 0:
                continue
            for phrase in phrases or []:
                toks = frozenset(re.findall(r"[a-z]+", str(phrase).lower()))
                if toks:
                    entries.append((ci, toks))
        index[set_name] = entries
    return index


def _fuzzy_class_idx(
    raw_label: str,
    prompt_set: str,
    phrase_index: dict[str, list[tuple[int, frozenset[str]]]],
    threshold: float = 0.8,
) -> int:
    """
    Matching de raw_label contra las frases de prompt_set basado en contención.

    Para cada frase P de cada clase C:
        score = |tokens(raw_label) ∩ tokens(P)| / |tokens(P)|   (contención)

    Devuelve el class_idx del ganador único (mejor score ≥ threshold y
    estrictamente mejor que todas las demás clases). Devuelve -1 si:
      - ninguna clase alcanza el umbral, o
      - dos o más clases empatan en el mejor score (etiqueta ambigua).
    """
    if not raw_label:
        return -1
    label_toks = frozenset(re.findall(r"[a-z]+", raw_label.lower()))
    if not label_toks:
        return -1

    entries = phrase_index.get(prompt_set, [])
    if not entries:
        return -1

    # Mejor score de contención por class_idx
    best_by_class: dict[int, float] = {}
    for ci, phrase_toks in entries:
        if not phrase_toks:
            continue
        score = len(label_toks & phrase_toks) / len(phrase_toks)
        if score > best_by_class.get(ci, 0.0):
            best_by_class[ci] = score

    if not best_by_class:
        return -1

    top_score = max(best_by_class.values())
    if top_score < threshold:
        return -1

    winners = [ci for ci, s in best_by_class.items() if s >= top_score - 1e-9]
    if len(winners) != 1:
        return -1  # ambiguo — dos clases coinciden igualmente bien

    return winners[0]


# ---------------------------------------------------------------------------
# Cargador compacto — evita la expansión de RAM ~30x de los dicts de Python
# ---------------------------------------------------------------------------


def _fast_load_compact(
    path: Path,
    label_to_idx: dict[str, int],
    prompt_to_idx: dict[str, int],
    class_to_idx: dict[str, int],
    label_key: str,
    phrase_fuzzy_index: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Carga un array JSON grande de detecciones en arrays numpy compactos.

    Usa mmap + orjson (parser en Rust) en lugar de json.JSONDecoder de Python.
    Clave: los dicts de detección NO tienen {} anidados (bbox es una lista []),
    por lo que el primer '}' tras cualquier '{' es siempre el cierre del objeto.
    mmap.find() localiza los límites en C; orjson parsea cada slice en Rust;
    los slices de memoryview son zero-copy. ~100x más rápido que raw_decode.

    Devuelve (img_ids, bboxes, scores, lbl_ids, prm_ids, cls_ids, n_parsed).
    """
    n_est = max(64, int(path.stat().st_size / 420 * 1.1))
    img_ids = np.empty(n_est, dtype=np.int32)
    bboxes = np.empty((n_est, 4), dtype=np.float32)
    scores = np.empty(n_est, dtype=np.float32)
    lbl_ids = np.empty(n_est, dtype=np.int16)
    prm_ids = np.empty(n_est, dtype=np.int8)
    cls_ids = np.empty(n_est, dtype=np.int8)

    n = 0
    n_parsed = 0

    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(mm)
        pos = 0

        while pos < size:
            start = mm.find(b"{", pos)
            if start == -1:
                break
            end = mm.find(b"}", start + 1)
            if end == -1:
                break

            try:
                obj = orjson.loads(mm[start : end + 1])
            except Exception:
                pos = end + 1
                continue

            n_parsed += 1
            if n_parsed % 5_000_000 == 0:
                print(
                    f"  {n_parsed / 1e6:.0f}M parsed" f" ({end / size * 100:.1f}%) ...",
                    flush=True,
                )

            if n >= len(img_ids):
                new_size = len(img_ids) * 2

                def _g1(a: np.ndarray) -> np.ndarray:
                    b = np.empty(new_size, dtype=a.dtype)
                    b[: len(a)] = a
                    return b

                def _g2(a: np.ndarray) -> np.ndarray:
                    b = np.empty((new_size, a.shape[1]), dtype=a.dtype)
                    b[: len(a)] = a
                    return b

                img_ids = _g1(img_ids)
                bboxes = _g2(bboxes)
                scores = _g1(scores)
                lbl_ids = _g1(lbl_ids)
                prm_ids = _g1(prm_ids)
                cls_ids = _g1(cls_ids)

            raw_lbl = str(obj.get(label_key, "")).lower().strip()
            pset = obj.get("prompt_set", "")
            img_ids[n] = obj["image_id"]
            bboxes[n] = obj["bbox"]
            scores[n] = obj["score"]
            lbl_ids[n] = label_to_idx.get(raw_lbl, -1)
            prm_ids[n] = prompt_to_idx.get(pset, -1)
            # Prioridad: class_name precalculado; luego matching difuso; luego mapeo exacto
            precomp = obj.get("class_name", "")
            if precomp and precomp in class_to_idx:
                cls_ids[n] = class_to_idx[precomp]
            elif phrase_fuzzy_index is not None:
                cls_ids[n] = _fuzzy_class_idx(raw_lbl, pset, phrase_fuzzy_index)
            else:
                cls_ids[n] = class_to_idx.get(precomp, -1)
            n += 1
            pos = end + 1

        mm.close()

    return (
        img_ids[:n],
        bboxes[:n],
        scores[:n],
        lbl_ids[:n],
        prm_ids[:n],
        cls_ids[:n],
        n_parsed,
    )


# ---------------------------------------------------------------------------
# Interfaz de línea de comandos
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster multi-prompt detections into one ensemble result per object."
    )
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Raw detection_results.json from DINO or Rex-Omni inference",
    )
    parser.add_argument(
        "--ann_file",
        type=Path,
        required=True,
        help="COCO annotation file to resolve category IDs",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/prompts/ensemble_prompts.yaml")
    )
    parser.add_argument(
        "--mode",
        choices=["score", "vote", "borda"],
        default="score",
        help="'score' for DINO (mean score per class), 'vote' for Rex-Omni, 'borda' for rank-based Borda voting",
    )
    parser.add_argument(
        "--label_key",
        default="dino_label",
        help="Field in each detection dict holding the raw prompt text",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=None,
        help="Override IoU cluster threshold from config",
    )
    parser.add_argument(
        "--cluster_by_class",
        action="store_true",
        help="Cluster within each class separately (use with per-class inference)",
    )
    parser.add_argument(
        "--cluster_method",
        choices=["union_find", "wbf"],
        default="union_find",
        help="union_find (default, componentes conexas por IoU) o wbf (Weighted Boxes Fusion)",
    )
    parser.add_argument(
        "--use_centroid_criterion",
        action="store_true",
        help="ATSS: usar el criterio del centroide como condicion adicional al IoU",
    )
    parser.add_argument(
        "--centroid_mode",
        choices=["or", "and"],
        default="or",
        help=(
            "Como combinar IoU y criterio centroide cuando use_centroid_criterion=True. "
            "'and' (semantica ATSS original): IoU>=thresh AND centroide_dentro -- filtro adicional. "
            "'or': IoU>=thresh OR centroide_dentro -- mas permisivo, riesgo de megaclusters en escenas densas."
        ),
    )
    parser.add_argument(
        "--wbf_conf_type",
        choices=["avg", "max", "box_and_model_avg", "absent_model_aware_avg"],
        default="avg",
        help="Solo para cluster_method=wbf: como combinar confianzas",
    )
    parser.add_argument(
        "--fusion_mode",
        choices=["mean", "score_weighted"],
        default="mean",
        help=(
            "Como combinar bboxes dentro de un cluster: 'mean' (media simple, cada"
            " miembro contribuye igual) o 'score_weighted' (media ponderada por score,"
            " estilo WBF: bbox_fused = sum(s_i * bbox_i) / sum(s_i))."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_to_class, class_names, class_to_id, iou_thresh = load_ensemble_config(
        args.config, coco_ann_file=args.ann_file
    )
    if args.iou_thresh is not None:
        iou_thresh = args.iou_thresh

    print(
        f"Ensemble config: {len(prompt_to_class)} prompts → {len(class_names)} classes"
    )
    print(f"Classes: {', '.join(class_names)}")
    print(f"Clustering IoU threshold: {iou_thresh}")
    print(f"Mode: {args.mode}  |  label key: {args.label_key}")
    print(
        f"Cluster method: {args.cluster_method}"
        + (f"  conf_type={args.wbf_conf_type}" if args.cluster_method == "wbf" else "")
        + (f"  fusion_mode={args.fusion_mode}")
        + (f"  +centroid" if args.use_centroid_criterion else "")
    )

    # image_sizes solo necesario para WBF (normalizacion a [0,1])
    image_sizes: dict[int, tuple[int, int]] | None = None
    if args.cluster_method == "wbf":
        with open(args.ann_file, encoding="utf-8") as fh:
            ann_data = json.load(fh)
        image_sizes = {
            img["id"]: (int(img["width"]), int(img["height"]))
            for img in ann_data.get("images", [])
        }
        print(f"Loaded image sizes for {len(image_sizes)} images")

    # Construir mapas de vocabulario desde la config (conocidos antes del streaming)
    with open(args.config, encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)
    vocab_labels: list[str] = sorted(prompt_to_class.keys())
    vocab_classes: list[str] = class_names
    vocab_prompts: list[str] = sorted(raw_cfg.get("prompt_sets", {}).keys())
    label_to_idx = {l: i for i, l in enumerate(vocab_labels)}
    prompt_to_idx = {p: i for i, p in enumerate(vocab_prompts)}
    class_to_idx_enc = {c: i for i, c in enumerate(vocab_classes)}

    phrase_fuzzy_index = _build_phrase_fuzzy_index(
        raw_cfg.get("prompt_sets", {}), class_names
    )
    n_phrases = sum(len(v) for v in phrase_fuzzy_index.values())
    print(
        f"Fuzzy index: {n_phrases} phrase entries across {len(phrase_fuzzy_index)} sets"
    )

    print(f"\nLoading detections from {args.detections} ...")
    img_ids, bboxes, scores, lbl_ids, prm_ids, cls_ids, n_parsed = _fast_load_compact(
        args.detections,
        label_to_idx=label_to_idx,
        prompt_to_idx=prompt_to_idx,
        class_to_idx=class_to_idx_enc,
        label_key=args.label_key,
        phrase_fuzzy_index=phrase_fuzzy_index,
    )
    matched = int(np.sum(cls_ids >= 0))
    print(
        f"  Class assigned: {matched}/{len(cls_ids)} = {matched/len(cls_ids)*100:.1f}%"
    )
    n_dets = len(img_ids)
    n_images = len(np.unique(img_ids))
    print(
        f"  {n_parsed} detections loaded into compact arrays"
        f" ({n_dets} stored, {n_images} images)"
    )
    mem_mb = (
        img_ids.nbytes
        + bboxes.nbytes
        + scores.nbytes
        + lbl_ids.nbytes
        + prm_ids.nbytes
        + cls_ids.nbytes
    ) / 1e6
    print(f"  Array memory: {mem_mb:.0f} MB")

    # Ordenar por image_id una vez para poder extraer bloques contiguos por imagen
    print("Sorting by image_id ...")
    order = np.argsort(img_ids, kind="stable")
    img_ids = img_ids[order]
    bboxes = bboxes[order]
    scores = scores[order]
    lbl_ids = lbl_ids[order]
    prm_ids = prm_ids[order]
    cls_ids = cls_ids[order]

    unique_ids, first_occ = np.unique(img_ids, return_index=True)

    print("Clustering and ensembling ...")
    merged: list[dict] = []
    for i, uid in enumerate(unique_ids):
        start = int(first_occ[i])
        end = int(first_occ[i + 1]) if i + 1 < len(unique_ids) else n_dets

        # Reconstruir dicts ligeros solo para las detecciones de esta imagen
        img_dets = [
            {
                "image_id": int(uid),
                "category_id": -1,
                "bbox": bboxes[k].tolist(),
                "score": float(scores[k]),
                args.label_key: vocab_labels[lbl_ids[k]] if lbl_ids[k] >= 0 else "",
                "prompt_set": vocab_prompts[prm_ids[k]] if prm_ids[k] >= 0 else "",
                "class_name": vocab_classes[cls_ids[k]] if cls_ids[k] >= 0 else "",
            }
            for k in range(start, end)
        ]

        img_size = image_sizes.get(int(uid)) if image_sizes is not None else None

        if args.cluster_by_class:
            # Modo de inferencia por clase: agrupar dentro de cada clase por separado
            # para que los solapamientos de IoU entre clases no diluyan la señal de clase.
            by_class: dict[str, list[dict]] = defaultdict(list)
            for det in img_dets:
                by_class[det.get("class_name") or "__unknown__"].append(det)
            for cls_dets in by_class.values():
                merged.extend(
                    ensemble_detections(
                        cls_dets,
                        prompt_to_class=prompt_to_class,
                        class_names=class_names,
                        class_to_id=class_to_id,
                        iou_thresh=iou_thresh,
                        mode=args.mode,
                        label_key=args.label_key,
                        cluster_method=args.cluster_method,
                        use_centroid_criterion=args.use_centroid_criterion,
                        centroid_mode=args.centroid_mode,
                        image_size=img_size,
                        wbf_conf_type=args.wbf_conf_type,
                        fusion_mode=args.fusion_mode,
                    )
                )
        else:
            merged.extend(
                ensemble_detections(
                    img_dets,
                    prompt_to_class=prompt_to_class,
                    class_names=class_names,
                    class_to_id=class_to_id,
                    iou_thresh=iou_thresh,
                    mode=args.mode,
                    label_key=args.label_key,
                    cluster_method=args.cluster_method,
                    use_centroid_criterion=args.use_centroid_criterion,
                    centroid_mode=args.centroid_mode,
                    image_size=img_size,
                    wbf_conf_type=args.wbf_conf_type,
                    fusion_mode=args.fusion_mode,
                )
            )
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(unique_ids)} images ...", flush=True)

    # Estadísticas resumen
    print(
        f"\n  {n_dets} raw detections → {len(merged)} merged clusters"
        f" across {n_images} images"
    )
    cluster_sizes = [d["n_cluster"] for d in merged]
    if cluster_sizes:
        print(
            f"  Cluster size: mean={np.mean(cluster_sizes):.2f}"
            f"  max={max(cluster_sizes)}"
            f"  singletons={sum(1 for s in cluster_sizes if s == 1)}"
        )
    mean_loc_unc = np.mean([d["loc_uncertainty"] for d in merged]) if merged else 0.0
    mean_cls_unc = np.mean([d["class_uncertainty"] for d in merged]) if merged else 0.0
    print(f"  Mean loc_uncertainty : {mean_loc_unc:.4f}")
    print(f"  Mean class_uncertainty: {mean_cls_unc:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"\nSaved to {args.output}")

    stats = {
        "model": args.label_key.replace("_label", ""),
        "raw_detections": n_dets,
        "merged_clusters": len(merged),
        "images_with_detections": n_images,
        "cluster_size_mean": round(float(np.mean(cluster_sizes)), 4)
        if cluster_sizes
        else 0,
        "cluster_size_max": int(max(cluster_sizes)) if cluster_sizes else 0,
        "singletons": int(sum(1 for s in cluster_sizes if s == 1)),
        "loc_uncertainty_mean": round(mean_loc_unc, 4),
        "class_uncertainty_mean": round(mean_cls_unc, 4),
    }
    stats_path = args.output.parent / "ensemble_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Stats  → {stats_path}")


if __name__ == "__main__":
    main()
