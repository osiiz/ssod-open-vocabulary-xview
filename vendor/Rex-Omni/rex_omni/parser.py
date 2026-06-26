#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Output parsing utilities for Rex Omni
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_prediction(
    text: str, w: int, h: int, task_type: str = "detection"
) -> Dict[str, List]:
    """
    Parse model output text to extract category-wise predictions.

    Args:
        text: Model output text
        w: Image width
        h: Image height
        task_type: Type of task ("detection", "keypoint", etc.)

    Returns:
        Dictionary with category as key and list of predictions as value
    """
    if task_type == "keypoint":
        return parse_keypoint_prediction(text, w, h)
    else:
        return parse_standard_prediction(text, w, h)


def parse_standard_prediction(text: str, w: int, h: int) -> Dict[str, List]:
    """
    Parse standard prediction output for detection, pointing, etc.

    Input format example:
    "<|object_ref_start|>person<|object_ref_end|><|box_start|><0><35><980><987>, <646><0><999><940><|box_end|>"

    Returns:
    {
        'category1': [{"type": "box/point/polygon", "coords": [...]}],
        'category2': [{"type": "box/point/polygon", "coords": [...]}],
        ...
    }
    """
    result = {}

    # Remove the end marker if present
    text = text.split("<|im_end|>")[0]
    if not text.endswith("<|box_end|>"):
        text = text + "<|box_end|>"

    # Use regex to find all object references and coordinate pairs
    pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>"
    matches = re.findall(pattern, text)

    for category, coords_text in matches:
        category = category.strip()

        # Find all coordinate tokens in the format <{number}>
        coord_pattern = r"<(\d+)>"
        coord_matches = re.findall(coord_pattern, coords_text)

        annotations = []
        # Split by comma to handle multiple coordinates for the same phrase
        coord_strings = coords_text.split(",")

        for coord_str in coord_strings:
            coord_nums = re.findall(coord_pattern, coord_str.strip())

            if len(coord_nums) == 2:
                # Point: <{x}><{y}>
                try:
                    x_bin = int(coord_nums[0])
                    y_bin = int(coord_nums[1])

                    # Convert from bins [0, 999] to absolute coordinates
                    x = (x_bin / 999.0) * w
                    y = (y_bin / 999.0) * h

                    annotations.append({"type": "point", "coords": [x, y]})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing point coordinates: {e}")
                    continue

            elif len(coord_nums) == 4:
                # Bounding box: <{x0}><{y0}><{x1}><{y1}>
                try:
                    x0_bin = int(coord_nums[0])
                    y0_bin = int(coord_nums[1])
                    x1_bin = int(coord_nums[2])
                    y1_bin = int(coord_nums[3])

                    # Convert from bins [0, 999] to absolute coordinates
                    x0 = (x0_bin / 999.0) * w
                    y0 = (y0_bin / 999.0) * h
                    x1 = (x1_bin / 999.0) * w
                    y1 = (y1_bin / 999.0) * h

                    annotations.append({"type": "box", "coords": [x0, y0, x1, y1]})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing box coordinates: {e}")
                    continue

            elif len(coord_nums) > 4 and len(coord_nums) % 2 == 0:
                # Polygon: <{x0}><{y0}><{x1}><{y1}>...
                try:
                    polygon_coords = []
                    for i in range(0, len(coord_nums), 2):
                        x_bin = int(coord_nums[i])
                        y_bin = int(coord_nums[i + 1])

                        # Convert from bins [0, 999] to absolute coordinates
                        x = (x_bin / 999.0) * w
                        y = (y_bin / 999.0) * h

                        polygon_coords.append([x, y])

                    annotations.append({"type": "polygon", "coords": polygon_coords})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing polygon coordinates: {e}")
                    continue

        if category not in result:
            result[category] = []
        result[category].extend(annotations)

    return result


_COORD_TOKEN_CACHE: Dict[int, dict] = {}


def _build_coord_id_to_bin(tokenizer) -> Dict[int, int]:
    """Mapeo {token_id: bin_number} para todos los tokens cuyo decode es '<N>'.

    Se construye decodificando todos los IDs del vocab y filtrando los que
    matchean el patron <NUM>. Es mas robusto que convert_tokens_to_ids cuando
    hay desfases entre el tokenizer y el vocab del modelo.
    """
    cache_key = id(tokenizer)
    if cache_key in _COORD_TOKEN_CACHE:
        return _COORD_TOKEN_CACHE[cache_key]

    coord_map: Dict[int, int] = {}
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for token, tid in vocab.items():
        if token.startswith("<") and token.endswith(">") and token[1:-1].isdigit():
            try:
                bin_num = int(token[1:-1])
                if 0 <= bin_num <= 999:
                    coord_map[tid] = bin_num
            except ValueError:
                pass
    _COORD_TOKEN_CACHE[cache_key] = coord_map
    return coord_map


def parse_prediction_with_scores(
    gen_ids,
    scores: Tuple,
    w: int,
    h: int,
    tokenizer,
) -> Dict[str, List]:
    """Parser token-level que extrae detecciones Y scores desde los logits.

    Args:
        gen_ids: secuencia de token IDs generados (lista o tensor 1D, ya sin el prompt)
        scores: tuple de tensores de logits (uno por paso de generacion); cada
                tensor es (vocab_size,) para una imagen
        w, h: dimensiones de la imagen en pixeles
        tokenizer: tokenizer del modelo

    Score por deteccion:
        score = cls_presence * geom_mean(P_coord_x0, P_coord_y0, P_coord_x1, P_coord_y1)

      donde:
        cls_presence = 1 - P('None') en el paso justo despues de '<|box_start|>'
                       (probabilidad de que el modelo decidiera generar coords
                        en lugar de declarar la clase ausente)
        P_coord_i    = softmax(logits)[token_generado] para el i-esimo coord token

    Devuelve: {cls_name: [{type, coords, score, cls_presence, loc_score}, ...]}
    """
    import torch

    # Resolver IDs de tokens estructurales
    special = {
        "obj_start": tokenizer.convert_tokens_to_ids("<|object_ref_start|>"),
        "obj_end": tokenizer.convert_tokens_to_ids("<|object_ref_end|>"),
        "box_start": tokenizer.convert_tokens_to_ids("<|box_start|>"),
        "box_end": tokenizer.convert_tokens_to_ids("<|box_end|>"),
    }
    none_ids = tokenizer.encode("None", add_special_tokens=False)
    none_id = none_ids[0] if none_ids else -1
    coord_id_to_bin = _build_coord_id_to_bin(tokenizer)
    coord_ids = set(coord_id_to_bin.keys())

    # Normalizar gen_ids a lista de ints
    if hasattr(gen_ids, "tolist"):
        gen_ids = gen_ids.tolist()
    gen_ids = list(gen_ids)
    n_tokens = len(gen_ids)
    n_steps = len(scores)

    def _prob_at(step_idx: int, token_id: int) -> float:
        """Softmax y devuelve la probabilidad del token_id en ese paso."""
        if step_idx >= n_steps or step_idx < 0:
            return 0.0
        logits = scores[step_idx]
        if logits.dim() == 2:
            logits = logits[0]
        probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[token_id])

    result: Dict[str, List] = {}
    i = 0
    while i < n_tokens:
        if gen_ids[i] != special["obj_start"]:
            i += 1
            continue

        # Recoger nombre de clase entre obj_start y obj_end
        i += 1
        cls_tok_ids = []
        while i < n_tokens and gen_ids[i] != special["obj_end"]:
            cls_tok_ids.append(gen_ids[i])
            i += 1
        cls_name = tokenizer.decode(cls_tok_ids, skip_special_tokens=False).strip()
        i += 1  # saltar obj_end

        # Avanzar hasta box_start
        while i < n_tokens and gen_ids[i] != special["box_start"]:
            i += 1
        if i >= n_tokens:
            break

        # i apunta a box_start. El token en i+1 es la BIFURCACION:
        #   - 'None' -> sin detecciones
        #   - coord token -> hay deteccion(es)
        i += 1  # mover a la posicion despues de box_start

        # cls_presence: prob de que el siguiente token sea cualquier cosa
        # menos 'None'. Equivalente a 1 - P(None) en este paso.
        cls_presence = 1.0 - _prob_at(i, none_id)

        annotations: List[dict] = []
        coord_buf: List[Tuple[int, float]] = []  # (token_id, prob) acumulados

        while i < n_tokens and gen_ids[i] != special["box_end"]:
            tid = gen_ids[i]
            if tid in coord_ids:
                p = _prob_at(i, tid)
                coord_buf.append((tid, p))
                if len(coord_buf) == 4:
                    # Bbox completo
                    tids = [c[0] for c in coord_buf]
                    probs = [c[1] for c in coord_buf]
                    try:
                        bins = [coord_id_to_bin[t] for t in tids]
                        x0 = bins[0] / 999.0 * w
                        y0 = bins[1] / 999.0 * h
                        x1 = bins[2] / 999.0 * w
                        y1 = bins[3] / 999.0 * h
                        # Geom mean de las 4 probs
                        loc_score = (probs[0] * probs[1] * probs[2] * probs[3]) ** 0.25
                        score = cls_presence * loc_score
                        annotations.append(
                            {
                                "type": "box",
                                "coords": [x0, y0, x1, y1],
                                "score": float(score),
                                "cls_presence": float(cls_presence),
                                "loc_score": float(loc_score),
                            }
                        )
                    except (ValueError, IndexError, KeyError):
                        pass
                    coord_buf = []
            i += 1

        if cls_name not in result:
            result[cls_name] = []
        result[cls_name].extend(annotations)

        if i < n_tokens and gen_ids[i] == special["box_end"]:
            i += 1  # saltar box_end

    return result


def parse_keypoint_prediction(text: str, w: int, h: int) -> Dict[str, List]:
    """
    Parse keypoint task JSON output to extract bbox and keypoints.

    Expected format:
    ```json
    {
        "person1": {
            "bbox": " <1> <36> <987> <984> ",
            "keypoints": {
                "nose": " <540> <351> ",
                "left eye": " <559> <316> ",
                "right eye": "unvisible",
                ...
            }
        },
        ...
    }
    ```

    Returns:
    Dict with category as key and list of keypoint instances as value
    """
    # Extract JSON content from markdown code blocks
    json_pattern = r"```json\s*(.*?)\s*```"
    json_matches = re.findall(json_pattern, text, re.DOTALL)

    if not json_matches:
        # Try to find JSON without markdown
        try:
            # Look for JSON-like structure
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
            else:
                return {}
        except:
            return {}
    else:
        json_str = json_matches[0]

    try:
        keypoint_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing keypoint JSON: {e}")
        return {}

    result = {}

    for instance_id, instance_data in keypoint_data.items():
        if "bbox" not in instance_data or "keypoints" not in instance_data:
            continue

        bbox = instance_data["bbox"]
        keypoints = instance_data["keypoints"]

        # Convert bbox coordinates from bins [0, 999] to absolute coordinates
        if isinstance(bbox, str) and bbox.strip():
            # Parse box tokens from string format like " <1> <36> <987> <984> "
            coord_pattern = r"<(\d+)>"
            coord_matches = re.findall(coord_pattern, bbox)

            if len(coord_matches) == 4:
                try:
                    x0_bin, y0_bin, x1_bin, y1_bin = [
                        int(match) for match in coord_matches
                    ]
                    x0 = (x0_bin / 999.0) * w
                    y0 = (y0_bin / 999.0) * h
                    x1 = (x1_bin / 999.0) * w
                    y1 = (y1_bin / 999.0) * h
                    converted_bbox = [x0, y0, x1, y1]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing bbox coordinates: {e}")
                    continue
            else:
                print(
                    f"Invalid bbox format for {instance_id}: expected 4 coordinates, got {len(coord_matches)}"
                )
                continue
        else:
            print(f"Invalid bbox format for {instance_id}: {bbox}")
            continue

        # Convert keypoint coordinates from bins to absolute coordinates
        converted_keypoints = {}
        for kp_name, kp_coords in keypoints.items():
            if kp_coords == "unvisible" or kp_coords is None:
                converted_keypoints[kp_name] = "unvisible"
            elif isinstance(kp_coords, str) and kp_coords.strip():
                # Parse box tokens from string format like " <540> <351> "
                coord_pattern = r"<(\d+)>"
                coord_matches = re.findall(coord_pattern, kp_coords)

                if len(coord_matches) == 2:
                    try:
                        x_bin, y_bin = [int(match) for match in coord_matches]
                        x = (x_bin / 999.0) * w
                        y = (y_bin / 999.0) * h
                        converted_keypoints[kp_name] = [x, y]
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing keypoint coordinates for {kp_name}: {e}")
                        converted_keypoints[kp_name] = "unvisible"
                else:
                    print(
                        f"Invalid keypoint format for {kp_name}: expected 2 coordinates, got {len(coord_matches)}"
                    )
                    converted_keypoints[kp_name] = "unvisible"
            else:
                converted_keypoints[kp_name] = "unvisible"

        # Group by category (assuming instance_id contains category info)
        # Try to extract category from instance_id (e.g., "person1" -> "person")
        category = "keypoint_instance"
        if instance_id:
            # Remove numbers from instance_id to get category
            category_match = re.match(r"^([a-zA-Z_]+)", instance_id)
            if category_match:
                category = category_match.group(1)

        if category not in result:
            result[category] = []

        result[category].append(
            {
                "type": "keypoint",
                "bbox": converted_bbox,
                "keypoints": converted_keypoints,
                "instance_id": instance_id,
            }
        )

    return result


def convert_boxes_to_normalized_bins(
    boxes: List[List[float]], ori_width: int, ori_height: int
) -> List[str]:
    """Convert boxes from absolute coordinates to normalized bins (0-999) and map to words."""
    word_mapped_boxes = []
    for box in boxes:
        x0, y0, x1, y1 = box

        # Normalize coordinates to [0, 1] range
        x0_norm = max(0.0, min(1.0, x0 / ori_width))
        x1_norm = max(0.0, min(1.0, x1 / ori_width))
        y0_norm = max(0.0, min(1.0, y0 / ori_height))
        y1_norm = max(0.0, min(1.0, y1 / ori_height))

        # Convert to bins [0, 999]
        x0_bin = max(0, min(999, int(x0_norm * 999)))
        y0_bin = max(0, min(999, int(y0_norm * 999)))
        x1_bin = max(0, min(999, int(x1_norm * 999)))
        y1_bin = max(0, min(999, int(y1_norm * 999)))

        # Map to words
        word_mapped_box = "".join(
            [
                f"<{x0_bin}>",
                f"<{y0_bin}>",
                f"<{x1_bin}>",
                f"<{y1_bin}>",
            ]
        )
        word_mapped_boxes.append(word_mapped_box)

    return word_mapped_boxes
