#!/usr/bin/env python3
"""
Compara tres estrategias de ensemble multi-prompt:
  1. IoU Clustering (actual)  — componentes conexas + media de bbox
  2. WBF                      — Weighted Boxes Fusion (ensemble-boxes)
  3. Quorum (K>=2)            — solo dets detectadas por >=2 prompt sets

Condiciones: score>=0.1, iou_thr=0.4, los 3 modelos.
"""
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

BENCH = Path("results/benchmark")
GT_FILE = BENCH / "benchmark_100_gt.json"
PROMPTS_CFG = "configs/prompts/ensemble_prompts.yaml"
IOT_THRESH = 0.4
SCORE_MIN = 0.1
QUORUM_K = 2

MODELS = {
    "dino": {"label_key": "dino_label", "mode": "score", "raw_dir": "raw_005"},
    "detic": {"label_key": "detic_label", "mode": "score", "raw_dir": "raw_005"},
    "gdsam2": {"label_key": "gdsam2_label", "mode": "score", "raw_dir": "raw"},
}
PROMPT_SETS = [
    "simple",
    "aerial_compact",
    "satellite_verbose",
    "scene_context",
    "original",
]

with open(GT_FILE) as f:
    gt = json.load(f)
img_dims = {img["id"]: (img["width"], img["height"]) for img in gt["images"]}
cat_names = {c["id"]: c["name"] for c in gt["categories"]}
name_to_id = {c["name"]: c["id"] for c in gt["categories"]}

# Mapeo prompt→clase desde config
import yaml

with open(PROMPTS_CFG) as f:
    cfg = yaml.safe_load(f)
prompt_to_class: dict[str, str] = {}
for set_classes in cfg.get("prompt_sets", {}).values():
    for cls, phrases in (set_classes or {}).items():
        for p in phrases or []:
            prompt_to_class[str(p).lower().strip()] = cls
class_names = sorted({v for v in prompt_to_class.values()})


def map_class(det: dict, label_key: str) -> str:
    if det.get("class_name") and det["class_name"] in name_to_id:
        return det["class_name"]
    return prompt_to_class.get(str(det.get(label_key, "")).lower().strip(), "")


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


def xywh2xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


# ─── Método 1: IoU Clustering (actual, vía subprocess) ───────────────────────
def method_clustering(all_dets, label_key, mode, tmpdir):
    raw_path = tmpdir / "raw.json"
    agg = tmpdir / "agg.json"
    with open(raw_path, "w") as f:
        json.dump(all_dets, f)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference.multi_prompt_ensemble",
            "--detections",
            str(raw_path),
            "--ann_file",
            str(GT_FILE),
            "--config",
            PROMPTS_CFG,
            "--mode",
            mode,
            "--label_key",
            label_key,
            "--iou_thresh",
            str(IOT_THRESH),
            "--output",
            str(agg),
        ],
        capture_output=True,
    )
    try:
        with open(agg) as f:
            return json.load(f)
    except:
        return []


# ─── Método 2: WBF ───────────────────────────────────────────────────────────
def method_wbf(
    dets_by_set: dict[str, list[dict]], label_key: str, img_id: int
) -> list[dict]:
    iw, ih = img_dims.get(img_id, (1, 1))
    if iw == 0 or ih == 0:
        return []

    boxes_list, scores_list, labels_list, set_dets_list = [], [], [], []

    for ps, dets in dets_by_set.items():
        if not dets:
            continue
        boxes, scores, labels = [], [], []
        for d in dets:
            x, y, w, h = d["bbox"]
            # Normalizar a [0,1], clamp
            x1 = max(0.0, min(1.0, x / iw))
            y1 = max(0.0, min(1.0, y / ih))
            x2 = max(0.0, min(1.0, (x + w) / iw))
            y2 = max(0.0, min(1.0, (y + h) / ih))
            if x2 <= x1 or y2 <= y1:
                continue
            cls = map_class(d, label_key)
            cls_idx = class_names.index(cls) if cls in class_names else 0
            boxes.append([x1, y1, x2, y2])
            scores.append(float(d.get("score", 0.0)))
            labels.append(cls_idx)
        if boxes:
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
            set_dets_list.append(dets)

    if not boxes_list:
        return []

    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=IOT_THRESH,
        skip_box_thr=0.0,
        conf_type="avg",
    )

    results = []
    for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
        x1, y1, x2, y2 = box
        cls_idx = int(round(label))
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else ""
        results.append(
            {
                "image_id": img_id,
                "category_id": name_to_id.get(cls_name, -1),
                "bbox": [
                    round(x1 * iw, 3),
                    round(y1 * ih, 3),
                    round((x2 - x1) * iw, 3),
                    round((y2 - y1) * ih, 3),
                ],
                "score": round(float(score), 6),
                "class_name": cls_name,
            }
        )
    return results


# ─── Método 3: Quorum K>=2 ───────────────────────────────────────────────────
def method_quorum(
    dets_by_set: dict[str, list[dict]], label_key: str, img_id: int
) -> list[dict]:
    """Mantiene una det si IoU>=threshold con al menos una det de OTRO set."""
    all_sets = list(dets_by_set.keys())
    results = []

    for ps_i, dets_i in dets_by_set.items():
        other_dets = [
            d for ps_j, dets_j in dets_by_set.items() if ps_j != ps_i for d in dets_j
        ]
        if not other_dets:
            continue
        other_boxes = [xywh2xyxy(d["bbox"]) for d in other_dets]

        for d in dets_i:
            box_i = xywh2xyxy(d["bbox"])
            has_support = any(iou_xyxy(box_i, ob) >= IOT_THRESH for ob in other_boxes)
            if has_support:
                cls = map_class(d, label_key)
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": name_to_id.get(cls, -1),
                        "bbox": d["bbox"],
                        "score": float(d.get("score", 0.0)),
                        "class_name": cls,
                    }
                )

    # Dedup: si dos dets del quórum se solapan entre sí, quedarse con la de mayor score
    if not results:
        return []
    results.sort(key=lambda x: -x["score"])
    kept = []
    for d in results:
        box = xywh2xyxy(d["bbox"])
        if not any(iou_xyxy(box, xywh2xyxy(k["bbox"])) >= IOT_THRESH for k in kept):
            kept.append(d)
    return kept


def evaluate(dets: list[dict], mode: str, tmpdir: Path) -> dict:
    det_path = tmpdir / f"dets_{mode}.json"
    eval_dir = tmpdir / f"eval_{mode}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(det_path, "w") as f:
        json.dump(dets, f)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference.ov_coco_eval",
            "--ann_file",
            str(GT_FILE),
            "--detection_results",
            str(det_path),
            "--output_folder",
            str(eval_dir),
            "--mode",
            mode,
        ],
        capture_output=True,
    )
    try:
        with open(eval_dir / "metrics.json") as f:
            return json.load(f)
    except:
        return {}


def fmt(v):
    return f"{v:.4f}" if v is not None else "  —   "


# ─── Main ────────────────────────────────────────────────────────────────────
print(f"score>={SCORE_MIN}, iou_thr={IOT_THRESH}, quorum K>={QUORUM_K}\n")

for model, cfg_m in MODELS.items():
    label_key = cfg_m["label_key"]

    # Cargar y filtrar checkpoints
    dets_by_set: dict[str, list[dict]] = {}
    for ps in PROMPT_SETS:
        ckpt = BENCH / model / cfg_m["raw_dir"] / f"_ckpt_{ps}.json"
        with open(ckpt) as f:
            raw = json.load(f)
        dets_by_set[ps] = [d for d in raw if d.get("score", 1.0) >= SCORE_MIN]

    all_filtered = [d for dets in dets_by_set.values() for d in dets]

    # Agrupar WBF y quórum por imagen
    by_img: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ps, dets in dets_by_set.items():
        for d in dets:
            by_img[d["image_id"]][ps].append(d)

    wbf_dets, quorum_dets = [], []
    for img_id, sets_dets in by_img.items():
        wbf_dets.extend(method_wbf(sets_dets, label_key, img_id))
        quorum_dets.extend(method_quorum(sets_dets, label_key, img_id))

    print(f"{'═'*72}")
    print(
        f"  {model.upper()}  |  n_filt={len(all_filtered)}  n_wbf={len(wbf_dets)}  n_quorum={len(quorum_dets)}"
    )
    print(
        f"  {'método':<18}  {'n_dets':>6}  {'AP50_aw':>8}  {'AR500_aw':>9}  {'AP50_ag':>8}  {'AR500_ag':>9}"
    )
    print(f"  {'-'*18}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*9}")

    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        for d in ["clust", "wbf", "quorum"]:
            (td / d).mkdir()

        # 1. Clustering (actual)
        clust = method_clustering(all_filtered, label_key, cfg_m["mode"], td / "clust")
        aw_c = evaluate(clust, "aware", td / "clust")
        ag_c = evaluate(clust, "agnostic", td / "clust")

        # 2. WBF
        aw_w = evaluate(wbf_dets, "aware", td / "wbf")
        ag_w = evaluate(wbf_dets, "agnostic", td / "wbf")

        # 3. Quorum
        aw_q = evaluate(quorum_dets, "aware", td / "quorum")
        ag_q = evaluate(quorum_dets, "agnostic", td / "quorum")

    for label, aw, ag, n in [
        ("1. Clustering", aw_c, ag_c, len(clust)),
        ("2. WBF", aw_w, ag_w, len(wbf_dets)),
        (f"3. Quorum K>={QUORUM_K}", aw_q, ag_q, len(quorum_dets)),
    ]:
        print(
            f"  {label:<18}  {n:>6}  {fmt(aw.get('AP50')):>8}  {fmt(aw.get('AR_500')):>9}  {fmt(ag.get('AP50')):>8}  {fmt(ag.get('AR_500')):>9}"
        )
    print()
