#!/usr/bin/env python3
"""
Filtra detecciones que ocupan más del MAX_AREA_RATIO de la imagen
y re-evalúa el ensemble para DINO, Detic y GDSAM2.
Compara con la línea base (sin filtro de área).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path("results/benchmark")
GT_FILE = BENCH / "benchmark_100_gt.json"
PROMPTS_CFG = "configs/prompts/ensemble_prompts.yaml"
IOT_THRESH = 0.4
SCORE_MIN = 0.1
MAX_AREA_RATIO = 0.9

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

# Construir lookup de dimensiones de imagen desde el GT
with open(GT_FILE) as f:
    gt = json.load(f)
img_dims = {img["id"]: (img["width"], img["height"]) for img in gt["images"]}


def area_ratio(det: dict) -> float:
    x, y, w, h = det["bbox"]
    iw, ih = img_dims.get(det["image_id"], (1, 1))
    return (w * h) / (iw * ih)


def filter_dets(dets: list) -> tuple[list, int]:
    filtered, n_removed = [], 0
    for d in dets:
        if d.get("score", 1.0) >= SCORE_MIN and area_ratio(d) <= MAX_AREA_RATIO:
            filtered.append(d)
        else:
            n_removed += 1
    return filtered, n_removed


def aggregate_and_eval(dets: list, label_key: str, mode: str, tmpdir: Path):
    raw_path = tmpdir / "raw.json"
    agg = tmpdir / "agg.json"
    eval_aw = tmpdir / "eval_aware"
    eval_ag = tmpdir / "eval_agnostic"
    eval_aw.mkdir(parents=True)
    eval_ag.mkdir(parents=True)

    with open(raw_path, "w") as f:
        json.dump(dets, f)

    r = subprocess.run(
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
        text=True,
    )
    if r.returncode != 0:
        print(f"  AGG error: {r.stderr[-200:]}", file=sys.stderr)
        return None, None, 0

    n_agg = len(json.load(open(agg)))

    for ev_mode, ev_dir in [("aware", eval_aw), ("agnostic", eval_ag)]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.inference.ov_coco_eval",
                "--ann_file",
                str(GT_FILE),
                "--detection_results",
                str(agg),
                "--output_folder",
                str(ev_dir),
                "--mode",
                ev_mode,
            ],
            capture_output=True,
            text=True,
        )

    def load(p):
        try:
            with open(p) as f:
                return json.load(f)
        except:
            return {}

    return load(eval_aw / "metrics.json"), load(eval_ag / "metrics.json"), n_agg


def fmt(v):
    return f"{v:.4f}" if v is not None else "  —   "


print(
    f"Filtro: score>={SCORE_MIN}, área<={MAX_AREA_RATIO*100:.0f}% de la imagen, iou_cluster={IOT_THRESH}"
)
print()

results = {}

for model, cfg in MODELS.items():
    # Cargar todos los checkpoints y filtrar
    all_raw, all_filtered = [], []
    n_score_removed = 0
    n_area_removed = 0

    for ps in PROMPT_SETS:
        ckpt = BENCH / model / cfg["raw_dir"] / f"_ckpt_{ps}.json"
        with open(ckpt) as f:
            dets = json.load(f)
        all_raw.extend(dets)
        for d in dets:
            score_ok = d.get("score", 1.0) >= SCORE_MIN
            area_ok = area_ratio(d) <= MAX_AREA_RATIO
            if score_ok and area_ok:
                all_filtered.append(d)
            elif score_ok and not area_ok:
                n_area_removed += 1
            else:
                n_score_removed += 1

    print(f"{'═'*70}")
    print(
        f"  {model.upper()}  raw={len(all_raw)}  score_removed={n_score_removed}  área_removed={n_area_removed}  final={len(all_filtered)}"
    )

    # Sin filtro de área (solo score)
    all_score_only = [d for d in all_raw if d.get("score", 1.0) >= SCORE_MIN]
    with tempfile.TemporaryDirectory() as td:
        aw_base, ag_base, n_agg_base = aggregate_and_eval(
            all_score_only, cfg["label_key"], cfg["mode"], Path(td)
        )

    # Con filtro de área
    with tempfile.TemporaryDirectory() as td:
        aw_filt, ag_filt, n_agg_filt = aggregate_and_eval(
            all_filtered, cfg["label_key"], cfg["mode"], Path(td)
        )

    results[model] = dict(
        aw_base=aw_base,
        ag_base=ag_base,
        aw_filt=aw_filt,
        ag_filt=ag_filt,
        n_area=n_area_removed,
    )

    print(
        f"  {'':20}  {'n_agg':>6}  {'AP50_aw':>8}  {'AR500_aw':>9}  {'AP50_ag':>8}  {'AR500_ag':>9}"
    )
    print(
        f"  {'sin filtro área':<20}  {n_agg_base:>6}  {fmt(aw_base.get('AP50')):>8}  {fmt(aw_base.get('AR_500')):>9}  {fmt(ag_base.get('AP50')):>8}  {fmt(ag_base.get('AR_500')):>9}"
    )
    print(
        f"  {'con filtro >=90%':<20}  {n_agg_filt:>6}  {fmt(aw_filt.get('AP50')):>8}  {fmt(aw_filt.get('AR_500')):>9}  {fmt(ag_filt.get('AP50')):>8}  {fmt(ag_filt.get('AR_500')):>9}"
    )

    def delta(a, b, k):
        va, vb = (a or {}).get(k), (b or {}).get(k)
        if va and vb:
            return f"{(vb-va)/va*100:+.1f}%"
        return "—"

    print(
        f"  {'Δ':<20}  {'':>6}  {delta(aw_base,aw_filt,'AP50'):>8}  {delta(aw_base,aw_filt,'AR_500'):>9}  {delta(ag_base,ag_filt,'AP50'):>8}  {delta(ag_base,ag_filt,'AR_500'):>9}"
    )
    print()
