#!/usr/bin/env python3
"""
Sweep IoU clustering thresholds + evaluate each prompt set individually.
Outputs two markdown tables.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path("results/benchmark")
GT = BENCH / "benchmark_100_gt.json"
PROMPTS_CFG = "configs/prompts/ensemble_prompts.yaml"

MODELS = {
    "dino": {"label_key": "dino_label", "mode": "score"},
    "detic": {"label_key": "detic_label", "mode": "score"},
    "gdsam2": {"label_key": "gdsam2_label", "mode": "score"},
}

PROMPT_SETS = [
    "simple",
    "aerial_compact",
    "satellite_verbose",
    "scene_context",
    "original",
]
IOT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def aggregate_and_eval(
    raw_json: Path, label_key: str, mode: str, iou_thresh: float, tmpdir: Path
) -> dict:
    agg = tmpdir / "agg.json"
    eval_dir = tmpdir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference.multi_prompt_ensemble",
            "--detections",
            str(raw_json),
            "--ann_file",
            str(GT),
            "--config",
            PROMPTS_CFG,
            "--mode",
            mode,
            "--label_key",
            label_key,
            "--iou_thresh",
            str(iou_thresh),
            "--output",
            str(agg),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"  [AGG ERROR] {r.stderr[-300:]}", file=sys.stderr)
        return {}

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference.ov_coco_eval",
            "--ann_file",
            str(GT),
            "--detection_results",
            str(agg),
            "--output_folder",
            str(eval_dir),
            "--mode",
            "aware",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"  [EVAL ERROR] {r.stderr[-300:]}", file=sys.stderr)
        return {}

    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def fmt(v):
    return f"{v:.4f}" if v else "—"


# ─── 1. IoU threshold sweep ───────────────────────────────────────────────────
print("\n# IoU clustering threshold sweep (AP50 aware)\n")
print(f"{'Model':<10}", end="")
for t in IOT_THRESHOLDS:
    print(f"  iou={t:.1f}", end="")
print()
print("-" * (10 + len(IOT_THRESHOLDS) * 9))

iou_results = {}

for model, cfg in MODELS.items():
    raw = BENCH / model / "raw" / "detection_results.json"
    print(f"{model:<10}", end="", flush=True)
    iou_results[model] = {}
    for iou_t in IOT_THRESHOLDS:
        with tempfile.TemporaryDirectory() as td:
            m = aggregate_and_eval(raw, cfg["label_key"], cfg["mode"], iou_t, Path(td))
            ap50 = m.get("AP50", None)
            iou_results[model][iou_t] = ap50
            print(f"  {fmt(ap50)}", end="", flush=True)
    print()

# ─── 2. Per-prompt-set evaluation ────────────────────────────────────────────
print("\n\n# Per-prompt-set AP50 aware (iou_thresh=0.3)\n")
header = f"{'Model':<10}  {'simple':>10}  {'aerial':>10}  {'sat_verb':>10}  {'scene':>10}  {'original':>10}"
print(header)
print("-" * len(header))

per_prompt_results = {}

for model, cfg in MODELS.items():
    row = f"{model:<10}"
    per_prompt_results[model] = {}
    for ps in PROMPT_SETS:
        ckpt = BENCH / model / "raw" / f"_ckpt_{ps}.json"
        with tempfile.TemporaryDirectory() as td:
            m = aggregate_and_eval(ckpt, cfg["label_key"], cfg["mode"], 0.3, Path(td))
            ap50 = m.get("AP50", None)
            per_prompt_results[model][ps] = ap50
            col = f"{fmt(ap50):>10}"
            row += f"  {col}"
    print(row)

print("\nDone.")
