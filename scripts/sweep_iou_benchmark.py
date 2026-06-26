#!/usr/bin/env python3
"""Mini IoU clustering threshold sweep on the 100-image benchmark."""
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

IOT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def aggregate_and_eval(
    raw_json: Path, label_key: str, mode: str, iou_thresh: float, tmpdir: Path
):
    agg = tmpdir / "agg.json"
    eval_aw = tmpdir / "eval_aware"
    eval_ag = tmpdir / "eval_agnostic"
    eval_aw.mkdir(parents=True)
    eval_ag.mkdir(parents=True)

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
        print(f"  AGG error (iou={iou_thresh}): {r.stderr[-200:]}", file=sys.stderr)
        return None, None

    for ev_mode, ev_dir in [("aware", eval_aw), ("agnostic", eval_ag)]:
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
                str(ev_dir),
                "--mode",
                ev_mode,
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(
                f"  EVAL error ({ev_mode}, iou={iou_thresh}): {r.stderr[-200:]}",
                file=sys.stderr,
            )

    def load(p):
        try:
            with open(p) as f:
                return json.load(f)
        except:
            return {}

    return load(eval_aw / "metrics.json"), load(eval_ag / "metrics.json")


def fmt(v):
    return f"{v:.4f}" if v is not None else "  —   "


results = {}  # model -> iou -> (aware, agnostic)

for model, cfg in MODELS.items():
    raw = BENCH / model / "raw" / "detection_results.json"
    results[model] = {}
    print(f"\n=== {model.upper()} ===")
    print(
        f"  {'iou':>5}  {'AP50_aw':>8}  {'AR500_aw':>9}  {'AP50_ag':>8}  {'AR500_ag':>9}"
    )
    for iou_t in IOT_THRESHOLDS:
        with tempfile.TemporaryDirectory() as td:
            aw, ag = aggregate_and_eval(
                raw, cfg["label_key"], cfg["mode"], iou_t, Path(td)
            )
        results[model][iou_t] = (aw, ag)
        ap50_aw = aw.get("AP50") if aw else None
        ar_aw = aw.get("AR_500") if aw else None
        ap50_ag = ag.get("AP50") if ag else None
        ar_ag = ag.get("AR_500") if ag else None
        marker = "  ← actual" if abs(iou_t - 0.3) < 0.01 else ""
        print(
            f"  {iou_t:>5.1f}  {fmt(ap50_aw):>8}  {fmt(ar_aw):>9}  {fmt(ap50_ag):>8}  {fmt(ar_ag):>9}{marker}"
        )

print("\n\n=== RESUMEN: AP50 aware ===")
header = f"{'iou':>5}" + "".join(f"  {m:>10}" for m in MODELS)
print(header)
print("-" * len(header))
for iou_t in IOT_THRESHOLDS:
    row = f"{iou_t:>5.1f}"
    for model in MODELS:
        aw = results[model][iou_t][0]
        row += f"  {fmt(aw.get('AP50') if aw else None):>10}"
    print(row)

print("\n=== RESUMEN: AP50 agnostic ===")
print(header)
print("-" * len(header))
for iou_t in IOT_THRESHOLDS:
    row = f"{iou_t:>5.1f}"
    for model in MODELS:
        ag = results[model][iou_t][1]
        row += f"  {fmt(ag.get('AP50') if ag else None):>10}"
    print(row)
