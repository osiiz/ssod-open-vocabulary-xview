#!/usr/bin/env python3
"""Per-prompt-set evaluation for each model on the 100-image benchmark.

DINO and Detic: no score pre-filter (inference already at higher threshold).
GDSAM2: run at score>=0.1 and score>=0.2 to compare — checkpoints were
        generated at score_thresh=0.05 so filtering is needed for consistency.
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
    "dino": {
        "label_key": "dino_label",
        "mode": "score",
        "score_filters": [0.1],
        "raw_dir": "raw_005",
    },
    "detic": {
        "label_key": "detic_label",
        "mode": "score",
        "score_filters": [0.1],
        "raw_dir": "raw_005",
    },
    "gdsam2": {
        "label_key": "gdsam2_label",
        "mode": "score",
        "score_filters": [0.1],
        "raw_dir": "raw",
    },
}

PROMPT_SETS = [
    "simple",
    "aerial_compact",
    "satellite_verbose",
    "scene_context",
    "original",
]
IOT_THRESH = 0.7


def filter_dets(dets: list, score_min: float | None) -> list:
    if score_min is None:
        return dets
    return [d for d in dets if d.get("score", 1.0) >= score_min]


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
            str(GT),
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
        print(f"  AGG error: {r.stderr[-300:]}", file=sys.stderr)
        return None, None, 0

    n_agg = 0
    try:
        with open(agg) as f:
            n_agg = len(json.load(f))
    except Exception:
        pass

    for ev_mode, ev_dir in [("aware", eval_aw), ("agnostic", eval_ag)]:
        subprocess.run(
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

    def load(p):
        try:
            with open(p) as f:
                return json.load(f)
        except:
            return {}

    return load(eval_aw / "metrics.json"), load(eval_ag / "metrics.json"), n_agg


def fmt(v):
    return f"{v:.4f}" if v is not None else "  —   "


def fmti(v):
    return f"{v:>6}" if v is not None else "     —"


for model, cfg in MODELS.items():
    for score_min in cfg["score_filters"]:
        sf_label = f"score>={score_min}" if score_min else "no filter"
        print(f"\n{'═'*80}")
        print(f"  {model.upper()}  (iou={IOT_THRESH}, {sf_label})")
        print(f"{'═'*80}")
        print(
            f"  {'prompt_set':<20}  {'n_raw':>6}  {'n_filt':>6}  {'n_agg':>6}"
            f"  {'AP50_aw':>8}  {'AR500_aw':>9}"
            f"  {'AP50_ag':>8}  {'AR500_ag':>9}"
        )
        print(
            f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*9}"
        )

        all_filtered = []

        for ps in PROMPT_SETS:
            ckpt = BENCH / model / cfg["raw_dir"] / f"_ckpt_{ps}.json"
            with open(ckpt) as f:
                raw_dets = json.load(f)
            filtered = filter_dets(raw_dets, score_min)
            all_filtered.extend(filtered)

            with tempfile.TemporaryDirectory() as td:
                aw, ag, n_agg = aggregate_and_eval(
                    filtered, cfg["label_key"], cfg["mode"], Path(td)
                )

            ap50_aw = aw.get("AP50") if aw else None
            ar_aw = aw.get("AR_500") if aw else None
            ap50_ag = ag.get("AP50") if ag else None
            ar_ag = ag.get("AR_500") if ag else None
            print(
                f"  {ps:<20}  {fmti(len(raw_dets))}  {fmti(len(filtered))}  {fmti(n_agg)}"
                f"  {fmt(ap50_aw):>8}  {fmt(ar_aw):>9}"
                f"  {fmt(ap50_ag):>8}  {fmt(ar_ag):>9}"
            )

        # Ensemble row
        with tempfile.TemporaryDirectory() as td:
            aw, ag, n_agg = aggregate_and_eval(
                all_filtered, cfg["label_key"], cfg["mode"], Path(td)
            )
        ap50_aw = aw.get("AP50") if aw else None
        ar_aw = aw.get("AR_500") if aw else None
        ap50_ag = ag.get("AP50") if ag else None
        ar_ag = ag.get("AR_500") if ag else None
        print(
            f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*9}"
        )
        print(
            f"  {'ENSEMBLE (5 sets)':<20}  {'':>6}  {fmti(len(all_filtered))}  {fmti(n_agg)}"
            f"  {fmt(ap50_aw):>8}  {fmt(ar_aw):>9}"
            f"  {fmt(ap50_ag):>8}  {fmt(ar_ag):>9}"
        )
