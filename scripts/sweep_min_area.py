#!/usr/bin/env python3
"""
Sweep de thresholds de area minima sobre el GT de train_unlabeled_eval.
Re-evalua DINO y Rex-Omni contra el GT filtrado para cada threshold.

Modos:
  (default)      Solo filtra el GT; las detecciones se pasan sin modificar.
                 Salida: results_filtered/sweep_gt_only.json
  --filter_dets  Filtra GT y detecciones por la misma area minima.
                 Las dets de objetos pequenos ya no cuentan como FP.
                 Salida: results_filtered/sweep_gt_and_dets.json

Metricas almacenadas: conjunto completo COCO estandar (AP, AP50, AP75,
AP_small/medium/large, AR_100/500/1000, AR_small/medium/large) por modelo y modo.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GT_FILE = Path(
    "results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json"
)

MODELS = {
    "dino": Path("results/dino/ensemble_argmax_aggregated/detection_results.json"),
    "rex": Path("results/rexomni/ensemble/detection_results.json"),
}

THRESHOLDS = [100, 225, 400, 625, 1024]
OUT_DIR = Path("results_filtered")

_ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
_CWD = str(Path(__file__).parent.parent)

METRIC_KEYS = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_100",
    "AR_500",
    "AR_1000",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _bbox_area(bbox) -> float:
    return bbox[2] * bbox[3]


def filter_gt(gt: dict, min_area: int) -> dict:
    kept_anns = [a for a in gt["annotations"] if _bbox_area(a["bbox"]) >= min_area]
    kept_img_ids = {a["image_id"] for a in kept_anns}
    kept_imgs = [img for img in gt["images"] if img["id"] in kept_img_ids]
    return {
        "images": kept_imgs,
        "annotations": kept_anns,
        "categories": gt["categories"],
    }


def filter_detections(detections: list, min_area: int) -> list:
    return [d for d in detections if _bbox_area(d["bbox"]) >= min_area]


def evaluate(det_path: Path, gt_path: Path, mode: str, eval_dir: Path) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference.ov_coco_eval",
            "--ann_file",
            str(gt_path),
            "--detection_results",
            str(det_path),
            "--output_folder",
            str(eval_dir),
            "--mode",
            mode,
        ],
        capture_output=True,
        text=True,
        cwd=_CWD,
        env=_ENV,
    )
    if r.returncode != 0:
        print(
            f"  ERROR ({mode}) rc={r.returncode}: {r.stderr[-400:]}",
            file=sys.stderr,
            flush=True,
        )
        return {}
    try:
        with open(eval_dir / "metrics.json") as f:
            m = json.load(f)
        if m.get("AP50") in (-1.0, None):
            print(
                f"  WARN ({mode}): AP50={m.get('AP50')} — stdout: {r.stdout[-600:]}",
                file=sys.stderr,
                flush=True,
            )
        return m
    except Exception as e:
        print(f"  ERROR leyendo metrics ({mode}): {e}", file=sys.stderr, flush=True)
        return {}


def extract_metrics(m: dict) -> dict:
    """Extrae el conjunto completo de metricas COCO del dict devuelto por ov_coco_eval."""
    keys = METRIC_KEYS.copy()
    # ov_coco_eval usa max_dets=1000 → AR_1000; torvis usa 1500 → AR_1500
    if "AR_1500" in m and "AR_1000" not in m:
        keys = ["AR_1500" if k == "AR_1000" else k for k in keys]
    return {k: m.get(k) for k in keys}


def fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "  -   "


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--filter_dets",
        action="store_true",
        help="Filtra tambien las detecciones por min_area (experiment GT+Dets).",
    )
    args = parser.parse_args()

    exp_label = "gt_and_dets" if args.filter_dets else "gt_only"
    out_file = OUT_DIR / f"sweep_{exp_label}.json"

    print(f"Experimento: {exp_label}", flush=True)
    print("Cargando GT...", flush=True)
    with open(GT_FILE) as f:
        gt_full = json.load(f)
    total_anns = len(gt_full["annotations"])
    print(
        f"GT: {total_anns} annotations, {len(gt_full['images'])} imagenes\n", flush=True
    )

    # Precarga dets en memoria si hay que filtrarlas (evita re-leer 2.6 GB x5 en subprocesos)
    dets_full: dict[str, list] = {}
    if args.filter_dets:
        for model, det_path in MODELS.items():
            print(f"Cargando detecciones {model} ({det_path}) ...", flush=True)
            with open(det_path) as f:
                dets_full[model] = json.load(f)
            print(f"  {len(dets_full[model]):,} detecciones cargadas", flush=True)
        print(flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_rows = []

    header = (
        f"{'min_area':>9}  {'n_ann':>7}  {'% ret':>6}  {'n_img':>6}  "
        f"{'DINO_aw':>8}  {'DINO_ag':>8}  {'Rex_aw':>8}  {'Rex_ag':>8}"
    )
    if args.filter_dets:
        header += f"  {'DINO_dets':>10}  {'Rex_dets':>9}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for min_area in THRESHOLDS:
        gt_filtered = filter_gt(gt_full, min_area)
        n_ann = len(gt_filtered["annotations"])
        n_img = len(gt_filtered["images"])
        pct = 100 * n_ann / total_anns

        row: dict = {
            "min_area_px2": min_area,
            "n_ann": n_ann,
            "pct_retained": round(pct, 2),
            "n_images": n_img,
        }

        with tempfile.TemporaryDirectory() as _td:
            td = Path(_td)
            gt_path = td / "gt.json"
            with open(gt_path, "w") as f:
                json.dump(gt_filtered, f)

            for model, det_path in MODELS.items():
                if args.filter_dets:
                    filtered = filter_detections(dets_full[model], min_area)
                    filtered_path = td / f"{model}_dets.json"
                    with open(filtered_path, "w") as f:
                        json.dump(filtered, f)
                    actual_path = filtered_path
                    row[f"{model}_n_dets"] = len(filtered)
                else:
                    actual_path = det_path

                for mode in ["aware", "agnostic"]:
                    m = evaluate(actual_path, gt_path, mode, td / model / mode)
                    row[f"{model}_{mode[:2]}"] = extract_metrics(m)

        output_rows.append(row)

        dino_aw = (row.get("dino_aw") or {}).get("AP50")
        dino_ag = (row.get("dino_ag") or {}).get("AP50")
        rex_aw = (row.get("rex_aw") or {}).get("AP50")
        rex_ag = (row.get("rex_ag") or {}).get("AP50")
        line = (
            f"{min_area:>9}  {n_ann:>7}  {pct:>5.1f}%  {n_img:>6}  "
            f"{fmt(dino_aw):>8}  {fmt(dino_ag):>8}  {fmt(rex_aw):>8}  {fmt(rex_ag):>8}"
        )
        if args.filter_dets:
            line += f"  {row.get('dino_n_dets', '-'):>10,}  {row.get('rex_n_dets', '-'):>9,}"
        print(line, flush=True)

        # Guarda parcialmente tras cada threshold por si el proceso se interrumpe
        with open(out_file, "w") as f:
            json.dump(output_rows, f, indent=2)

    print(f"\nTabla guardada en {out_file}", flush=True)
    print(
        "Elige min_area_px2 y registralo en params.yaml bajo size_filter.min_area_px2",
        flush=True,
    )


if __name__ == "__main__":
    main()
