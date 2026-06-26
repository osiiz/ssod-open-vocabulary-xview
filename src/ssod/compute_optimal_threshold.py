"""
Barre umbrales de score sobre un detection_results.json y encuentra el que
maximiza AP50 contra un fichero de ground-truth COCO.

También calcula LRP, OCE, D-ECE y LA-ECE0 en cada umbral.

    LRP    : Localisation-Recall-Precision (Oksuz et al. 2022).
             0 = perfecto; extraído de los datos de matching de COCOeval.
    OCE    : Object-level Calibration Error (Park et al. 2024).
             Puntuación Brier media por objeto GT; FN contribuye 1.0.
    D-ECE  : Detection ECE (Küppers et al. 2020).
             Agrupa predicciones por confianza; accuracy = 1 si TP, 0 si FP.
    LA-ECE0: Variante 0 de ECE sensible a localización.
             Como D-ECE pero accuracy = valor IoU (continuo) en lugar de {0,1}.

Uso:
    python -m src.ssod.compute_optimal_threshold \\
        --detection_results results/dino/r90_dino_raw/detection_results.json \\
        --gt_ann_file results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \\
        --output_file results/ssod/thresholds/dino_threshold.json
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# nombre_métrica -> (mayor_es_mejor, etiqueta_display, color_línea)
METRICS_META: dict[str, tuple[bool, str, str]] = {
    "ap50": (True, "AP50", "#2196F3"),
    "lrp": (False, "LRP", "#E53935"),
    "oce": (False, "OCE", "#FB8C00"),
    "d_ece": (False, "D-ECE", "#8E24AA"),
    "la_ece0": (False, "LA-ECE0", "#00897B"),
}


# ---------------------------------------------------------------------------
# Ejecutor de COCOeval
# ---------------------------------------------------------------------------


def _run_cocoeval(
    detections: list, coco_gt: COCO, iou_thresh: float = 0.5
) -> Optional[COCOeval]:
    if not detections:
        return None
    try:
        coco_dt = coco_gt.loadRes(detections)
    except Exception:
        return None
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.iouThrs = np.array([iou_thresh])
    ev.params.maxDets = [100, 500, 1500]  # match ov_coco_eval --max_dets 1500
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


# ---------------------------------------------------------------------------
# Extracción de matches del estado interno de COCOeval
# ---------------------------------------------------------------------------


def _extract_matches(
    ev: COCOeval,
) -> tuple[list, list, list, int]:
    """
    Tras ev.evaluate(), parsea evalImgs para obtener los resultados por detección.

    Devuelve
    --------
    tp_scores : puntuaciones de confianza de las detecciones TP
    tp_ious   : valor IoU de cada TP con su GT coincidente
    fp_scores : puntuaciones de confianza de las detecciones FP
    n_fn      : número total de objetos GT no coincididos (perdidos)
    """
    tp_scores: list = []
    tp_ious: list = []
    fp_scores: list = []
    n_fn = 0
    t = 0  # índice único de umbral IoU (siempre se fija iouThrs a un solo valor)

    all_area_rng = ev.params.areaRng[0]  # [0, 1e10] = "all sizes"

    for eval_img in ev.evalImgs:
        if eval_img is None:
            continue
        # Procesar solo el rango "all area" para evitar doble conteo
        if eval_img["aRng"] != all_area_rng:
            continue

        img_id = eval_img["image_id"]
        cat_id = eval_img["category_id"]

        dt_scores = eval_img["dtScores"]  # lista, ordenada por score descendente
        gt_ids = eval_img["gtIds"]  # lista de IDs GT en este img/cat
        dt_matches = eval_img["dtMatches"][t]  # ID GT emparejado con cada det (0 = FP)
        dt_ignore = eval_img["dtIgnore"][t]  # array de bools
        gt_matches = eval_img["gtMatches"][t]  # ID det emparejado con cada GT (0 = FN)
        gt_ignore = eval_img["gtIgnore"]  # array de bools

        ious = ev.ious.get((img_id, cat_id))  # ndarray shape (n_det, n_gt) or None
        gt_id_to_idx = {gid: gi for gi, gid in enumerate(gt_ids)}

        for di, (score, match_id, ignore) in enumerate(
            zip(dt_scores, dt_matches, dt_ignore)
        ):
            if ignore:
                continue
            if match_id > 0:
                gi = gt_id_to_idx.get(int(match_id))
                if gi is not None and ious is not None and ious.size > 0:
                    iou_val = float(ious[di, gi])
                else:
                    iou_val = float(ev.params.iouThrs[0])  # fallback
                tp_scores.append(float(score))
                tp_ious.append(iou_val)
            else:
                fp_scores.append(float(score))

        for match_id, ignore in zip(gt_matches, gt_ignore):
            if not ignore and match_id == 0:
                n_fn += 1

    return tp_scores, tp_ious, fp_scores, n_fn


# ---------------------------------------------------------------------------
# Implementaciones de métricas
# ---------------------------------------------------------------------------


def _lrp(tp_ious: list, n_fp: int, n_fn: int, iou_thresh: float = 0.5) -> float:
    """
    LRP = (Σ_TP e_loc + FP + FN) / (TP + FP + FN)
    e_loc = (1 - IoU) / (1 - iou_thresh)  ∈ [0, 1]
    Rango [0, 1]; 0 = perfecto.
    """
    n_tp = len(tp_ious)
    total = n_tp + n_fp + n_fn
    if total == 0:
        return 0.0
    denom = max(1.0 - iou_thresh, 1e-9)
    loc_errors = sum((1.0 - iou) / denom for iou in tp_ious)
    return float((loc_errors + n_fp + n_fn) / total)


def _extract_oce_brier_scores(ev: COCOeval, iou_thresh: float) -> list:
    """
    Para cada objeto GT no ignorado: encuentra TODAS las predicciones con IoU >= iou_thresh,
    promedia sus puntuaciones de confianza, calcula Brier = (avg_conf - 1)^2.
    Si ninguna predicción se solapa (FN): Brier = 1.0.

    Esta es la agregación por objeto correcta de Park et al. (2024) §V Ec. 4.
    Usar todas las predicciones solapadas (no solo el match 1-a-1 de COCOeval)
    penaliza los regímenes de umbral bajo donde las predicciones secundarias inflan el ruido.
    """
    brier_scores: list = []
    all_area_rng = ev.params.areaRng[0]

    for eval_img in ev.evalImgs:
        if eval_img is None:
            continue
        if eval_img["aRng"] != all_area_rng:
            continue

        img_id = eval_img["image_id"]
        cat_id = eval_img["category_id"]

        dt_scores = np.array(eval_img["dtScores"], dtype=float)
        gt_ignore = eval_img["gtIgnore"]
        raw_ious = ev.ious.get((img_id, cat_id))
        ious_arr = np.asarray(raw_ious) if raw_ious is not None else None

        for gi, ignore in enumerate(gt_ignore):
            if ignore:
                continue
            if ious_arr is not None and ious_arr.size > 0 and len(dt_scores) > 0:
                overlapping = dt_scores[ious_arr[:, gi] >= iou_thresh]
            else:
                overlapping = np.array([])

            if len(overlapping) > 0:
                avg_conf = float(overlapping.mean())
                brier_scores.append((avg_conf - 1.0) ** 2)
            else:
                brier_scores.append(1.0)

    return brier_scores


def _oce(brier_scores: list) -> float:
    """OCE = puntuación Brier media sobre todos los objetos GT. Rango [0, 1]; 0 = perfecto."""
    if not brier_scores:
        return 0.0
    return float(np.mean(brier_scores))


def _binned_ece(scores: list, targets: list, n_bins: int = 15) -> float:
    """
    ECE = Σ_b (n_b / N) |mean_conf_b - mean_target_b|
    `targets` puede ser binario (D-ECE) o valores IoU continuos (LA-ECE0).
    """
    if not scores:
        return 0.0
    s = np.asarray(scores, dtype=float)
    a = np.asarray(targets, dtype=float)
    n = len(s)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (s >= lo) & (s <= hi) if i == n_bins - 1 else (s >= lo) & (s < hi)
        if not mask.any():
            continue
        conf_b = s[mask].mean()
        acc_b = a[mask].mean()
        ece += (mask.sum() / n) * abs(conf_b - acc_b)
    return float(ece)


def _d_ece(tp_scores: list, fp_scores: list, n_bins: int = 15) -> float:
    scores = tp_scores + fp_scores
    targets = [1.0] * len(tp_scores) + [0.0] * len(fp_scores)
    return _binned_ece(scores, targets, n_bins)


def _la_ece0(
    tp_scores: list, tp_ious: list, fp_scores: list, n_bins: int = 15
) -> float:
    scores = tp_scores + fp_scores
    targets = list(tp_ious) + [0.0] * len(fp_scores)
    return _binned_ece(scores, targets, n_bins)


# ---------------------------------------------------------------------------
# Agregación por umbral
# ---------------------------------------------------------------------------

_NULL_METRICS = {"ap50": 0.0, "lrp": 1.0, "oce": 1.0, "d_ece": 1.0, "la_ece0": 1.0}


def _compute_metrics(detections: list, coco_gt: COCO, iou_thresh: float = 0.5) -> dict:
    ev = _run_cocoeval(detections, coco_gt, iou_thresh)
    if ev is None:
        return dict(_NULL_METRICS)

    ap50 = max(0.0, float(ev.stats[1]))
    tp_s, tp_iou, fp_s, n_fn = _extract_matches(ev)
    brier_scores = _extract_oce_brier_scores(ev, iou_thresh)

    return {
        "ap50": round(ap50, 6),
        "lrp": round(_lrp(tp_iou, len(fp_s), n_fn, iou_thresh), 6),
        "oce": round(_oce(brier_scores), 6),
        "d_ece": round(_d_ece(tp_s, fp_s), 6),
        "la_ece0": round(_la_ece0(tp_s, tp_iou, fp_s), 6),
    }


def sweep_thresholds(
    all_detections: list,
    coco_gt: COCO,
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
    iou_thresh: float = 0.5,
) -> list[dict]:
    thresholds = np.arange(
        threshold_start, threshold_stop + threshold_step * 0.5, threshold_step
    )
    thresholds = np.round(thresholds, 10)

    results = []
    for t in thresholds:
        filtered = [d for d in all_detections if d["score"] >= t]
        m = _compute_metrics(filtered, coco_gt, iou_thresh)
        print(
            f"  threshold={t:.2f}  n={len(filtered):>8}"
            f"  AP50={m['ap50']:.4f}"
            f"  LRP={m['lrp']:.4f}"
            f"  OCE={m['oce']:.4f}"
            f"  D-ECE={m['d_ece']:.4f}"
            f"  LA-ECE0={m['la_ece0']:.4f}"
        )
        results.append(
            {
                "threshold": round(float(t), 4),
                "n_detections_at_thresh": len(filtered),
                **m,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------


def save_sweep_chart(
    sweep_results: list[dict], out_path: Path, title: str = ""
) -> None:
    """
    Gráfico de líneas con dos paneles:
      - Superior: AP50 (↑ mayor es mejor)
      - Inferior: LRP, OCE, D-ECE, LA-ECE0 (↓ menor es mejor)

    El umbral óptimo de cada métrica se marca con una estrella + anotación.
    Bajo ambos paneles: tabla con todos los valores numéricos por umbral.
    """
    thresholds = [r["threshold"] for r in sweep_results]

    fig = plt.figure(figsize=(13, 12))
    # 3 filas: gráfico AP50 / gráfico de errores / tabla
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1.4], hspace=0.45)
    ax_ap = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])
    ax_tbl = fig.add_subplot(gs[2])
    ax_tbl.axis("off")

    # ---- auxiliares ----
    def _best_thresh(metric: str) -> dict:
        higher = METRICS_META[metric][0]
        return (
            max(sweep_results, key=lambda r: r[metric])
            if higher
            else min(sweep_results, key=lambda r: r[metric])
        )

    def _plot_metric(ax, metric: str) -> None:
        _, label, color = METRICS_META[metric]
        vals = [r[metric] for r in sweep_results]
        ax.plot(
            thresholds,
            vals,
            marker="o",
            markersize=4,
            color=color,
            linewidth=1.8,
            label=label,
        )
        best = _best_thresh(metric)
        bt, bv = best["threshold"], best[metric]
        ax.scatter([bt], [bv], marker="*", s=200, color=color, zorder=5)
        ax.annotate(
            f"T={bt:.2f}\n{label}={bv:.3f}",
            xy=(bt, bv),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec=color, lw=0.8
            ),
        )

    # ---- panel AP50 ----
    _plot_metric(ax_ap, "ap50")
    ax_ap.set_ylabel("AP50", fontsize=10)
    ax_ap.set_title(
        f"AP50 vs score threshold  (↑ higher is better){('  —  ' + title) if title else ''}",
        fontsize=10,
    )
    ax_ap.set_ylim(bottom=0)
    ax_ap.set_xticks(thresholds)
    ax_ap.grid(axis="both", linestyle="--", alpha=0.4)
    ax_ap.legend(fontsize=9)

    # ---- panel de métricas de error ----
    for metric in ("lrp", "oce", "d_ece", "la_ece0"):
        _plot_metric(ax_err, metric)
    ax_err.set_ylabel("Error (↓ lower is better)", fontsize=10)
    ax_err.set_xlabel("Score threshold", fontsize=10)
    ax_err.set_title(
        "Calibration & localisation error metrics vs score threshold", fontsize=10
    )
    ax_err.set_ylim(0, 1.05)
    ax_err.set_xticks(thresholds)
    ax_err.grid(axis="both", linestyle="--", alpha=0.4)
    ax_err.legend(fontsize=9)

    # ---- tabla numérica ----
    col_labels = [
        "Thresh",
        "n_dets",
        "AP50 ↑",
        "LRP ↓",
        "OCE ↓",
        "D-ECE ↓",
        "LA-ECE0 ↓",
    ]
    table_data = [
        [
            f"{r['threshold']:.2f}",
            f"{r['n_detections_at_thresh']:,}",
            f"{r['ap50']:.4f}",
            f"{r['lrp']:.4f}",
            f"{r['oce']:.4f}",
            f"{r['d_ece']:.4f}",
            f"{r['la_ece0']:.4f}",
        ]
        for r in sweep_results
    ]

    # Resaltar las mejores celdas por columna de métrica
    best_rows = {
        2: sweep_results.index(_best_thresh("ap50")),
        3: sweep_results.index(_best_thresh("lrp")),
        4: sweep_results.index(_best_thresh("oce")),
        5: sweep_results.index(_best_thresh("d_ece")),
        6: sweep_results.index(_best_thresh("la_ece0")),
    }
    metric_colors = {
        2: METRICS_META["ap50"][2],
        3: METRICS_META["lrp"][2],
        4: METRICS_META["oce"][2],
        5: METRICS_META["d_ece"][2],
        6: METRICS_META["la_ece0"][2],
    }

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#EEEEEE")
            cell.set_text_props(fontweight="bold")
        elif col in best_rows and row - 1 == best_rows[col]:
            # aclarar el color de la métrica
            import matplotlib.colors as mcolors

            rgba = mcolors.to_rgba(metric_colors[col])
            cell.set_facecolor((*rgba[:3], 0.25))
            cell.set_text_props(fontweight="bold")

    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Auxiliar de mejor valor por métrica
# ---------------------------------------------------------------------------


def _best_per_metric(sweep_results: list[dict]) -> dict:
    """Devuelve un dict que mapea cada nombre de métrica a la entrada del sweep con su valor óptimo."""
    bests = {}
    for metric, (higher, _, _) in METRICS_META.items():
        bests[metric] = (
            max(sweep_results, key=lambda r: r[metric])
            if higher
            else min(sweep_results, key=lambda r: r[metric])
        )
    return bests


# ---------------------------------------------------------------------------
# Interfaz de línea de comandos
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep thresholds; optimise AP50; report LRP/OCE/D-ECE/LA-ECE0."
    )
    parser.add_argument("--detection_results", type=Path, required=True)
    parser.add_argument("--gt_ann_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--threshold_start", type=float, default=0.1)
    parser.add_argument("--threshold_stop", type=float, default=0.9)
    parser.add_argument("--threshold_step", type=float, default=0.1)
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="IoU threshold for TP/FP/FN matching",
    )
    args = parser.parse_args()

    print(f"Loading detections from {args.detection_results} ...")
    with args.detection_results.open() as fh:
        all_detections = json.load(fh)
    print(f"  {len(all_detections)} detections loaded.")

    print(f"Loading GT from {args.gt_ann_file} ...")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(args.gt_ann_file))

    print(
        f"Sweeping [{args.threshold_start:.2f} .. {args.threshold_stop:.2f}] "
        f"step {args.threshold_step:.2f}, IoU@{args.iou_thresh} ..."
    )
    sweep_results = sweep_thresholds(
        all_detections,
        coco_gt,
        args.threshold_start,
        args.threshold_stop,
        args.threshold_step,
        args.iou_thresh,
    )

    bests = _best_per_metric(sweep_results)

    # Tabla resumen en consola
    header = f"{'Metric':<10} {'Best thresh':>12} {'Value':>8}  {'Direction'}"
    print(f"\n{'='*55}")
    print(header)
    print("-" * 55)
    for metric, (higher, label, _) in METRICS_META.items():
        b = bests[metric]
        direction = "↑ max" if higher else "↓ min"
        print(f"  {label:<8} {b['threshold']:>12.2f} {b[metric]:>8.4f}  {direction}")
    print(f"{'='*55}")

    # JSON de salida: nivel superior = mejor por AP50 (compatible con build_pe_dataset),
    # más bloque "best_by_metric" con el ganador de cada métrica por separado.
    best_ap50 = bests["ap50"]
    output = {
        **best_ap50,
        "best_by_metric": {metric: bests[metric] for metric in METRICS_META},
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nSaved to {args.output_file}")

    # JSON completo del sweep
    sweep_file = args.output_file.with_stem(args.output_file.stem + "_sweep")
    with sweep_file.open("w") as fh:
        json.dump(sweep_results, fh, indent=2)
    print(f"Full sweep saved to {sweep_file}")

    # Gráfico
    chart_file = args.output_file.with_stem(
        args.output_file.stem + "_chart"
    ).with_suffix(".png")
    save_sweep_chart(
        sweep_results,
        chart_file,
        title=args.detection_results.parts[-2]
        if len(args.detection_results.parts) >= 2
        else "",
    )
    print(f"Chart saved to {chart_file}")


if __name__ == "__main__":
    main()
