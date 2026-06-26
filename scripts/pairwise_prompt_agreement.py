#!/usr/bin/env python3
"""
Matriz 5x5 de acuerdo entre prompt sets.

Para cada par (i, j): usa las detecciones del set i como pseudo-GT (class-aware)
y evalua el set j contra ellas con COCOeval. Metrica: AP50.
Diagonal (i==i): auto-evaluacion, trivialmente 1.0.

Uso:
  python scripts/pairwise_prompt_agreement.py \
      --ann_file results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json \
      --per_set_dir results/dino/single_term_aggregated/per_set \
      --output docs/results_reports/pairwise_prompt_agreement_dino.json
"""
import argparse
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SETS = [
    "set1_direct",
    "set2_synonyms_a",
    "set3_synonyms_b",
    "set4_synonyms_c",
    "set5_synonyms_d",
]
SET_SHORT = ["set1\ndirect", "set2\nsynon_a", "set3\nsynon_b", "set4\nsynon_c", "set5\nsynon_d"]


def build_fake_gt(real_ann: dict, dets: list) -> dict:
    annotations = []
    for idx, det in enumerate(dets):
        x, y, w, h = det["bbox"]
        annotations.append(
            {
                "id": idx + 1,
                "image_id": det["image_id"],
                "category_id": det["category_id"],
                "bbox": [x, y, max(w, 0.0), max(h, 0.0)],
                "area": max(w, 0.0) * max(h, 0.0),
                "iscrowd": 0,
            }
        )
    return {
        "images": real_ann["images"],
        "annotations": annotations,
        "categories": real_ann["categories"],
    }


def eval_pair(coco_gt: COCO, dt_dets: list) -> float:
    if not dt_dets:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt_dets)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [1, 10, 1500]
    ev.params.iouThrs = np.array([0.5])  # solo AP50 → ~10x más rápido
    ev.params.areaRng = [[0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.evaluate()
    ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    return float(ev.stats[1])  # AP50: _summarize(iouThr=.50, maxDets=1500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", required=True)
    parser.add_argument("--per_set_dir", required=True)
    parser.add_argument("--sets", nargs="+", default=SETS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    real_ann = json.loads(Path(args.ann_file).read_text())
    per_set_dir = Path(args.per_set_dir)
    sets = args.sets
    n = len(sets)

    print("Cargando detecciones por set...")
    all_dets: dict[str, list] = {}
    for s in sets:
        path = per_set_dir / s / "detection_results.json"
        all_dets[s] = json.loads(path.read_text())
        print(f"  {s}: {len(all_dets[s]):>8} dets")

    matrix = np.zeros((n, n))

    for i, s_gt in enumerate(sets):
        print(f"\n[{i+1}/{n}] GT = {s_gt} ({len(all_dets[s_gt])} anotaciones)")
        fake_gt = build_fake_gt(real_ann, all_dets[s_gt])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(fake_gt, f)
            tmp_path = f.name

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                coco_gt = COCO(tmp_path)
            for j, s_dt in enumerate(sets):
                print(f"  DT={s_dt}...", end=" ", flush=True)
                ap50 = eval_pair(coco_gt, all_dets[s_dt])
                matrix[i, j] = ap50
                print(f"AP50={ap50:.4f}")
        finally:
            os.unlink(tmp_path)

    # Print table
    col_w = 14
    print("\nMatriz AP50 (fila=pseudo-GT, columna=evaluado):")
    print(" " * 22 + "".join(f"{s:>{col_w}}" for s in sets))
    for i, s_gt in enumerate(sets):
        row = f"{s_gt:>22}"
        for j in range(n):
            cell = f"{matrix[i,j]:.4f}"
            if i == j:
                cell = f"[{cell}]"
            row += f"{cell:>{col_w}}"
        print(row)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"sets": sets, "matrix_ap50": matrix.tolist()}
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nGuardado: {out_path}")

    # Save markdown
    md_path = out_path.with_suffix(".md")
    rows = [
        "# Matriz de acuerdo entre prompt sets (AP50 class-aware)\n",
        "Fila = pseudo-GT, columna = set evaluado. "
        "Diagonal = auto-evaluacion (trivialmente ~1.0).\n",
        "| | " + " | ".join(sets) + " |",
        "|" + "---|" * (n + 1),
    ]
    for i, s_gt in enumerate(sets):
        cells = []
        for j in range(n):
            val = f"{matrix[i,j]:.4f}"
            cells.append(f"**{val}**" if i == j else val)
        rows.append(f"| **{s_gt}** | " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(rows) + "\n")
    print(f"Markdown: {md_path}")

    # Save image
    img_path = out_path.with_suffix(".png")
    _save_heatmap(matrix, sets, img_path)
    print(f"Imagen:   {img_path}")


def _save_heatmap(matrix: np.ndarray, sets: list[str], out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [s.replace("_", "\n") for s in sets]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="AP50")

    ax.set_xticks(range(len(sets)))
    ax.set_yticks(range(len(sets)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Set evaluado (DT)")
    ax.set_ylabel("Pseudo-GT")
    ax.set_title("Acuerdo entre prompt sets (AP50 class-aware)\nFila=pseudo-GT, columna=DT")

    for i in range(len(sets)):
        for j in range(len(sets)):
            v = matrix[i, j]
            color = "white" if v > 0.6 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
