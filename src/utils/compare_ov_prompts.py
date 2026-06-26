"""
Compara 3 variantes de prompt de Rex-Omni sobre N tiles y agrega métricas.

Modos:
  - tile única:  --image_path img.tif
  - N tiles:     --n_samples 20  (muestreo estratificado por categoría)

Salidas:
  - PNG por tile (GT + 3 prompts)
  - aggregate_results.json  con métricas por tile y globales
  - aggregate_summary.png   con boxplots de recall y precision

Uso:
    python -m src.utils.compare_ov_prompts \
        --ann_file   results/preprocess/tile_images/train_sampled/COCO_annotations.json \
        --img_dir    results/preprocess/tile_images/train_sampled/images \
        --n_samples  20 \
        --output_dir debug/prompt_comparison
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.inference.rex_inference import parse_rexomni_detection_output

# ---------------------------------------------------------------------------
# Conjuntos de prompts
# ---------------------------------------------------------------------------

PROMPT_SETS = {
    "simple": [
        "Aircraft",
        "Building",
        "Engineering Vehicle",
        "Heavy Vehicle",
        "Light Vehicle",
        "Maritime Vessel",
        "Railway Vehicle",
        "Storage Tank",
        "Tower & Pylon",
    ],
    "aerial_compact": [
        "aerial view of aircraft",
        "aerial view of building",
        "aerial view of construction vehicle",
        "aerial view of truck",
        "aerial view of car",
        "aerial view of ship",
        "aerial view of train",
        "aerial view of storage tank",
        "aerial view of tower",
    ],
    "aerial_verbose": [
        "aircraft seen from above",
        "airplane on the ground seen from satellite",
        "building rooftop seen from above",
        "building seen from satellite",
        "construction vehicle seen from above",
        "bulldozer seen from satellite",
        "truck seen from above",
        "large vehicle seen from satellite",
        "small car seen from above",
        "light vehicle seen from satellite",
        "ship seen from above",
        "train seen from above",
        "circular storage tank seen from satellite",
        "tower or pylon seen from above",
    ],
}

COLORS = {
    "gt": "#00cc44",
    "simple": "#ff4444",
    "aerial_compact": "#4488ff",
    "aerial_verbose": "#ff9900",
}


# ---------------------------------------------------------------------------
# Auxiliares de IoU
# ---------------------------------------------------------------------------


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def recall_at_iou(gt_boxes: list, det_boxes: list, iou_thresh: float = 0.5) -> float:
    if not gt_boxes:
        return float("nan")
    return sum(
        1 for gt in gt_boxes if any(_iou(gt, d) >= iou_thresh for d in det_boxes)
    ) / len(gt_boxes)


def precision_at_iou(gt_boxes: list, det_boxes: list, iou_thresh: float = 0.5) -> float:
    if not det_boxes:
        return float("nan")
    return sum(
        1 for d in det_boxes if any(_iou(gt, d) >= iou_thresh for gt in gt_boxes)
    ) / len(det_boxes)


# ---------------------------------------------------------------------------
# Muestreo: estratificado por categoría para garantizar cobertura de todas las clases
# ---------------------------------------------------------------------------


def sample_tiles(
    coco: dict, n: int, img_dir: Path, seed: int, stratified: bool = True
) -> list[dict]:
    rng = random.Random(seed)
    cats_by_id = {c["id"]: c["name"] for c in coco["categories"]}

    img_cats: dict[int, set[str]] = defaultdict(set)
    for ann in coco["annotations"]:
        img_cats[ann["image_id"]].add(cats_by_id[ann["category_id"]])

    valid = [
        img
        for img in coco["images"]
        if img_cats[img["id"]] and (img_dir / img["file_name"]).exists()
    ]

    if n >= len(valid):
        return valid

    if not stratified:
        return rng.sample(valid, n)

    # Estratificado voraz: primero seleccionar imágenes que maximicen la cobertura
    # de categorías, luego rellenar aleatoriamente.
    selected: list[dict] = []
    covered: set[str] = set()
    remaining = list(valid)

    for cat_name in cats_by_id.values():
        candidates = [
            img
            for img in remaining
            if cat_name in img_cats[img["id"]] and cat_name not in covered
        ]
        if candidates and len(selected) < n:
            pick = rng.choice(candidates)
            selected.append(pick)
            remaining.remove(pick)
            covered.update(img_cats[pick["id"]])

    rng.shuffle(remaining)
    selected += remaining[: n - len(selected)]
    return selected[:n]


# ---------------------------------------------------------------------------
# Auxiliares de dibujo
# ---------------------------------------------------------------------------


def _draw_boxes(ax, image: Image.Image, boxes: list, color: str, title: str) -> None:
    ax.imshow(image)
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.2,
                edgecolor=color,
                facecolor="none",
            )
        )
    ax.set_title(f"{title}\n({len(boxes)} boxes)", fontsize=8)
    ax.axis("off")


def save_tile_viz(
    image: Image.Image,
    gt_boxes: list,
    results: dict,
    out_path: Path,
    stem: str,
    iou_thresh: float,
) -> None:
    # 4 paneles (GT + 3 prompts) → rejilla 2×2
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()
    _draw_boxes(
        axes_flat[0], image, gt_boxes, COLORS["gt"], f"GT ({len(gt_boxes)} boxes)"
    )
    for ax, (name, det_boxes) in zip(axes_flat[1:], results.items()):
        rec = recall_at_iou(gt_boxes, det_boxes, iou_thresh)
        prec = precision_at_iou(gt_boxes, det_boxes, iou_thresh)
        r = f"{rec:.2f}" if not np.isnan(rec) else "N/A"
        p = f"{prec:.2f}" if not np.isnan(prec) else "N/A"
        _draw_boxes(ax, image, det_boxes, COLORS[name], f"{name}\nR={r} P={p}")
    plt.suptitle(f"Rex-Omni prompt comparison — {stem}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_aggregate_viz(per_tile: list[dict], out_path: Path, iou_thresh: float) -> None:
    prompt_names = list(PROMPT_SETS.keys())
    recall_vals = {p: [] for p in prompt_names}
    prec_vals = {p: [] for p in prompt_names}

    for tile_data in per_tile:
        for p in prompt_names:
            r = tile_data["metrics"][p]["recall"]
            pr = tile_data["metrics"][p]["precision"]
            if r is not None:
                recall_vals[p].append(float(r))
            if pr is not None:
                prec_vals[p].append(float(pr))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals_dict, ylabel in [
        (axes[0], recall_vals, f"Recall@{iou_thresh}"),
        (axes[1], prec_vals, f"Precision@{iou_thresh}"),
    ]:
        data = [vals_dict[p] for p in prompt_names]
        bp = ax.boxplot(data, labels=prompt_names, patch_artist=True)
        for patch, name in zip(bp["boxes"], prompt_names):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.7)
        for p, d in zip(prompt_names, data):
            if d:
                ax.scatter(
                    [prompt_names.index(p) + 1] * len(d),
                    d,
                    color=COLORS[p],
                    alpha=0.4,
                    s=20,
                    zorder=3,
                )
                ax.text(
                    prompt_names.index(p) + 1,
                    max(d) + 0.02,
                    f"μ={np.mean(d):.3f}",
                    ha="center",
                    fontsize=8,
                )
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    n = len(per_tile)
    plt.suptitle(f"Rex-Omni prompt variants — aggregate over {n} tiles", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", type=Path, required=True)
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("debug/prompt_comparison")
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        default=None,
        help="Single tile override (skips --n_samples)",
    )
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        help="Pure random sampling instead of stratified",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", type=str, default="IDEA-Research/Rex-Omni")
    args = parser.parse_args()

    with args.ann_file.open() as fh:
        coco = json.load(fh)

    cats_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # Seleccionar tiles
    if args.image_path is not None:
        fname = args.image_path.name
        img_meta = next(
            (
                i
                for i in coco["images"]
                if i["file_name"] == fname or Path(i["file_name"]).name == fname
            ),
            None,
        )
        if img_meta is None:
            raise ValueError(f"'{fname}' not found in {args.ann_file}")
        selected = [img_meta]
    else:
        selected = sample_tiles(
            coco,
            args.n_samples,
            args.img_dir,
            args.seed,
            stratified=not args.random_sampling,
        )

    print(f"Tiles selected: {len(selected)}")

    # Cargar Rex-Omni
    print(f"\nLoading Rex-Omni from {args.model_id} ...")
    sys.path.insert(0, str(Path("vendor/Rex-Omni")))
    from rex_omni import RexOmniWrapper

    model = RexOmniWrapper(
        model_path=args.model_id,
        backend="transformers",
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
        attn_implementation="eager",
        torch_dtype="float16",
        device_map="auto",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Ejecutar todos los prompts por lote por prompt (evita recargar el modelo)
    # Estructura: detections_by_prompt[prompt_name][tile_idx] = lista de cajas
    detections_by_prompt: dict[str, list[list]] = {}

    for prompt_name, categories in PROMPT_SETS.items():
        print(f"\n=== Prompt: {prompt_name} ({len(categories)} phrases) ===")
        all_det_boxes: list[list] = []

        for batch_start in range(0, len(selected), args.batch_size):
            batch_metas = selected[batch_start : batch_start + args.batch_size]
            images = [
                Image.open(args.img_dir / m["file_name"]).convert("RGB")
                for m in batch_metas
            ]
            raw_batch = model.inference(
                images=images, task="detection", categories=categories
            )
            for img, raw in zip(images, raw_batch):
                W, H = img.size
                dets = parse_rexomni_detection_output(raw, image_size=(W, H))
                all_det_boxes.append([d["box_xyxy"] for d in dets])
            print(
                f"  batch {batch_start // args.batch_size + 1}/"
                f"{(len(selected) + args.batch_size - 1) // args.batch_size} done"
            )

        detections_by_prompt[prompt_name] = all_det_boxes

    # Calcular métricas por tile y guardar visualizaciones
    per_tile: list[dict] = []

    for tile_idx, img_meta in enumerate(selected):
        image = Image.open(args.img_dir / img_meta["file_name"]).convert("RGB")
        gt_anns = anns_by_img[img_meta["id"]]
        gt_boxes = [
            [
                a["bbox"][0],
                a["bbox"][1],
                a["bbox"][0] + a["bbox"][2],
                a["bbox"][1] + a["bbox"][3],
            ]
            for a in gt_anns
        ]

        tile_results = {p: detections_by_prompt[p][tile_idx] for p in PROMPT_SETS}
        tile_metrics = {}
        for p, det_boxes in tile_results.items():
            tile_metrics[p] = {
                "recall": recall_at_iou(gt_boxes, det_boxes, args.iou_thresh),
                "precision": precision_at_iou(gt_boxes, det_boxes, args.iou_thresh),
                "n_dets": len(det_boxes),
            }

        stem = Path(img_meta["file_name"]).stem
        viz_path = args.output_dir / f"{stem}_comparison.png"
        save_tile_viz(image, gt_boxes, tile_results, viz_path, stem, args.iou_thresh)

        def _safe_float(v):
            try:
                return None if np.isnan(float(v)) else float(v)
            except (TypeError, ValueError):
                return v

        per_tile.append(
            {
                "file_name": img_meta["file_name"],
                "n_gt": len(gt_boxes),
                "metrics": {
                    p: {k: _safe_float(v) for k, v in m.items()}
                    for p, m in tile_metrics.items()
                },
            }
        )

    # Resumen agregado
    print("\n=== AGGREGATE SUMMARY ===")
    print(
        f"{'Prompt':<20} {'mean Recall':>12} {'mean Precision':>15} {'mean Dets':>10}"
    )
    print("-" * 62)
    for p in PROMPT_SETS:
        recalls = [
            t["metrics"][p]["recall"]
            for t in per_tile
            if t["metrics"][p]["recall"] is not None
        ]
        precs = [
            t["metrics"][p]["precision"]
            for t in per_tile
            if t["metrics"][p]["precision"] is not None
        ]
        dets = [t["metrics"][p]["n_dets"] for t in per_tile]
        print(
            f"{p:<20} {np.mean(recalls):>12.3f} {np.mean(precs):>15.3f} {np.mean(dets):>10.1f}"
        )

    # Guardar JSON agregado
    agg_json = args.output_dir / "aggregate_results.json"
    with agg_json.open("w") as fh:
        json.dump(per_tile, fh, indent=2)
    print(f"\nPer-tile results saved to {agg_json}")

    # Guardar boxplot agregado
    agg_viz = args.output_dir / "aggregate_summary.png"
    save_aggregate_viz(per_tile, agg_viz, args.iou_thresh)
    print(f"Aggregate chart saved to {agg_viz}")


if __name__ == "__main__":
    main()
