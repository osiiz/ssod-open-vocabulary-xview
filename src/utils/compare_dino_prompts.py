"""
Compara 3 variantes de prompt de Grounding DINO sobre N tiles y agrega métricas.

Prompts:
  - simple:         9 nombres de macro-clase directos
  - aerial_compact: "aerial view of X" para cada macro-clase
  - original:       prompt fino actual de dino_prompts.yaml (referencia)

Salidas:
  - PNG por tile (GT + 3 prompts)
  - aggregate_results.json  con métricas por tile y globales
  - aggregate_summary.png   con boxplots de recall y precision

Uso:
    python -m src.utils.compare_dino_prompts \
        --ann_file   results/preprocess/tile_images/train_sampled/COCO_annotations.json \
        --img_dir    results/preprocess/tile_images/train_sampled/images \
        --n_samples  20 \
        --output_dir debug/dino_prompt_comparison
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# ---------------------------------------------------------------------------
# Conjuntos de prompts  (formato DINO: "frase1 . frase2 . ... .")
# ---------------------------------------------------------------------------

_SIMPLE_PHRASES = [
    "aircraft",
    "light vehicle",
    "heavy vehicle",
    "railway vehicle",
    "maritime vessel",
    "engineering vehicle",
    "building",
    "storage tank",
    "tower",
]

_AERIAL_COMPACT_PHRASES = [
    "aerial view of aircraft",
    "aerial view of car",
    "aerial view of truck",
    "aerial view of train",
    "aerial view of ship",
    "aerial view of construction vehicle",
    "aerial view of building",
    "aerial view of storage tank",
    "aerial view of tower",
]


def _aerial_fine_phrases(original_phrases: list[str]) -> list[list[str]]:
    """Divide el prompt fino en dos mitades y prefija con 'aerial view of'.

    Grounding DINO tiene un límite de ~256 tokens de texto. El prompt fino
    original (~40 frases) ya roza ese límite; añadir 'aerial view of' a cada
    frase lo supera. Se divide en dos grupos que se procesan por separado y
    cuyas detecciones se fusionan por imagen.
    """
    prefixed = [f"aerial view of {p.strip().lower()}" for p in original_phrases]
    mid = len(prefixed) // 2
    return [prefixed[:mid], prefixed[mid:]]


def _load_original_phrases(prompt_file: Path) -> list[str]:
    try:
        import yaml

        with prompt_file.open() as fh:
            data = yaml.safe_load(fh)
    except ImportError:
        import re

        text = prompt_file.read_text()
        data = {"prompt_groups": []}
        for line in text.splitlines():
            m = re.match(r"\s+-\s+(.+)", line)
            if m and not line.strip().startswith("name:"):
                data["prompt_groups"][-1]["phrases"].append(m.group(1).strip())
            elif "name:" in line:
                data["prompt_groups"].append({"phrases": []})

    phrases = []
    for group in data.get("prompt_groups", []):
        phrases.extend(group.get("phrases", []))
    return phrases


def _phrases_to_prompt(phrases: list[str]) -> str:
    return " . ".join(p.strip().lower() for p in phrases) + " ."


COLORS = {
    "gt": "#00cc44",
    "simple": "#ff4444",
    "aerial_compact": "#4488ff",
    "original": "#ff9900",
    "aerial_fine": "#aa44ff",
}


# ---------------------------------------------------------------------------
# Auxiliares de IoU
# ---------------------------------------------------------------------------


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def recall_at_iou(gt_boxes, det_boxes, iou_thresh=0.5) -> float:
    if not gt_boxes:
        return float("nan")
    return sum(
        1 for gt in gt_boxes if any(_iou(gt, d) >= iou_thresh for d in det_boxes)
    ) / len(gt_boxes)


def precision_at_iou(gt_boxes, det_boxes, iou_thresh=0.5) -> float:
    if not det_boxes:
        return float("nan")
    return sum(
        1 for d in det_boxes if any(_iou(gt, d) >= iou_thresh for gt in gt_boxes)
    ) / len(det_boxes)


# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------


def sample_tiles(
    coco: dict, n: int, img_dir: Path, seed: int, stratified: bool = True
) -> list[dict]:
    rng = random.Random(seed)
    cats_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    img_cats: dict[int, set] = defaultdict(set)
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

    selected, covered, remaining = [], set(), list(valid)
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
# Inferencia por lotes de DINO
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_dino_batch(
    processor,
    model,
    images: list[Image.Image],
    prompt: str,
    score_thresh: float,
    text_thresh: float,
    device,
) -> list[list[list[float]]]:
    target_sizes = [(img.height, img.width) for img in images]
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=score_thresh,
        text_threshold=text_thresh,
        target_sizes=target_sizes,
    )
    return [[box.tolist() for box in r.get("boxes", [])] for r in results]


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------


def _draw_boxes(ax, image, boxes, color, title):
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


def save_tile_viz(image, gt_boxes, results, out_path, stem, iou_thresh):
    # 5 panels (GT + 4 prompts) → 2×3 grid (last cell empty)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    _draw_boxes(
        axes_flat[0], image, gt_boxes, COLORS["gt"], f"GT ({len(gt_boxes)} boxes)"
    )
    for ax, (name, det_boxes) in zip(axes_flat[1:], results.items()):
        r = recall_at_iou(gt_boxes, det_boxes, iou_thresh)
        p = precision_at_iou(gt_boxes, det_boxes, iou_thresh)
        rs = f"{r:.2f}" if not np.isnan(r) else "N/A"
        ps = f"{p:.2f}" if not np.isnan(p) else "N/A"
        _draw_boxes(ax, image, det_boxes, COLORS[name], f"{name}\nR={rs} P={ps}")
    axes_flat[-1].axis("off")  # última celda vacía
    plt.suptitle(f"Grounding DINO prompt comparison — {stem}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_aggregate_viz(per_tile, out_path, iou_thresh, prompt_names):
    recall_vals = {p: [] for p in prompt_names}
    prec_vals = {p: [] for p in prompt_names}
    for tile in per_tile:
        for p in prompt_names:
            r = tile["metrics"][p]["recall"]
            pr = tile["metrics"][p]["precision"]
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
        for i, (p, d) in enumerate(zip(prompt_names, data)):
            if d:
                ax.scatter(
                    [i + 1] * len(d), d, color=COLORS[p], alpha=0.4, s=20, zorder=3
                )
                ax.text(
                    i + 1, max(d) + 0.02, f"μ={np.mean(d):.3f}", ha="center", fontsize=8
                )
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(f"Grounding DINO prompts — {len(per_tile)} tiles", fontsize=11)
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
        "--output_dir", type=Path, default=Path("debug/dino_prompt_comparison")
    )
    parser.add_argument(
        "--prompt_file", type=Path, default=Path("configs/prompts/dino_prompts.yaml")
    )
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        help="Pure random sampling instead of stratified",
    )
    parser.add_argument("--score_thresh", type=float, default=0.08)
    parser.add_argument("--text_thresh", type=float, default=0.08)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_id", type=str, default="IDEA-Research/grounding-dino-base"
    )
    args = parser.parse_args()

    with args.ann_file.open() as fh:
        coco = json.load(fh)

    cats_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    selected = sample_tiles(
        coco,
        args.n_samples,
        args.img_dir,
        args.seed,
        stratified=not args.random_sampling,
    )
    print(f"Tiles selected: {len(selected)}")

    original_phrases = _load_original_phrases(args.prompt_file)
    aerial_fine_groups = _aerial_fine_phrases(
        original_phrases
    )  # lista de 2 grupos de frases
    PROMPT_SETS = {
        "simple": _phrases_to_prompt(_SIMPLE_PHRASES),
        "aerial_compact": _phrases_to_prompt(_AERIAL_COMPACT_PHRASES),
        "original": _phrases_to_prompt(original_phrases),
        # aerial_fine se trata aparte (2 pases fusionados)
    }
    for name, text in PROMPT_SETS.items():
        print(f"\n[{name}] {text[:120]}{'...' if len(text) > 120 else ''}")

    print(f"\nLoading Grounding DINO ({args.model_id}) ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
        .to(device)
        .eval()
    )
    print("Model loaded.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Ejecutar todos los prompts por lote
    detections_by_prompt: dict[str, list[list]] = {p: [] for p in PROMPT_SETS}
    for prompt_name, prompt_text in PROMPT_SETS.items():
        print(f"\n=== Prompt: {prompt_name} ===")
        for batch_start in range(0, len(selected), args.batch_size):
            batch = selected[batch_start : batch_start + args.batch_size]
            images = [
                Image.open(args.img_dir / m["file_name"]).convert("RGB") for m in batch
            ]
            batch_boxes = run_dino_batch(
                processor,
                model,
                images,
                prompt_text,
                args.score_thresh,
                args.text_thresh,
                device,
            )
            detections_by_prompt[prompt_name].extend(batch_boxes)
            print(
                f"  batch {batch_start // args.batch_size + 1}/"
                f"{(len(selected) + args.batch_size - 1) // args.batch_size} done"
            )

    # aerial_fine: dos pasadas con las dos mitades del prompt, fusionar por imagen
    print("\n=== Prompt: aerial_fine (2 grupos fusionados) ===")
    aerial_fine_dets: list[list] = [[] for _ in selected]
    for g_idx, group_phrases in enumerate(aerial_fine_groups):
        group_prompt = _phrases_to_prompt(group_phrases)
        print(f"  Grupo {g_idx+1}/2: {group_prompt[:80]}...")
        for batch_start in range(0, len(selected), args.batch_size):
            batch = selected[batch_start : batch_start + args.batch_size]
            images = [
                Image.open(args.img_dir / m["file_name"]).convert("RGB") for m in batch
            ]
            batch_boxes = run_dino_batch(
                processor,
                model,
                images,
                group_prompt,
                args.score_thresh,
                args.text_thresh,
                device,
            )
            for tile_offset, boxes in enumerate(batch_boxes):
                aerial_fine_dets[batch_start + tile_offset].extend(boxes)
        print(f"    batch done")
    detections_by_prompt["aerial_fine"] = aerial_fine_dets

    # Métricas + visualizaciones por tile
    all_prompt_names = list(PROMPT_SETS.keys()) + ["aerial_fine"]
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
        tile_results = {p: detections_by_prompt[p][tile_idx] for p in all_prompt_names}
        tile_metrics = {
            p: {
                "recall": recall_at_iou(gt_boxes, dets, args.iou_thresh),
                "precision": precision_at_iou(gt_boxes, dets, args.iou_thresh),
                "n_dets": len(dets),
            }
            for p, dets in tile_results.items()
        }
        stem = Path(img_meta["file_name"]).stem
        save_tile_viz(
            image,
            gt_boxes,
            tile_results,
            args.output_dir / f"{stem}_comparison.png",
            stem,
            args.iou_thresh,
        )

        def _sf(v):
            try:
                return None if np.isnan(float(v)) else float(v)
            except:
                return v

        per_tile.append(
            {
                "file_name": img_meta["file_name"],
                "n_gt": len(gt_boxes),
                "metrics": {
                    p: {k: _sf(v) for k, v in m.items()}
                    for p, m in tile_metrics.items()
                },
            }
        )

    # Agregado
    prompt_names = all_prompt_names
    print("\n=== AGGREGATE SUMMARY ===")
    print(
        f"{'Prompt':<20} {'mean Recall':>12} {'mean Precision':>15} {'mean Dets':>10}"
    )
    print("-" * 62)
    for p in prompt_names:
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

    agg_json = args.output_dir / "aggregate_results.json"
    with agg_json.open("w") as fh:
        json.dump(per_tile, fh, indent=2)
    print(f"\nPer-tile results: {agg_json}")

    agg_viz = args.output_dir / "aggregate_summary.png"
    save_aggregate_viz(per_tile, agg_viz, args.iou_thresh, all_prompt_names)
    print(f"Aggregate chart:  {agg_viz}")


if __name__ == "__main__":
    main()
