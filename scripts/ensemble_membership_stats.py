#!/usr/bin/env python3
"""
Histograma de membresía del ensemble: para cada deteccion del ensemble,
cuantos de los N prompt sets contribuyeron al cluster.

Modo primario (exacto): lee el campo 'contributing_sets' del JSON del ensemble,
generado por multi_prompt_ensemble.py >= v2. No requiere per_set_dir ni IoU.

Modo de compatibilidad (retroactivo, aproximado): si el JSON no tiene el campo,
hace matching IoU entre cada deteccion del ensemble y las detecciones por set.
AVISO: este modo no es fiel cuando union-find encadena detecciones transitivamente
y el centroide fusionado difiere de los contribuyentes originales.

Uso:
  python scripts/ensemble_membership_stats.py \
      --ensemble results/dino/single_term_aggregated/detection_results.json \
      --output docs/results_reports/ensemble_membership_stats_dino.json

  # Solo necesario para JSONs antiguos (sin contributing_sets):
  python scripts/ensemble_membership_stats.py \
      --ensemble results/dino/single_term_aggregated/detection_results.json \
      --per_set_dir results/dino/single_term_aggregated/per_set \
      --output docs/results_reports/ensemble_membership_stats_dino.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SETS = [
    "set1_direct",
    "set2_synonyms_a",
    "set3_synonyms_b",
    "set4_synonyms_c",
    "set5_synonyms_d",
]


# ---------------------------------------------------------------------------
# Modo retroactivo (compatibilidad con JSONs sin contributing_sets)
# ---------------------------------------------------------------------------

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU entre dos conjuntos de cajas en formato xywh. Devuelve (N, M)."""
    ax1 = boxes_a[:, 0]
    ay1 = boxes_a[:, 1]
    ax2 = boxes_a[:, 0] + boxes_a[:, 2]
    ay2 = boxes_a[:, 1] + boxes_a[:, 3]
    bx1 = boxes_b[:, 0]
    by1 = boxes_b[:, 1]
    bx2 = boxes_b[:, 0] + boxes_b[:, 2]
    by2 = boxes_b[:, 1] + boxes_b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def index_by_image_and_cat(dets: list) -> dict:
    """Devuelve {(image_id, category_id): np.ndarray de bboxes xywh}."""
    groups: dict = defaultdict(list)
    for det in dets:
        groups[(det["image_id"], det["category_id"])].append(det["bbox"])
    return {k: np.array(v, dtype=np.float32) for k, v in groups.items()}


def compute_membership_retroactive(
    ensemble: list[dict],
    sets: list[str],
    per_set_dir: Path,
    iou_thr: float,
) -> np.ndarray:
    """Fallback: matching IoU retroactivo. Aproximado — puede subestimar sets."""
    print(
        f"  [AVISO] Modo retroactivo (IoU>={iou_thr}). No es exacto para clusters\n"
        "  transitivos. Vuelve a generar el ensemble para obtener 'contributing_sets'."
    )
    n_sets = len(sets)
    print("Cargando per-set detecciones...")
    set_indices: list[dict] = []
    for s in sets:
        path = per_set_dir / s / "detection_results.json"
        dets = json.loads(path.read_text())
        set_indices.append(index_by_image_and_cat(dets))
        print(f"  {s}: {len(dets)} dets")

    n_contributing = np.zeros(len(ensemble), dtype=np.int8)

    ens_groups: dict = defaultdict(list)
    for idx, det in enumerate(ensemble):
        key = (det["image_id"], det["category_id"])
        ens_groups[key].append((idx, det["bbox"]))

    total_keys = len(ens_groups)
    for k_idx, (key, ens_list) in enumerate(ens_groups.items()):
        if k_idx % 50000 == 0:
            print(f"  {k_idx}/{total_keys} grupos...", end="\r")
        ens_idxs = [item[0] for item in ens_list]
        ens_boxes = np.array([item[1] for item in ens_list], dtype=np.float32)
        for s_idx in range(n_sets):
            set_boxes = set_indices[s_idx].get(key)
            if set_boxes is None:
                continue
            iou = iou_matrix(ens_boxes, set_boxes)
            has_match = (iou >= iou_thr).any(axis=1)
            for local_i, global_i in enumerate(ens_idxs):
                if has_match[local_i]:
                    n_contributing[global_i] += 1

    print(f"\n  Listo. {len(ensemble)} detecciones procesadas.")
    return n_contributing


# ---------------------------------------------------------------------------
# Modo exacto
# ---------------------------------------------------------------------------

def compute_membership_exact(ensemble: list[dict], sets: list[str]) -> np.ndarray:
    """Lee contributing_sets del JSON — exacto, sin IoU."""
    set_index = {s: i for i, s in enumerate(sets)}
    n_contributing = np.zeros(len(ensemble), dtype=np.int8)
    for idx, det in enumerate(ensemble):
        cs = det.get("contributing_sets", [])
        n_contributing[idx] = sum(1 for s in cs if s in set_index)
    return n_contributing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--per_set_dir", default=None,
                        help="Solo necesario para JSONs sin 'contributing_sets'")
    parser.add_argument("--sets", nargs="+", default=SETS)
    parser.add_argument("--iou_thr", type=float, default=0.7,
                        help="IoU para modo retroactivo (debe coincidir con iou_cluster del ensemble)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sets = args.sets
    n_sets = len(sets)

    print("Cargando ensemble...")
    ensemble = json.loads(Path(args.ensemble).read_text())
    print(f"  {len(ensemble)} detecciones")

    has_field = any("contributing_sets" in d for d in ensemble[:100])

    print("\nCalculando membresía...")
    if has_field:
        print("  Modo exacto: leyendo campo 'contributing_sets'.")
        n_contributing = compute_membership_exact(ensemble, sets)
        method = "contributing_sets"
    else:
        if args.per_set_dir is None:
            raise ValueError(
                "El ensemble no tiene 'contributing_sets'. "
                "Proporciona --per_set_dir para el modo retroactivo, "
                "o vuelve a generar el ensemble con la version actualizada."
            )
        n_contributing = compute_membership_retroactive(
            ensemble, sets, Path(args.per_set_dir), args.iou_thr
        )
        method = f"iou_retroactivo_{args.iou_thr}"

    # Histograma global
    hist = np.bincount(n_contributing, minlength=n_sets + 1)[1:]  # indices 1..n_sets
    total = len(ensemble)
    print(f"\nHistograma de membresía ({method}):")
    print(f"{'Sets':>6}  {'Count':>10}  {'%':>7}")
    for k in range(1, n_sets + 1):
        bar = "#" * int(40 * hist[k - 1] / total)
        print(f"  {k:>4}  {hist[k-1]:>10}  {100*hist[k-1]/total:>6.1f}%  {bar}")

    # Desglose por categoria
    cat_ids = sorted({det["category_id"] for det in ensemble})
    cat_names = {det["category_id"]: det.get("class_name", str(det["category_id"])) for det in ensemble}

    cat_hist: dict = {}
    for cat_id in cat_ids:
        mask = np.array([det["category_id"] == cat_id for det in ensemble], dtype=bool)
        counts = n_contributing[mask]
        cat_hist[cat_names[cat_id]] = np.bincount(counts, minlength=n_sets + 1)[1:].tolist()

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "sets": sets,
        "method": method,
        "total_detections": total,
        "histogram": hist.tolist(),
        "histogram_pct": (hist / total * 100).tolist(),
        "per_category": cat_hist,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nGuardado: {out_path}")

    # Markdown
    md_path = out_path.with_suffix(".md")
    lines = [
        "# Histograma de membresía del ensemble\n",
        f"Método: `{method}`.\n",
        "| Sets contribuyentes | Detecciones | % |",
        "|---|---|---|",
    ]
    for k in range(1, n_sets + 1):
        lines.append(f"| {k} | {hist[k-1]:,} | {100*hist[k-1]/total:.1f}% |")

    lines += [
        "",
        "## Desglose por categoria\n",
        "| Categoria | " + " | ".join(f"{k} set{'s' if k>1 else ''}" for k in range(1, n_sets + 1)) + " |",
        "|" + "---|" * (n_sets + 1),
    ]
    for cat_name, counts in cat_hist.items():
        cat_total = sum(counts)
        cells = [f"{c:,} ({100*c/cat_total:.0f}%)" if cat_total else "0" for c in counts]
        lines.append(f"| {cat_name} | " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Markdown: {md_path}")

    # Save image
    img_path = out_path.with_suffix(".png")
    _save_histogram(hist, cat_hist, n_sets, total, method, img_path)
    print(f"Imagen:   {img_path}")


def _save_histogram(
    hist: np.ndarray,
    cat_hist: dict,
    n_sets: int,
    total: int,
    method: str,
    out_path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(range(1, n_sets + 1))
    pcts = [100 * hist[k - 1] / total for k in ks]

    cat_names = list(cat_hist.keys())
    n_cats = len(cat_names)
    colors = plt.cm.tab10.colors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bars = ax1.bar(ks, pcts, color="#4878CF", edgecolor="white")
    for bar, p in zip(bars, pcts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_xlabel("Número de sets contribuyentes")
    ax1.set_ylabel("% de detecciones ensemble")
    ax1.set_title(f"Membresía del ensemble\n{total:,} dets — {method}")
    ax1.set_xticks(ks)
    ax1.set_ylim(0, max(pcts) * 1.15)

    bottom = np.zeros(n_cats)
    for k_idx, k in enumerate(ks):
        vals = np.array([cat_hist[c][k_idx] / max(sum(cat_hist[c]), 1) * 100
                         for c in cat_names])
        ax2.bar(range(n_cats), vals, bottom=bottom,
                label=f"{k} set{'s' if k > 1 else ''}", color=colors[k_idx])
        bottom += vals

    ax2.set_xticks(range(n_cats))
    ax2.set_xticklabels([c.replace(" ", "\n") for c in cat_names], fontsize=7)
    ax2.set_ylabel("% de detecciones por categoría")
    ax2.set_title("Membresía por categoría")
    ax2.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=8,
    )
    ax2.set_ylim(0, 110)

    fig.suptitle("Histograma de membresía del ensemble — DINO single-term", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
