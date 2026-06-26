"""Visualiza el comportamiento del prompt ensemble de DINO por imagen.

Para cada imagen seleccionada genera una figura con 3 paneles:
  - Panel 1 (GT):       anotaciones ground-truth coloreadas por clase.
  - Panel 2 (Prompts):  detecciones individuales de los 5 prompt sets superpuestas
                        en la misma imagen (cada prompt = color distinto).
  - Panel 3 (Ensemble): resultado de la fusión con uncertainty anotada sobre cada caja.

Criterio de selección: imágenes con al menos una detección ensemble grande
(area ≥ 9216 px²) de la categoría objetivo (Aircraft o Vessel).

Uso:
    python scripts/visualize_ensemble_prompts.py --category aircraft --n_images 8
    python scripts/visualize_ensemble_prompts.py --category vessel --n_images 8
    python scripts/visualize_ensemble_prompts.py --category aircraft vessel --n_images 5
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Rutas por modelo
# ---------------------------------------------------------------------------
PATHS = {
    "dino": {
        "full":     Path("results/dino/ensemble_argmax_aggregated/detection_results.json"),
        "so":       Path("results/dino/ensemble_argmax_so_aggregated/detection_results.json"),
        "synonyms": Path("results/dino/ensemble_argmax_synonyms_aggregated/detection_results.json"),
        "sos":      Path("results/dino/ensemble_argmax_sos_aggregated/detection_results.json"),
        "ckpt_dir": Path("results/dino/ensemble_argmax_raw"),
    },
    "dino_single": {
        "baseline":              Path("results/dino/single_term_aggregated/detection_results.json"),
        "uf_score_weighted":           Path("results/dino/single_term_aggregated_uf_score_weighted/detection_results.json"),
        "borda":                       Path("results/dino/single_term_aggregated_borda/detection_results.json"),
        "wbf_lib":                     Path("results/dino/single_term_aggregated_wbf_lib/detection_results.json"),
        "uf_centroid_and":             Path("results/dino/single_term_aggregated_uf_centroid_and/detection_results.json"),
        "uf_centroid_and_score_weighted": Path("results/dino/single_term_aggregated_uf_centroid_and_score_weighted/detection_results.json"),
        "ckpt_dir": Path("results/dino/single_term_raw"),
    },
    "rex": {
        "full":     Path("results/rexomni/ensemble/detection_results.json"),
        "ckpt_dir": Path("results/rexomni/ensemble_raw"),
    },
    "rex_single": {
        "vote":      Path("results/rex/single_term_aggregated/detection_results.json"),
        "ckpt_dir":  Path("results/rex/single_term_raw"),
    },
}

# Mantener alias para compatibilidad con código existente (se sobreescriben en main)
ENSEMBLE_DETS         = PATHS["dino"]["full"]
ENSEMBLE_SO_DETS      = PATHS["dino"]["so"]
ENSEMBLE_SYNONYMS_DETS = PATHS["dino"]["synonyms"]
ENSEMBLE_SOS_DETS     = PATHS["dino"]["sos"]
CKPT_DIR = PATHS["dino"]["ckpt_dir"]
ANN_FILE = Path(
    "results/preprocess/tile_images/train_unlabeled_eval/COCO_annotations.json"
)
IMG_DIR = Path("results/preprocess/tile_images/train_unlabeled_eval/images")
OUT_BASE = Path("docs/visualizations/ensemble_prompts")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
LARGE_AREA_THRESH = 9216  # 96×96 px²
SCORE_THRESH = 0.1

PROMPT_COLORS = {
    # ensemble argmax (sets originales)
    "simple": "#1f77b4",
    "aerial_compact": "#ff7f0e",
    "satellite_verbose": "#2ca02c",
    "scene_context": "#d62728",
    "original": "#9467bd",
    "synonyms": "#8c564b",
    # single-term sets
    "set1_direct": "#1f77b4",
    "set2_synonyms_a": "#ff7f0e",
    "set3_synonyms_b": "#2ca02c",
    "set4_synonyms_c": "#d62728",
    "set5_synonyms_d": "#9467bd",
}

CLASS_COLORS = {
    "Aircraft": "#e6194b",
    "Light Vehicle": "#3cb44b",
    "Heavy Vehicle": "#ffe119",
    "Railway Vehicle": "#4363d8",
    "Maritime Vessel": "#f58231",
    "Engineering Vehicle": "#911eb4",
    "Building": "#42d4f4",
    "Storage Tank": "#f032e6",
    "Tower & Pylon": "#bfef45",
}
GT_COLOR = "#ffffff"

CATEGORY_IDS = {
    "aircraft": 1,
    "vessel": 5,
}
CATEGORY_LABELS = {
    "aircraft": "Aircraft",
    "vessel": "Maritime Vessel",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def bbox_area(bbox):
    return bbox[2] * bbox[3]


def draw_boxes_gt(ax, img: Image.Image, gt_anns: list, cat_names: dict):
    ax.imshow(img)
    ax.axis("off")
    for ann in gt_anns:
        x, y, w, h = ann["bbox"]
        cls = cat_names.get(ann["category_id"], "?")
        color = CLASS_COLORS.get(cls, "#aaaaaa")
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="square,pad=0",
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_title("GT", fontsize=9, pad=2)


def draw_boxes_prompts(ax, img: Image.Image, per_prompt: dict[str, list]):
    """Dibuja detecciones de cada prompt en su color; filtra score >= SCORE_THRESH."""
    ax.imshow(img)
    ax.axis("off")
    for prompt_set, dets in per_prompt.items():
        color = PROMPT_COLORS.get(prompt_set, "#aaaaaa")
        for d in dets:
            if d.get("score", 1.0) < SCORE_THRESH:
                continue
            x, y, w, h = d["bbox"]
            rect = mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="square,pad=0",
                linewidth=1.0,
                edgecolor=color,
                facecolor="none",
                alpha=0.8,
            )
            ax.add_patch(rect)

    # Solo mostrar en leyenda los prompts que realmente aparecen en per_prompt
    active_prompts = [ps for ps in PROMPT_COLORS if ps in per_prompt]
    n_by_prompt = {
        ps: sum(1 for d in per_prompt[ps] if d.get("score", 1.0) >= SCORE_THRESH)
        for ps in active_prompts
    }
    legend_patches = [
        mpatches.Patch(color=PROMPT_COLORS[ps], label=f"{ps} ({n_by_prompt[ps]})")
        for ps in active_prompts
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=5.5,
        framealpha=0.7,
        edgecolor="gray",
    )
    total = sum(n_by_prompt.values())
    ax.set_title(
        f"5 prompt sets — score≥{SCORE_THRESH} ({total} dets)", fontsize=9, pad=2
    )


def draw_boxes_ensemble(ax, img: Image.Image, ensemble_dets: list):
    ax.imshow(img)
    ax.axis("off")
    for d in ensemble_dets:
        if d.get("score", 1.0) < SCORE_THRESH:
            continue
        x, y, w, h = d["bbox"]
        cls = d.get("class_name", "?")
        color = CLASS_COLORS.get(cls, "#aaaaaa")
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="square,pad=0",
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        score = d.get("score", 0)
        loc_unc = d.get("loc_uncertainty", 0)
        cls_unc = d.get("class_uncertainty", 0)
        # sc = mean cluster score  |  σ = bbox std dev (px)  |  H = class vote entropy (nats)
        label = f"sc={score:.2f}  σ={loc_unc:.1f}  H={cls_unc:.2f}"
        ax.text(
            x,
            max(y - 3, 0),
            label,
            fontsize=5.5,
            color="yellow",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.6, pad=1.5, linewidth=0),
        )

    n = sum(1 for d in ensemble_dets if d.get("score", 1.0) >= SCORE_THRESH)
    ax.set_title(
        f"Ensemble fusión — score≥{SCORE_THRESH} ({n} dets)", fontsize=10, pad=3
    )
    # Nota explicativa de métricas
    ax.text(
        0.01,
        0.01,
        "sc = score medio del cluster\nσ = incert. localización (px)\nH = entropía de votos de clase (nats)",
        transform=ax.transAxes,
        fontsize=6,
        color="white",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.6, pad=3, linewidth=0),
    )


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def load_annotations():
    data = json.loads(ANN_FILE.read_text())
    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    img_meta = {img["id"]: img for img in data["images"]}
    gt_by_image: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    return cat_names, img_meta, gt_by_image


def load_ensemble_dets(path: Path = ENSEMBLE_DETS):
    dets = json.loads(path.read_text())
    by_image: dict[int, list] = defaultdict(list)
    for d in dets:
        by_image[d["image_id"]].append(d)
    return by_image


def load_checkpoint_dets(
    target_image_ids: set[int], prompt_sets: set[str] | None = None
) -> dict[int, dict[str, list]]:
    """Carga checkpoints y devuelve {image_id: {prompt_set: [dets]}} solo para target_image_ids.

    Soporta dos estructuras:
    - DINO: ficheros individuales _ckpt_{prompt_set}.json en CKPT_DIR.
    - Rex: único detection_results.json con campo 'prompt_set' en cada detección.

    prompt_sets: si se especifica, solo carga los checkpoints de esos prompt sets.
    """
    print(f"Cargando checkpoints para {len(target_image_ids)} imágenes...")
    per_image: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    ckpt_files = sorted(f for f in CKPT_DIR.glob("_ckpt_*.json") if "probs" not in f.name)
    merged_file = CKPT_DIR / "detection_results.json"

    if ckpt_files:
        # Estructura DINO: un fichero por prompt set
        for ckpt_path in ckpt_files:
            ps_name = ckpt_path.stem.removeprefix("_ckpt_")
            if prompt_sets is not None and ps_name not in prompt_sets:
                continue
            print(f"  {ckpt_path.name} ...")
            dets = json.loads(ckpt_path.read_text())
            for d in dets:
                if d["image_id"] in target_image_ids:
                    ps = d.get("prompt_set", ps_name)
                    per_image[d["image_id"]][ps].append(d)
    elif merged_file.exists():
        # Estructura Rex: fichero único con campo prompt_set por detección
        print(f"  {merged_file.name} (fichero único con todos los prompts)...")
        dets = json.loads(merged_file.read_text())
        for d in dets:
            if d["image_id"] not in target_image_ids:
                continue
            ps = d.get("prompt_set", "unknown")
            if prompt_sets is not None and ps not in prompt_sets:
                continue
            per_image[d["image_id"]][ps].append(d)
    else:
        print(f"  [WARN] No se encontraron checkpoints en {CKPT_DIR}")

    return per_image


# ---------------------------------------------------------------------------
# Selección de imágenes
# ---------------------------------------------------------------------------


def select_images(
    ensemble_by_image: dict,
    gt_by_image: dict,
    cat_id: int,
    n: int,
) -> list[int]:
    """Elige las n imágenes con GT confirmado de la categoría y menos dets ensemble totales.

    Criterios:
      1. La imagen tiene al menos 1 anotación GT grande (area >= LARGE_AREA_THRESH)
         de la categoría objetivo.
      2. El ensemble produce al menos 1 det grande (score >= SCORE_THRESH) de esa categoría.
      3. Ordenar por total de dets ensemble con score >= SCORE_THRESH (ascendente),
         para mostrar las imágenes más limpias primero.
    """
    scored = []
    for img_id, gt_anns in gt_by_image.items():
        large_gt = [
            a
            for a in gt_anns
            if a["category_id"] == cat_id
            and a.get("area", a["bbox"][2] * a["bbox"][3]) >= LARGE_AREA_THRESH
        ]
        if not large_gt:
            continue

        ens_dets = ensemble_by_image.get(img_id, [])
        ens_thresh = [d for d in ens_dets if d.get("score", 1.0) >= SCORE_THRESH]
        large_ens = [
            d
            for d in ens_thresh
            if d.get("category_id") == cat_id
            and bbox_area(d["bbox"]) >= LARGE_AREA_THRESH
        ]
        if not large_ens:
            continue

        scored.append((img_id, len(ens_thresh)))

    # Las n imágenes con menos detecciones totales (más limpias)
    scored.sort(key=lambda x: x[1])
    selected = [img_id for img_id, _ in scored[:n]]
    print(
        f"  Imágenes candidatas (GT+ens, score≥{SCORE_THRESH}): {len(scored)}, seleccionadas: {len(selected)}"
    )
    if scored[:n]:
        print(
            f"  Rango de dets totales: {scored[0][1]}–{scored[min(n,len(scored))-1][1]}"
        )
    return selected


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def draw_class_legend(ax):
    ax.axis("off")
    patches = [
        mpatches.Patch(color=color, label=cls) for cls, color in CLASS_COLORS.items()
    ]
    ax.legend(
        handles=patches,
        loc="center",
        fontsize=8,
        framealpha=0.9,
        edgecolor="gray",
        title="Clases",
        title_fontsize=9,
    )


def render_image(
    image_id: int,
    img_meta: dict,
    cat_names: dict,
    gt_anns: list,
    per_prompt: dict[str, list],
    ensemble_dets: list,
    out_path: Path,
):
    fname = img_meta["file_name"]
    img_path = IMG_DIR / fname
    if not img_path.exists():
        print(f"  [SKIP] imagen no encontrada: {img_path}")
        return

    img = Image.open(img_path).convert("RGB")

    fig, axes = plt.subplots(
        1, 4, figsize=(24, 6), gridspec_kw={"width_ratios": [4, 4, 4, 1]}, dpi=200
    )
    fig.suptitle(f"Image {image_id} — {fname}", fontsize=11, y=1.01)

    draw_boxes_gt(axes[0], img, gt_anns, cat_names)
    draw_boxes_prompts(axes[1], img, per_prompt)
    draw_boxes_ensemble(axes[2], img, ensemble_dets)
    draw_class_legend(axes[3])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Guardado: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        nargs="+",
        default=["aircraft", "vessel"],
        choices=["aircraft", "vessel"],
    )
    parser.add_argument("--n_images", type=int, default=10)
    parser.add_argument(
        "--model",
        choices=list(PATHS.keys()),
        default="dino",
        help="Modelo: dino | dino_single | rex | rex_single",
    )
    parser.add_argument(
        "--ensemble",
        default=None,
        help=(
            "Variante de ensemble. Opciones por modelo — "
            "dino: full|so|synonyms|sos; "
            "dino_single: baseline|uf_score_weighted|borda|wbf_lib|uf_centroid_and|uf_centroid_and_score_weighted; "
            "rex/rex_single: full|vote. "
            "Si se omite, usa la primera variante disponible."
        ),
    )
    args = parser.parse_args()

    model_paths = PATHS[args.model]
    available = [k for k in model_paths if k != "ckpt_dir"]

    if args.ensemble is None:
        args.ensemble = available[0]
        print(f"  --ensemble no especificado, usando: {args.ensemble}")

    if args.ensemble not in model_paths:
        print(f"[ERROR] El modelo '{args.model}' no tiene ensemble '{args.ensemble}'. "
              f"Opciones: {available}")
        return

    global CKPT_DIR
    CKPT_DIR = model_paths["ckpt_dir"]

    print("Cargando anotaciones GT...")
    cat_names, img_meta, gt_by_image = load_annotations()

    ens_path = model_paths[args.ensemble]
    print(f"Cargando ensemble detecciones ({args.model}/{args.ensemble}): {ens_path}")
    ensemble_by_image = load_ensemble_dets(ens_path)

    # Para modelos single-term cargamos todos los sets; para los otros, filtro por variante
    prompt_filter = {
        "so": {"simple", "original"},
        "synonyms": {"synonyms"},
        "sos": {"simple", "original", "synonyms"},
    }.get(args.ensemble)  # None = todos los prompts disponibles

    for category in args.category:
        cat_id = CATEGORY_IDS[category]
        cat_label = CATEGORY_LABELS[category]
        print(f"\n=== Categoría: {cat_label} (id={cat_id}) ===")

        selected_ids = select_images(
            ensemble_by_image, gt_by_image, cat_id, args.n_images
        )
        if not selected_ids:
            print(f"  No se encontraron imágenes con dets grandes de {cat_label}")
            continue

        ckpt_dets = load_checkpoint_dets(set(selected_ids), prompt_sets=prompt_filter)

        out_dir = OUT_BASE / f"{category}_{args.model}_{args.ensemble}"
        for img_id in selected_ids:
            meta = img_meta.get(img_id)
            if meta is None:
                continue
            gt_anns = gt_by_image.get(img_id, [])
            per_prompt = dict(ckpt_dets.get(img_id, {}))
            ens_dets = ensemble_by_image.get(img_id, [])

            out_path = out_dir / f"img_{img_id}.png"
            render_image(
                img_id, meta, cat_names, gt_anns, per_prompt, ens_dets, out_path
            )

    print("\nListo.")


if __name__ == "__main__":
    main()
