"""
Visualizacion del benchmark multi-detector.

Genera para cada imagen:
  - PNG grid: filas = detectores, columnas = prompt sets (5) + ensemble. Ultima fila = GT.
  - HTML report estatico navegable con filtros por clase y detector.

Uso
---
    python -m src.utils.visualize_benchmark \\
        --gt_file   results/benchmark/benchmark_100_gt.json \\
        --img_dir   results/preprocess/tile_images/train_unlabeled_eval/images \\
        --dets_dir  results/benchmark \\
        --output_dir results/benchmark/viz
"""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paleta de colores por clase (consistente entre detectores)
# ---------------------------------------------------------------------------

_CLASS_COLORS = {
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
_DEFAULT_COLOR = "#aaaaaa"
_GT_COLOR = "#ffffff"

_DETECTORS = ["dino", "detic", "gdsam2"]
_DETECTOR_LABELS = {
    "dino": "DINO argmax",
    "detic": "Detic",
    "gdsam2": "Grounded SAM2+SAHI",
}
_PROMPT_SETS = [
    "simple",
    "aerial_compact",
    "satellite_verbose",
    "scene_context",
    "original",
]
_LABEL_KEYS = {"dino": "dino_label", "detic": "detic_label", "gdsam2": "gdsam2_label"}

# ---------------------------------------------------------------------------
# Carga de detecciones
# ---------------------------------------------------------------------------


def _load_detections(dets_dir: Path) -> dict:
    """
    Carga todas las detecciones en estructura:
      dets[detector][prompt_set | "ensemble"] = list[dict]
    """
    dets: dict[str, dict[str, list[dict]]] = {}
    for det in _DETECTORS:
        dets[det] = {}
        raw_path = dets_dir / det / "raw" / "detection_results.json"
        agg_path = dets_dir / det / "aggregated" / "detection_results.json"

        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            for ps in _PROMPT_SETS:
                dets[det][ps] = [d for d in raw if d.get("prompt_set") == ps]

        if agg_path.exists():
            dets[det]["ensemble"] = json.loads(agg_path.read_text())

    return dets


# ---------------------------------------------------------------------------
# Dibujado de bboxes sobre un ax de matplotlib
# ---------------------------------------------------------------------------


def _draw_boxes(
    ax,
    img: Image.Image,
    detections: list[dict],
    label_key: str,
    score_thresh: float = 0.0,
    linewidth: float = 1.0,
    is_gt: bool = False,
):
    ax.imshow(img)
    ax.axis("off")
    for d in detections:
        if d.get("score", 1.0) < score_thresh:
            continue
        x, y, w, h = d["bbox"]
        if is_gt:
            cls = d.get("category_name", "?")
            color = _GT_COLOR
        else:
            cls = d.get("class_name") or d.get(label_key, "?")
            color = _CLASS_COLORS.get(cls, _DEFAULT_COLOR)
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


def _draw_legend(ax, class_names: list[str]):
    ax.axis("off")
    patches = [
        mpatches.Patch(color=_CLASS_COLORS.get(cn, _DEFAULT_COLOR), label=cn)
        for cn in class_names
    ]
    patches.append(mpatches.Patch(color=_GT_COLOR, label="GT"))
    ax.legend(
        handles=patches,
        loc="center",
        fontsize=6,
        ncol=2,
        framealpha=0.8,
        edgecolor="gray",
    )


# ---------------------------------------------------------------------------
# PNG grid por imagen
# ---------------------------------------------------------------------------


def render_image_grid(
    image_id: int,
    img: Image.Image,
    dets: dict,
    gt_anns: list[dict],
    cat_id_to_name: dict[int, str],
    output_path: Path,
    score_thresh: float = 0.0,
) -> None:
    gt_with_names = [
        {**a, "category_name": cat_id_to_name.get(a["category_id"], "?"), "score": 1.0}
        for a in gt_anns
    ]

    cols = _PROMPT_SETS + ["ensemble"]
    n_cols = len(cols) + 1  # +1 para leyenda
    n_rows = len(_DETECTORS) + 1  # +1 para GT

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.5),
        dpi=80,
        gridspec_kw={"width_ratios": [1] * len(cols) + [0.6]},
    )
    fig.patch.set_facecolor("#1a1a1a")

    col_headers = [ps.replace("_", "\n") for ps in _PROMPT_SETS] + ["ENSEMBLE"]

    for ci, header in enumerate(col_headers):
        axes[0, ci].set_title(header, color="white", fontsize=7, pad=3)

    for ri, det in enumerate(_DETECTORS):
        axes[ri, -1].text(
            0.5,
            0.5,
            _DETECTOR_LABELS[det],
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            transform=axes[ri, -1].transAxes,
            wrap=True,
        )
        axes[ri, -1].axis("off")
        axes[ri, -1].set_facecolor("#1a1a1a")

        for ci, col in enumerate(cols):
            ax = axes[ri, ci]
            ax.set_facecolor("#1a1a1a")
            img_dets = [
                d for d in dets.get(det, {}).get(col, []) if d["image_id"] == image_id
            ]
            lw = 1.5 if col == "ensemble" else 1.0
            _draw_boxes(
                ax,
                img,
                img_dets,
                _LABEL_KEYS.get(det, "class_name"),
                score_thresh=score_thresh,
                linewidth=lw,
            )
            n_d = len(img_dets)
            ax.set_xlabel(f"{n_d}", color="#aaaaaa", fontsize=6, labelpad=1)

    # Fila GT
    gt_row = len(_DETECTORS)
    _draw_legend(axes[gt_row, -1], list(_CLASS_COLORS.keys()))
    axes[gt_row, -1].set_facecolor("#1a1a1a")
    axes[gt_row, 0].set_facecolor("#1a1a1a")
    _draw_boxes(
        axes[gt_row, 0], img, gt_with_names, "category_name", linewidth=1.5, is_gt=True
    )
    axes[gt_row, 0].set_title("GT", color="white", fontsize=7, pad=3)
    for ci in range(1, len(cols)):
        axes[gt_row, ci].axis("off")
        axes[gt_row, ci].set_facecolor("#1a1a1a")
        axes[gt_row, ci].imshow(img, alpha=0.2)

    for ax in axes.flat:
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    fig.suptitle(f"Image {image_id}", color="white", fontsize=9, y=1.0)
    plt.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Thumbnail en base64 para HTML
# ---------------------------------------------------------------------------


def _img_to_b64(path: Path, max_size: int = 200) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def _grid_to_b64(path: Path) -> str:
    buf = BytesIO()
    img = Image.open(path).convert("RGB")
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def render_html_report(
    image_ids: list[int],
    img_dir: Path,
    gt_coco: dict,
    dets: dict,
    grids_dir: Path,
    output_path: Path,
    metrics: dict | None = None,
) -> None:
    id_to_meta = {m["id"]: m for m in gt_coco["images"]}
    id_to_anns: dict[int, list] = {}
    cat_id_to_name = {c["id"]: c["name"] for c in gt_coco["categories"]}
    for ann in gt_coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Calcular clases presentes por imagen
    img_classes: dict[int, list[str]] = {}
    for iid in image_ids:
        cls_set = {
            cat_id_to_name.get(a["category_id"], "?") for a in id_to_anns.get(iid, [])
        }
        img_classes[iid] = sorted(cls_set)

    all_classes = sorted(_CLASS_COLORS.keys())

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    # Métricas tabla
    metrics_html = ""
    if metrics:
        rows = ""
        for det_name, det_metrics in metrics.items():
            aw = det_metrics.get("aware", {})
            ag = det_metrics.get("agnostic", {})
            rows += f"""<tr>
              <td>{_DETECTOR_LABELS.get(det_name, det_name)}</td>
              <td>{_fmt(aw.get('AP50'))}</td>
              <td>{_fmt(ag.get('AP50'))}</td>
              <td>{_fmt(aw.get('AR_500'))}</td>
            </tr>"""
        metrics_html = f"""
        <table class="metrics">
          <thead><tr><th>Detector</th><th>AP50 aware</th><th>AP50 agnostic</th><th>AR@500 aware</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    # Cards de imágenes
    cards_js_data = []
    for iid in image_ids:
        meta = id_to_meta.get(iid, {})
        fname = meta.get("file_name", "")
        img_path = img_dir / fname
        grid_path = grids_dir / f"img_{iid}.png"

        thumb_b64 = _img_to_b64(img_path) if img_path.exists() else ""
        grid_b64 = _grid_to_b64(grid_path) if grid_path.exists() else ""
        n_gt = len(id_to_anns.get(iid, []))
        classes = img_classes.get(iid, [])

        cards_js_data.append(
            {
                "id": iid,
                "fname": fname,
                "thumb": thumb_b64,
                "grid": grid_b64,
                "classes": classes,
                "n_gt": n_gt,
            }
        )

    cards_json = json.dumps(cards_js_data)

    class_filter_html = "".join(
        f'<label><input type="checkbox" class="cls-filter" value="{c}" checked> {c}</label>\n'
        for c in all_classes
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Benchmark Multi-Detector OV</title>
<style>
  body {{ background:#111; color:#eee; font-family:sans-serif; margin:0; padding:16px; }}
  h1 {{ color:#fff; margin-bottom:8px; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:16px; margin-bottom:16px; background:#1e1e1e;
               padding:12px; border-radius:8px; }}
  .controls label {{ font-size:13px; cursor:pointer; }}
  .metrics {{ border-collapse:collapse; margin-bottom:16px; font-size:13px; }}
  .metrics th, .metrics td {{ padding:6px 12px; border:1px solid #444; text-align:right; }}
  .metrics th {{ background:#2a2a2a; color:#aaa; }}
  .metrics td:first-child {{ text-align:left; }}
  #gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:10px; }}
  .card {{ background:#1e1e1e; border-radius:6px; overflow:hidden; cursor:pointer;
           border:2px solid transparent; transition:border .15s; }}
  .card:hover {{ border-color:#555; }}
  .card img.thumb {{ width:100%; display:block; }}
  .card .info {{ padding:6px 8px; font-size:11px; color:#aaa; }}
  .card .cls-chips {{ display:flex; flex-wrap:wrap; gap:3px; padding:4px 8px 6px; }}
  .chip {{ font-size:9px; padding:1px 5px; border-radius:10px; color:#000; font-weight:bold; }}
  #modal {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.85);
            z-index:100; overflow:auto; padding:20px; }}
  #modal img {{ max-width:100%; border-radius:4px; }}
  #modal .close {{ position:fixed; top:12px; right:20px; font-size:28px;
                   cursor:pointer; color:#fff; background:#333; border:none;
                   border-radius:50%; width:36px; height:36px; line-height:34px;
                   text-align:center; }}
  #modal .caption {{ color:#aaa; font-size:12px; margin-top:8px; }}
</style>
</head>
<body>
<h1>Benchmark Multi-Detector OV &mdash; 100 im&aacute;genes</h1>
{metrics_html}
<div class="controls">
  <div>
    <strong style="display:block;margin-bottom:4px;font-size:12px;color:#888">Filtrar por clase GT</strong>
    <div style="display:flex;flex-wrap:wrap;gap:6px">{class_filter_html}</div>
  </div>
  <div>
    <button onclick="selectAll(true)" style="margin-right:4px">Todas</button>
    <button onclick="selectAll(false)">Ninguna</button>
    &nbsp;
    <input id="search" placeholder="Buscar imagen..." style="background:#222;color:#eee;border:1px solid #444;padding:4px 8px;border-radius:4px;">
  </div>
</div>
<div id="gallery"></div>
<div id="modal">
  <button class="close" onclick="closeModal()">&#x2715;</button>
  <img id="modal-img" src="">
  <div class="caption" id="modal-caption"></div>
</div>
<script>
const DATA = {cards_json};
const COLORS = {json.dumps(_CLASS_COLORS)};

function chipColor(cls) {{ return COLORS[cls] || '#aaa'; }}

function renderCards(data) {{
  const gallery = document.getElementById('gallery');
  gallery.innerHTML = '';
  data.forEach(card => {{
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `
      <img class="thumb" src="data:image/jpeg;base64,${{card.thumb}}" loading="lazy">
      <div class="cls-chips">${{card.classes.map(c =>
        `<span class="chip" style="background:${{chipColor(c)}}">${{c.split(' ')[0]}}</span>`
      ).join('')}}</div>
      <div class="info">ID ${{card.id}} &bull; GT: ${{card.n_gt}} anot.</div>`;
    div.onclick = () => openModal(card);
    gallery.appendChild(div);
  }});
}}

function openModal(card) {{
  document.getElementById('modal-img').src = 'data:image/jpeg;base64,' + card.grid;
  document.getElementById('modal-caption').textContent = card.fname + ' (ID ' + card.id + ')';
  document.getElementById('modal').style.display = 'block';
}}
function closeModal() {{ document.getElementById('modal').style.display = 'none'; }}
document.getElementById('modal').addEventListener('click', e => {{
  if(e.target === document.getElementById('modal')) closeModal();
}});

function getActiveClasses() {{
  return [...document.querySelectorAll('.cls-filter:checked')].map(el => el.value);
}}
function selectAll(v) {{
  document.querySelectorAll('.cls-filter').forEach(el => el.checked = v);
  applyFilters();
}}
function applyFilters() {{
  const active = new Set(getActiveClasses());
  const query = document.getElementById('search').value.toLowerCase();
  const filtered = DATA.filter(card => {{
    const hasClass = card.classes.some(c => active.has(c));
    const matchesSearch = query === '' || card.fname.toLowerCase().includes(query) ||
                          String(card.id).includes(query);
    return hasClass && matchesSearch;
  }});
  renderCards(filtered);
}}

document.querySelectorAll('.cls-filter').forEach(el => el.addEventListener('change', applyFilters));
document.getElementById('search').addEventListener('input', applyFilters);
renderCards(DATA);
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"HTML report → {output_path}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def visualize_benchmark(
    gt_file: Path,
    img_dir: Path,
    dets_dir: Path,
    output_dir: Path,
    score_thresh: float = 0.0,
    max_images: int | None = None,
) -> None:
    with gt_file.open(encoding="utf-8") as fh:
        gt_coco = json.load(fh)

    image_ids = [m["id"] for m in gt_coco["images"]]
    if max_images:
        image_ids = image_ids[:max_images]

    cat_id_to_name = {c["id"]: c["name"] for c in gt_coco["categories"]}
    id_to_anns: dict[int, list] = {}
    for ann in gt_coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)
    id_to_meta = {m["id"]: m for m in gt_coco["images"]}

    print(f"Cargando detecciones desde {dets_dir} ...")
    dets = _load_detections(dets_dir)

    grids_dir = output_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generando {len(image_ids)} PNG grids ...")
    for i, iid in enumerate(image_ids):
        meta = id_to_meta[iid]
        img_path = img_dir / meta["file_name"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        grid_path = grids_dir / f"img_{iid}.png"
        render_image_grid(
            image_id=iid,
            img=img,
            dets=dets,
            gt_anns=id_to_anns.get(iid, []),
            cat_id_to_name=cat_id_to_name,
            output_path=grid_path,
            score_thresh=score_thresh,
        )
        print(f"  {i+1}/{len(image_ids)}  img_{iid}.png", end="\r")
    print()

    # Cargar métricas si existen
    metrics: dict = {}
    for det in _DETECTORS:
        m_aw = dets_dir / det / "eval_aware" / "metrics.json"
        m_ag = dets_dir / det / "eval_agnostic" / "metrics.json"
        if m_aw.exists() or m_ag.exists():
            metrics[det] = {}
            if m_aw.exists():
                metrics[det]["aware"] = json.loads(m_aw.read_text())
            if m_ag.exists():
                metrics[det]["agnostic"] = json.loads(m_ag.read_text())

    print("Generando HTML report ...")
    render_html_report(
        image_ids=image_ids,
        img_dir=img_dir,
        gt_coco=gt_coco,
        dets=dets,
        grids_dir=grids_dir,
        output_path=output_dir / "report.html",
        metrics=metrics or None,
    )
    print(f"Listo. Abre {output_dir / 'report.html'} en el navegador.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_file", type=Path, default=Path("results/benchmark/benchmark_100_gt.json")
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=Path("results/preprocess/tile_images/train_unlabeled_eval/images"),
    )
    parser.add_argument("--dets_dir", type=Path, default=Path("results/benchmark"))
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/benchmark/viz")
    )
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    visualize_benchmark(
        gt_file=args.gt_file,
        img_dir=args.img_dir,
        dets_dir=args.dets_dir,
        output_dir=args.output_dir,
        score_thresh=args.score_thresh,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
