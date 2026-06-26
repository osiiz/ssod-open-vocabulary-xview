"""
Compara la distribución de scores DINO por clase entre dos configuraciones
de prompt:

    - single-term  : 1 término por clase, agregado sobre todos los _ckpt_*.json
    - synonyms     : 3 sinónimos por clase (config anterior, raw aggregated)

Salida: PNG con histograma por clase (single-term vs synonyms en cada subplot)
y una tabla MD con estadísticas (media, mediana, p25, p75, p90, p95) por clase.

Útil para verificar empíricamente la hipótesis: "single-term sube los scores
porque elimina la competencia entre tokens de frases sinónimas en el bloque
de cross-attention de Grounding DINO".

Uso:
    python scripts/compare_dino_scores.py \
        --single_term_dir results/dino/single_term_raw/ \
        --synonyms_file   results/dino/ensemble_argmax_synonyms_raw/detection_results.json \
        --output_dir      docs/results_reports/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def _load_scores_by_class(detections: list) -> dict[str, np.ndarray]:
    """Agrupa scores por class_name. Devuelve dict {class_name: np.ndarray}."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for d in detections:
        cls = d.get("class_name")
        if not cls:
            continue
        buckets[cls].append(float(d["score"]))
    return {c: np.asarray(v, dtype=float) for c, v in buckets.items()}


def _load_single_term(path_dir: Path) -> dict[str, np.ndarray]:
    """
    Single-term: 5 archivos _ckpt_set*.json, uno por prompt_set.
    Concatena scores de las 5 inferencias.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    files = sorted(path_dir.glob("_ckpt_set*.json"))
    if not files:
        raise FileNotFoundError(f"No se encontraron _ckpt_set*.json en {path_dir}")
    print(f"Cargando single-term: {len(files)} archivos")
    for f in files:
        with f.open() as fh:
            for d in json.load(fh):
                cls = d.get("class_name")
                if cls:
                    buckets[cls].append(float(d["score"]))
        print(f"  OK: {f.name}")
    return {c: np.asarray(v, dtype=float) for c, v in buckets.items()}


def _load_synonyms(path: Path) -> dict[str, np.ndarray]:
    """Synonyms: un único detection_results.json agregado."""
    print(f"Cargando synonyms: {path}")
    with path.open() as fh:
        data = json.load(fh)
    return _load_scores_by_class(data)


# ---------------------------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------------------------


PERCENTILES = [25, 50, 75, 90, 95]


def _stats(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"n": 0, "mean": float("nan"), **{f"p{p}": float("nan") for p in PERCENTILES}}
    return {
        "n": int(scores.size),
        "mean": float(scores.mean()),
        **{f"p{p}": float(np.percentile(scores, p)) for p in PERCENTILES},
    }


# ---------------------------------------------------------------------------
# Gráfica
# ---------------------------------------------------------------------------


def _plot_comparison(
    single_term: dict[str, np.ndarray],
    synonyms: dict[str, np.ndarray],
    output: Path,
    bins: int = 60,
) -> None:
    all_classes = sorted(set(single_term) | set(synonyms))
    n = len(all_classes)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, cls in enumerate(all_classes):
        ax = axes[i]
        st = single_term.get(cls, np.array([]))
        sy = synonyms.get(cls, np.array([]))

        # Mismo rango para comparar visualmente
        joint = np.concatenate([st, sy]) if (st.size + sy.size) else np.array([0.0, 1.0])
        lo, hi = float(joint.min()), float(joint.max())
        if hi <= lo:
            hi = lo + 1e-3
        edges = np.linspace(lo, hi, bins + 1)

        if sy.size:
            ax.hist(
                sy, bins=edges, alpha=0.55, color="#E53935",
                label=f"synonyms (n={sy.size:,}, med={np.median(sy):.3f})",
                density=True,
            )
        if st.size:
            ax.hist(
                st, bins=edges, alpha=0.55, color="#1E88E5",
                label=f"single-term (n={st.size:,}, med={np.median(st):.3f})",
                density=True,
            )
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel("score", fontsize=9)
        ax.set_ylabel("densidad", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Distribución de scores DINO por clase: single-term vs synonyms (raw)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"PNG: {output}")


def _write_md(
    single_term: dict[str, np.ndarray],
    synonyms: dict[str, np.ndarray],
    md_path: Path,
) -> None:
    all_classes = sorted(set(single_term) | set(synonyms))
    lines = [
        "# Comparación de scores DINO: single-term vs synonyms (raw detections)",
        "",
        "Scores extraídos de las inferencias raw (sin ensemble ni umbral) agrupados por `class_name`.",
        "",
        "**single-term**: concatenación de los 5 ckpt (set1..set5), 1 término por clase y prompt set.",
        "**synonyms**: aggregated raw de la config anterior (4 prompt sets × 3 sinónimos por clase).",
        "",
        "| Clase | Config | N | mean | p25 | p50 | p75 | p90 | p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for cls in all_classes:
        for label, scores in [("single-term", single_term.get(cls, np.array([]))),
                              ("synonyms", synonyms.get(cls, np.array([])))]:
            s = _stats(scores)
            if s["n"] == 0:
                lines.append(f"| {cls} | {label} | 0 | — | — | — | — | — | — |")
            else:
                lines.append(
                    f"| {cls} | {label} | {s['n']:,} | "
                    f"{s['mean']:.4f} | {s['p25']:.4f} | {s['p50']:.4f} | "
                    f"{s['p75']:.4f} | {s['p90']:.4f} | {s['p95']:.4f} |"
                )

    lines.extend([
        "",
        "## Resumen",
        "",
        "Para cada clase, compara `mean(single-term) - mean(synonyms)` y `p50(single-term) - p50(synonyms)`.",
        "Una diferencia positiva consistente indica que single-term concentra la atención de los tokens",
        "(eliminando la competencia entre sinónimos), produciendo `sigmoid(max_token)` más altos.",
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD : {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara distribuciones de scores DINO entre single-term y synonyms.",
    )
    parser.add_argument(
        "--single_term_dir", type=Path,
        default=Path("results/dino/single_term_raw"),
        help="Directorio con los _ckpt_set*.json de single-term",
    )
    parser.add_argument(
        "--synonyms_file", type=Path,
        default=Path("results/dino/ensemble_argmax_synonyms_raw/detection_results.json"),
        help="Archivo aggregated raw de synonyms",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("docs/results_reports"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    st = _load_single_term(args.single_term_dir)
    sy = _load_synonyms(args.synonyms_file)

    print()
    print(f"Clases single-term: {sorted(st)}")
    print(f"Clases synonyms:    {sorted(sy)}")
    print()

    _plot_comparison(st, sy, args.output_dir / "dino_score_comparison.png")
    _write_md(st, sy, args.output_dir / "dino_score_comparison.md")


if __name__ == "__main__":
    main()
