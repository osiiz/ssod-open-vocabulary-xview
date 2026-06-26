"""Gráfico de barras agrupadas: efecto do prompt-ensembling (con vs sen) no AP50 sobre test.

Xera unha figura comparando, para cada combinación con fonte de vocabulario aberto, o AP50 do
modelo con ensembling multi-prompt fronte á variante cun único prompt (sufixo _noens), coa cota
inferior (baseline) como referencia. Saída por defecto: memoria/figuras/comparison_noens.png.

Uso: python scripts/plot_noens_comparison.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ap50(path: Path) -> float:
    return json.load(open(path))["AP50"] * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe_root", type=Path, default=Path("results/inference_test_ssod_pe"))
    ap.add_argument("--baseline", type=Path,
                    default=Path("results/inference_test_ssod_baseline/metrics.json"))
    ap.add_argument("--output", type=Path, default=Path("memoria/figuras/comparison_noens.png"))
    args = ap.parse_args()

    combos = ["FT_GD", "FT_RO", "FT_GD_RO"]
    con = [ap50(args.pe_root / c / "metrics.json") for c in combos]
    sen = [ap50(args.pe_root / f"{c}_noens" / "metrics.json") for c in combos]
    base = ap50(args.baseline)

    x = np.arange(len(combos))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, con, w, label="Con prompt-ensembling", color="#1f77b4", alpha=0.9)
    b2 = ax.bar(x + w / 2, sen, w, label="Sen prompt-ensembling", color="#9ecae1", alpha=0.9)
    ax.axhline(base, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Liña base ({base:.2f})")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(combos)
    ax.set_ylabel("AP50 (base 100)")
    ax.set_title("Efecto do prompt-ensembling sobre o AP50 en test")
    ax.set_ylim(0, max(con) * 1.18)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"saved {args.output}")
    for c, a, b in zip(combos, con, sen):
        print(f"  {c}: con={a:.2f} sen={b:.2f} delta={a-b:+.2f}")


if __name__ == "__main__":
    main()
