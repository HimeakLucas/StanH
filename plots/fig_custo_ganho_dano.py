"""F8 - Cost x gain x damage: the thesis figure.

The whole axis of the work in one image: 320 -> 6.9 M -> 22 M -> 75 M TRAINABLE parameters
against the target-domain gain, with source-domain damage as a third dimension (marker size
and a dB annotation).

The x axis is TRAINABLE PARAMETERS, never "storage cost": those are two different cost
dimensions, and the materialized delta size is a single-point annotation, not an axis.

The `full` cell is a RANGE in both coordinates (two replicas): drawn as two connected
points, never as one.

`encoder_hyper` is an intermediate adapter, never a "lightweight" one. Its cross damage on
x-ray does not exist in the unit of this figure: the published value (+5.43%) is BD-Rate,
precisely the unit this work forbids for cross damage. The point enters with the position it
has (cost and gain) and the hollow marker declares the third dimension as not measured --
computing it here would make a number originate in a figure.

Usage:
    python plots/fig_custo_ganho_dano.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AZUL, VERMELHO, VERDE, ROSA, AMBAR, PRETO = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000")
DELTA_ENCODER = "results/a3_delta_sizes.json"

# label, trainable params, target BD (%), cross damage (dB) or None, color, marker.
# Every value comes from Table I of the report (patient-disjoint ruler).
PONTOS = [
    ("Só quantizador", 320, [1.23], [-0.06], ROSA, "s"),
    ("Encoder", 6.9e6, [2.38], [0.01], VERMELHO, "v"),
    ("Decoder", 6.9e6, [-3.43], [-3.13], AMBAR, "D"),
    ("encoder_hyper", 22e6, [-2.69], None, AZUL, "P"),
    ("Full backbone", 75e6, [-5.07, -6.48], [-1.25, -1.62], PRETO, "*"),
]
REF_DANO = 3.13     # maior |dano| do conjunto, ancora da escala de tamanho


def num(v, casas):
    return f"{v:.{casas}f}".replace(".", ",")


def tamanho(dano):
    return 26 + 210 * abs(dano) / REF_DANO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/figs_relatorio/f8_custo_ganho_dano.png")
    args = ap.parse_args()
    with open(os.path.join(ROOT, DELTA_ENCODER)) as fh:
        deltas = json.load(fh)
    mb = sorted({v["fp16_MB"] for v in deltas.values()})
    ancora = 301.7
    if len(mb) != 1:
        raise SystemExit(f"delta fp16 não é único entre os λ: {mb} — não plotado.")
    print(f"delta materializado do encoder: {mb[0]:.2f} MB em fp16, "
          f"{ancora / mb[0]:.1f}× menor que a âncora de {ancora} MB")

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.7,
    })
    fig, eixo = plt.subplots(figsize=(3.05, 3.0))

    for rotulo, params, bds, danos, cor, mk in PONTOS:
        if len(bds) > 1:                     # replicated cell: a range, not a point
            eixo.plot([params, params], bds, color=cor, linewidth=1.2, zorder=2)
        for i, bd in enumerate(bds):
            if danos is None:
                eixo.scatter([params], [bd], s=70, facecolors="none", edgecolors=cor,
                             marker=mk, linewidths=1.3, zorder=3)
            else:
                eixo.scatter([params], [bd], s=tamanho(danos[i]), color=cor,
                             marker=mk, zorder=3, edgecolors="white", linewidths=0.5)

    # Labels go to the legend rather than per-point annotations: at 3.4 in wide five
    # annotations overlap, and the dB damage is what has to stay readable.
    rotulos = {
        "Só quantizador": "Só quantizador · −0,06 dB",
        "Encoder": "Encoder · +0,01 dB",
        "Decoder": "Decoder · −3,13 dB",
        "encoder_hyper": "encoder_hyper · não medido",
        "Full backbone": "Full · −1,25 a −1,62 dB",
    }
    marcas = []
    for rotulo, _p, _b, danos, cor, mk in PONTOS:
        marcas.append(plt.Line2D([], [], marker=mk, linestyle="none", markersize=6,
                                 color=cor,
                                 markerfacecolor="none" if danos is None else cor,
                                 markeredgewidth=1.3 if danos is None else 0.5,
                                 markeredgecolor=cor if danos is None else "white"))
    fig.legend(marcas, [rotulos[p[0]] for p in PONTOS], loc="lower center", ncol=2,
               frameon=False, handlelength=1.0, columnspacing=0.8, fontsize=7.6,
               labelspacing=0.35, bbox_to_anchor=(0.52, 0.005))

    eixo.axhline(0, color=PRETO, linewidth=0.8)
    eixo.set_xscale("log")
    eixo.set_xlim(120, 4.5e8)
    eixo.set_ylim(-8.6, 5.6)
    eixo.set_xlabel("Parâmetros treináveis")
    eixo.set_ylabel("BD-Rate no raio-X (%)")
    eixo.set_xticks([1e3, 1e5, 1e7])
    eixo.set_xticklabels(["10³", "10⁵", "10⁷"])
    eixo.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: num(v, 0)))
    eixo.grid(True, alpha=0.25, linewidth=0.4)
    eixo.set_axisbelow(True)
    eixo.text(160, 4.75, "pior que a genérica ↑", fontsize=7.6, color="0.35")
    eixo.text(160, -8.15, "melhor ↓  ·  área ∝ |dano no Kodak|",
              fontsize=7.6, color="0.35")
    eixo.annotate(f"delta materializado:\n{num(mb[0], 2)} MB em fp16,\n{num(ancora / mb[0], 1)}× menor que a âncora",
                  xy=(6.9e6, 2.38), xytext=(170, -5.4), fontsize=7.6, color=VERMELHO,
                  ha="left", va="center",
                  arrowprops=dict(arrowstyle="->", lw=0.6, color=VERMELHO,
                                  shrinkA=3, shrinkB=6,
                                  connectionstyle="arc3,rad=-0.12"))

    fig.subplots_adjust(left=0.155, right=0.985, top=0.97, bottom=0.275)
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
