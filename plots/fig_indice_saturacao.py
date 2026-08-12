"""F6 - The saturation index was pre-registered and does NOT predict.

Claim: the 2x2 "the encoder wins rate-limited domains" is DESCRIPTION, not rule. Index S was
built from the GENERIC curve alone, before any adaptation, and pre-registered; the criterion
required a threshold in S separating the winners perfectly, in the predicted direction
(higher S => encoder wins). It fails, and fails at the extreme: OCT has the HIGHEST S in the
study -- the least saturated curve -- and is won by the DECODER.

The figure must not suggest a threshold that "almost" works: the pre-registered criterion
required PERFECT separation. Hence the in-figure text states where any cut fails, instead of
letting the reader draw an approximate one.

Source: `results/_exp_30jul/saturation_index_verdict.json` (S, winner, BD and window floors
per domain, plus the pre-registered criterion as text).

Usage:
    python plots/fig_indice_saturacao.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTE = "results/_exp_30jul/saturation_index_verdict.json"
AZUL, VERMELHO, PRETO, CINZA = "#0072B2", "#D55E00", "#000000", "0.45"
ROTULO = {"raio-X": "Raio-X", "OCT": "OCT", "retina": "Retina",
          "documentos": "Documentos", "DIOR": "DIOR", "RICO": "RICO"}


def num(v, casas):
    return f"{v:.{casas}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/figs_relatorio/f6_indice_saturacao.png")
    args = ap.parse_args()
    with open(os.path.join(ROOT, FONTE)) as fh:
        verdito = json.load(fh)
    tabela = sorted(verdito["table"], key=lambda r: r["S"])
    print("veredito:", verdito["verdict"], "| separa por S:",
          verdito["separates_primary_S"], "| por R:", verdito["separates_secondary_R"])
    for r in tabela:
        print(f"  {r['domain']:<12} S={r['S']:.3f}  vencedor={r['winner']}")

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "axes.linewidth": 0.7,
    })
    # One row per domain, ordered by S: four of the six fall between 0.111 and 0.159, and
    # in a single-row scatter the labels overlap. If a rule existed, a VERTICAL CUT would
    # separate the markers; there is none.
    fig, eixo = plt.subplots(figsize=(3.16, 2.1))
    ys = list(range(len(tabela)))

    for y, r in zip(ys, tabela):
        excluido = r["winner"].startswith("EXCLU")
        if excluido:
            cor, mk, face = CINZA, "x", CINZA
        elif r["winner"] == "encoder":
            cor, mk, face = AZUL, "^", "white"
        else:
            cor, mk, face = VERMELHO, "o", VERMELHO
        eixo.plot([0.04, r["S"]], [y, y], color="0.85", linewidth=0.6, zorder=1)
        eixo.plot([r["S"]], [y], marker=mk, color=cor, markerfacecolor=face,
                  markersize=7, markeredgewidth=1.3, zorder=3)
        eixo.annotate(num(r["S"], 3), xy=(r["S"], y), xytext=(9, 0),
                      textcoords="offset points", va="center", fontsize=7.3,
                      color=CINZA if excluido else PRETO)

    eixo.set_yticks(ys)
    eixo.set_yticklabels([ROTULO[r["domain"]] for r in tabela])
    eixo.set_ylim(-0.85, len(tabela) - 0.05)
    eixo.set_xlim(0.04, 0.80)
    eixo.set_xlabel("Índice de saturação S (pré-registrado)")
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: num(v, 1)))
    eixo.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    eixo.set_axisbelow(True)

    eixo.annotate("maior S do estudo — a 2×2\nmandaria o codificador vencer",
                  xy=(tabela[-1]["S"], len(tabela) - 1), xytext=(0.055, len(tabela) - 0.6),
                  fontsize=7.3, color=VERMELHO, ha="left", va="center",
                  arrowprops=dict(arrowstyle="->", lw=0.7, color=VERMELHO,
                                  shrinkA=6, shrinkB=5,
                                  connectionstyle="arc3,rad=-0.15"))
    eixo.text(0.055, -0.62, "nenhum corte vertical separa os marcadores",
              fontsize=7.3, color="0.3", va="center")

    marcas = [plt.Line2D([], [], marker="^", color=AZUL, markerfacecolor="white",
                         markeredgewidth=1.3, linestyle="none", markersize=7),
              plt.Line2D([], [], marker="o", color=VERMELHO, linestyle="none",
                         markersize=7),
              plt.Line2D([], [], marker="x", color=CINZA, linestyle="none",
                         markersize=7, markeredgewidth=1.3)]
    fig.legend(marcas, ["codificador vence", "decodificador vence",
                        "excluído pela guarda de janela"],
               loc="lower center", ncol=2, frameon=False, handlelength=1.0,
               columnspacing=1.2, fontsize=7.3, bbox_to_anchor=(0.5, -0.03))

    fig.subplots_adjust(left=0.245, right=0.985, top=0.97, bottom=0.34)
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
