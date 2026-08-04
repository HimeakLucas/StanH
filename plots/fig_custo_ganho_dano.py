"""F8 - Custo x ganho x dano: a figura-tese.

Afirma: o eixo inteiro do trabalho numa imagem -- 320 -> 6,9 M -> 22 M -> 75 M
parametros TREINAVEIS contra o ganho no dominio alvo, com o dano no dominio de
origem como terceira dimensao (tamanho do marcador e anotacao em dB).

⚠ O eixo x e PARAMETROS TREINAVEIS, nunca "custo de armazenamento": sao duas
dimensoes de custo diferentes, e o tamanho materializado do delta e uma anotacao de
um ponto so, nao um eixo.

⚠ A celula `full` e FAIXA nas duas coordenadas (duas replicas): desenhada como dois
pontos ligados, nunca como um ponto.

⚠ O `encoder_hyper` e "adaptador intermediario", nunca "leve". E o dano cross dele
no raio-X NAO existe na unidade da figura: o valor publicado (+5,43%) e BD-Rate, que
e justamente a unidade que o trabalho proibe para dano cross. O ponto entra com a
posicao que tem (custo e ganho) e o marcador vazado declara a terceira dimensao
como nao medida -- computa-la aqui faria um numero nascer numa figura.

Uso:
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

# rotulo, params treinaveis, BD alvo (%), dano cross (dB) ou None, cor, marcador
# Todos os valores estao na Tabela I do relatorio (regua disjunta por paciente).
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
        if len(bds) > 1:                     # célula com réplica: faixa, não ponto
            eixo.plot([params, params], bds, color=cor, linewidth=1.2, zorder=2)
        for i, bd in enumerate(bds):
            if danos is None:
                eixo.scatter([params], [bd], s=70, facecolors="none", edgecolors=cor,
                             marker=mk, linewidths=1.3, zorder=3)
            else:
                eixo.scatter([params], [bd], s=tamanho(danos[i]), color=cor,
                             marker=mk, zorder=3, edgecolors="white", linewidths=0.5)

    # Rotulos vao para a legenda, e nao como anotacoes por ponto: a 3,4 in de
    # largura cinco anotacoes se sobrepoem, e o dano em dB e o que precisa ser lido.
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
