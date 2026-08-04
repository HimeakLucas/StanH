"""F0 - Espectro de adaptadores no raio-X, na amostra DISJUNTA por paciente.

Substitui `pibic-paper/fig_spectrum.png`, que fora gerado em 03/07 a partir das
curvas da amostra vazada (`results/*_on_xray_rd.json`) enquanto a Tabela I do
relatorio ja estava na regua disjunta (`results/*_on_xray_disjoint_rd.json`).
Aqui as duas ficam na mesma regua.

A celula `full` tem duas replicas (`runBfix` e `runB8`), que compartilham o
checkpoint de menor taxa; sao desenhadas na mesma cor, com estilos de linha
diferentes, porque o relatorio reporta a FAIXA e nao um valor.

Uso:
    export PYTHONPATH=src
    python plots/fig_espectro_disjunto.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Okabe-Ito: segura para daltonismo e separavel em tons de cinza.
AZUL, VERMELHO, VERDE, ROSA, AMBAR, PRETO = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000")


def load(name):
    with open(os.path.join(ROOT, name)) as fh:
        return json.load(fh)


def virgula(casas):
    """Formatador de eixo com virgula decimal (relatorio em portugues)."""
    return FuncFormatter(lambda v, _pos: f"{v:.{casas}f}".replace(".", ","))


# O corpo desenha esta figura em `0.96\textwidth` FIXO, entao a largura em
# polegadas nao muda o tamanho impresso: ela muda o quanto a figura e reduzida,
# e portanto a ALTURA que ela ocupa na pagina. A versao de 7,0 in ficava com
# aspecto h/w = 0,473 contra 0,382 da figura que ela substituiu.
#
# Por que alargar, e o que isso NAO faz. Medido em 04/08, no mesmo tectonic e na
# mesma maquina: sobre o .tex PRE-figuras, o documento compila em 12 paginas com
# a figura antiga e 13 com a de 7,0 in -- a geometria sozinha custava uma pagina.
# Sobre o .tex ATUAL, com as dez figuras dentro, os tres aspectos (0,473 / 0,382 /
# 0,377) dao 16 paginas, os tres. Ou seja: a economia NAO se realiza hoje, porque
# o custo de pagina por figura nao e aditivo -- e efeito de limiar, e com dez
# floats o LaTeX tem folga para absorver ~1% de altura sem cruzar fronteira.
# Mantido assim por paridade de geometria com a figura que substitui, e porque e
# de graca; nao por economia medida.
#
# O preco de alargar e que a reducao ate a largura de destino passa de 1,02x para
# ~1,26x, o que encolheria as fontes abaixo do piso de 7 pt. Por isso tudo que e
# medido em pontos escala junto, por ESCALA: no papel os rotulos de eixo ficam em
# 7,36 pt nas duas versoes, identicos.
LARGURA_IN = 8.7
ESCALA = LARGURA_IN / 7.0


def estilo():
    plt.rcParams.update({
        "font.size": 8.5 * ESCALA, "axes.labelsize": 8.5 * ESCALA,
        "axes.titlesize": 9.5 * ESCALA,
        "xtick.labelsize": 7.5 * ESCALA, "ytick.labelsize": 7.5 * ESCALA,
        "legend.fontsize": 7.5 * ESCALA,
        "axes.linewidth": 0.7 * ESCALA, "grid.linewidth": 0.4 * ESCALA,
        "xtick.major.width": 0.6 * ESCALA, "ytick.major.width": 0.6 * ESCALA,
    })


# rotulo, json alvo, json cross, cor, marcador, estilo de linha
SERIES = [
    ("Genérica (autores)", "results/xray_generic_disjoint_rd.json",
     "results/kodak_rd.json", AZUL, "o", "-"),
    ("Só quantizador (320)", "results/v4_finetuned_on_xray_disjoint_rd.json",
     "results/v4_finetuned_on_kodak_rd.json", ROSA, "s", (0, (1, 1.2))),
    ("Decoder (6,9 M)", "results/v7_decoder_on_xray_disjoint_rd.json",
     "results/v7_decoder_on_kodak_rd.json", AMBAR, "D", (0, (4, 1.6))),
    ("Encoder (6,9 M)", "results/v8_encoder_on_xray_disjoint_rd.json",
     "results/v8_encoder_on_kodak_rd.json", VERMELHO, "v", (0, (3, 1.2, 1, 1.2))),
    ("Full backbone (75 M), treino A", "results/v6_full_runBfix_on_xray_disjoint_rd.json",
     "results/v6_full_runBfix_on_kodak_rd.json", PRETO, "*", "-"),
    ("Full backbone (75 M), treino B", "results/v6_full_runB8_on_xray_disjoint_rd.json",
     "results/v6_full_runB8_on_kodak_rd.json", PRETO, "*", (0, (5, 2))),
]
VTM = ("VTM (H.266)", "results/xray_vtm_disjoint_rd.json", "results/vtm_kodak_rd.json",
       VERDE, "^", (0, (2, 1.4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/figs_relatorio/f0_espectro.png")
    args = ap.parse_args()
    estilo()

    # ⚠ A ALTURA nao escala: e alargar a altura constante que derruba o aspecto,
    # e o aspecto e o que decide quantas paginas a figura ocupa.
    fig, ax = plt.subplots(1, 2, figsize=(LARGURA_IN, 3.15))
    for rotulo, alvo, cross, cor, mk, ls in SERIES:
        for eixo, caminho in ((ax[0], alvo), (ax[1], cross)):
            d = load(caminho)
            largura = (1.5 if rotulo.startswith("Genérica") else 1.1) * ESCALA
            tamanho = (5.5 if mk == "*" else 3.6) * ESCALA
            eixo.plot(d["bpp"], d["psnr"], marker=mk, linestyle=ls, color=cor,
                      label=rotulo, markersize=tamanho, linewidth=largura,
                      markeredgewidth=0.6 * ESCALA)
    for eixo, caminho in ((ax[0], VTM[1]), (ax[1], VTM[2])):
        d = load(caminho)
        eixo.plot(d["bpp"], d["psnr"], marker=VTM[4], linestyle=VTM[5], color=VTM[3],
                  label=VTM[0], markersize=3.6 * ESCALA, linewidth=1.1 * ESCALA,
                  markeredgewidth=0.6 * ESCALA)

    # O lambda=0,13 do `full` e dominado nas duas replicas (mais taxa, menos PSNR):
    # o recuo e sistematico, nao ruido, e nao deve ser escondido.
    d = load(SERIES[4][1])
    i = d["lambdas"].index("lambda_0.13")
    ax[0].annotate("λ=0,13\ndominado", xy=(d["bpp"][i], d["psnr"][i]),
                   xytext=(d["bpp"][i] + 0.075, d["psnr"][i] - 1.75),
                   fontsize=7.2 * ESCALA, ha="center", va="top", color=PRETO,
                   arrowprops=dict(arrowstyle="->", lw=0.6 * ESCALA, color=PRETO,
                                   shrinkA=0, shrinkB=2))

    ax[0].set_title("Raio-X (domínio alvo)")
    ax[1].set_title("Kodak (domínio de origem)")
    for eixo, casas in ((ax[0], 2), (ax[1], 1)):
        eixo.set_xlabel("bpp")
        eixo.set_ylabel("PSNR (dB)")
        eixo.set_xlim(left=0)
        eixo.grid(True, alpha=0.3, linewidth=0.4 * ESCALA)
        eixo.xaxis.set_major_formatter(virgula(casas))
        eixo.yaxis.set_major_formatter(virgula(0))
    # Uma legenda so, embaixo: as duas curvas usam o mesmo mapeamento e a
    # duplicacao roubava o canto util dos dois paineis.
    linhas, rotulos = ax[0].get_legend_handles_labels()
    fig.legend(linhas, rotulos, loc="lower center", ncol=4, frameon=False,
               handlelength=2.8, columnspacing=1.4, bbox_to_anchor=(0.5, -0.012))

    fig.tight_layout(pad=0.4 * ESCALA, w_pad=1.2 * ESCALA, rect=(0, 0.175, 1, 1))
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
