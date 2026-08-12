"""F0 - Adapter spectrum on x-ray, over the patient-DISJOINT sample.

Replaces an earlier figure drawn from the leaked sample (`results/*_on_xray_rd.json`)
while the report table was already on the disjoint one
(`results/*_on_xray_disjoint_rd.json`); both are on the same ruler here.

The `full` cell has two replicas (`runBfix`, `runB8`) sharing the lowest-rate
checkpoint. They are drawn in one color with different line styles, because the report
gives a RANGE rather than a single value.

Usage:
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

# Okabe-Ito: colorblind-safe and separable in grayscale.
AZUL, VERMELHO, VERDE, ROSA, AMBAR, PRETO = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000")


def load(name):
    with open(os.path.join(ROOT, name)) as fh:
        return json.load(fh)


def virgula(casas):
    """Axis formatter with a decimal comma (the report is in Portuguese)."""
    return FuncFormatter(lambda v, _pos: f"{v:.{casas}f}".replace(".", ","))


# The report includes this figure at a FIXED `0.96\textwidth`, so the width in inches does
# not change the printed size: it changes how much the figure is scaled down, and therefore
# the HEIGHT it takes on the page. This width matches the aspect ratio (h/w = 0.38) of the
# figure it replaces.
#
# The price of the wider canvas is a 1.26x reduction instead of 1.02x, which would push
# fonts below the 7 pt floor. Everything measured in points is therefore scaled by ESCALA,
# so axis labels land at 7.36 pt either way.
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


# label, target json, cross json, color, marker, line style
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

    # Height does NOT scale: widening at constant height is what lowers the aspect ratio,
    # which is what decides how many pages the figure costs.
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

    # lambda=0.13 of `full` is dominated in both replicas (more rate, less PSNR): the
    # regression is systematic rather than noise, and is not hidden.
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
    # Single legend at the bottom: both panels share the mapping, and duplicating it stole
    # the useful corner of each.
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
