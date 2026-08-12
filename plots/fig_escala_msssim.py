"""F1 - The single ruler in MS-SSIM: adapter scale.

Claim: the parametric quantizer leaves a residual an order of magnitude smaller than any
adapter that works -- and that is a fact of MS-SSIM (in PSNR the signs are opposite and the
ratio against the decoder is 2.8x, not 15x).

The encoder is NOT on the ladder: in MS-SSIM it is worth +0.98%, on the wrong side of zero,
so it sits above the divider with a marker of its own.

Reads `results/_exp_01ago/xray_msssim_bd_summary.json` (`bd_rate_pct` and `ci95` per cell)
without recomputing anything. The annotated scale ratios are |BD| of the cell over |BD| of
the quantizer, from the same fields.

Usage:
    python plots/fig_escala_msssim.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTE = "results/_exp_01ago/xray_msssim_bd_summary.json"

AZUL, VERMELHO, ROSA, PRETO = "#0072B2", "#D55E00", "#CC79A7", "#000000"

# (JSON key, label, color, marker), bottom to top on the y axis
LINHAS = [
    ("full_runB8", "Full 75 M · treino B", PRETO, "o"),
    ("full_runBfix", "Full 75 M · treino A", PRETO, "o"),
    ("encoder_hyper", "encoder_hyper ≈22 M", AZUL, "o"),
    ("decoder", "Decoder 6,9 M", AZUL, "o"),
    ("quantizador", "Só quantizador 320", ROSA, "s"),
    ("encoder", "Encoder 6,9 M", VERMELHO, "^"),
]
SEPARA_ACIMA_DE = "quantizador"   # divider between the encoder and the rest


def virgula(casas):
    return FuncFormatter(lambda v, _pos: f"{v:.{casas}f}".replace(".", ","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/figs_relatorio/f1_escala_msssim.png")
    args = ap.parse_args()
    with open(os.path.join(ROOT, FONTE)) as fh:
        celulas = json.load(fh)["cells"]

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 8, "axes.linewidth": 0.7,
    })
    fig, eixo = plt.subplots(figsize=(3.05, 2.5))

    ref = abs(celulas["quantizador"]["bd_rate_pct"])
    for y, (chave, rotulo, cor, mk) in enumerate(LINHAS):
        c = celulas[chave]
        v, (lo, hi) = c["bd_rate_pct"], c["ci95"]
        eixo.plot([lo, hi], [y, y], color=cor, linewidth=1.5, solid_capstyle="butt")
        for x in (lo, hi):
            eixo.plot([x, x], [y - 0.16, y + 0.16], color=cor, linewidth=1.0)
        eixo.plot([v], [y], marker=mk, color=cor, markersize=5,
                  markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        if chave not in ("quantizador", "encoder"):
            eixo.annotate(f"{abs(v) / ref:.0f}×", xy=(hi, y), xytext=(4, 0),
                          textcoords="offset points", va="center", fontsize=7.5,
                          color=cor)

    # The quantizer CI is smaller than its marker: at this scale the bar vanishes and the
    # figure would seem to say "zero". The value is written out so it does not.
    q = celulas["quantizador"]
    eixo.annotate(f"{q['bd_rate_pct']:.2f} [{q['ci95'][0]:.2f}; {q['ci95'][1]:.2f}]"
                  .replace(".", ","),
                  xy=(q["ci95"][1], 4), xytext=(7, 0), textcoords="offset points",
                  va="center", fontsize=7.2, color=ROSA)

    y_sep = [i for i, l in enumerate(LINHAS) if l[0] == SEPARA_ACIMA_DE][0] + 0.5
    eixo.axhline(y_sep, color="0.55", linewidth=0.7, linestyle=(0, (3, 2)))
    eixo.axvline(0, color=PRETO, linewidth=0.8)

    eixo.set_yticks(range(len(LINHAS)))
    eixo.set_yticklabels([l[1] for l in LINHAS])
    eixo.set_ylim(-0.7, len(LINHAS) - 0.2)
    eixo.set_xlim(-9.6, 6.4)
    eixo.set_xticks([-8, -6, -4, -2, 0, 2])
    eixo.set_xlabel("BD-Rate em MS-SSIM (%)")
    eixo.xaxis.set_major_formatter(virgula(0))
    eixo.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    eixo.set_axisbelow(True)
    eixo.text(6.2, len(LINHAS) - 0.42, "pior que a genérica", ha="right", va="center",
              fontsize=7.2, color=VERMELHO)
    eixo.text(-9.2, -0.55, "negativo = melhor", fontsize=7.2, color="0.35")

    fig.tight_layout(pad=0.35)
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)
    print("razões de escala (|BD| / |BD do quantizador|):")
    for chave, rotulo, _c, _m in LINHAS:
        print(f"   {rotulo:34s} {celulas[chave]['bd_rate_pct']:+7.3f}%  "
              f"{abs(celulas[chave]['bd_rate_pct']) / ref:5.1f}×")


if __name__ == "__main__":
    main()
