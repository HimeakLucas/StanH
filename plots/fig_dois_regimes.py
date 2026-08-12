"""F2 - A regra do esquecimento nos 6 dominios: dois regimes.

Afirma: o decodificador causa mais dano cross-domain que o codificador em 6 de 6
dominios, e as magnitudes se partem em dois regimes separados por um vao vazio
(leve ate 0,16 dB; colapso a partir de 3,1 dB).

Unidade: ΔPSNR medio casado por bpp no Kodak, NUNCA BD-Rate -- o BD do raio-X
repousa sobre janela de 0,61-0,66 dB e reprova a guarda de janela, e em documentos
e OCT as curvas nem se sobrepoem. Por isso a figura nao tem segunda escala em %.

Os valores sao RE-DERIVADOS aqui com `matched_mean` (a implementacao de referencia,
importada de `plots/decompose_ycbcr.py`) e conferidos contra a Tabela II do
relatorio; se algum ponto divergir alem da tolerancia, o script para sem plotar.

O eixo e quebrado porque -0,03 e -7,19 dB nao convivem num eixo linear. A quebra
cai DENTRO do vao vazio: nao ha dominio nenhum entre -0,16 e -3,13 dB.

Uso:
    export PYTHONPATH=src
    python plots/fig_dois_regimes.py
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plots"))
from decompose_ycbcr import matched_mean  # noqa: E402

AZUL, VERMELHO, PRETO = "#0072B2", "#D55E00", "#000000"
GENERICA = "results/kodak_rd.json"

# label, monochrome?, encoder json, decoder json, (enc, dec) as published in Table II
# of the report, and whether the encoder cell has a CI crossing zero.
DOMINIOS = [
    ("Tela (RICO)", False, "results/rico_encoder_on_cross_rd.json",
     "results/rico_decoder_on_cross_rd.json", (+0.11, -0.03), False),
    ("Aéreo (DIOR)", False, "results/dior_encoder_on_cross_rd.json",
     "results/dior_decoder_on_cross_rd.json", (+0.08, -0.05), False),
    ("Retina", False, "results/retina_encoder_on_cross_rd.json",
     "results/retina_decoder_on_cross_rd.json", (+0.04, -0.16), False),
    ("Raio-X", True, "results/v8_encoder_on_kodak_rd.json",
     "results/v7_decoder_on_kodak_rd.json", (+0.01, -3.13), True),
    ("Documentos", True, "results/documents_encoder_on_cross_rd.json",
     "results/documents_decoder_on_cross_rd.json", (-0.12, -6.01), False),
    ("OCT", True, "results/oct_encoder_on_cross_rd.json",
     "results/oct_decoder_on_cross_rd.json", (-0.20, -7.19), False),
]
TOL = 0.005          # a Tabela II imprime 2 casas; meia casa e o limite razoavel
VAO = (-3.13, -0.16)  # extremos medidos do vao entre regimes


def load(nome):
    with open(os.path.join(ROOT, nome)) as fh:
        return json.load(fh)


def virgula(casas):
    return FuncFormatter(lambda v, _pos: f"{v:.{casas}f}".replace(".", ","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/figs_relatorio/f2_dois_regimes.png")
    args = ap.parse_args()
    gen = load(GENERICA)

    medidos, pior = [], 0.0
    print(f"{'domínio':<14}{'enc publ':>9}{'enc med':>9}{'dec publ':>10}{'dec med':>9}")
    for rotulo, mono, je, jd, publicados, _est in DOMINIOS:
        valores = []
        for caminho in (je, jd):
            t = load(caminho)
            v, _n = matched_mean(gen["bpp"], gen["psnr"], t["bpp"], t["psnr"])
            valores.append(v)
        pior = max(pior, max(abs(v - p) for v, p in zip(valores, publicados)))
        medidos.append(valores)
        print(f"{rotulo:<14}{publicados[0]:>+9.2f}{valores[0]:>+9.3f}"
              f"{publicados[1]:>+10.2f}{valores[1]:>+9.3f}")
    print(f"pior |diferença| vs Tabela II: {pior:.4f} dB (tolerância {TOL})")
    if pior > TOL:
        raise SystemExit("re-derivação NÃO reproduz a Tabela II — não plotado.")

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 8, "legend.fontsize": 7.6, "axes.linewidth": 0.7,
    })
    fig, (esq, dir_) = plt.subplots(
        1, 2, figsize=(3.10, 2.75), sharey=True,
        gridspec_kw=dict(width_ratios=[1.3, 1.0], wspace=0.07))
    alt = 0.36
    ys = list(range(len(DOMINIOS)))[::-1]   # first domain on top

    for eixo in (esq, dir_):
        for y, (rotulo, mono, _je, _jd, _p, estrela) in zip(ys, DOMINIOS):
            enc, dec = medidos[len(DOMINIOS) - 1 - y]
            hachura = "////" if mono else None
            eixo.barh(y + alt / 2 + 0.02, enc, height=alt, color="white",
                      edgecolor=AZUL, linewidth=0.8, hatch=hachura, zorder=2)
            # a hachura desenha na cor da borda: nas barras cheias ela so aparece
            # se a borda for clara.
            eixo.barh(y - alt / 2 - 0.02, dec, height=alt, color=VERMELHO,
                      edgecolor="white" if mono else VERMELHO,
                      linewidth=0.8 if mono else 0.8, hatch=hachura, zorder=2)
        eixo.axvline(0, color=PRETO, linewidth=0.8, zorder=3)
        eixo.axhline(2.5, color="0.55", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        eixo.grid(True, axis="x", alpha=0.25, linewidth=0.4)
        eixo.set_axisbelow(True)
        eixo.xaxis.set_major_formatter(virgula(1))

    esq.set_xlim(-7.9, -2.9)
    esq.set_xticks([-7, -6, -5, -4, -3])
    dir_.set_xlim(-0.30, 0.19)
    dir_.set_xticks([-0.2, -0.1, 0.0, 0.1])
    dir_.xaxis.set_major_formatter(virgula(1))
    esq.set_yticks(ys)
    esq.set_yticklabels([d[0] for d in DOMINIOS])
    esq.set_ylim(-0.72, len(DOMINIOS) - 0.28)

    # axis-break marks, on the facing edges
    esq.spines["right"].set_visible(False)
    dir_.spines["left"].set_visible(False)
    dir_.tick_params(left=False)
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5, linestyle="none",
              color=PRETO, mec=PRETO, mew=0.8, clip_on=False)
    esq.plot([1, 1], [0, 1], transform=esq.transAxes, **kw)
    dir_.plot([0, 0], [0, 1], transform=dir_.transAxes, **kw)

    # the gap between regimes: no domain lands there, and that is what the break spans
    esq.text(-3.02, 2.60, "vão vazio: {:.2f} a {:.2f} dB".format(*VAO).replace(".", ","),
             ha="right", va="bottom", fontsize=7.6, color="0.3")

    # the only cell in this column whose CI crosses zero (Table II)
    dir_.text(medidos[3][0] + 0.012, 2 + alt / 2 + 0.02, "*", fontsize=9,
              color=AZUL, va="center")

    barras = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=AZUL, lw=0.8),
              plt.Rectangle((0, 0), 1, 1, facecolor=VERMELHO, edgecolor=VERMELHO, lw=0.8),
              plt.Rectangle((0, 0), 1, 1, facecolor="0.93", edgecolor="0.35",
                            lw=0.8, hatch="////")]
    fig.subplots_adjust(left=0.275, right=0.985, top=0.985, bottom=0.30)
    centro = (0.275 + 0.985) / 2
    fig.text(centro, 0.185, "ΔPSNR casado por bpp no Kodak (dB)",
             ha="center", fontsize=8.2)
    fig.text(centro, 0.105, "* IC de 95% cruza zero", ha="center", fontsize=7.6,
             color="0.3")
    fig.legend(barras, ["Codificador", "Decodificador", "monocromático"],
               loc="lower center", ncol=3, frameon=False, handlelength=1.4,
               handleheight=0.9, columnspacing=0.9, bbox_to_anchor=(centro, -0.012))
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
