"""F7 - O bootstrap bimodal da retina: por que "indefinido" nao e evasiva.

Afirma: o BD-Rate do codificador na retina nao e uma medicao com incerteza grande --
e um SORTEIO ENTRE DOIS VALORES. Duas modas separadas por um vale quase vazio.

⚠ A causa e o `drop_dominated` ser DESCONTINUO: ele descarta pontos nao-Pareto ANTES
de a janela ser fixada, entao 0,001 bpp de diferenca em qual ponto tem o menor bpp
muda quais pontos sobrevivem, o que move o piso da janela e o BD em dezenas de p.p.
A legenda tem de atribuir a bimodalidade a isso, e nao a "os dados da retina sao
ruidosos", que e outra afirmacao, e falsa.

Alvo de reproducao, publicado no relatorio: 64,4% das reamostragens abaixo de -15%,
35,3% acima de zero e 0,3% entre os dois. Se as tres fracoes nao reproduzirem, o
script para sem plotar.

Uso:
    python plots/fig_retina_bimodal.py
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plots"))
from analyze_finetuned import bd_rate  # noqa: E402

AZUL, VERMELHO, PRETO = "#0072B2", "#D55E00", "#000000"
REFERENCIA = "results/retina_generic_disjoint_rd.json"
TESTE = "results/retina_encoder_on_retina_disjoint_rd.json"
B, SEED = 1000, 42
PUBLICADO = dict(abaixo=64.4, acima=35.3, meio=0.3)   # relatório, seção da retina
TOL = 0.05  # pontos percentuais


def num(v, casas):
    return f"{v:.{casas}f}".replace(".", ",")


def chaves(js):
    return js.get("levels") or js.get("lambdas")


def pilha(js, campo):
    return np.array([js["per_image"][k][campo] for k in chaves(js)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/figs_relatorio/f7_retina_bimodal.png")
    args = ap.parse_args()
    with open(os.path.join(ROOT, REFERENCIA)) as fh:
        ref = json.load(fh)
    with open(os.path.join(ROOT, TESTE)) as fh:
        teste = json.load(fh)
    if ref["files"] != teste["files"]:
        raise SystemExit("conjuntos de imagens diferentes: reamostreio pareado indefinido.")

    rb, rp = pilha(ref, "bpp"), pilha(ref, "psnr")
    tb, tp = pilha(teste, "bpp"), pilha(teste, "psnr")
    n = rb.shape[1]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bd, _j = bd_rate(rb[:, i].mean(1), rp[:, i].mean(1),
                         tb[:, i].mean(1), tp[:, i].mean(1))
        if np.isfinite(bd):
            vals.append(bd)
    v = np.array(vals)
    fr = dict(abaixo=100 * np.mean(v < -15), acima=100 * np.mean(v > 0),
              meio=100 * np.mean((v >= -15) & (v <= 0)))
    _bd_nom, (lo, hi) = bd_rate(ref["bpp"], ref["psnr"], teste["bpp"], teste["psnr"])
    print(f"{len(v)} reamostragens finitas de {B}")
    for k in ("abaixo", "acima", "meio"):
        print(f"  {k:<7} {fr[k]:5.1f}%  (publicado {PUBLICADO[k]:.1f}%)")
    print(f"  janela de sobreposição da curva nominal: {hi - lo:.3f} dB")
    if max(abs(fr[k] - PUBLICADO[k]) for k in PUBLICADO) > TOL:
        raise SystemExit("as três frações NÃO reproduzem o publicado — não plotado.")

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.linewidth": 0.7,
    })
    fig, eixo = plt.subplots(figsize=(3.28, 2.15))
    bordas = np.linspace(np.floor(v.min()), np.ceil(v.max()), 61)
    cores = [VERMELHO if (b + bordas[1] - bordas[0]) <= -15 else AZUL for b in bordas[:-1]]
    n_bin, _b, barras = eixo.hist(v, bins=bordas, color=AZUL, edgecolor="none")
    for bar, cor in zip(barras, cores):
        bar.set_facecolor(cor)

    eixo.axvline(-15, color=PRETO, linewidth=0.8, linestyle=(0, (3, 2)))
    eixo.axvline(0, color=PRETO, linewidth=0.8)
    topo = n_bin.max()
    eixo.annotate(f"{num(fr['abaixo'], 1)}%\nabaixo de −15%",
                  xy=(-29.5, topo * 0.87), fontsize=7.5, color=VERMELHO,
                  ha="left", va="center")
    eixo.annotate(f"{num(fr['acima'], 1)}%\nacima de zero",
                  xy=(bordas[-1] - 0.5, topo * 0.87), fontsize=7.5, color=AZUL,
                  ha="right", va="center")
    eixo.annotate(f"vale: {num(fr['meio'], 1)}%",
                  xy=(-7.5, topo * 0.10), xytext=(-7.5, topo * 0.42),
                  fontsize=7.5, color="0.3", ha="center", va="bottom",
                  arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"))

    eixo.set_xlabel("BD-Rate da reamostragem (%)")
    eixo.set_ylabel("reamostragens")
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: num(x, 0)))
    eixo.set_xlim(bordas[0], bordas[-1])
    eixo.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    eixo.set_axisbelow(True)

    fig.subplots_adjust(left=0.155, right=0.985, top=0.97, bottom=0.19)
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
