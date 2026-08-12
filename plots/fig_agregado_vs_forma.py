"""F5 - The aggregate moves, the curve shape does not (methodological contribution).

Claim: aggregate BD-Rate can shift by percentage points while the RD curve stays put --
including with NO TRAINING AT ALL, just changing the storage precision of the same trained
model (fp32 -> fp16, 0.42 p.p.), which is more than a fresh seed moves it.

The figure must not say "the aggregate is unstable" unqualified: that would generalize from
the most unstable cell in the study, the very vice this contribution denounces. Hence both
cells with a replicated aggregate appear together, and one of them -- `xray_encoder`,
0.27 p.p. -- is STABLE.

The RICO replica is excluded: it has a single lambda, so it has no aggregate, and plotting
it on the left panel would mix two domains on the same RD axes.

Usage:
    python plots/fig_agregado_vs_forma.py
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
from analyze_finetuned import bd_rate  # noqa: E402

AZUL, VERMELHO, VERDE, PRETO = "#0072B2", "#D55E00", "#009E73", "#000000"
GENERICA = "results/xray_generic_disjoint_rd.json"
TOL = 0.02   # p.p.; published values carry 2 decimals

# group, label, json, color, marker, line style, published BD
CURVAS = [
    ("full", "treino A", "results/v6_full_runBfix_on_xray_disjoint_rd.json",
     PRETO, "*", "-", -5.07),
    ("full", "treino B", "results/v6_full_runB8_on_xray_disjoint_rd.json",
     PRETO, "*", (0, (5, 2)), -6.48),
    ("encoder", "fp32", "results/v8_encoder_on_xray_disjoint_rd.json",
     AZUL, "v", "-", +2.38),
    ("encoder", "fp16 (mesmo modelo)", "results/v8_encoder_fp16_on_xray_disjoint_rd.json",
     VERMELHO, "v", (0, (3, 1.2, 1, 1.2)), +1.96),
    ("encoder", "fp16, 2ª semente", "results/v8_encoder_runB_on_xray_disjoint_rd.json",
     VERDE, "v", (0, (1, 1.2)), +1.69),
]
GRUPOS = [("full", "Full 75 M", "{} p.p. (duas sementes)"),
          ("encoder", "Encoder 6,9 M",
           "{} p.p. no total: 0,27 entre sementes\ne 0,42 só trocando fp32→fp16")]


def load(nome):
    with open(os.path.join(ROOT, nome)) as fh:
        return json.load(fh)


def num(v, casas):
    """Decimal comma on the number only: a `replace` over the whole sentence would
    turn `p.p.` into `p,p,`."""
    return f"{v:.{casas}f}".replace(".", ",")


def virgula(casas):
    return FuncFormatter(lambda v, _pos: num(v, casas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/figs_relatorio/f5_agregado_vs_forma.png")
    args = ap.parse_args()
    gen = load(GENERICA)

    medidos, pior = [], 0.0
    for grupo, rotulo, caminho, cor, mk, ls, publicado in CURVAS:
        d = load(caminho)
        bd, _janela = bd_rate(gen["bpp"], gen["psnr"], d["bpp"], d["psnr"])
        pior = max(pior, abs(bd - publicado))
        medidos.append((grupo, rotulo, d, cor, mk, ls, bd))
        print(f"{grupo:>8} {rotulo:<22} BD {bd:+7.3f}%  (publicado {publicado:+.2f}%)")
    print(f"pior |diferença| vs publicado: {pior:.4f} p.p. (tolerância {TOL})")
    if pior > TOL:
        raise SystemExit("re-derivação NÃO reproduz os BD-Rates publicados — não plotado.")

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
    })
    fig, (esq, dir_) = plt.subplots(
        1, 2, figsize=(7.0, 2.75),
        gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.26))

    esq.plot(gen["bpp"], gen["psnr"], "o-", color="0.55", markersize=3,
             linewidth=1.0, label="Genérica (referência)", zorder=1)
    for _g, rotulo, d, cor, mk, ls, _bd in medidos:
        esq.plot(d["bpp"], d["psnr"], marker=mk, linestyle=ls, color=cor,
                 markersize=5 if mk == "*" else 3.4, linewidth=1.1, label=rotulo,
                 markeredgewidth=0.6, zorder=2)
    esq.set_xlabel("bpp")
    esq.set_ylabel("PSNR (dB)")
    esq.set_xlim(left=0)
    esq.grid(True, alpha=0.3, linewidth=0.4)
    esq.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    esq.xaxis.set_major_formatter(virgula(1))
    esq.yaxis.set_major_formatter(virgula(0))
    esq.legend(loc="lower right", framealpha=0.9, handlelength=2.6, borderpad=0.35)
    esq.set_title("As curvas (raio-X, amostra disjunta)", fontsize=8.5)

    for y, (grupo, rotulo, molde) in enumerate(GRUPOS[::-1]):
        pontos = [m for m in medidos if m[0] == grupo]
        xs = [m[6] for m in pontos]
        dir_.plot([min(xs), max(xs)], [y, y], color="0.6", linewidth=1.0, zorder=1)
        for _g, _r, _d, cor, mk, _ls, bd in pontos:
            dir_.plot([bd], [y], marker=mk, color=cor,
                      markersize=8 if mk == "*" else 5.5, zorder=3,
                      markeredgecolor="white", markeredgewidth=0.5)
        dir_.annotate(molde.format(num(max(xs) - min(xs), 2)),
                      xy=((min(xs) + max(xs)) / 2, y), xytext=(0, 10),
                      textcoords="offset points", ha="center", fontsize=7.3,
                      color="0.25")
    dir_.axvline(0, color=PRETO, linewidth=0.8)
    dir_.set_yticks([0, 1])
    dir_.set_yticklabels([g[1] for g in GRUPOS[::-1]])
    dir_.set_ylim(-0.6, 1.85)
    dir_.set_xlim(-7.9, 5.6)
    dir_.set_xlabel("BD-Rate agregado no raio-X (%)")
    dir_.set_xticks([-6, -4, -2, 0, 2, 4])
    dir_.xaxis.set_major_formatter(virgula(0))
    dir_.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    dir_.set_axisbelow(True)
    dir_.set_title("O agregado das mesmas curvas", fontsize=8.5)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.24)
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
