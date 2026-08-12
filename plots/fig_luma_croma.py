"""F3 - Mechanism: luma vs chroma damage, plus the controls that bound the causal reading.

Two halves of one claim. In the collapsing domains the decoder damage is chroma-dominated
and the output inter-channel correlation rises towards 1. But "luma damage proves damage
beyond color" does NOT follow: `div2k_gray` -- natural content, domain distance ~ 0, only
R=G=B -- loses luma of the same order. The control bars are not optional; without them the
figure asserts the revoked reading.

Three control arms over the same 800 DIV2K images, varying only the color statistics:
  - `div2k_color`  negative control, validates the apparatus: no damage, xcorr unmoved;
  - `div2k_gray`   monochrome alone explains luma damage of the observed magnitude;
  - `div2k_decorr` collapses (-2.64 dB) with color intact and xcorr BELOW the generic one
                   -- the counterexample forbidding "high xcorr <=> collapse".

Never plot the CbCr/Y ratio: DIOR is RGB, does not collapse, and its ratio (5.94) is
practically the x-ray one (5.86). Magnitude carries the argument, not the ratio.

Sources: `results/ycbcr_decomposition_summary.json` (six domains; it deliberately excludes
the control and replay arms) and `results/ycbcr_div2k_{color,gray,decorr}_on_kodak.json`
(controls, re-derived here and checked against the published values).

Bootstrap B differs by source: 1000 for the six domains (recorded in the summary), 2000 for
the control arms.

Usage:
    export PYTHONPATH=src
    python plots/fig_luma_croma.py
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
from decompose_ycbcr import matched_mean, matched_support, stack  # noqa: E402

AZUL, VERMELHO, PRETO = "#0072B2", "#D55E00", "#000000"
SUMMARY = "results/ycbcr_decomposition_summary.json"
GENERICA = "results/ycbcr_generic_on_kodak.json"

DOMINIOS = [("rico", "Tela (RICO)"), ("dior", "Aéreo (DIOR)"), ("retina", "Retina"),
            ("xray", "Raio-X"), ("documents", "Documentos"), ("oct", "OCT")]
# Published control values, used here as REPRODUCTION TARGETS: no number may originate in
# this figure. `dy_ci` exists only where the luma CI is published (`gray`); for the other
# two the target is the point, and the plotted CI comes from the same resampling.
BOOT, SEED = 2000, 42
CONTROLES = [
    dict(arm="div2k_color", rot="DIV2K colorido\n(controle −)", rot_col="DIV2K cor\n(controle −)",
         dy=+0.032, dcbcr=+0.002, xcorr_faixa=(0.848, 0.849)),
    dict(arm="div2k_gray", rot="DIV2K em cinza\n(controle)", rot_col="DIV2K cinza\n(controle)",
         dy=-1.717, dy_ci=(-2.402, -1.119), dcbcr=-11.4, xcorr_faixa=(0.915, 0.955)),
    dict(arm="div2k_decorr", rot="DIV2K descorrel.\n(controle)", rot_col="DIV2K desc.\n(controle)",
         dy=-1.081, dcbcr=-6.846, xcorr_faixa=(0.837, 0.839)),
]
TOL, TOL_CI, TOL_1CASA, TOL_XCORR = 0.002, 0.005, 0.05, 0.0011


def load(nome):
    with open(os.path.join(ROOT, nome)) as fh:
        return json.load(fh)


def virgula(casas):
    return FuncFormatter(lambda v, _pos: f"{v:.{casas}f}".replace(".", ","))


def bootstrap_matched(gen, teste, canal, B, seed):
    """Same per-image paired resampling used by `decompose_ycbcr.analyze`."""
    gb, gm = stack(gen, "bpp"), stack(gen, canal)
    tb, tm = stack(teste, "bpp"), stack(teste, canal)
    n = gb.shape[1]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(matched_mean(gb[:, i].mean(1), gm[:, i].mean(1),
                                 tb[:, i].mean(1), tm[:, i].mean(1))[0])
    v = np.array(vals)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/figs_relatorio/f3_luma_croma.png")
    ap.add_argument("--col", action="store_true",
                    help="single-column IEEE geometry (3.4 in), rendered at final "
                         "width so 7 pt labels stay 7 pt on the page; the mild-regime "
                         "annotation moves to the caption")
    args = ap.parse_args()
    S = load(SUMMARY)
    gen = load(GENERICA)

    controles = []
    for pub in CONTROLES:
        ctl = load(f"results/ycbcr_{pub['arm']}_on_kodak.json")
        dy, _n = matched_mean(gen["bpp"], gen["psnr_y"], ctl["bpp"], ctl["psnr_y"])
        dc, n_ctl = matched_mean(gen["bpp"], gen["psnr_cbcr"],
                                 ctl["bpp"], ctl["psnr_cbcr"])
        ci_y = bootstrap_matched(gen, ctl, "psnr_y", BOOT, SEED)
        faixa = (min(ctl["xcorr"]), max(ctl["xcorr"]))
        print(f"controle {pub['arm']}: ΔY {dy:+.3f} (publicado {pub['dy']:+.3f}), "
              f"IC [{ci_y[0]:+.3f}, {ci_y[1]:+.3f}]"
              + (f" (publicado [{pub['dy_ci'][0]:+.3f}, {pub['dy_ci'][1]:+.3f}])"
                 if "dy_ci" in pub else " (IC de luma não publicado)")
              + f", n={n_ctl} pontos")
        print(f"controle {pub['arm']}: ΔCbCr {dc:+.3f} (publicado {pub['dcbcr']:+.3f}), "
              f"xcorr por λ {[round(x, 4) for x in ctl['xcorr']]} → faixa "
              f"{faixa[0]:.3f}–{faixa[1]:.3f} (publicada "
              f"{pub['xcorr_faixa'][0]:.3f}–{pub['xcorr_faixa'][1]:.3f})")
        falhas = [
            abs(dy - pub["dy"]) > TOL,
            "dy_ci" in pub and max(abs(a - b) for a, b in zip(ci_y, pub["dy_ci"])) > TOL_CI,
            abs(dc - pub["dcbcr"]) > TOL_1CASA,
            max(abs(a - b) for a, b in zip(faixa, pub["xcorr_faixa"])) > TOL_XCORR,
        ]
        if any(falhas):
            raise SystemExit(f"controle {pub['arm']} NÃO reproduz os valores "
                             "publicados — não plotado.")
        # Chroma CIs of the controls are not drawn: they are not published, and no
        # number may originate in this figure.
        controles.append((pub, dy, ci_y, dc, list(ctl["xcorr"])))

    # Right panel: matched-support mean from the summary for the six domains. The controls
    # are not in the summary, so their own per-lambda values are used instead.
    rotulos = dict(DOMINIOS)
    if args.col:   # short names: in one column the tick labels eat the drawing area
        rotulos.update(rico="Tela", dior="Aéreo", xray="Raio-X", documents="Docs.")
    linhas = ([(rotulos[k], S["curves"][k]["psnr_y"], S["curves"][k]["psnr_cbcr"],
                [S["xcorr"][k]["mean"]], False) for k, _rot in DOMINIOS]
              + [(pub["rot_col"] if args.col else pub["rot"],
                  {"delta": dy, "ci95": list(ci_y)},
                  {"delta": dc, "ci95": [np.nan, np.nan]}, xc, True)
                 for pub, dy, ci_y, dc, xc in controles])

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 7.5 if args.col else 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7 if args.col else 7.5, "ytick.labelsize": 7.5 if args.col else 8,
        "legend.fontsize": 7 if args.col else 7.5,
        "axes.linewidth": 0.7,
    })
    if args.col:
        fig, (esq, dir_) = plt.subplots(
            1, 2, figsize=(3.4, 2.78), sharey=True,
            gridspec_kw=dict(width_ratios=[1.9, 1.0], wspace=0.07))
    else:
        fig, (esq, dir_) = plt.subplots(
            1, 2, figsize=(7.0, 3.20), sharey=True,
            gridspec_kw=dict(width_ratios=[1.75, 1.0], wspace=0.06))
    ys = list(range(len(linhas)))[::-1]
    alt = 0.36

    for y, (rot, y_ch, c_ch, xc, controle) in zip(ys, linhas):
        esq.barh(y + alt / 2 + 0.02, y_ch["delta"], height=alt, color=AZUL,
                 edgecolor=AZUL, linewidth=0.7, zorder=2)
        esq.barh(y - alt / 2 - 0.02, c_ch["delta"], height=alt, color=VERMELHO,
                 edgecolor=VERMELHO, linewidth=0.7, zorder=2)
        esq.plot(y_ch["ci95"], [y + alt / 2 + 0.02] * 2, color=PRETO, linewidth=0.8,
                 zorder=3)
        if np.isfinite(c_ch["ci95"][0]):
            esq.plot(c_ch["ci95"], [y - alt / 2 - 0.02] * 2, color=PRETO,
                     linewidth=0.8, zorder=3)
        dir_.plot([S["xcorr"]["generic"]["mean"], max(xc)], [y, y], color="0.6",
                  linewidth=0.7, zorder=2)
        if controle:
            dir_.plot(xc, [y] * len(xc), marker="o", linestyle="none", color=PRETO,
                      markersize=4.5, markerfacecolor="white", markeredgewidth=0.9,
                      zorder=3)
        else:
            dir_.plot(xc, [y] * len(xc), marker="o", linestyle="none", color=PRETO,
                      markersize=4.5, zorder=3)

    y_sep = len(CONTROLES) - 0.5   # separates the six domains from the three controls
    for eixo in (esq, dir_):
        eixo.axhline(y_sep, color="0.55", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        eixo.grid(True, axis="x", alpha=0.25, linewidth=0.4)
        eixo.set_axisbelow(True)
    esq.axvline(0, color=PRETO, linewidth=0.8, zorder=3)
    dir_.axvline(S["xcorr"]["generic"]["mean"], color=AZUL, linewidth=1.0,
                 linestyle=(0, (4, 1.6)), zorder=3)

    esq.set_yticks(ys)
    esq.set_yticklabels([l[0] for l in linhas])
    esq.set_ylim(-0.75, len(linhas) - 0.25)
    esq.set_xlim(-15.6, 1.2)
    esq.set_xlabel("Δ PSNR casado no Kodak (dB)" if args.col
                   else "Δ PSNR casado por bpp no Kodak, por canal (dB)")
    esq.xaxis.set_major_formatter(virgula(0))
    if args.col:
        esq.set_xticks([-15, -10, -5, 0])
    else:
        esq.text(-15.3, len(linhas) - 1.45,
                 "regime leve: as três primeiras\nbarras vão de −0,02 a −0,21 dB",
                 fontsize=7, color="0.35", va="center")

    dir_.set_xlim(0.830, 0.985)
    dir_.set_xticks([0.85, 0.95] if args.col else [0.85, 0.90, 0.95])
    dir_.xaxis.set_major_formatter(virgula(2))
    dir_.set_xlabel("corr. inter-canal" if args.col
                    else "correlação inter-canal da saída")
    dir_.tick_params(left=False)
    dir_.annotate("genérica\n{:.4f}".format(S["xcorr"]["generic"]["mean"]).replace(".", ","),
                  xy=(S["xcorr"]["generic"]["mean"], len(linhas) - 0.55),
                  xytext=(4, 0), textcoords="offset points", fontsize=7,
                  color=AZUL, va="center")

    if not args.col:   # no room in one column: the note moves to the .tex caption
        dir_.text(0.856, -0.62, "○ controle: um ponto por λ", fontsize=7, color="0.35")
    barras = [plt.Rectangle((0, 0), 1, 1, facecolor=AZUL),
              plt.Rectangle((0, 0), 1, 1, facecolor=VERMELHO)]
    esq.legend(barras, ["luma (Y)", "croma (CbCr)"], loc="upper left",
               frameon=False, handlelength=1.4, handleheight=0.9,
               bbox_to_anchor=(0.005, 1.02))

    if args.col:
        fig.subplots_adjust(left=0.275, right=0.985, top=0.985, bottom=0.215)
    else:
        fig.subplots_adjust(left=0.135, right=0.99, top=0.985, bottom=0.19)
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
