"""F3 - O mecanismo: luma x croma, e os controles que limitam a leitura causal.

Afirma duas metades da MESMA coisa: nos dominios de colapso o dano do decodificador
e dominado por CROMA e a correlacao inter-canal da saida sobe para perto de 1 (o
decodificador desaprendeu a renderizar cor); e que a leitura causal "o dano em luma
prova dano alem da cor" esta REVOGADA, porque `div2k_gray` -- conteudo natural,
distancia de dominio ~ 0, unica manipulacao R=G=B -- perde luma na mesma ordem.
As barras de controle nao sao opcionais: sem elas a figura afirma a leitura revogada.

O controle tem TRES bracos, sobre as MESMAS 800 imagens do DIV2K, variando so a
estatistica de cor, e a figura desenha os tres (ate 06/08 desenhava so o `gray`,
enquanto o texto descrevia os tres):
  - `div2k_color`  controle negativo, valida o aparato: nao danifica e nao move a xcorr;
  - `div2k_gray`   monocromia basta para o dano em luma da magnitude observada;
  - `div2k_decorr` colapsa (-2,64 dB) com a cor intacta e SEM levantar a xcorr, que
                   fica ABAIXO da generica -- e o contraexemplo que proibe ler
                   "xcorr alta <=> colapso" na metade direita da figura.

NUNCA desenhar a razao CbCr/Y: o DIOR e RGB, nao colapsa, e tem razao 5,94,
praticamente igual a do raio-X (5,86). E a magnitude que carrega o argumento.

Fontes: `results/ycbcr_decomposition_summary.json` para os seis dominios (o summary
NAO contem os bracos de controle nem os de replay, de proposito) e
`results/ycbcr_div2k_{color,gray,decorr}_on_kodak.json` para os controles, cujos
pontos e IC sao re-derivados aqui e conferidos contra os valores publicados.

⚠ B do bootstrap difere por fonte: 1000 nos seis dominios (gravado no summary) e
2000 nos bracos de controle (e o valor com que o IC publicado do `gray` reproduz;
nos outros dois o IC publicado nao distingue 1000 de 2000).

Uso:
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
# Valores ja publicados dos bracos de controle, usados aqui como ALVO DE REPRODUCAO:
# nenhum numero desta figura pode nascer nela. `dy_ci` so existe onde o IC de luma
# esta publicado (o `gray`); nos outros dois o alvo e o ponto, e o IC desenhado sai
# do mesmo reamostreio.
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
    """Mesmo reamostreio pareado por imagem que `decompose_ycbcr.analyze` usa."""
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
                    help="geometria de UMA coluna IEEE (3,4 in). Gera na largura "
                         "final, sem reducao no .tex, para que os rotulos de 7 pt "
                         "continuem com 7 pt na pagina. A anotacao do regime leve "
                         "sai da figura e passa para a legenda.")
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
        # O IC de croma dos controles NAO e desenhado: nao esta publicado, e a
        # convencao da figura e que nenhum numero nasca nela (o de luma do `gray`
        # esta, e e ele quem valida o reamostreio usado nos outros dois).
        controles.append((pub, dy, ci_y, dc, list(ctl["xcorr"])))

    # O painel da direita mostra, para os seis dominios, a media no suporte casado
    # gravada no summary. Os controles NAO estao no summary (`analyze` nao os inclui,
    # e re-roda-lo reescreveria o JSON): deles vao os proprios valores por lambda,
    # que sao o que esta publicado.
    rotulos = dict(DOMINIOS)
    if args.col:   # nomes curtos: numa coluna o rotulo do eixo come a area de desenho
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

    y_sep = len(CONTROLES) - 0.5   # separa os seis domínios dos três controles
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

    if not args.col:   # numa coluna nao ha espaco: a nota vai para a legenda do .tex
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
