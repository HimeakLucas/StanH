"""F4 - O antidoto: replay em 3 dominios, e o preco no raio-X.

Afirma: replay DIV2K com alfa=0,8 elimina o colapso do decodificador nos tres
dominios em que foi testado -- incluindo o pior do estudo -- e no raio-X isso custa
cerca de um terco do ganho no alvo.

⚠ Vocabulario. No OCT o alvo e PRESERVADO (+0,233 contra +0,226 dB: sete milesimos,
nunca "melhorou"). No raio-X ele e MAJORITARIAMENTE PRESERVADO, AO PRECO DE CERCA DE
UM TERCO -- nunca "preservado", que e o verbo do OCT. Os dois casos levam anotacao
propria, para que a figura nao apague a distincao.

⚠ Unidade: ΔPSNR casado por bpp. O BD-Rate do alvo destas celulas NAO e reportavel
(OCT: janela 0,988 dB, piso 0,958; raio-X: 0,768 e 0,744; reprovam a 1a guarda) e
por isso nao aparece aqui em hipotese alguma.

⚠ Suportes diferentes por linha, declarados na figura: documentos compara os 8 lambda
(as duas curvas os tem), OCT e raio-X comparam os 3 lambda do braco com replay.

Todos os valores sao lidos de JSON ou re-derivados e conferidos contra o publicado;
se algum nao reproduzir, o script para sem plotar.

Uso:
    export PYTHONPATH=src
    python plots/fig_replay.py
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
from decompose_ycbcr import matched_mean, matched_support  # noqa: E402

AZUL, VERMELHO, PRETO = "#0072B2", "#D55E00", "#000000"
KODAK = "results/kodak_rd.json"
B, SEED, TOL = 1000, 42, 0.01

# dominio, json sem replay, json com replay, (publicado sem, publicado com)
CROSS = [
    ("Documentos", "results/documents_decoder_on_cross_rd.json",
     "results/documents_decoder_replay_on_cross_rd.json", (-6.01, -0.02)),
    ("OCT", "results/oct_decoder_on_cross_rd.json",
     "results/oct_decoder_replay_on_cross_rd.json", (-8.010, -0.058)),
    ("Raio-X", "results/v7_decoder_on_kodak_rd.json",
     "results/_exp_01ago/xray_decoder_replay_on_cross_rd.json", (-4.158, +0.024)),
]
# o preco no alvo: lido dos JSONs de contraste do lote de 01/08, sem recalcular
ALVO = [
    ("OCT", "results/_exp_01ago/_control_b2_oct_target_grid.json",
     "oct_sem_replay", "oct_com_replay", "preservado"),
    ("Raio-X", "results/_exp_01ago/e2_target_grid3.json",
     "xray_decoder_sem_replay", "xray_decoder_com_replay", "≈1/3 do ganho abandonado"),
]


def load(nome):
    with open(os.path.join(ROOT, nome)) as fh:
        return json.load(fh)


def virgula(casas):
    return FuncFormatter(lambda v, _pos: f"{v:.{casas}f}".replace(".", ","))


def restringe(js, lambdas):
    i = [js["lambdas"].index(l) for l in lambdas]
    out = {k: [js[k][j] for j in i] for k in ("lambdas", "bpp", "psnr")}
    out["per_image"] = {l: js["per_image"][l] for l in lambdas}
    return out


def pilha(js, chave):
    return np.array([js["per_image"][l][chave] for l in js["lambdas"]])


def casado(ref, teste):
    """ΔPSNR medio casado por bpp, com IC bootstrap pareado por imagem."""
    v, _ = matched_mean(ref["bpp"], ref["psnr"], teste["bpp"], teste["psnr"])
    n_sup = len(matched_support(ref["bpp"], ref["psnr"], teste["bpp"]))
    rb, rp = pilha(ref, "bpp"), pilha(ref, "psnr")
    tb, tp = pilha(teste, "bpp"), pilha(teste, "psnr")
    n = rb.shape[1]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(matched_mean(rb[:, i].mean(1), rp[:, i].mean(1),
                                 tb[:, i].mean(1), tp[:, i].mean(1))[0])
    o = np.array(vals)
    o = o[np.isfinite(o)]
    return v, (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))), n_sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/figs_relatorio/f4_replay.png")
    args = ap.parse_args()

    kod = load(KODAK)
    kod["lambdas"] = kod["levels"]

    dados, pior = [], 0.0
    print(f"{'domínio':<12}{'sem replay':>26}{'com replay':>26}{'λ':>4}{'n sup':>7}")
    for rotulo, j_sem, j_com, publ in CROSS:
        sem, com = load(j_sem), load(j_com)
        if len(com["lambdas"]) < len(sem["lambdas"]):      # like-for-like
            sem = restringe(sem, com["lambdas"])
        v_s, ci_s, n_s = casado(kod, sem)
        v_c, ci_c, n_c = casado(kod, com)
        pior = max(pior, abs(v_s - publ[0]), abs(v_c - publ[1]))
        dados.append((rotulo, v_s, ci_s, v_c, ci_c, len(com["lambdas"]), min(n_s, n_c)))
        print(f"{rotulo:<12}{v_s:>+9.3f} [{ci_s[0]:+.3f},{ci_s[1]:+.3f}]"
              f"{v_c:>+9.3f} [{ci_c[0]:+.3f},{ci_c[1]:+.3f}]"
              f"{len(com['lambdas']):>4}{min(n_s, n_c):>7}")
    print(f"pior |diferença| vs publicado: {pior:.4f} dB (tolerância {TOL})")
    if pior > TOL:
        raise SystemExit("re-derivação NÃO reproduz os valores publicados — não plotado.")

    alvo = []
    for rotulo, caminho, k_sem, k_com, nota in ALVO:
        g = load(caminho)["common_grid"]
        alvo.append((rotulo, g[k_sem]["dpsnr"], g[k_com]["dpsnr"], nota))
        print(f"alvo {rotulo:<8} {g[k_sem]['dpsnr']:+.3f} → {g[k_com]['dpsnr']:+.3f} dB  ({nota})")

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
    })
    fig, (esq, dir_) = plt.subplots(
        1, 2, figsize=(7.0, 2.35),
        gridspec_kw=dict(width_ratios=[2.05, 1.0], wspace=0.30))

    ys = [2, 1, 0]
    for y, (rotulo, v_s, ci_s, v_c, ci_c, n_lam, n_sup) in zip(ys, dados):
        esq.annotate("", xy=(v_c, y), xytext=(v_s, y),
                     arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.45",
                                     shrinkA=4.5, shrinkB=4.5))
        esq.plot(ci_s, [y, y], color=VERMELHO, linewidth=1.0, zorder=3)
        esq.plot(ci_c, [y, y], color=AZUL, linewidth=1.0, zorder=3)
        esq.plot([v_s], [y], marker="o", color=VERMELHO, markersize=5.5, zorder=4)
        esq.plot([v_c], [y], marker="s", color="white", markeredgecolor=AZUL,
                 markeredgewidth=1.2, markersize=5.5, zorder=4)
        esq.annotate(f"{n_lam} λ · {n_sup} no suporte", xy=(0.35, y), fontsize=7,
                     color="0.35", va="center")
    esq.axvline(0, color=PRETO, linewidth=0.8)
    esq.set_yticks(ys)
    esq.set_yticklabels([d[0] for d in dados])
    esq.set_ylim(-0.6, 2.6)
    esq.set_xlim(-9.6, 2.6)
    esq.set_xticks([-8, -6, -4, -2, 0])
    esq.xaxis.set_major_formatter(virgula(0))
    esq.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    esq.set_axisbelow(True)
    esq.set_xlabel("Dano no domínio de origem: ΔPSNR casado no Kodak (dB)")

    ys2 = [1, 0]
    for y, (rotulo, v_s, v_c, nota) in zip(ys2, alvo):
        dir_.annotate("", xy=(v_c, y), xytext=(v_s, y),
                      arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.45",
                                      shrinkA=4.5, shrinkB=4.5))
        dir_.plot([v_s], [y], marker="o", color=VERMELHO, markersize=5.5, zorder=4)
        dir_.plot([v_c], [y], marker="s", color="white", markeredgecolor=AZUL,
                  markeredgewidth=1.2, markersize=5.5, zorder=4)
        dir_.annotate(nota, xy=((v_s + v_c) / 2, y), xytext=(0, 9),
                      textcoords="offset points", fontsize=7, color="0.25", ha="center")
    dir_.axvline(0, color=PRETO, linewidth=0.8)
    dir_.set_yticks(ys2)
    dir_.set_yticklabels([a[0] for a in alvo])
    dir_.set_ylim(-0.6, 1.75)
    dir_.set_xlim(-0.01, 0.30)
    dir_.set_xticks([0.0, 0.1, 0.2])
    dir_.xaxis.set_major_formatter(virgula(1))
    dir_.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    dir_.set_axisbelow(True)
    dir_.set_xlabel("Ganho no alvo (dB)")

    marcas = [plt.Line2D([], [], marker="o", color=VERMELHO, linestyle="none",
                         markersize=5.5),
              plt.Line2D([], [], marker="s", color="white", markeredgecolor=AZUL,
                         markeredgewidth=1.2, linestyle="none", markersize=5.5)]
    fig.legend(marcas, ["sem replay", "com replay (α=0,8)"], loc="lower center",
               ncol=2, frameon=False, handlelength=1.1, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(left=0.105, right=0.99, top=0.97, bottom=0.30)
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
