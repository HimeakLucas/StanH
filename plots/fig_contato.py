"""Contact sheet das figuras do relatorio: todas as miniaturas lado a lado.

Nao e figura do relatorio -- e a folha de escolha, para decidir de relance o que
entra quando o corte vier.

Uso:
    python plots/fig_contato.py
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRETORIO = "results/plots/figs_relatorio"
SAIDA = "results/plots/figs_relatorio/_contato.png"
TITULO = {
    "f0_espectro": "F0 · espectro na régua disjunta\n(substitui a Fig. 2)",
    "f1_escala_msssim": "F1 · escala do adaptador\n(MS-SSIM, régua única)",
    "f2_dois_regimes": "F2 · esquecimento em 6 domínios\n(dois regimes)",
    "f3_luma_croma": "F3 · mecanismo luma × croma\n(+ controle DIV2K cinza)",
    "f4_replay": "F4 · replay em 3 domínios\n(e o preço no raio-X)",
    "f5_agregado_vs_forma": "F5 · o agregado move-se,\na forma não",
    "f6_indice_saturacao": "F6 · índice pré-registrado\nnão prevê o vencedor",
    "f7_retina_bimodal": "F7 · bootstrap bimodal\nda retina",
    "f8_custo_ganho_dano": "F8 · custo × ganho × dano\n(figura-tese)",
    "f9_colapso_cor": "F9 · o colapso de cor,\nvisível (qualitativa)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=SAIDA)
    args = ap.parse_args()
    caminhos = sorted(p for p in glob.glob(os.path.join(ROOT, DIRETORIO, "*.png"))
                      if not os.path.basename(p).startswith("_"))
    if not caminhos:
        raise SystemExit(f"nenhuma figura em {DIRETORIO}")

    colunas = 3
    linhas = -(-len(caminhos) // colunas)
    fig, ax = plt.subplots(linhas, colunas, figsize=(9.5, 3.1 * linhas))
    ax = ax.ravel()
    for eixo, caminho in zip(ax, caminhos):
        chave = os.path.basename(caminho).replace(".png", "")
        eixo.imshow(Image.open(caminho))
        eixo.set_title(TITULO.get(chave, chave), fontsize=9, pad=5)
        eixo.set_xticks([]); eixo.set_yticks([])
        for lado in eixo.spines.values():
            lado.set_color("0.8")
    for eixo in ax[len(caminhos):]:
        eixo.axis("off")

    fig.tight_layout(pad=0.6)
    destino = os.path.join(ROOT, args.out)
    fig.savefig(destino, dpi=110, bbox_inches="tight")
    print(f"salvo {args.out}  ({len(caminhos)} figuras)")


if __name__ == "__main__":
    main()
