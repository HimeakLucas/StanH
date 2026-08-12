"""F9 - Qualitative: the color collapse of the adapted decoder, and the replay fix.

ILLUSTRATES an already measured fact (chroma damage and output inter-channel correlation);
it measures nothing. No number originates here.

Reconstructions come from `plots/decompose_ycbcr.py decompose --save_recon`, i.e. the same
code path that produced the published curves -- the images shown are the images measured.

The crop is not picked by eye: for each image it is the square window (64 px stride) with
the highest mean chroma energy in the ORIGINAL, which is where color loss has something to
show. Nearest-neighbour zoom, so the visible artifact belongs to the codec and not to the
resampling.

How to generate the reconstructions (2 Kodak images in a separate directory):
    export PYTHONPATH=src
    for c in generic oct oct_replay; do
      python plots/decompose_ycbcr.py decompose --curve $c \\
        --dataset <dir_com_as_2_imagens> --limit 2 \\
        --out_dir <dir_temporario> --save_recon <dir_recon>
    done
    python plots/fig_colapso_cor.py --recon_dir <dir_recon> --kodak_dir <dir_com_as_2_imagens>
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGENS = ["kodim03.png", "kodim23.png"]
# curve -> (level recorded by --save_recon, column label)
# Rate-matched operating point: the three curves sit at mean bpp 0.44-0.47
# (`results/kodak_rd.json`, `results/oct_decoder{,_replay}_on_cross_rd.json`).
COLUNAS = [
    (None, None, "Original"),
    ("generic", "D11-A040.pth.tar", "Genérica"),
    ("oct", "lambda_0.02", "Decodificador\nadaptado ao OCT"),
    ("oct_replay", "lambda_0.02", "Decodificador\n+ replay (α=0,8)"),
]
LADO = 224   # lado do recorte, em pixels do original
PASSO = 64


def crop_mais_cromatico(arr, lado=LADO, passo=PASSO):
    """Janela quadrada de maior energia cromatica media no original (BT.601)."""
    a = arr.astype(np.float64)
    cb = -0.168736 * a[..., 0] - 0.331264 * a[..., 1] + 0.5 * a[..., 2]
    cr = 0.5 * a[..., 0] - 0.418688 * a[..., 1] - 0.081312 * a[..., 2]
    energia = cb ** 2 + cr ** 2
    soma = np.cumsum(np.cumsum(energia, 0), 1)
    soma = np.pad(soma, ((1, 0), (1, 0)))
    h, w = energia.shape
    melhor, pos = -1.0, (0, 0)
    for y in range(0, h - lado + 1, passo):
        for x in range(0, w - lado + 1, passo):
            s = (soma[y + lado, x + lado] - soma[y, x + lado]
                 - soma[y + lado, x] + soma[y, x])
            if s > melhor:
                melhor, pos = s, (y, x)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", required=True,
                    help="diretório passado a --save_recon (contém generic/, oct/, oct_replay/)")
    ap.add_argument("--kodak_dir", required=True, help="diretório com os originais do Kodak")
    ap.add_argument("--out", default="results/plots/figs_relatorio/f9_colapso_cor.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5})
    fig, ax = plt.subplots(len(IMAGENS), len(COLUNAS), figsize=(7.0, 4.05))

    for i, nome in enumerate(IMAGENS):
        original = np.array(Image.open(os.path.join(args.kodak_dir, nome)).convert("RGB"))
        y, x = crop_mais_cromatico(original)
        for j, (curva, nivel, titulo) in enumerate(COLUNAS):
            if curva is None:
                img = original
            else:
                img = np.array(Image.open(os.path.join(
                    args.recon_dir, curva, f"{nivel}__{nome}")).convert("RGB"))
            eixo = ax[i, j]
            eixo.imshow(img[y:y + LADO, x:x + LADO], interpolation="nearest")
            eixo.set_xticks([]); eixo.set_yticks([])
            for lado in eixo.spines.values():
                lado.set_linewidth(0.6)
            if i == 0:
                eixo.set_title(titulo, fontsize=8.5, pad=4)
        ax[i, 0].set_ylabel(nome.replace(".png", ""), fontsize=8)

    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.35)
    destino = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    print("salvo", args.out)


if __name__ == "__main__":
    main()
