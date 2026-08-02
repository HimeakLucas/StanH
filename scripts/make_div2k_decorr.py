"""B1 — terceiro braco do controle de monocromia: `div2k_decorr`.

PROBLEMA. O controle G4 tem dois bracos, `div2k_color` (RGB original) e `div2k_gray`
(R=G=B), e mostra que a MONOCROMIA BASTA para causar o dano em luma. Ele nao mostra que
a distancia seja irrelevante, e nao pode: `gray` x `color` diferem numa variavel que E um
deslocamento de dominio, ao longo exatamente do eixo que o PSNR-RGB mede.

O braco que falta e um dominio COLORIDO e DISTANTE: preserva croma, desloca a estatistica.

    braco     croma   distancia | se "e monocromia"  se "e distancia"
    color     sim     ~0        | sem dano           sem dano
    gray      nao     alta      | COLAPSA            colapsa
    decorr    SIM     ALTA      | sem colapso        COLAPSA        <- discrimina

TRANSFORMACAO. Uma unica matriz 3x3 FIXA, a mesma para todas as imagens dos dois splits:
a rotacao de Hotelling/PCA (KLT) da covariancia inter-canal, estimada UMA VEZ sobre uma
amostra do split de treino e depois congelada. E ortogonal por construcao (autovetores de
uma matriz simetrica), e forcamos det(W) = +1 para que seja rotacao propria, nao reflexao.
Ela diagonaliza a covariancia entre canais: a saida tem canais DESCORRELACIONADOS, que e
a estatistica mais distante da natural (onde R, G e B tem correlacao ~0,85) sem tirar
croma nenhum.

MAPEAMENTO DE FAIXA — e aqui esta a unica decisao de projeto, declarada:
uma rotacao do cubo [0,1]^3 nao cai dentro do cubo, entao a saida precisa voltar a faixa.
Fazemos isso com um afim FIXO POR CANAL, cujos limites sao calculados ANALITICAMENTE dos
extremos de cada funcional linear sobre o cubo (os extremos de um funcional linear num
cubo estao nos vertices):

    lo_i = sum_j min(W_ij, 0)      hi_i = sum_j max(W_ij, 0)
    z_i  = (W x - lo_i) / (hi_i - lo_i)   em [0,1] EXATAMENTE, zero clipping

⚠ DESVIO DECLARADO em relacao a "preserve a energia": a escala e POR CANAL, entao o mapa
e uma rotacao seguida de escalas de eixo diferentes, e nao preserva energia literalmente.
A alternativa (escala COMUM aos tres canais, que preservaria energia) comprimiria os dois
eixos de croma para ~10% da faixa de 8 bits, porque a variancia natural de croma na base
KLT e uma fracao pequena da de luminancia. Isso deixaria o braco `decorr` com dois canais
quase constantes — isto e, PARECIDO COM O BRACO `gray` exatamente na variavel sob teste —
alem de introduzir um artefato de quantizacao meu. Escolhi preservar a DESCORRELACAO e a
PRESENCA DE CROMA, que sao o que o desenho experimental precisa, e declarar a perda de
isometria. As escalas por canal ficam gravadas no JSON de parametros, para que a magnitude
do desvio seja auditavel.

O que NAO se faz: normalizacao por imagem (proibida pelo desenho — tornaria
a transformacao dependente do conteudo, e o braco deixaria de ser um deslocamento fixo).

PARTICAO: identica a dos outros dois bracos (700/100), lida dos proprios diretorios de
`div2k_color`, nao re-sorteada.

Uso:
    PYTHONPATH=src python scripts/make_div2k_decorr.py
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "datasets/div2k_color"
DST = "datasets/div2k_decorr"


def fit_rotation(files, max_pixels_per_image=200_000, seed=42):
    """Rotacao de Hotelling/PCA da covariancia inter-canal, estimada uma vez."""
    rng = np.random.default_rng(seed)
    acc_n = 0
    acc_sum = np.zeros(3, dtype=np.float64)
    acc_ss = np.zeros((3, 3), dtype=np.float64)
    for i, f in enumerate(files):
        x = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
        if x.shape[0] > max_pixels_per_image:
            idx = rng.choice(x.shape[0], max_pixels_per_image, replace=False)
            x = x[idx]
        acc_n += x.shape[0]
        acc_sum += x.sum(0)
        acc_ss += x.T @ x
        if (i + 1) % 100 == 0:
            print(f"  covariancia: {i+1}/{len(files)} imagens")
    mean = acc_sum / acc_n
    cov = acc_ss / acc_n - np.outer(mean, mean)
    evals, evecs = np.linalg.eigh(cov)          # simetrica -> evecs ortogonal
    order = np.argsort(evals)[::-1]             # variancia decrescente: PC1 = luminancia
    W = evecs[:, order].T                       # linhas = componentes principais
    if np.linalg.det(W) < 0:                    # rotacao propria, nao reflexao
        W[2, :] *= -1.0
    return W, cov, evals[order], mean


def cube_bounds(W):
    """Extremos exatos de cada canal de saida sobre o cubo [0,1]^3."""
    lo = np.minimum(W, 0.0).sum(1)
    hi = np.maximum(W, 0.0).sum(1)
    return lo, hi


def transform(img, W, lo, hi):
    x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    y = x @ W.T.astype(np.float32)
    z = (y - lo.astype(np.float32)) / (hi - lo).astype(np.float32)
    # `z` esta em [0,1] por construcao; o clip so protege contra ruido de ponto flutuante.
    return Image.fromarray(np.round(np.clip(z, 0.0, 1.0) * 255.0).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--fit_images", type=int, default=200, help="imagens do treino para estimar a covariancia")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = os.path.join(ROOT, args.src)
    dst = os.path.join(ROOT, args.dst)
    if os.path.exists(dst):
        raise SystemExit(f"RECUSADO: {dst} ja existe.")

    splits = {}
    for sp in ("train", "val"):
        d = os.path.join(src, sp, "data")
        splits[sp] = sorted(os.listdir(d))
        print(f"{sp}: {len(splits[sp])} imagens (particao herdada de {args.src})")

    fit_files = [os.path.join(src, "train", "data", n)
                 for n in splits["train"][:args.fit_images]]
    print(f"\nEstimando a rotacao KLT em {len(fit_files)} imagens do treino...")
    W, cov, evals, mean = fit_rotation(fit_files, seed=args.seed)
    lo, hi = cube_bounds(W)

    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    print(f"\ncorrelacao inter-canal da FONTE (div2k_color):")
    print(f"   RG={corr[0,1]:.4f}  RB={corr[0,2]:.4f}  GB={corr[1,2]:.4f}")
    print(f"variancia por componente principal: {evals}")
    print(f"rotacao W (det={np.linalg.det(W):+.6f}, ortogonal? "
          f"{np.allclose(W @ W.T, np.eye(3), atol=1e-10)}):\n{W}")
    print(f"limites por canal no cubo: lo={lo}, hi={hi}, escalas={(hi-lo)}")

    params = {
        "generated_by": "scripts/make_div2k_decorr.py (B1, 30/07/2026)",
        "source": args.src, "partition": "herdada de div2k_color (700/100), nao re-sorteada",
        "rotation_W_rows_are_principal_components": W.tolist(),
        "det_W": float(np.linalg.det(W)),
        "orthogonal": bool(np.allclose(W @ W.T, np.eye(3), atol=1e-10)),
        "fit_images": len(fit_files), "seed": args.seed,
        "source_channel_corr": {"RG": float(corr[0, 1]), "RB": float(corr[0, 2]),
                                "GB": float(corr[1, 2])},
        "pc_variances": evals.tolist(), "source_channel_mean": mean.tolist(),
        "cube_bounds_lo": lo.tolist(), "cube_bounds_hi": hi.tolist(),
        "per_channel_scale": (hi - lo).tolist(),
        "declared_deviation": ("escala POR CANAL (nao comum), logo o mapa nao preserva "
                               "energia literalmente; ver docstring do script para o motivo "
                               "(escala comum colapsaria os eixos de croma para ~10% da "
                               "faixa e aproximaria este braco do braco `gray`)"),
        "clipping": "zero por construcao: limites analiticos exatos sobre o cubo [0,1]^3",
    }

    for sp, names in splits.items():
        outdir = os.path.join(dst, sp, "data")
        os.makedirs(outdir, exist_ok=True)
        for i, n in enumerate(names):
            img = Image.open(os.path.join(src, sp, "data", n))
            transform(img, W, lo, hi).save(os.path.join(outdir, n))
            if (i + 1) % 100 == 0:
                print(f"  {sp}: {i+1}/{len(names)}")
        print(f"{sp}: {len(names)} escritas em {outdir}")

    with open(os.path.join(dst, "transform_params.json"), "w") as f:
        json.dump(params, f, indent=4)
    print(f"\nparametros em {dst}/transform_params.json")


if __name__ == "__main__":
    main()
