"""Third arm of the color control: `div2k_decorr`.

PROBLEM. The other two arms, `div2k_color` (original RGB) and `div2k_gray` (R=G=B), show
that MONOCHROME IS ENOUGH to cause the luma damage. They cannot show that domain distance
is irrelevant: `gray` vs `color` differ in a variable that IS a domain shift, along exactly
the axis RGB-PSNR measures. The missing arm is a COLORED and DISTANT domain -- chroma
preserved, statistics displaced:

    arm       chroma  distance | if "it is monochrome"  if "it is distance"
    color     yes     ~0       | no damage              no damage
    gray      no      high     | COLLAPSES              collapses
    decorr    YES     HIGH     | no collapse            COLLAPSES     <- discriminates

TRANSFORM. A single FIXED 3x3 matrix, shared by every image of both splits: the
Hotelling/PCA (KLT) rotation of the inter-channel covariance, estimated ONCE over a sample
of the training split and then frozen. It is orthogonal by construction, with det(W) = +1
forced so it is a proper rotation rather than a reflection. It diagonalizes the channel
covariance, so the output has DECORRELATED channels -- the statistic furthest from natural
(where R, G and B correlate ~0.85) without removing any chroma.

RANGE MAPPING, the one design decision, declared here: a rotation of the [0,1]^3 cube does
not land inside the cube, so the output has to be mapped back. This uses a FIXED PER-CHANNEL
affine whose limits are computed ANALYTICALLY from the extrema of each linear functional
over the cube (which lie on its vertices):

    lo_i = sum_j min(W_ij, 0)      hi_i = sum_j max(W_ij, 0)
    z_i  = (W x - lo_i) / (hi_i - lo_i)   in [0,1] EXACTLY, no clipping

Declared deviation from "preserve energy": the scale is PER CHANNEL, so the map is a
rotation followed by per-axis scaling and does not preserve energy literally. A scale COMMON
to the three channels would preserve it, but would squeeze both chroma axes into ~10% of the
8-bit range (natural chroma variance in the KLT basis is a small fraction of luminance),
leaving `decorr` with two near-constant channels -- i.e. close to `gray` in exactly the
variable under test -- and adding a quantization artifact. Decorrelation and chroma presence
are what the design needs, so those are preserved; the per-channel scales are written to the
parameter JSON so the size of the deviation stays auditable.

Not done: per-image normalization (forbidden by the design -- it would make the transform
content-dependent, and the arm would stop being a fixed shift).

SPLIT: identical to the other two arms (700/100), read from the `div2k_color` directories
rather than re-drawn.

Usage:
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
    """Hotelling/PCA rotation of the inter-channel covariance, estimated once."""
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
    if np.linalg.det(W) < 0:                    # proper rotation, not a reflection
        W[2, :] *= -1.0
    return W, cov, evals[order], mean


def cube_bounds(W):
    """Exact extrema of each output channel over the [0,1]^3 cube."""
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
        "declared_deviation": ("PER-CHANNEL scale (not common), so the map does not preserve "
                               "energy literally; see the module docstring for why (a common "
                               "scale would squeeze the chroma axes into ~10% of the range "
                               "and push this arm towards the `gray` one)"),
        "clipping": "zero by construction: exact analytic bounds over the [0,1]^3 cube",
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
