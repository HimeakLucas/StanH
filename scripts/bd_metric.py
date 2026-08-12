"""BD-Rate with a bootstrap CI over an ARBITRARY quality metric (PSNR or MS-SSIM).

`plots/analyze_finetuned.py` computes BD-Rate + per-image paired bootstrap CI, but only over
`psnr`, because `per_image` used to store just `bpp` and `psnr`. The three evaluators now
also record `ms-ssim` in `per_image`, in dB; this script is the corresponding generic reader.

Reports per cell: BD-Rate, 95% CI, number of post-Pareto points of both curves, integration
window and the bootstrap FLOOR of that window (2.5th percentile) -- the two validity guards.

The 1 dB threshold of the first guard was calibrated in PSNR. In MS-SSIM-dB it is NOT
calibrated: the script prints the floor, not a verdict.

Uso:
  python scripts/bd_metric.py --ref results/..._rd.json --metric ms-ssim \
      --test nome=results/..._rd.json [--test ...] --out results/_exp_XX/saida.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"))
from analyze_finetuned import bd_rate, drop_dominated  # noqa: E402  (mesma funcao do repo)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


def per_image(curve, metric):
    keys = curve.get("levels") or curve.get("lambdas")
    if metric not in curve["per_image"][keys[0]]:
        raise SystemExit(f"per_image sem a chave '{metric}' (JSON anterior ao A5?)")
    b = np.array([curve["per_image"][k]["bpp"] for k in keys])
    q = np.array([curve["per_image"][k][metric] for k in keys])
    return b, q


def analyse(ref, test, metric="ms-ssim", B=1000, seed=42):
    bd, (lo, hi) = bd_rate(ref["bpp"], ref[metric], test["bpp"], test[metric])
    n_ref = len(drop_dominated(ref["bpp"], ref[metric])[0])
    n_test = len(drop_dominated(test["bpp"], test[metric])[0])
    out = {
        "bd_rate_pct": float(bd),
        "window": {"lo": float(lo), "hi": float(hi), "width": float(hi - lo)},
        "n_pareto_ref": n_ref,
        "n_pareto_test": n_test,
        "n_points_ref": len(ref["bpp"]),
        "n_points_test": len(test["bpp"]),
    }
    if "per_image" not in ref or "per_image" not in test or ref.get("files") != test.get("files"):
        out["ci95"] = None
        return out
    rb, rq = per_image(ref, metric)
    tb, tq = per_image(test, metric)
    n = rb.shape[1]
    rng = np.random.default_rng(seed)
    vals, widths = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v, (l, h) = bd_rate(rb[:, idx].mean(1), rq[:, idx].mean(1),
                            tb[:, idx].mean(1), tq[:, idx].mean(1))
        if np.isfinite(v):
            vals.append(v)
            widths.append(h - l)
    out["ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    out["n_resamples_valid"] = len(vals)
    out["window_bootstrap_floor"] = float(np.percentile(widths, 2.5))
    out["excludes_zero"] = bool(out["ci95"][0] * out["ci95"][1] > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", action="append", required=True, help="nome=caminho.json")
    ap.add_argument("--metric", default="ms-ssim")
    ap.add_argument("-B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = load(args.ref)
    res = {}
    print(f"metrica: {args.metric}   referencia: {args.ref}\n")
    print(f"{'celula':<16} {'BD':>8}  {'IC 95%':>20}  {'pos-Pareto':>10}  {'janela':>7}  {'piso':>6}")
    for spec in args.test:
        name, path = spec.split("=", 1)
        r = analyse(ref, load(path), args.metric, args.B, args.seed)
        r["json"] = path
        res[name] = r
        ci = f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" if r["ci95"] else "(sem per_image)"
        print(f"{name:<16} {r['bd_rate_pct']:>+7.2f}%  {ci:>20}  "
              f"{r['n_pareto_test']}/{r['n_points_test']} ({r['n_pareto_ref']}/{r['n_points_ref']})  "
              f"{r['window']['width']:>7.3f}  {r.get('window_bootstrap_floor', float('nan')):>6.3f}")

    if args.out:
        path = os.path.join(ROOT, args.out)
        if os.path.exists(path):
            raise SystemExit(f"recusando sobrescrever {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"metric": args.metric, "reference": args.ref, "B": args.B,
                   "seed": args.seed, "cells": res,
                   "warning": ("the 1 dB threshold of the first guard was calibrated in PSNR; "
                               "in MS-SSIM-dB it is not calibrated -- the floor is reported, "
                               "not judged")},
                  open(path, "w"), indent=2, ensure_ascii=False)
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
