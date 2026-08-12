"""How noisy is the signal that selects the checkpoint?

Checkpoint selection uses the validation loss measured on `--val_images` images (default
24) in a CENTRAL 256x256 crop, over a split whose groups are mostly shared with training.
That is the most likely mechanistic explanation for the seed instability reported in the
work, and it was only a hypothesis until measured here.

This script does NOT retrain. It reloads the saved `_best` and `_last` checkpoints and
recomputes the SAME validation loss with the training n (24) and with the maximum available,
to measure whether the choice between `_best` and `_last` WOULD CHANGE under a larger
validation set.

It reproduces the trainer path exactly:
  - same loss: lmbda * 255^2 * MSE + bpp        (train_xray_full.py:42-53)
  - same transform: CenterCrop(256) + ToTensor  (train_xray_full.py:121)
  - same image order: ImageFolder takes the FIRST `num_images` of `iterdir()`
    (src/compress/datasets/utils.py), so n=24 is the subset training saw
  - same model loading as eval/eval_full.py (anchor + delta, or state_dict)

Usage:
    PYTHONPATH=src python scripts/val_noise.py --out results/_exp_30jul/val_noise.json
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from compress.datasets.utils import ImageFolder
from compress.models.cnn_multiStanh import WACNNMultiSTanH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = "models/original_paper/STanH/anchor/0728_last_.pth.tar"

# Cells with both `_best` and `_last` saved, one per checkpoint regime: `full` stores the
# whole model, `encoder` stores a delta over the anchor.
CELLS = {
    "xray_full_v6_runB": {"dataset": "datasets/xrays", "mode": "full"},
    "xray_encoder_finetuning_v8": {"dataset": "datasets/xrays", "mode": "encoder"},
}


def rd_loss(out, target, lmbda):
    n, _, h, w = target.size()
    npx = n * h * w
    mse = torch.nn.functional.mse_loss(out["x_hat"], target)
    bpp = sum((torch.log(l).sum() / (-math.log(2) * npx)) for l in out["likelihoods"].values())
    return (lmbda * 255 ** 2 * mse + bpp).item(), mse.item(), bpp.item()


def build(ckpt_path, device, anchor_sd):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WACNNMultiSTanH(N=192, M=320, num_stanh=1,
                            factorized_configuration=ck["factorized_configuration"],
                            gaussian_configuration=ck["gaussian_configuration"]).to(device)
    model.update(device=torch.device(device))
    if ck.get("is_delta"):
        model.load_state_dict(anchor_sd(), state_dicts_stanh=None)
        merged = model.state_dict()
        merged.update({k: v.to(device=device, dtype=merged[k].dtype)
                       for k, v in ck["delta"].items() if k in merged})
        model.load_state_dict(merged, state_dicts_stanh=None)
    else:
        model.load_state_dict(ck["state_dict"], state_dicts_stanh=None)
    model.update(device=torch.device(device))
    model.eval()
    return model


@torch.no_grad()
def val_loss(model, loader, lmbda, device, keep_per_image=False):
    tot = tot_mse = tot_bpp = 0.0
    n = 0
    per_image = []
    for d in loader:
        d = d.to(device)
        out = model(d, training=False, stanh_level=0)
        l, m, b = rd_loss(out, d, lmbda)
        tot += l; tot_mse += m; tot_bpp += b; n += 1
        if keep_per_image:
            per_image.append(l)
        if device == "cuda":
            torch.cuda.empty_cache()
    n = max(n, 1)
    return tot / n, tot_mse / n, tot_bpp / n, per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/_exp_30jul/val_noise.json")
    ap.add_argument("--small", type=int, default=24, help="n do treino")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="reamostragens de quais 24 imagens; 0 desliga")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dst = os.path.join(ROOT, args.out)
    if os.path.exists(dst):
        raise SystemExit(f"RECUSADO: {dst} ja existe.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _a = {"sd": None}

    def anchor_sd():
        if _a["sd"] is None:
            _a["sd"] = torch.load(os.path.join(ROOT, ANCHOR), map_location=device,
                                  weights_only=False)["state_dict"]
        return _a["sd"]

    tf = transforms.Compose([transforms.CenterCrop((args.patch, args.patch)),
                             transforms.ToTensor()])

    out = {"generated_by": "scripts/val_noise.py (A4/T3.2, 30/07/2026)",
           "question": ("would the choice between _best and _last change if validation used "
                        "every image instead of the 24 seen during training?"),
           "loss": "lmbda * 255^2 * MSE + bpp, central 256x256 crop (identical to the trainer)",
           "note_order": ("ImageFolder takes the FIRST num_images of iterdir(), not a random "
                          "sample: n=24 is exactly the subset training used."),
           "cells": {}}

    for cell, cfg in CELLS.items():
        ds_root = os.path.join(ROOT, cfg["dataset"])
        n_all = len(list((os.path.join(ds_root, "val", "data"), ) and
                         os.listdir(os.path.join(ds_root, "val", "data"))))
        loaders = {}
        for tag, n in (("n24", args.small), ("nall", n_all)):
            ds = ImageFolder(ds_root, num_images=n, split="val", transform=tf)
            loaders[tag] = (DataLoader(ds, batch_size=1, num_workers=4, shuffle=False), len(ds))
        print(f"\n=== {cell} | val n24={loaders['n24'][1]}  nall={loaders['nall'][1]} ===")

        rec = {"dataset": cfg["dataset"], "mode": cfg["mode"],
               "n_small": loaders["n24"][1], "n_all": loaders["nall"][1], "lambdas": {}}

        for best in sorted(glob.glob(os.path.join(ROOT, "models", cell, "*_best.pth.tar"))):
            lam_s = os.path.basename(best).replace("lambda_", "").replace("_best.pth.tar", "")
            last = best.replace("_best.pth.tar", "_last.pth.tar")
            if not os.path.exists(last):
                print(f"  {lam_s}: sem _last, pulado"); continue
            lmbda = float(lam_s)
            vals = {}
            per_img = {}
            for which, path in (("best", best), ("last", last)):
                model = build(path, device, anchor_sd)
                for tag in ("n24", "nall"):
                    dl, _ = loaders[tag]
                    keep = args.bootstrap > 0 and tag == "nall"
                    l, m, b, pi = val_loss(model, dl, lmbda, device, keep_per_image=keep)
                    vals[f"{which}_{tag}"] = {"loss": l, "mse": m, "bpp": b}
                    if keep:
                        per_img[which] = pi
                del model
                torch.cuda.empty_cache()
            pick24 = "best" if vals["best_n24"]["loss"] <= vals["last_n24"]["loss"] else "last"
            pickall = "best" if vals["best_nall"]["loss"] <= vals["last_nall"]["loss"] else "last"
            entry = {
                **vals,
                "margin_n24": vals["last_n24"]["loss"] - vals["best_n24"]["loss"],
                "margin_nall": vals["last_nall"]["loss"] - vals["best_nall"]["loss"],
                "pick_n24": pick24, "pick_nall": pickall, "flips": pick24 != pickall}

            # best vs last is a coarse binary: a wide margin never flips, and such a
            # negative does not measure the NOISE of the signal, only that this pair is not
            # a close call. The direct test is another one: how often would the choice change
            # had training drawn ANOTHER 24 validation images? Resampling without replacement
            # of 24 out of `n_all`, over the per-image losses.
            if args.bootstrap > 0 and "best" in per_img and "last" in per_img:
                pb = np.array(per_img["best"]); pl = np.array(per_img["last"])
                rng = np.random.default_rng(args.seed)
                n_pool = pb.size
                flips = 0
                for _ in range(args.bootstrap):
                    idx = rng.choice(n_pool, size=args.small, replace=False)
                    if pl[idx].mean() < pb[idx].mean():
                        flips += 1
                entry["boot_flip_rate"] = flips / args.bootstrap
                entry["boot_B"] = args.bootstrap
                # Worst-case effect: the largest advantage `last` ever shows.
                entry["per_image_mean_gap"] = float((pl - pb).mean())
                entry["per_image_sd_gap"] = float((pl - pb).std(ddof=1))

            rec["lambdas"][lam_s] = entry
            flag = "  <-- INVERTE" if pick24 != pickall else ""
            br = f" | P(inverte com outras 24) = {entry['boot_flip_rate']:.3f}" \
                 if "boot_flip_rate" in entry else ""
            print(f"  lambda={lam_s:8s} n24: best {vals['best_n24']['loss']:.5f} / last "
                  f"{vals['last_n24']['loss']:.5f} -> {pick24:4s} | nall: best "
                  f"{vals['best_nall']['loss']:.5f} / last {vals['last_nall']['loss']:.5f} "
                  f"-> {pickall:4s}{flag}{br}")
        rec["n_flips"] = sum(1 for v in rec["lambdas"].values() if v["flips"])
        rec["n_lambdas"] = len(rec["lambdas"])
        out["cells"][cell] = rec
        print(f"  => {rec['n_flips']} inversoes em {rec['n_lambdas']} lambdas")

    out["total_flips"] = sum(c["n_flips"] for c in out["cells"].values())
    out["total_lambdas"] = sum(c["n_lambdas"] for c in out["cells"].values())
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=4)
    print(f"\nTOTAL: {out['total_flips']} inversoes em {out['total_lambdas']} celulas-lambda")
    print(f"escrito: {dst}")


if __name__ == "__main__":
    main()
