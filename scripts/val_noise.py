"""A4 / T3.2 — quanto ruido tem o sinal que escolhe o checkpoint?

A selecao de checkpoint usa a perda de validacao medida em `--val_images` imagens
(default **24**) em recorte CENTRAL 256x256 (train/train_xray_full.py:113,121,124),
num split cujos grupos sao majoritariamente compartilhados com o treino. Isso e a
explicacao mecanistica mais provavel da instabilidade de semente que virou a
contribuicao (3), e ate hoje era so hipotese.

Este script NAO retreina. Ele recarrega os checkpoints `_best` e `_last` ja salvos e
recomputa a MESMA perda de validacao com o n do treino (24) e com o maximo disponivel,
para medir se a escolha entre `_best` e `_last` MUDARIA com uma validacao maior.

Reproduz exatamente o caminho do treinador:
  - mesma perda: lmbda * 255^2 * MSE + bpp        (train_xray_full.py:42-53)
  - mesmo transform: CenterCrop(256) + ToTensor   (train_xray_full.py:121)
  - mesma ordem de imagens: ImageFolder pega as `num_images` PRIMEIRAS de
    `iterdir()` (src/compress/datasets/utils.py), entao n=24 e o mesmo subconjunto
    que o treino viu
  - mesmo carregamento de modelo que eval/eval_full.py (ancora + delta ou state_dict)

Uso:
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

# Celulas com `_best` E `_last` salvos. Uma de cada regime de checkpoint:
# `full` grava o modelo inteiro, `encoder` grava delta sobre a ancora.
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
           "question": ("A escolha entre _best e _last mudaria se a validacao usasse "
                        "todas as imagens em vez das 24 do treino?"),
           "loss": "lmbda * 255^2 * MSE + bpp, recorte central 256x256 (identica ao treinador)",
           "note_order": ("ImageFolder toma as num_images PRIMEIRAS de iterdir(), nao uma "
                          "amostra aleatoria: n=24 e exatamente o subconjunto do treino."),
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

            # A comparacao best x last e um binario grosso: se a margem for larga ela nunca
            # inverte, e um negativo assim nao mede o RUIDO do sinal, so diz que este par
            # nao e um caso apertado. O teste direto de T3.2 e outro: quantas vezes a
            # escolha mudaria se o treino tivesse sorteado OUTRAS 24 imagens de validacao?
            # Reamostragem SEM reposicao de 24 entre as `n_all`, sobre as perdas por imagem.
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
                # Efeito no PIOR caso: qual a maior vantagem que `last` chega a mostrar.
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
