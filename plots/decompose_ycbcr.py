"""
Y/CbCr decomposition of the decoder's cross-domain damage on Kodak.

Makes executable a measurement that previously existed only as hand-computed
notes: how much of the cross-domain PSNR-RGB loss of a domain-adapted DECODER is
chroma collapse and how much is luma (statistics) damage. Grayscale domains are trained on 3x-replicated gray, so a decoder that
spent 20 epochs seeing R=G=B unlearns colour; RGB-PSNR alone cannot tell that
apart from ordinary forgetting.

Two stages, one file:

  1) decompose (GPU) - rebuild each checkpoint, compress/decompress Kodak and
     store PER IMAGE: bpp, PSNR-RGB, PSNR-Y, PSNR-Cb, PSNR-Cr, PSNR-CbCr and the
     mean inter-channel correlation of the reconstruction.

       python plots/decompose_ycbcr.py decompose --all
       python plots/decompose_ycbcr.py decompose --curve oct

  2) analyze (CPU) - matched-bpp deltas against the generic curve, per channel,
     with the same paired bootstrap over images used everywhere else in the
     project, plus the adjacent-gap tests and the point-by-point confrontation
     with the 24/07 numbers.

       python plots/decompose_ycbcr.py analyze

Conventions kept identical to eval/eval_full.py so the RGB column is directly
comparable to results/*_on_cross_rd.json: padding to a multiple of 64, entropy
estimation by default, metrics on rounded 0-255 values, PSNR over the unpadded
image. Y/CbCr uses the full-range BT.601 (JPEG) matrix; --matrix bt709 exists so
a convention mismatch can be diagnosed instead of guessed.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "plots"))
from compressai.ops import compute_padding  # noqa: E402
from compress.models.cnn_multiStanh import WACNNMultiSTanH  # noqa: E402
from analyze_finetuned import drop_dominated  # noqa: E402

ANCHOR = "models/original_paper/STanH/anchor/0728_last_.pth.tar"
DERIVATIONS = "models/original_paper/STanH/derivations"

# The eight curves: the generic (authors') derivations, the six adapted decoders
# that make up the forgetting rule, and the replay antidote.
# `rgb_ref` is the already-published RGB curve, used as a pipeline check: the
# recomputed PSNR-RGB must reproduce it point by point.
CURVES = {
    "generic":   dict(kind="generic", rgb_ref="results/kodak_rd.json"),
    "rico":      dict(kind="ckpt", models_dir="models/rico_decoder",
                      rgb_ref="results/rico_decoder_on_cross_rd.json"),
    "dior":      dict(kind="ckpt", models_dir="models/dior_decoder",
                      rgb_ref="results/dior_decoder_on_cross_rd.json"),
    "retina":    dict(kind="ckpt", models_dir="models/retina_decoder",
                      rgb_ref="results/retina_decoder_on_cross_rd.json"),
    "xray":      dict(kind="ckpt", models_dir="models/xray_decoder_finetuning_v7",
                      rgb_ref="results/v7_decoder_on_kodak_rd.json"),
    "documents": dict(kind="ckpt", models_dir="models/documents_decoder",
                      rgb_ref="results/documents_decoder_on_cross_rd.json"),
    "oct":       dict(kind="ckpt", models_dir="models/oct_decoder",
                      rgb_ref="results/oct_decoder_on_cross_rd.json"),
    "replay":    dict(kind="ckpt", models_dir="models/documents_decoder_replay",
                      rgb_ref="results/documents_decoder_replay_on_cross_rd.json"),
}

# Ordering of the forgetting rule (light regime first). `replay` is not part of
# the ordering: it is the antidote applied to `documents`.
ORDER = ["rico", "dior", "retina", "xray", "documents", "oct"]
LABEL = {"rico": "RICO", "dior": "DIOR", "retina": "Retina", "xray": "Raio-X",
         "documents": "Documentos", "oct": "OCT", "replay": "Replay (docs a=0,8)"}

# Reference table from an earlier hand-run of the modality test. These numbers
# had no artefact in the repo; this constant exists so the comparison is
# executable and any divergence surfaces as a diff, not as a memory.
REF_24JUL = {
    "rico":      dict(dy=-0.022, dy_ci=[-0.03, -0.01], dcbcr=-0.060, dcbcr_ci=[-0.08, -0.04],
                      ratio=2.8, drgb=-0.031, xcorr=0.849),
    "dior":      dict(dy=-0.026, dy_ci=[-0.04, -0.02], dcbcr=-0.157, dcbcr_ci=[-0.20, -0.11],
                      ratio=5.9, drgb=-0.053, xcorr=0.849),
    "retina":    dict(dy=-0.152, dy_ci=[-0.19, -0.12], dcbcr=-0.205, dcbcr_ci=[-0.24, -0.17],
                      ratio=1.3, drgb=-0.161, xcorr=0.849),
    "xray":      dict(dy=-1.331, dy_ci=[-2.05, -0.91], dcbcr=-7.808, dcbcr_ci=[-9.47, -6.80],
                      ratio=5.9, drgb=-3.131, xcorr=0.902),
    "documents": dict(dy=-1.909, dy_ci=[-2.61, -1.31], dcbcr=-12.525, dcbcr_ci=[-13.73, -11.23],
                      ratio=6.6, drgb=-6.013, xcorr=0.952),
    "oct":       dict(dy=-3.053, dy_ci=[-3.89, -2.29], dcbcr=-14.128, dcbcr_ci=[-15.34, -12.84],
                      ratio=4.6, drgb=-7.193, xcorr=0.967),
}
REF_XCORR_ORIGINAL = 0.848   # Kodak originals, 24/07
REF_XCORR_GENERIC = 0.849    # generic reconstruction, 24/07

# Full-range (JPEG) matrices. Only the linear part matters for a PSNR of
# differences, so the +128 offsets of Cb/Cr are irrelevant and omitted.
MATRIX = {
    "bt601": [[0.299, 0.587, 0.114],
              [-0.168736, -0.331264, 0.5],
              [0.5, -0.418688, -0.081312]],
    "bt709": [[0.2126, 0.7152, 0.0722],
              [-0.114572, -0.385428, 0.5],
              [0.5, -0.454153, -0.045847]],
}


# ----------------------------------------------------------------- metrics ---
def _psnr_from_mse(mse, max_val=255.0):
    return 20 * math.log10(max_val) - 10 * math.log10(max(mse, 1e-12))


def channel_metrics(org, rec, mat, max_val=255):
    """PSNR of RGB, Y, Cb, Cr and CbCr (joint), plus the reconstruction's mean
    inter-channel correlation. `org`/`rec` are (1,3,H,W) in [0,1]."""
    o = (org * max_val).clamp(0, max_val).round()[0]
    r = (rec * max_val).clamp(0, max_val).round()[0]
    d = (o - r).reshape(3, -1)
    m = torch.tensor(mat, dtype=d.dtype, device=d.device)
    dy_cb_cr = m @ d
    out = {
        "psnr": _psnr_from_mse(d.pow(2).mean().item(), max_val),
        "psnr_y": _psnr_from_mse(dy_cb_cr[0].pow(2).mean().item(), max_val),
        "psnr_cb": _psnr_from_mse(dy_cb_cr[1].pow(2).mean().item(), max_val),
        "psnr_cr": _psnr_from_mse(dy_cb_cr[2].pow(2).mean().item(), max_val),
        "psnr_cbcr": _psnr_from_mse(dy_cb_cr[1:].pow(2).mean().item(), max_val),
        "xcorr": interchannel_corr(r),
    }
    return out


def interchannel_corr(img):
    """Mean Pearson correlation of the three channel pairs of one image."""
    v = img.reshape(3, -1).double()
    v = v - v.mean(dim=1, keepdim=True)
    s = v.norm(dim=1).clamp_min(1e-12)
    c = (v @ v.t()) / (s[:, None] * s[None, :])
    return float((c[0, 1] + c[0, 2] + c[1, 2]).item() / 3.0)


def read_image(fp):
    return transforms.ToTensor()(Image.open(fp).convert("RGB"))


# --------------------------------------------------------------- decompose ---
def build_generic(device):
    """Anchor backbone + the authors' external STanH derivations (one level each),
    exactly as eval/evaluate_xray.py loads them."""
    files = sorted(f for f in os.listdir(os.path.join(ROOT, DERIVATIONS)) if f.endswith(".pth.tar"))
    ck = torch.load(os.path.join(ROOT, ANCHOR), map_location=device, weights_only=False)
    model = WACNNMultiSTanH(N=192, M=320, num_stanh=len(files),
                            factorized_configuration=ck["factorized_configuration"],
                            gaussian_configuration=ck["gaussian_configuration"]).to(device)
    model.eval()
    model.update(device=torch.device(device))
    model.load_state_dict(ck["state_dict"], state_dicts_stanh=None)
    for i, f in enumerate(files):
        sd = torch.load(os.path.join(ROOT, DERIVATIONS, f), map_location=device,
                        weights_only=False)["state_dict"]
        for holder, key in ((model.gaussian_conditional[i], "gaussian_conditional"),
                            (model.entropy_bottleneck[i], "entropy_bottleneck")):
            holder.sos.w = torch.nn.Parameter(sd[key]["w"].to(device))
            holder.sos.b = torch.nn.Parameter(sd[key]["b"].to(device))
            holder.sos.update_state(device)
    model.update(device=torch.device(device))
    return model, files


def build_ckpt(path, device, anchor_cache):
    """One adapted checkpoint: full state_dict, or anchor + delta when is_delta."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = WACNNMultiSTanH(N=192, M=320, num_stanh=1,
                            factorized_configuration=ck["factorized_configuration"],
                            gaussian_configuration=ck["gaussian_configuration"]).to(device)
    model.update(device=torch.device(device))
    if ck.get("is_delta"):
        if anchor_cache["sd"] is None:
            anchor_cache["sd"] = torch.load(os.path.join(ROOT, ANCHOR), map_location=device,
                                            weights_only=False)["state_dict"]
        model.load_state_dict(anchor_cache["sd"], state_dicts_stanh=None)
        merged = model.state_dict()
        merged.update({k: v.to(device=device, dtype=merged[k].dtype)
                       for k, v in ck["delta"].items() if k in merged})
        model.load_state_dict(merged, state_dicts_stanh=None)
    else:
        model.load_state_dict(ck["state_dict"], state_dicts_stanh=None)
    model.update(device=torch.device(device))
    model.eval()
    return model


def run_curve(name, args):
    spec = CURVES[name]
    device = args.device
    mat = MATRIX[args.matrix]
    files = sorted(f for f in glob.glob(os.path.join(ROOT, args.dataset, "*"))
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")))
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    print(f"[{name}] {len(files)} images from {args.dataset} ({args.matrix}, "
          f"{'entropy estimation' if not args.real_coding else 'real coding'})")

    # Inter-channel correlation of the ORIGINALS: the reference the reconstruction
    # is supposed to preserve (a decoder that lost colour drifts towards 1.0).
    xcorr_org = [interchannel_corr((read_image(f) * 255).round()) for f in files]

    if spec["kind"] == "generic":
        model, keys = build_generic(device)
        levels = [(k, i) for i, k in enumerate(keys)]
        key_name = "levels"
    else:
        paths = sorted(glob.glob(os.path.join(ROOT, spec["models_dir"], "*_best.pth.tar")))
        levels = [(os.path.basename(p).replace("_best.pth.tar", ""), p) for p in paths]
        key_name = "lambdas"
        model = None
    anchor_cache = {"sd": None}

    keys, per_image = [], {}
    agg = {k: [] for k in ("bpp", "psnr", "psnr_y", "psnr_cb", "psnr_cr", "psnr_cbcr", "xcorr")}
    for label, ref in levels:
        if spec["kind"] == "generic":
            level_idx = ref
        else:
            model = build_ckpt(ref, device, anchor_cache)
            level_idx = 0
        rows = {k: [] for k in agg}
        for fp in files:
            x = read_image(fp).unsqueeze(0).to(device)
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2 ** 6)
            xp = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            with torch.no_grad():
                if args.real_coding:
                    data = model.compress(xp, stanh_level=level_idx)
                    out = model.decompress(data, stanh_level=level_idx)
                    out["x_hat"] = torch.nn.functional.pad(out["x_hat"], unpad).clamp_(0., 1.)
                    npx = out["x_hat"].size(0) * out["x_hat"].size(2) * out["x_hat"].size(3)
                    bpp = (len(data["strings"][0]) * 8.0) / npx + sum(
                        (len(data["strings"][1][i]) * 8.0) / npx
                        for i in range(len(data["strings"][1])))
                else:
                    out = model(xp, training=False, stanh_level=level_idx)
                    out["x_hat"] = torch.nn.functional.pad(out["x_hat"], unpad).clamp_(0., 1.)
                    npx = x.size(0) * x.size(2) * x.size(3)
                    bpp = sum((torch.log(l).sum() / (-math.log(2) * npx))
                              for l in out["likelihoods"].values()).item()
                met = channel_metrics(x, out["x_hat"], mat)
            if device == "cuda":
                torch.cuda.empty_cache()
            rows["bpp"].append(bpp)
            for k, v in met.items():
                rows[k].append(v)
        keys.append(label)
        per_image[label] = rows
        for k in agg:
            agg[k].append(float(np.mean(rows[k])))
        print(f"  {label:>18} bpp {agg['bpp'][-1]:.4f} | RGB {agg['psnr'][-1]:.3f} | "
              f"Y {agg['psnr_y'][-1]:.3f} | CbCr {agg['psnr_cbcr'][-1]:.3f} | "
              f"xcorr {agg['xcorr'][-1]:.4f}")

    out = {"curve": name, "matrix": args.matrix, "dataset": args.dataset,
           "real_coding": bool(args.real_coding), key_name: keys,
           "files": [os.path.basename(f) for f in files],
           "xcorr_original": float(np.mean(xcorr_org)),
           "per_image_xcorr_original": xcorr_org,
           "per_image": per_image}
    out.update({k: v for k, v in agg.items()})
    path = os.path.join(ROOT, args.out_dir, f"ycbcr_{name}_on_kodak.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=4)
    print(f"  -> {os.path.relpath(path, ROOT)}")
    return out


# ----------------------------------------------------------------- analyze ---
def matched_mean(bpp_ref, met_ref, bpp_test, met_test):
    """Mean matched-bpp delta: adapted minus generic interpolated at the same bpp,
    over the adapted points that fall inside the (Pareto-filtered) generic range.
    Same definition used for the RGB unit elsewhere in the analysis."""
    br, mr = drop_dominated(bpp_ref, met_ref)
    ds = [m - float(np.interp(b, br, mr)) for b, m in zip(bpp_test, met_test)
          if min(br) <= b <= max(br)]
    if not ds:
        return float("nan"), 0
    return float(np.mean(ds)), len(ds)


def stack(js, key):
    ks = js.get("levels") or js.get("lambdas")
    return np.array([js["per_image"][k][key] for k in ks])


def matched_support(bpp_ref, met_ref, bpp_test):
    """Indices of the adapted points that enter the matched delta. The
    inter-channel correlation has to be averaged over THIS support, not over all
    lambdas: outside the generic bpp range there is no delta to explain, and for
    the X-ray decoder the two excluded points are the most colour-collapsed ones
    (0,912 over 8 lambdas vs 0,902 over the 6 matched)."""
    br, mr = drop_dominated(bpp_ref, met_ref)
    return [i for i, b in enumerate(bpp_test) if min(br) <= b <= max(br)]


def analyze(args):
    gen = json.load(open(os.path.join(ROOT, args.out_dir, "ycbcr_generic_on_kodak.json")))
    names = [n for n in ORDER + ["replay"]
             if os.path.exists(os.path.join(ROOT, args.out_dir, f"ycbcr_{n}_on_kodak.json"))]
    cur = {n: json.load(open(os.path.join(ROOT, args.out_dir, f"ycbcr_{n}_on_kodak.json")))
           for n in names}
    channels = ["psnr", "psnr_y", "psnr_cb", "psnr_cr", "psnr_cbcr"]

    for n, js in cur.items():
        if js["files"] != gen["files"]:
            raise SystemExit(f"{n}: image list differs from the generic; pairing undefined.")

    gb = stack(gen, "bpp")
    gm = {c: stack(gen, c) for c in channels}
    tb = {n: stack(js, "bpp") for n, js in cur.items()}
    tm = {n: {c: stack(js, c) for c in channels} for n, js in cur.items()}

    point = {n: {c: matched_mean(gen["bpp"], gen[c], js["bpp"], js[c]) for c in channels}
             for n, js in cur.items()}

    # One paired bootstrap over images for every curve and channel at once, so
    # adjacent gaps are differences within the SAME resample.
    n_img = gb.shape[1]
    rng = np.random.default_rng(args.seed)
    draws = {n: {c: [] for c in channels} for n in names}
    for _ in range(args.boot):
        idx = rng.integers(0, n_img, n_img)
        gb_i = gb[:, idx].mean(1)
        gm_i = {c: gm[c][:, idx].mean(1) for c in channels}
        for n in names:
            tb_i = tb[n][:, idx].mean(1)
            for c in channels:
                draws[n][c].append(matched_mean(gb_i, gm_i[c], tb_i, tm[n][c][:, idx].mean(1))[0])
    draws = {n: {c: np.array(v) for c, v in d.items()} for n, d in draws.items()}

    def ci(vals):
        v = vals[np.isfinite(vals)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size)]

    summary = {"generated_by": "plots/decompose_ycbcr.py analyze", "matrix": gen["matrix"],
               "n_images": n_img, "bootstrap": {"B": args.boot, "seed": args.seed},
               "definition": ("matched-bpp delta = mean over adapted points inside the "
                              "Pareto-filtered generic bpp range of (adapted - generic@bpp); "
                              "negative = damage"),
               "curves": {}, "gaps": {}, "xcorr": {}, "confronto_24jul": {}}

    for n in names:
        d = {}
        for c in channels:
            val, npts = point[n][c]
            lo, hi, nv = ci(draws[n][c])
            d[c] = {"delta": val, "ci95": [lo, hi], "n_points": npts, "n_resamples": nv}
        dy, dc = d["psnr_y"]["delta"], d["psnr_cbcr"]["delta"]
        d["ratio_cbcr_over_y"] = dc / dy if dy else float("nan")
        summary["curves"][n] = d
        sup = matched_support(gen["bpp"], gen["psnr"], cur[n]["bpp"])
        summary["xcorr"][n] = {
            "mean": float(np.mean([cur[n]["xcorr"][i] for i in sup])),
            "mean_all_lambdas": float(np.mean(cur[n]["xcorr"])),
            "support": sup, "per_lambda": cur[n]["xcorr"]}
    summary["xcorr"]["generic"] = {"mean": float(np.mean(gen["xcorr"])),
                                   "mean_all_lambdas": float(np.mean(gen["xcorr"])),
                                   "per_lambda": gen["xcorr"]}
    summary["xcorr"]["original"] = {"mean": gen["xcorr_original"]}

    order = [n for n in ORDER if n in names]
    for a, b in zip(order, order[1:]):
        g = {}
        for c in channels:
            diff = summary["curves"][b][c]["delta"] - summary["curves"][a][c]["delta"]
            dv = draws[b][c] - draws[a][c]
            lo, hi, nv = ci(dv)
            g[c] = {"delta": diff, "ci95": [lo, hi], "n_resamples": nv,
                    "resolved": bool(lo < 0 and hi < 0) or bool(lo > 0 and hi > 0)}
        summary["gaps"][f"{LABEL[b]}-{LABEL[a]}"] = g

    # ---- printing -----------------------------------------------------------
    print("=" * 96)
    print(f"Decoder damage on Kodak, matched-bpp deltas ({gen['matrix']}, n={n_img} images, "
          f"paired bootstrap B={args.boot}, seed {args.seed})")
    print("=" * 96)
    print(f"{'domain':<12}{'dY':>22}{'dCbCr':>24}{'CbCr/Y':>9}{'dRGB':>10}{'pts':>6}{'xcorr':>8}")
    for n in names:
        d = summary["curves"][n]
        y, cb = d["psnr_y"], d["psnr_cbcr"]
        print(f"{LABEL[n]:<12}"
              f"{y['delta']:>+8.3f} [{y['ci95'][0]:+.3f},{y['ci95'][1]:+.3f}]"
              f"{cb['delta']:>+9.3f} [{cb['ci95'][0]:+.3f},{cb['ci95'][1]:+.3f}]"
              f"{d['ratio_cbcr_over_y']:>9.1f}{d['psnr']['delta']:>+10.3f}"
              f"{y['n_points']:>6}{summary['xcorr'][n]['mean']:>8.4f}")
    print(f"\ninter-channel correlation (xcorr column above = matched support): "
          f"originals {summary['xcorr']['original']['mean']:.4f} | "
          f"generic reconstruction {summary['xcorr']['generic']['mean']:.4f}")
    print("  over ALL lambdas instead: " + " | ".join(
        f"{LABEL[n]} {summary['xcorr'][n]['mean_all_lambdas']:.4f}" for n in names))

    print("\nAdjacent gaps (same resample; 'resolved' = 95% CI excludes zero):")
    for k, g in summary["gaps"].items():
        for c, tag in (("psnr_y", "Y"), ("psnr", "RGB")):
            print(f"  {k:<22} {tag:<4}{g[c]['delta']:>+8.3f} "
                  f"[{g[c]['ci95'][0]:+.3f}, {g[c]['ci95'][1]:+.3f}]  "
                  f"{'resolvido' if g[c]['resolved'] else 'NAO RESOLVIDO'}")

    # ---- point-by-point comparison against the reference table ---------------
    print("\n" + "=" * 96)
    print("Confronto ponto a ponto com a tabela de referencia")
    print("=" * 96)
    print(f"{'domain':<12}{'quantity':<10}{'ref':>10}{'medido':>10}{'diff':>10}   status")
    worst = {"dy": 0.0, "dcbcr": 0.0, "drgb": 0.0, "ratio": 0.0, "xcorr": 0.0}
    for n in ORDER:
        if n not in names:
            continue
        ref, got = REF_24JUL[n], summary["curves"][n]
        pairs = [("dY", "dy", got["psnr_y"]["delta"], args.tol),
                 ("dCbCr", "dcbcr", got["psnr_cbcr"]["delta"], args.tol),
                 ("dRGB", "drgb", got["psnr"]["delta"], args.tol),
                 ("CbCr/Y", "ratio", got["ratio_cbcr_over_y"], args.tol_ratio),
                 ("xcorr", "xcorr", summary["xcorr"][n]["mean"], args.tol_xcorr)]
        rows = {}
        for label, key, val, tol in pairs:
            diff = val - ref[key]
            ok = abs(diff) <= tol
            worst[key] = max(worst[key], abs(diff))
            rows[key] = {"ref": ref[key], "medido": val, "diff": diff, "within_tol": bool(ok)}
            print(f"{LABEL[n]:<12}{label:<10}{ref[key]:>10.3f}{val:>10.3f}{diff:>+10.3f}   "
                  f"{'ok' if ok else 'DIVERGE'}")
        summary["confronto_24jul"][n] = rows
    gen_x = summary["xcorr"]["generic"]["mean"]
    org_x = summary["xcorr"]["original"]["mean"]
    summary["confronto_24jul"]["xcorr_original"] = {
        "ref": REF_XCORR_ORIGINAL, "medido": org_x, "diff": org_x - REF_XCORR_ORIGINAL,
        "within_tol": bool(abs(org_x - REF_XCORR_ORIGINAL) <= args.tol_xcorr)}
    summary["confronto_24jul"]["xcorr_generic"] = {
        "ref": REF_XCORR_GENERIC, "medido": gen_x, "diff": gen_x - REF_XCORR_GENERIC,
        "within_tol": bool(abs(gen_x - REF_XCORR_GENERIC) <= args.tol_xcorr)}
    print(f"{'-':<12}{'xcorr org':<10}{REF_XCORR_ORIGINAL:>10.3f}{org_x:>10.3f}"
          f"{org_x - REF_XCORR_ORIGINAL:>+10.3f}   "
          f"{'ok' if abs(org_x - REF_XCORR_ORIGINAL) <= args.tol_xcorr else 'DIVERGE'}")
    print(f"{'-':<12}{'xcorr gen':<10}{REF_XCORR_GENERIC:>10.3f}{gen_x:>10.3f}"
          f"{gen_x - REF_XCORR_GENERIC:>+10.3f}   "
          f"{'ok' if abs(gen_x - REF_XCORR_GENERIC) <= args.tol_xcorr else 'DIVERGE'}")
    summary["confronto_24jul"]["worst_abs_diff"] = worst
    print(f"\nworst |diff|: dY {worst['dy']:.4f} | dCbCr {worst['dcbcr']:.4f} | "
          f"dRGB {worst['drgb']:.4f} dB | CbCr/Y {worst['ratio']:.3f} | xcorr {worst['xcorr']:.4f} "
          f"(tol {args.tol} dB / {args.tol_ratio} / {args.tol_xcorr})")

    # ---- pipeline check: recomputed RGB curve vs the published one -----------
    print("\nCheck de pipeline: PSNR-RGB recomputado vs curva publicada (por ponto)")
    checks = {}
    for n, js in [("generic", gen)] + [(k, cur[k]) for k in names]:
        ref_path = CURVES[n]["rgb_ref"]
        if not os.path.exists(os.path.join(ROOT, ref_path)):
            continue
        pub = json.load(open(os.path.join(ROOT, ref_path)))
        dp = float(np.max(np.abs(np.array(js["psnr"]) - np.array(pub["psnr"]))))
        db = float(np.max(np.abs(np.array(js["bpp"]) - np.array(pub["bpp"]))))
        checks[n] = {"ref": ref_path, "max_abs_dpsnr": dp, "max_abs_dbpp": db}
        print(f"  {LABEL.get(n, n):<22} max|dPSNR| {dp:.2e} dB   max|dbpp| {db:.2e}   "
              f"({os.path.basename(ref_path)})")
    summary["check_rgb_vs_published"] = checks

    path = os.path.join(ROOT, args.out_dir, "ycbcr_decomposition_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary saved to {os.path.relpath(path, ROOT)}")


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decompose", help="run the models on Kodak and store per-channel metrics")
    d.add_argument("--curve", choices=sorted(CURVES), help="single curve (default: --all)")
    d.add_argument("--all", action="store_true", help="all eight curves")
    d.add_argument("--dataset", default="datasets/kodak")
    d.add_argument("--limit", type=int, default=24)
    d.add_argument("--matrix", choices=sorted(MATRIX), default="bt601")
    d.add_argument("--real_coding", action="store_true",
                   help="arithmetic coding instead of the default entropy estimation "
                        "(the published cross curves use entropy estimation)")
    d.add_argument("--out_dir", default="results")
    d.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    a = sub.add_parser("analyze", help="matched-bpp deltas, gaps and the 24/07 confrontation")
    a.add_argument("--out_dir", default="results")
    a.add_argument("--boot", type=int, default=1000)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--tol", type=float, default=0.005, help="dB tolerance vs the 24/07 table")
    a.add_argument("--tol_ratio", type=float, default=0.05)
    a.add_argument("--tol_xcorr", type=float, default=0.001)

    args = ap.parse_args()
    if args.cmd == "decompose":
        todo = sorted(CURVES) if (args.all or not args.curve) else [args.curve]
        for name in todo:
            run_curve(name, args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
