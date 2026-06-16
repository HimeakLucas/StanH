"""
Evaluate Zou22 + STanH (paper anchor + derivations) on an X-ray test set.

Uses the SAME pipeline as eval/evaluate_kodak.py so the X-ray and Kodak curves
are apples-to-apples with each other and with the VTM baseline:
  - loads the 300MB anchor (backbone) and then overwrites the STanH layers
    (sos.w / sos.b) of each level with the external derivation weights;
  - pads images to a multiple of 64 (NOT resize, which would resample content);
  - real arithmetic coding by default (matches VTM / the paper), with an
    --entropy_estimation flag for a faster idealized-bpp estimate.

X-ray PNGs are grayscale; .convert("RGB") replicates the channel so the
RGB-trained anchor sees the expected 3-channel input. That domain gap is
exactly what the PIBIC project measures.
"""
import torch
import os
import json
import math
import sys
import argparse
import glob
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_msssim import ms_ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from compressai.ops import compute_padding
from compress.models.cnn_multiStanh import WACNNMultiSTanH


def bpp_calculation(out_net, out_enc):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    bpp_1 = (len(out_enc[0]) * 8.0) / num_pixels
    bpp_2 = sum((len(out_enc[1][i]) * 8.0) / num_pixels for i in range(len(out_enc[1])))
    return bpp_1 + bpp_2


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(org, rec, max_val: int = 255):
    metrics = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def read_image(filepath):
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Zou22+STanH on an X-ray test set")
    parser.add_argument("--model", default="models/original_paper/STanH/anchor/0728_last_.pth.tar", help="Path to anchor model")
    parser.add_argument("--stanh_dir", default="models/original_paper/STanH/derivations", help="Path to derivation weights")
    parser.add_argument("--dataset", default="datasets/xrays/test/data", help="Path to X-ray test images")
    parser.add_argument("--limit", type=int, default=24, help="Number of images to evaluate (0 = all)")
    parser.add_argument("--out_json", default="results/xray_stanh_rd.json", help="Output JSON results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--entropy_estimation", action="store_true", help="Use entropy estimation (fast) instead of arithmetic coding")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    print(f"Using device: {args.device}")

    # Discover STanH derivations (same convention as evaluate_kodak.py)
    stanh_files = sorted([f for f in os.listdir(args.stanh_dir) if f.endswith('.pth.tar')])
    stanh_paths = [os.path.join(args.stanh_dir, f) for f in stanh_files]
    print(f"Found {len(stanh_paths)} STanH derivation levels: {stanh_files}")

    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)

    # Initialize multi-level model with the anchor's quantizer configuration
    model = WACNNMultiSTanH(
        N=192,
        M=320,
        num_stanh=len(stanh_paths),
        factorized_configuration=checkpoint["factorized_configuration"],
        gaussian_configuration=checkpoint["gaussian_configuration"],
    )
    model = model.to(args.device)
    model.eval()

    # Update buffers before loading state dict to prevent size mismatches
    model.update(device=torch.device(args.device))

    # Load anchor weights (full backbone)
    model.load_state_dict(checkpoint["state_dict"], state_dicts_stanh=None)

    # Overwrite each level's STanH layer with the external derivation weights
    print("Loading external STanH derivations...")
    loaded_w = []
    for i, sc in enumerate(stanh_paths):
        stanhs = torch.load(sc, map_location=args.device, weights_only=False)
        gw = stanhs["state_dict"]["gaussian_conditional"]["w"].to(args.device)
        gb = stanhs["state_dict"]["gaussian_conditional"]["b"].to(args.device)
        ew = stanhs["state_dict"]["entropy_bottleneck"]["w"].to(args.device)
        eb = stanhs["state_dict"]["entropy_bottleneck"]["b"].to(args.device)

        model.gaussian_conditional[i].sos.w = torch.nn.Parameter(gw)
        model.gaussian_conditional[i].sos.b = torch.nn.Parameter(gb)
        model.gaussian_conditional[i].sos.update_state(args.device)
        model.entropy_bottleneck[i].sos.w = torch.nn.Parameter(ew)
        model.entropy_bottleneck[i].sos.b = torch.nn.Parameter(eb)
        model.entropy_bottleneck[i].sos.update_state(args.device)
        loaded_w.append(gw.detach().float().mean().item())

    model.update(device=torch.device(args.device))

    # Sanity check: derivations must actually differ across levels
    if len(set(round(v, 6) for v in loaded_w)) == 1:
        print("WARNING: all derivations produced identical STanH weights — "
              "loading is likely broken (results would be meaningless).")

    image_files = sorted(glob.glob(os.path.join(args.dataset, "*.png")))
    if args.limit and args.limit > 0:
        image_files = image_files[:args.limit]
    print(f"Evaluating on {len(image_files)} images from {args.dataset}")
    mode = "entropy estimation" if args.entropy_estimation else "real arithmetic coding"
    print(f"Mode: {mode}")

    results_bpp, results_psnr, results_mssim = [], [], []

    for level_idx in range(len(stanh_paths)):
        print(f"\n--- Testing Level {level_idx} ({stanh_files[level_idx]}) ---")
        avg_bpp, avg_psnr, avg_mssim = 0, 0, 0

        for idx, img_path in enumerate(image_files):
            x = read_image(img_path).unsqueeze(0).to(args.device)
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2 ** 6)
            x_padded = torch.nn.functional.pad(x, pad, mode="constant", value=0)

            with torch.no_grad():
                if not args.entropy_estimation:
                    data = model.compress(x_padded, stanh_level=level_idx)
                    out_dec = model.decompress(data, stanh_level=level_idx)
                    out_dec["x_hat"] = torch.nn.functional.pad(out_dec["x_hat"], unpad)
                    out_dec["x_hat"].clamp_(0., 1.)
                    metrics = compute_metrics(x, out_dec["x_hat"], 255)
                    bpp = bpp_calculation(out_dec, data["strings"])
                else:
                    out_dec = model(x_padded, training=False, stanh_level=level_idx)
                    out_dec["x_hat"] = torch.nn.functional.pad(out_dec["x_hat"], unpad)
                    out_dec["x_hat"].clamp_(0., 1.)
                    num_pixels = x.size(0) * x.size(2) * x.size(3)
                    bpp = sum((torch.log(l).sum() / (-math.log(2) * num_pixels)) for l in out_dec["likelihoods"].values()).item()
                    metrics = compute_metrics(x, out_dec["x_hat"], 255)

            if args.device == "cuda":
                torch.cuda.empty_cache()

            sys.stdout.write(f"\r  Img {idx+1}/{len(image_files)}: BPP={bpp:.3f}, PSNR={metrics['psnr']:.2f}")
            sys.stdout.flush()

            avg_bpp += bpp
            avg_psnr += metrics["psnr"]
            avg_mssim += -10 * math.log10(1 - metrics["ms-ssim"])

        n = len(image_files)
        avg_bpp, avg_psnr, avg_mssim = avg_bpp / n, avg_psnr / n, avg_mssim / n
        print(f"\n  Average -> BPP: {avg_bpp:.4f}, PSNR: {avg_psnr:.3f} dB")
        results_bpp.append(avg_bpp)
        results_psnr.append(avg_psnr)
        results_mssim.append(avg_mssim)

    out_data = {
        "levels": stanh_files,
        "bpp": results_bpp,
        "psnr": results_psnr,
        "ms-ssim": results_mssim,
    }
    with open(args.out_json, "w") as f:
        json.dump(out_data, f, indent=4)
    print(f"\nSaved results to {args.out_json}")


if __name__ == "__main__":
    main()
