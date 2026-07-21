"""
Evaluate X-ray fine-tuned STanH derivations (PIBIC) against any test set.

Loads the frozen paper anchor (backbone) once, then for each fine-tuned
STanH-only checkpoint (saved by train/train_xray_stanh.py as
{epoch, state_dict:{gaussian_conditional:{w,b}, entropy_bottleneck:{w,b}}})
swaps in its STanH layer and evaluates. Same pipeline as evaluate_kodak.py
(padding + real arithmetic coding by default) for apples-to-apples curves.

Run twice to get both PIBIC curves:
  python eval/eval_finetuned.py --dataset datasets/xrays/test/data --out_json results/xray_finetuned_on_xray_rd.json
  python eval/eval_finetuned.py --dataset datasets/kodak          --out_json results/xray_finetuned_on_kodak_rd.json
"""
import os
import sys
import glob
import json
import argparse
import math
import torch
from torchvision import transforms
from PIL import Image
from pytorch_msssim import ms_ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from compressai.ops import compute_padding
from compress.models.cnn_multiStanh import WACNNMultiSTanH


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(org, rec, max_val: int = 255):
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    return {
        "psnr": psnr(org, rec).item(),
        "ms-ssim": ms_ssim(org, rec, data_range=max_val).item(),
    }


def bpp_calculation(out_net, out_enc):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    bpp_1 = (len(out_enc[0]) * 8.0) / num_pixels
    bpp_2 = sum((len(out_enc[1][i]) * 8.0) / num_pixels for i in range(len(out_enc[1])))
    return bpp_1 + bpp_2


def read_image(filepath):
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned STanH derivations")
    parser.add_argument("--models_dir", default="models/xray_stanh_finetuning_new", help="Dir with fine-tuned *_best.pth.tar (STanH-only)")
    parser.add_argument("--anchor", default="models/original_paper/STanH/anchor/0728_last_.pth.tar", help="Frozen anchor checkpoint")
    parser.add_argument("--dataset", default="datasets/xrays/test/data", help="Path to test images")
    parser.add_argument("--limit", type=int, default=24, help="Number of images (0 = all)")
    parser.add_argument("--out_json", default="results/xray_finetuned_on_xray_rd.json", help="Output JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--entropy_estimation", action="store_true", help="Use entropy estimation instead of arithmetic coding")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    image_files = sorted(f for f in glob.glob(os.path.join(args.dataset, "*"))
                         if f.lower().endswith(EXTS))
    if args.limit and args.limit > 0:
        image_files = image_files[:args.limit]
    print(f"Evaluating on {len(image_files)} images from {args.dataset}")

    model_paths = sorted(glob.glob(os.path.join(args.models_dir, "*_best.pth.tar")))
    if not model_paths:
        print(f"No '*_best.pth.tar' found in {args.models_dir}. Train derivations first.")
        return
    print(f"Found {len(model_paths)} fine-tuned models: {[os.path.basename(m) for m in model_paths]}")

    # Load the anchor backbone ONCE; swap only the STanH layer per checkpoint.
    anchor_ckpt = torch.load(args.anchor, map_location=device, weights_only=False)
    model = WACNNMultiSTanH(
        N=192, M=320, num_stanh=1,
        factorized_configuration=anchor_ckpt["factorized_configuration"],
        gaussian_configuration=anchor_ckpt["gaussian_configuration"],
    ).to(device)
    model.update(device=torch.device(device))
    model.load_state_dict(anchor_ckpt["state_dict"], state_dicts_stanh=None)
    model.eval()

    # per_image enables bootstrapped BD-Rate confidence intervals downstream
    results = {"lambdas": [], "bpp": [], "psnr": [], "ms-ssim": [],
               "files": [os.path.basename(f) for f in image_files], "per_image": {}}
    seen_w = []

    for ckpt_path in model_paths:
        name = os.path.basename(ckpt_path).replace("_best.pth.tar", "")
        finetuned = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = finetuned["state_dict"]
        if "gaussian_conditional" not in sd or "w" not in sd["gaussian_conditional"]:
            print(f"  SKIP {name}: not a STanH-only checkpoint "
                  f"(expected nested gaussian_conditional/{{w,b}}, got keys {list(sd.keys())[:4]}).")
            continue

        model.upload_stanh_values(sd, index=0)
        model.update(device=torch.device(device))
        seen_w.append(round(sd["gaussian_conditional"]["w"].detach().float().mean().item(), 6))

        avg_bpp, avg_psnr, avg_mssim = 0, 0, 0
        im_bpp, im_psnr = [], []
        for img_path in image_files:
            x = read_image(img_path).unsqueeze(0).to(device)
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2 ** 6)
            x_padded = torch.nn.functional.pad(x, pad, mode="constant", value=0)

            with torch.no_grad():
                if not args.entropy_estimation:
                    data = model.compress(x_padded, stanh_level=0)
                    out_dec = model.decompress(data, stanh_level=0)
                    out_dec["x_hat"] = torch.nn.functional.pad(out_dec["x_hat"], unpad).clamp_(0., 1.)
                    bpp = bpp_calculation(out_dec, data["strings"])
                else:
                    out_dec = model(x_padded, training=False, stanh_level=0)
                    out_dec["x_hat"] = torch.nn.functional.pad(out_dec["x_hat"], unpad).clamp_(0., 1.)
                    num_pixels = x.size(0) * x.size(2) * x.size(3)
                    bpp = sum((torch.log(l).sum() / (-math.log(2) * num_pixels)) for l in out_dec["likelihoods"].values()).item()

                metrics = compute_metrics(x, out_dec["x_hat"], 255)

            if args.device == "cuda":
                torch.cuda.empty_cache()
            avg_bpp += bpp
            avg_psnr += metrics["psnr"]
            avg_mssim += -10 * math.log10(1 - metrics["ms-ssim"])
            im_bpp.append(bpp)
            im_psnr.append(metrics["psnr"])

        n = len(image_files)
        avg_bpp, avg_psnr, avg_mssim = avg_bpp / n, avg_psnr / n, avg_mssim / n
        results["per_image"][name] = {"bpp": im_bpp, "psnr": im_psnr}
        results["lambdas"].append(name)
        results["bpp"].append(avg_bpp)
        results["psnr"].append(avg_psnr)
        results["ms-ssim"].append(avg_mssim)
        print(f"  {name} -> BPP: {avg_bpp:.4f}, PSNR: {avg_psnr:.2f} dB")

    if len(seen_w) > 1 and len(set(seen_w)) == 1:
        print("WARNING: every checkpoint has identical STanH weights — the fine-tuned "
              "derivations are not distinct (check training output dir).")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    main()
