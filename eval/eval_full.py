"""
Evaluate FULL fine-tuned models (train/train_xray_full.py) on a test set.

Unlike eval_finetuned.py (which loads one frozen anchor and only swaps the STanH
layer), here every checkpoint is a complete model with its OWN backbone, so we
rebuild the model from the checkpoint's configs and load the full state_dict.
Same metrics/pipeline (padding, entropy-estimation default off) as the others.

  python eval/eval_full.py --models_dir models/xray_full_finetuning \
      --dataset datasets/xrays/test/data --entropy_estimation \
      --out_json results/v5_fullft_on_xray_rd.json
"""
import os, sys, glob, json, argparse, math
import torch
from torchvision import transforms
from PIL import Image
from pytorch_msssim import ms_ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from compressai.ops import compute_padding
from compress.models.cnn_multiStanh import WACNNMultiSTanH


def psnr(a, b, max_val=255):
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(org, rec, max_val=255):
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    return {"psnr": psnr(org, rec).item(), "ms-ssim": ms_ssim(org, rec, data_range=max_val).item()}


def read_image(fp):
    return transforms.ToTensor()(Image.open(fp).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="models/xray_full_finetuning")
    ap.add_argument("--dataset", default="datasets/xrays/test/data")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--out_json", default="results/v5_fullft_on_xray_rd.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--entropy_estimation", action="store_true")
    ap.add_argument("--anchor", default="models/original_paper/STanH/anchor/0728_last_.pth.tar",
                    help="Frozen anchor used to reconstruct delta checkpoints (anchor + delta).")
    args = ap.parse_args()
    device = args.device
    print(f"Using device: {device}")

    # Lazily-loaded shared frozen backbone, so delta checkpoints reuse one copy.
    _anchor_sd = {"sd": None}
    def anchor_state():
        if _anchor_sd["sd"] is None:
            ac = torch.load(args.anchor, map_location=device, weights_only=False)
            _anchor_sd["sd"] = ac["state_dict"]
        return _anchor_sd["sd"]

    EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    image_files = sorted(f for f in glob.glob(os.path.join(args.dataset, "*"))
                         if f.lower().endswith(EXTS))
    if args.limit and args.limit > 0:
        image_files = image_files[:args.limit]
    print(f"Evaluating on {len(image_files)} images from {args.dataset}")

    model_paths = sorted(glob.glob(os.path.join(args.models_dir, "*_best.pth.tar")))
    if not model_paths:
        print(f"No '*_best.pth.tar' in {args.models_dir}."); return
    print(f"Found {len(model_paths)} full models: {[os.path.basename(m) for m in model_paths]}")

    # per_image keeps the raw per-image metrics so BD-Rate uncertainty can be
    # bootstrapped over images downstream (aggregates alone can't be resampled).
    results = {"lambdas": [], "bpp": [], "psnr": [], "ms-ssim": [],
               "files": [os.path.basename(f) for f in image_files], "per_image": {}}
    for ckpt_path in model_paths:
        name = os.path.basename(ckpt_path).replace("_best.pth.tar", "")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "factorized_configuration" not in ck:
            print(f"  SKIP {name}: not a full checkpoint (no configs)."); continue
        model = WACNNMultiSTanH(N=192, M=320, num_stanh=1,
                                factorized_configuration=ck["factorized_configuration"],
                                gaussian_configuration=ck["gaussian_configuration"]).to(device)
        model.update(device=torch.device(device))
        if ck.get("is_delta"):
            # Reconstruct full model = shared frozen anchor + this derivation's small delta.
            model.load_state_dict(anchor_state(), state_dicts_stanh=None)
            merged = model.state_dict()
            merged.update({k: v.to(device=device, dtype=merged[k].dtype)
                           for k, v in ck["delta"].items() if k in merged})
            model.load_state_dict(merged, state_dicts_stanh=None)
            print(f"  [delta] {name}: {len(ck['delta'])} tensors over anchor ({ck.get('mode')})")
        else:
            model.load_state_dict(ck["state_dict"], state_dicts_stanh=None)
        model.update(device=torch.device(device))
        model.eval()

        avg_bpp = avg_psnr = avg_mssim = 0.0
        im_bpp, im_psnr = [], []
        for img_path in image_files:
            x = read_image(img_path).unsqueeze(0).to(device)
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2 ** 6)
            xp = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            with torch.no_grad():
                if not args.entropy_estimation:
                    data = model.compress(xp, stanh_level=0)
                    out = model.decompress(data, stanh_level=0)
                    out["x_hat"] = torch.nn.functional.pad(out["x_hat"], unpad).clamp_(0., 1.)
                    npx = out["x_hat"].size(0) * out["x_hat"].size(2) * out["x_hat"].size(3)
                    bpp = (len(data["strings"][0]) * 8.0) / npx + sum(
                        (len(data["strings"][1][i]) * 8.0) / npx for i in range(len(data["strings"][1])))
                else:
                    out = model(xp, training=False, stanh_level=0)
                    out["x_hat"] = torch.nn.functional.pad(out["x_hat"], unpad).clamp_(0., 1.)
                    npx = x.size(0) * x.size(2) * x.size(3)
                    bpp = sum((torch.log(l).sum() / (-math.log(2) * npx)) for l in out["likelihoods"].values()).item()
                metrics = compute_metrics(x, out["x_hat"], 255)
            if device == "cuda":
                torch.cuda.empty_cache()
            avg_bpp += bpp; avg_psnr += metrics["psnr"]
            avg_mssim += -10 * math.log10(1 - metrics["ms-ssim"])
            im_bpp.append(bpp); im_psnr.append(metrics["psnr"])

        n = len(image_files)
        results["per_image"][name] = {"bpp": im_bpp, "psnr": im_psnr}
        results["lambdas"].append(name)
        results["bpp"].append(avg_bpp / n)
        results["psnr"].append(avg_psnr / n)
        results["ms-ssim"].append(avg_mssim / n)
        print(f"  {name} -> BPP: {avg_bpp/n:.4f}, PSNR: {avg_psnr/n:.2f} dB")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    main()
