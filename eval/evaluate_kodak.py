import torch 
import os 
from torchvision import transforms
from PIL import Image
import json
import math
import numpy as np
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from compressai.ops import compute_padding
from pytorch_msssim import ms_ssim
from compress.models.cnn_multiStanh import WACNNMultiSTanH
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("tab10")

torch.backends.cudnn.benchmark = True

def bpp_calculation(out_net, out_enc):
    size = out_net['x_hat'].size() 
    num_pixels = size[0] * size[2] * size[3]
    bpp_1 = (len(out_enc[0]) * 8.0 ) / num_pixels
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
    parser = argparse.ArgumentParser(description="Evaluate STanH on Kodak")
    parser.add_argument("--model", default="models/original_paper/STanH/anchor/0728_last_.pth.tar", help="Path to anchor model")
    parser.add_argument("--stanh_dir", default="models/original_paper/STanH/derivations", help="Path to derivation weights")
    parser.add_argument("--dataset", default="datasets/kodak", help="Path to Kodak images")
    parser.add_argument("--out_json", default="results/kodak_rd.json", help="Output JSON results")
    parser.add_argument("--out_plot", default="results/plots/kodak_rd.png", help="Output RD curve plot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--entropy_estimation", action="store_true", help="Use entropy estimation (fast) instead of arithmetic coding")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)

    print(f"Using device: {args.device}")

    # Discover STanH derivations
    stanh_files = sorted([f for f in os.listdir(args.stanh_dir) if f.endswith('.pth.tar')])
    stanh_paths = [os.path.join(args.stanh_dir, f) for f in stanh_files]
    print(f"Found {len(stanh_paths)} STanH derivation levels: {stanh_files}")

    checkpoint = torch.load(args.model, map_location=args.device)
    
    # Initialize multi-level model
    model = WACNNMultiSTanH(
        N=192, 
        M=320, 
        num_stanh=len(stanh_paths),
        factorized_configuration=checkpoint["factorized_configuration"], 
        gaussian_configuration=checkpoint["gaussian_configuration"]
    )
    model = model.to(args.device)
    model.eval()
    
    # Update buffers before loading state dict to prevent size mismatches
    model.update(device=torch.device(args.device))

    # Load anchor weights (base network + pre-populated levels)
    model.load_state_dict(checkpoint["state_dict"], state_dicts_stanh=None)

    # Overwrite the STanH levels with external derivations explicitly
    print("Loading external STanH derivations...")
    for i, sc in enumerate(stanh_paths):
        stanhs = torch.load(sc, map_location=args.device)
        model.gaussian_conditional[i].sos.w = torch.nn.Parameter(stanhs["state_dict"]["gaussian_conditional"]["w"].to(args.device))
        model.gaussian_conditional[i].sos.b = torch.nn.Parameter(stanhs["state_dict"]["gaussian_conditional"]["b"].to(args.device))
        model.gaussian_conditional[i].sos.update_state(args.device)

        model.entropy_bottleneck[i].sos.w = torch.nn.Parameter(stanhs["state_dict"]["entropy_bottleneck"]["w"].to(args.device))
        model.entropy_bottleneck[i].sos.b = torch.nn.Parameter(stanhs["state_dict"]["entropy_bottleneck"]["b"].to(args.device))
        model.entropy_bottleneck[i].sos.update_state(args.device)
        
    model.update(device=torch.device(args.device))

    image_files = sorted([os.path.join(args.dataset, f) for f in os.listdir(args.dataset) if f.endswith('.png')])
    print(f"Evaluating on {len(image_files)} images from {args.dataset}")

    results_bpp = []
    results_psnr = []
    results_mssim = []

    for level_idx in range(len(stanh_paths)):
        print(f"\n--- Testing Level {level_idx} ({stanh_files[level_idx]}) ---")
        avg_bpp, avg_psnr, avg_mssim = 0, 0, 0
        
        for idx, img_path in enumerate(image_files):
            x = read_image(img_path).unsqueeze(0).to(args.device)
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)
            x_padded = torch.nn.functional.pad(x, pad, mode="constant", value=0)

            if not args.entropy_estimation:
                # Real compression mode
                data = model.compress(x_padded, stanh_level=level_idx)
                out_dec = model.decompress(data, stanh_level=level_idx)
                
                out_dec["x_hat"] = torch.nn.functional.pad(out_dec["x_hat"], unpad)
                out_dec["x_hat"].clamp_(0., 1.)
                
                metrics = compute_metrics(x, out_dec["x_hat"], 255)
                bpp = bpp_calculation(out_dec, data["strings"])
            else:
                # Entropy estimation mode (faster)
                with torch.no_grad():
                    out_dec = model(x_padded, training=False, stanh_level=level_idx)
                    
                out_dec["x_hat"] = torch.nn.functional.pad(out_dec["x_hat"], unpad)
                out_dec["x_hat"].clamp_(0., 1.)
                
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_dec["likelihoods"].values())
                bpp = bpp.item()
                metrics = compute_metrics(x, out_dec["x_hat"], 255)

            sys.stdout.write(f"\r  Img {idx+1}/{len(image_files)}: BPP={bpp:.3f}, PSNR={metrics['psnr']:.2f}")
            sys.stdout.flush()
            
            avg_bpp += bpp
            avg_psnr += metrics["psnr"]
            avg_mssim += -10 * math.log10(1 - metrics["ms-ssim"])
            
        avg_bpp /= len(image_files)
        avg_psnr /= len(image_files)
        avg_mssim /= len(image_files)
        
        print(f"\n  Average -> BPP: {avg_bpp:.3f}, PSNR: {avg_psnr:.3f} dB")
        results_bpp.append(avg_bpp)
        results_psnr.append(avg_psnr)
        results_mssim.append(avg_mssim)

    # Save to JSON
    out_data = {
        "levels": stanh_files,
        "bpp": results_bpp,
        "psnr": results_psnr,
        "ms-ssim": results_mssim
    }
    with open(args.out_json, 'w') as f:
        json.dump(out_data, f, indent=4)
        
    print(f"\nSaved results to {args.out_json}")

    # Plot
    plt.figure(figsize=(10, 7))
    plt.plot(results_bpp, results_psnr, 'o-', color=palette[0], markersize=8, linewidth=2, label="Zou22 + STanH")
    
    # Annotate points
    for i, txt in enumerate(stanh_files):
        plt.annotate(txt.split('-')[0], (results_bpp[i], results_psnr[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Bit-rate [bpp]', fontsize=14)
    plt.ylabel('PSNR [dB]', fontsize=14)
    plt.title('Rate-Distortion Performance on Kodak Dataset', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig(args.out_plot, bbox_inches='tight')
    print(f"Saved plot to {args.out_plot}")

if __name__ == '__main__':
    main()
