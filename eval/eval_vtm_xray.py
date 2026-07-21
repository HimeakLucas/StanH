import os
import sys
import json
import argparse
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from compress.utils.standard.codecs import VTM

def main():
    parser = argparse.ArgumentParser(description="Evaluate VTM on X-ray test dataset")
    parser.add_argument("--dataset", default="datasets/xrays/test/data", help="Path to X-ray images")
    parser.add_argument("--limit", type=int, default=24, help="Number of images to evaluate")
    parser.add_argument("--build_dir", default=os.path.expanduser("~/VVCSoftware_VTM/bin"), help="VTM build dir")
    parser.add_argument("--config", default=os.path.expanduser("~/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg"), help="VTM config")
    parser.add_argument("--out_json", default="results/xray_vtm_rd.json", help="Output JSON")
    parser.add_argument("--qps", nargs="+", type=int, default=[22, 27, 32, 37, 42, 47], help="QP values")
    args = parser.parse_args()

    vtm_args = argparse.Namespace(build_dir=args.build_dir, config=args.config, rgb=False)
    vtm = VTM(vtm_args)
    
    EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    image_files = sorted(f for f in glob.glob(os.path.join(args.dataset, "*"))
                         if f.lower().endswith(EXTS))
    if args.limit and args.limit > 0:
        image_files = image_files[:args.limit]
    print(f"\nEvaluating on {len(image_files)} images from {args.dataset}")
    
    # per_image enables bootstrapped BD-Rate confidence intervals downstream
    results = {"qp": [], "bpp": [], "psnr": [],
               "files": [os.path.basename(f) for f in image_files], "per_image": {}}

    for qp in args.qps:
        print(f"\n--- QP {qp} ---")
        total_bpp = 0.0
        total_psnr = 0.0
        im_bpp, im_psnr = [], []

        for i, img_path in enumerate(image_files):
            # RGB -> YCbCr444 -> VTM -> YCbCr444 -> RGB -> RGB-PSNR
            info = vtm.run(img_path, qp, metrics=["psnr-rgb"])

            total_bpp += info["bpp"]
            total_psnr += info["psnr-rgb"]
            im_bpp.append(info["bpp"])
            im_psnr.append(info["psnr-rgb"])

            print(f"  [{i+1}/{len(image_files)}] {os.path.basename(img_path)}: "
                  f"BPP={info['bpp']:.4f}, PSNR={info['psnr-rgb']:.2f} dB")

        avg_bpp = total_bpp / len(image_files)
        avg_psnr = total_psnr / len(image_files)

        results["per_image"][str(qp)] = {"bpp": im_bpp, "psnr": im_psnr}
        results["qp"].append(qp)
        results["bpp"].append(avg_bpp)
        results["psnr"].append(avg_psnr)
        
        print(f"  Average -> BPP: {avg_bpp:.4f}, PSNR: {avg_psnr:.2f} dB")
    
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {args.out_json}")

if __name__ == "__main__":
    main()
