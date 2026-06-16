"""
Evaluate VTM (H.266) on Kodak using the EXACT same pipeline as the paper authors.

Uses the VTM class from compress.utils.standard.codecs which:
1. Converts RGB -> YCbCr 4:4:4 
2. Encodes with VTM --InputChromaFormat=444
3. Decodes
4. Converts YCbCr -> RGB
5. Computes RGB-PSNR (same metric as the neural network)
6. Computes BPP from the actual bitstream file size
"""
import os
import sys
import json
import argparse
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from compress.utils.standard.codecs import VTM


def main():
    parser = argparse.ArgumentParser(description="Evaluate VTM on Kodak (author's pipeline)")
    parser.add_argument("--dataset", default="datasets/kodak", help="Path to Kodak images")
    parser.add_argument("--build_dir", default=os.path.expanduser("~/VVCSoftware_VTM/bin"), help="VTM build dir")
    parser.add_argument("--config", default=os.path.expanduser("~/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg"), help="VTM config")
    parser.add_argument("--out_json", default="results/vtm_kodak_rd.json", help="Output JSON")
    parser.add_argument("--qps", nargs="+", type=int, default=[22, 27, 32, 37, 42, 47], help="QP values")
    args = parser.parse_args()

    # Setup VTM codec using the authors' class
    vtm_args = argparse.Namespace(build_dir=args.build_dir, config=args.config, rgb=False)
    vtm = VTM(vtm_args)
    
    # Verify paths
    print(f"Encoder: {vtm.encoder_path}")
    print(f"Decoder: {vtm.decoder_path}")
    print(f"Config:  {vtm.config_path}")
    assert os.path.exists(vtm.encoder_path), f"Encoder not found: {vtm.encoder_path}"
    assert os.path.exists(vtm.decoder_path), f"Decoder not found: {vtm.decoder_path}"

    # Get all Kodak images
    image_files = sorted(glob.glob(os.path.join(args.dataset, "*.png")))
    print(f"\nEvaluating on {len(image_files)} images from {args.dataset}")
    
    results = {"qp": [], "bpp": [], "psnr": []}
    
    for qp in args.qps:
        print(f"\n--- QP {qp} ---")
        total_bpp = 0.0
        total_psnr = 0.0
        
        for i, img_path in enumerate(image_files):
            # vtm.run() does the full pipeline:
            # RGB -> YCbCr444 -> VTM encode -> decode -> YCbCr -> RGB -> compute psnr-rgb
            info = vtm.run(img_path, qp, metrics=["psnr-rgb"])
            
            total_bpp += info["bpp"]
            total_psnr += info["psnr-rgb"]
            
            print(f"  [{i+1}/{len(image_files)}] {os.path.basename(img_path)}: "
                  f"BPP={info['bpp']:.4f}, PSNR={info['psnr-rgb']:.2f} dB")
        
        avg_bpp = total_bpp / len(image_files)
        avg_psnr = total_psnr / len(image_files)
        
        results["qp"].append(qp)
        results["bpp"].append(avg_bpp)
        results["psnr"].append(avg_psnr)
        
        print(f"  Average -> BPP: {avg_bpp:.4f}, PSNR: {avg_psnr:.2f} dB")
    
    # Save results
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
