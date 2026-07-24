"""
Low-rate penalty of a fine-tuned point against the generic curve.

At low lambda the fine-tuned encoder can land BELOW the generic curve. BD-Rate
hides that (it integrates over the whole overlap), so we read it point by point:
for each (bpp, psnr) of the fine-tuned model, interpolate the generic curve at
the SAME bpp and report the PSNR gap. Negative = the adapter is worse there.

  python plots/lowlam_penalty.py --generic results/xray_stanh_rd.json \
      --json results/g1_xray_encoder_armA_rd.json --label "encoder arm A"
"""
import json, argparse
import numpy as np
from scipy.interpolate import PchipInterpolator


def load(path):
    d = json.load(open(path))
    b, p = np.asarray(d["bpp"], float), np.asarray(d["psnr"], float)
    o = np.argsort(b)
    return b[o], p[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generic", default="results/xray_stanh_rd.json")
    ap.add_argument("--json", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    gb, gp = load(args.generic)
    curve = PchipInterpolator(gb, gp)
    fb, fp = load(args.json)

    print(f"=== {args.label or args.json} ===")
    print(f"{'bpp':>9} {'PSNR':>8} {'generic@bpp':>12} {'penalty(dB)':>12}")
    for b, p in zip(fb, fp):
        if b < gb[0] or b > gb[-1]:
            print(f"{b:9.4f} {p:8.2f} {'out of range':>12} {'-':>12}")
            continue
        g = float(curve(b))
        print(f"{b:9.4f} {p:8.2f} {g:12.2f} {p - g:+12.2f}")


if __name__ == "__main__":
    main()
