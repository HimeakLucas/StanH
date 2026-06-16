"""
Generic RD analysis for a STanH fine-tuning round.

Compares a fine-tuned set of derivations against the authors' GENERIC derivations
(and VTM on X-ray), on the target domain (X-ray) and cross-domain (Kodak):
  - BD-Rate (Bjontegaard, PSNR), negative = fewer bits at equal quality (better);
  - matched-bpp PSNR deltas;
  - a comparison plot with the anchor (native STanH = D10) marked.

Usage:
  python plots/analyze_finetuned.py --tag v4 \
      --xray_json results/v4_finetuned_on_xray_rd.json \
      --kodak_json results/v4_finetuned_on_kodak_rd.json
"""
import json, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


def drop_dominated(bpp, psnr):
    """Keep only Pareto-efficient, monotonic-increasing points (sorted by bpp)."""
    pts = sorted(zip(bpp, psnr))
    keep, best = [], -1e9
    for b, p in pts:
        if p > best:
            keep.append((b, p)); best = p
    return [x[0] for x in keep], [x[1] for x in keep]


def bd_rate(bpp_ref, psnr_ref, bpp_test, psnr_test):
    """BD-Rate (%) of test vs ref over the overlapping PSNR range."""
    br, pr = drop_dominated(bpp_ref, psnr_ref)
    bt, pt = drop_dominated(bpp_test, psnr_test)
    # Adaptive degree: a cubic needs >=4 points; with few points (e.g. the 3-point
    # full-FT curve) drop to a lower degree to avoid an ill-conditioned fit.
    cr = np.polyfit(pr, np.log(br), min(3, len(pr) - 1))
    ct = np.polyfit(pt, np.log(bt), min(3, len(pt) - 1))
    lo, hi = max(min(pr), min(pt)), min(max(pr), max(pt))
    if hi <= lo:
        return float("nan"), (lo, hi)
    x = np.linspace(lo, hi, 200)
    diff = (np.trapz(np.polyval(ct, x), x) - np.trapz(np.polyval(cr, x), x)) / (hi - lo)
    return (np.exp(diff) - 1.0) * 100.0, (lo, hi)


def matched_deltas(bpp_ref, psnr_ref, bpp_test, psnr_test):
    br, pr = drop_dominated(bpp_ref, psnr_ref)
    out = []
    for b, p in zip(bpp_test, psnr_test):
        pr_i = np.interp(b, br, pr) if min(br) <= b <= max(br) else None
        out.append((b, p, pr_i, (p - pr_i) if pr_i is not None else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--xray_json", required=True)
    ap.add_argument("--kodak_json", required=True)
    ap.add_argument("--label", default=None, help="Curve label (default derived from tag)")
    ap.add_argument("--drop_lambdas", default="", help="comma-sep lambda names to omit from the PLOTTED curve (e.g. the anchor-rate anomaly)")
    args = ap.parse_args()
    drop = {s.strip() for s in args.drop_lambdas.split(",") if s.strip()}
    label = args.label or f"STanH fine-tuned {args.tag} (X-ray)"

    gen_x = load("results/xray_stanh_rd.json")
    gen_k = load("results/kodak_rd.json")
    vtm_x = load("results/xray_vtm_rd.json")
    vtm_k = load("results/vtm_kodak_rd.json")
    ft_x = load(args.xray_json)
    ft_k = load(args.kodak_json)

    bd_x, rng_x = bd_rate(gen_x["bpp"], gen_x["psnr"], ft_x["bpp"], ft_x["psnr"])
    bd_k, rng_k = bd_rate(gen_k["bpp"], gen_k["psnr"], ft_k["bpp"], ft_k["psnr"])

    print("=" * 66)
    print(f"[{args.tag}] BD-Rate vs GENERIC derivations  (negative = better)")
    print(f"  X-ray (target) : {bd_x:+.2f}%   over PSNR [{rng_x[0]:.2f}, {rng_x[1]:.2f}] dB")
    print(f"  Kodak (cross)  : {bd_k:+.2f}%   over PSNR [{rng_k[0]:.2f}, {rng_k[1]:.2f}] dB")
    print("=" * 66)
    print("\nMatched-bpp PSNR delta on X-RAY (fine-tuned - generic@same bpp):")
    for b, p, pr, dl in matched_deltas(gen_x["bpp"], gen_x["psnr"], ft_x["bpp"], ft_x["psnr"]):
        s = f"{dl:+.3f} dB" if dl is not None else "(out of generic range)"
        ref = f"{pr:.2f}" if pr is not None else " n/a"
        print(f"  bpp {b:.4f}  ft {p:.2f} dB | generic {ref} -> {s}")

    # anchor (native STanH = D10) operating point on X-ray
    d10 = gen_x["levels"].index("D10-A040.pth.tar")
    a_bpp, a_psnr = gen_x["bpp"][d10], gen_x["psnr"][d10]

    def plot_pts(res):  # filter out dropped lambdas from the plotted curve
        keep = [i for i, l in enumerate(res["lambdas"]) if l not in drop]
        return [res["bpp"][i] for i in keep], [res["psnr"][i] for i in keep]
    px_bpp, px_psnr = plot_pts(ft_x)
    pk_bpp, pk_psnr = plot_pts(ft_k)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(gen_x["bpp"], gen_x["psnr"], "o-", color="tab:blue", label="STanH generic (authors)")
    ax[0].plot(px_bpp, px_psnr, "s-", color="tab:red", label=label)
    ax[0].plot(vtm_x["bpp"], vtm_x["psnr"], "^--", color="tab:green", label="VTM (H.266)")
    ax[0].scatter([a_bpp], [a_psnr], marker="*", s=320, color="black", zorder=5,
                  label="Anchor (native STanH = D10)")
    ax[0].set_title(f"X-ray (target domain)\nBD-Rate {args.tag} vs generic: {bd_x:+.2f}%")
    ax[0].set_xlabel("bpp"); ax[0].set_ylabel("PSNR (dB)")
    ax[0].set_xlim(left=0); ax[0].grid(True, alpha=0.3); ax[0].legend()  # autoscale right to fit full curve

    ax[1].plot(gen_k["bpp"], gen_k["psnr"], "o-", color="tab:blue", label="STanH generic (authors)")
    ax[1].plot(pk_bpp, pk_psnr, "s-", color="tab:red", label=label)
    ax[1].plot(vtm_k["bpp"], vtm_k["psnr"], "^--", color="tab:green", label="VTM (H.266)")
    ax[1].set_title(f"Kodak (cross-domain)\nBD-Rate {args.tag} vs generic: {bd_k:+.2f}%")
    ax[1].set_xlabel("bpp"); ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_xlim(left=0); ax[1].grid(True, alpha=0.3); ax[1].legend()

    fig.suptitle(f"STanH domain adaptation ({args.tag}): fine-tuned vs generic", fontsize=13)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "plots", f"{args.tag}_rd_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
