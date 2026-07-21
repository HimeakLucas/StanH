"""
Domain-agnostic RD analysis for a STanH fine-tuning round.

Compares a fine-tuned set of derivations against the authors' GENERIC derivations
(and, if available, VTM), on a TARGET domain and a CROSS domain:
  - BD-Rate (Bjontegaard, PSNR), negative = fewer bits at equal quality (better);
  - matched-bpp PSNR deltas;
  - a comparison plot, with the anchor (native STanH) marked if present.

Defaults reproduce the original X-ray vs Kodak setup; override the baselines/names
for any other domain (e.g. documents, retina). VTM curves are optional.

Examples:
  # X-ray (defaults)
  python plots/analyze_finetuned.py --tag v8 \
      --target_json results/v8_encoder_on_xray_rd.json \
      --cross_json  results/v8_encoder_on_kodak_rd.json

  # documents (custom target baseline; no VTM yet)
  python plots/analyze_finetuned.py --tag docs_encoder --target_name Documents \
      --target_json results/docs_encoder_on_documents_rd.json \
      --cross_json  results/docs_encoder_on_kodak_rd.json \
      --target_baseline results/documents_generic_rd.json \
      --target_vtm "" --cross_vtm ""
"""
import json, os, argparse
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


def load_opt(name):
    """Load a JSON if the path is non-empty and exists, else None (optional curves)."""
    if not name:
        return None
    p = os.path.join(ROOT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def drop_dominated(bpp, psnr):
    """Keep only Pareto-efficient, monotonic-increasing points (sorted by bpp)."""
    pts = sorted(zip(bpp, psnr))
    keep, best = [], -1e9
    for b, p in pts:
        if p > best:
            keep.append((b, p)); best = p
    return [x[0] for x in keep], [x[1] for x in keep]


def bd_rate(bpp_ref, psnr_ref, bpp_test, psnr_test):
    """BD-Rate (%) of test vs ref over the overlapping PSNR range.

    Piecewise-cubic monotone interpolation (PCHIP, as in JVET practice) instead of
    a single global cubic polyfit: the global fit oscillates with 7-8 points and
    was measured to swing the aggregate by several percent.
    """
    br, pr = drop_dominated(bpp_ref, psnr_ref)
    bt, pt = drop_dominated(bpp_test, psnr_test)
    lo, hi = max(min(pr), min(pt)), min(max(pt), max(pr))
    if hi <= lo or len(pr) < 2 or len(pt) < 2:
        return float("nan"), (lo, hi)
    fr = PchipInterpolator(pr, np.log(br))
    ft = PchipInterpolator(pt, np.log(bt))
    x = np.linspace(lo, hi, 200)
    diff = np.trapz(ft(x) - fr(x), x) / (hi - lo)
    return (np.exp(diff) - 1.0) * 100.0, (lo, hi)


def bootstrap_bd(ref, test, B=1000, seed=42):
    """95% CI for BD-Rate by paired bootstrap over images.

    Requires per-image metrics in both JSONs, computed on the SAME image set
    (paired resampling is undefined otherwise). Returns (lo, hi, n_valid) or None.
    """
    if "per_image" not in ref or "per_image" not in test:
        return None
    if "files" in ref and "files" in test and ref["files"] != test["files"]:
        return None
    rkeys = ref.get("levels") or ref.get("lambdas")
    tkeys = test.get("levels") or test.get("lambdas")
    rb = np.array([ref["per_image"][k]["bpp"] for k in rkeys])
    rp = np.array([ref["per_image"][k]["psnr"] for k in rkeys])
    tb = np.array([test["per_image"][k]["bpp"] for k in tkeys])
    tp = np.array([test["per_image"][k]["psnr"] for k in tkeys])
    if rb.shape[1] != tb.shape[1]:
        return None
    n = rb.shape[1]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        bd, _ = bd_rate(rb[:, idx].mean(1), rp[:, idx].mean(1),
                        tb[:, idx].mean(1), tp[:, idx].mean(1))
        if np.isfinite(bd):
            vals.append(bd)
    if len(vals) < B // 2:  # mostly-degenerate resamples -> CI meaningless
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


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
    # fine-tuned curves on the two domains (accept legacy --xray_json/--kodak_json aliases)
    ap.add_argument("--target_json", "--xray_json", dest="target_json", required=True)
    ap.add_argument("--cross_json", "--kodak_json", dest="cross_json", required=True)
    # baselines (defaults reproduce X-ray vs Kodak)
    ap.add_argument("--target_baseline", default="results/xray_stanh_rd.json")
    ap.add_argument("--cross_baseline", default="results/kodak_rd.json")
    ap.add_argument("--target_vtm", default="results/xray_vtm_rd.json", help="'' to disable")
    ap.add_argument("--cross_vtm", default="results/vtm_kodak_rd.json", help="'' to disable")
    ap.add_argument("--target_name", default="X-ray")
    ap.add_argument("--cross_name", default="Kodak")
    ap.add_argument("--anchor_level", default="D10-A040.pth.tar", help="native-STanH level to star, '' to skip")
    ap.add_argument("--label", default=None, help="Curve label (default derived from tag)")
    ap.add_argument("--drop_lambdas", default="", help="comma-sep lambda names to omit from the PLOTTED curve")
    ap.add_argument("--out", default=None, help="output PNG (default results/plots/<tag>_rd_comparison.png)")
    args = ap.parse_args()
    drop = {s.strip() for s in args.drop_lambdas.split(",") if s.strip()}
    label = args.label or f"STanH fine-tuned {args.tag} ({args.target_name})"

    gen_t = load(args.target_baseline)
    gen_c = load(args.cross_baseline)
    vtm_t = load_opt(args.target_vtm)
    vtm_c = load_opt(args.cross_vtm)
    ft_t = load(args.target_json)
    ft_c = load(args.cross_json)

    bd_t, rng_t = bd_rate(gen_t["bpp"], gen_t["psnr"], ft_t["bpp"], ft_t["psnr"])
    bd_c, rng_c = bd_rate(gen_c["bpp"], gen_c["psnr"], ft_c["bpp"], ft_c["psnr"])
    ci_t = bootstrap_bd(gen_t, ft_t)
    ci_c = bootstrap_bd(gen_c, ft_c)

    def ci_str(ci):
        return f"   95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] ({ci[2]} resamples)" if ci else ""
    print("=" * 66)
    print(f"[{args.tag}] BD-Rate vs GENERIC derivations  (negative = better)")
    print(f"  {args.target_name} (target): {bd_t:+.2f}%   over PSNR [{rng_t[0]:.2f}, {rng_t[1]:.2f}] dB{ci_str(ci_t)}")
    print(f"  {args.cross_name} (cross) : {bd_c:+.2f}%   over PSNR [{rng_c[0]:.2f}, {rng_c[1]:.2f}] dB{ci_str(ci_c)}")
    print("=" * 66)
    print(f"\nMatched-bpp PSNR delta on {args.target_name} (fine-tuned - generic@same bpp):")
    for b, p, pr, dl in matched_deltas(gen_t["bpp"], gen_t["psnr"], ft_t["bpp"], ft_t["psnr"]):
        s = f"{dl:+.3f} dB" if dl is not None else "(out of generic range)"
        ref = f"{pr:.2f}" if pr is not None else " n/a"
        print(f"  bpp {b:.4f}  ft {p:.2f} dB | generic {ref} -> {s}")

    def plot_pts(res):  # filter out dropped lambdas from the plotted curve
        keep = [i for i, l in enumerate(res["lambdas"]) if l not in drop]
        return [res["bpp"][i] for i in keep], [res["psnr"][i] for i in keep]
    px_bpp, px_psnr = plot_pts(ft_t)
    pk_bpp, pk_psnr = plot_pts(ft_c)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(gen_t["bpp"], gen_t["psnr"], "o-", color="tab:blue", label="STanH generic (authors)")
    ax[0].plot(px_bpp, px_psnr, "s-", color="tab:red", label=label)
    if vtm_t is not None:
        ax[0].plot(vtm_t["bpp"], vtm_t["psnr"], "^--", color="tab:green", label="VTM (H.266)")
    # anchor (native STanH) operating point, if the baseline lists levels and contains it
    if args.anchor_level and "levels" in gen_t and args.anchor_level in gen_t["levels"]:
        i = gen_t["levels"].index(args.anchor_level)
        ax[0].scatter([gen_t["bpp"][i]], [gen_t["psnr"][i]], marker="*", s=320,
                      color="black", zorder=5, label="Anchor (native STanH)")
    ax[0].set_title(f"{args.target_name} (target domain)\nBD-Rate {args.tag} vs generic: {bd_t:+.2f}%")
    ax[0].set_xlabel("bpp"); ax[0].set_ylabel("PSNR (dB)")
    ax[0].set_xlim(left=0); ax[0].grid(True, alpha=0.3); ax[0].legend()

    ax[1].plot(gen_c["bpp"], gen_c["psnr"], "o-", color="tab:blue", label="STanH generic (authors)")
    ax[1].plot(pk_bpp, pk_psnr, "s-", color="tab:red", label=label)
    if vtm_c is not None:
        ax[1].plot(vtm_c["bpp"], vtm_c["psnr"], "^--", color="tab:green", label="VTM (H.266)")
    ax[1].set_title(f"{args.cross_name} (cross-domain)\nBD-Rate {args.tag} vs generic: {bd_c:+.2f}%")
    ax[1].set_xlabel("bpp"); ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_xlim(left=0); ax[1].grid(True, alpha=0.3); ax[1].legend()

    fig.suptitle(f"STanH domain adaptation ({args.tag}): {args.target_name} target, {args.cross_name} cross", fontsize=13)
    fig.tight_layout()
    out = args.out or os.path.join("results", "plots", f"{args.tag}_rd_comparison.png")
    out = out if os.path.isabs(out) else os.path.join(ROOT, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
