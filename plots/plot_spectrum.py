"""
Consolidated 'adapter spectrum' figure (domain-agnostic).

Overlays any set of fine-tuning variants against the authors' generic STanH (and,
if available, VTM), on a TARGET domain and a CROSS domain. One figure tells the
story: left = where the domain gain lives; right = cross-domain forgetting.

Variants are passed as repeatable --variant "label:target_json:cross_json[:color:marker]"
(color/marker optional). With no --variant args, reproduces the X-ray spectrum.

Example (documents, no VTM yet):
  python plots/plot_spectrum.py --domain documents --target_name Documentos \
     --target_baseline results/documents_generic_rd.json --target_vtm "" --cross_vtm "" --target_xlim 0.8 \
     --variant "Encoder (6,9M):results/docs_encoder_on_documents_rd.json:results/docs_encoder_on_kodak_rd.json:tab:red:v-" \
     --variant "Decoder (6,9M):results/docs_decoder_on_documents_rd.json:results/docs_decoder_on_kodak_rd.json:tab:orange:D-"
"""
import json, os, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(n): return json.load(open(os.path.join(ROOT, n)))
def load_opt(n):
    if not n: return None
    p = os.path.join(ROOT, n); return json.load(open(p)) if os.path.exists(p) else None

# Default X-ray spectrum (label, target_json, cross_json, color, marker)
DEFAULT_VARIANTS = [
    ("Só quantizador (320)", "results/v4_finetuned_on_xray_rd.json", "results/v4_finetuned_on_kodak_rd.json", "tab:purple", "s-"),
    ("Decoder (6,9M)",       "results/v7_decoder_on_xray_rd.json",   "results/v7_decoder_on_kodak_rd.json",   "tab:orange", "D-"),
    ("Encoder (6,9M)",       "results/v8_encoder_on_xray_rd.json",   "results/v8_encoder_on_kodak_rd.json",   "tab:red",    "v-"),
    ("Full backbone (75M)",  "results/v6_fullft_on_xray_rd.json",    "results/v6_fullft_on_kodak_rd.json",    "black",      "*-"),
]


def parse_variant(s):
    # "label:target_json:cross_json[:color:marker]" (color may contain ':' e.g. tab:red)
    parts = s.split(":")
    label, tj, cj = parts[0], parts[1], parts[2]
    rest = parts[3:]
    if rest and (rest[-1].endswith("-") or rest[-1] in (".", "o", "s", "v", "D", "^", "*")):
        marker = rest[-1]; color_tokens = rest[:-1]
    else:
        marker = "o-"; color_tokens = rest
    color = ":".join(color_tokens) if color_tokens else None
    return (label, tj, cj, color, marker)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="xray", help="used for output filename")
    ap.add_argument("--target_name", default="Raio-X")
    ap.add_argument("--cross_name", default="Kodak")
    ap.add_argument("--target_baseline", default="results/xray_stanh_rd.json")
    ap.add_argument("--cross_baseline", default="results/kodak_rd.json")
    ap.add_argument("--target_vtm", default="results/xray_vtm_rd.json", help="'' to disable")
    ap.add_argument("--cross_vtm", default="results/vtm_kodak_rd.json", help="'' to disable")
    ap.add_argument("--target_xlim", type=float, default=0.0, help="right x-limit for target panel; <=0 autoscales")
    ap.add_argument("--variant", action="append", default=[], help="label:target_json:cross_json[:color:marker]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variants = [parse_variant(v) for v in args.variant] if args.variant else DEFAULT_VARIANTS
    gen_t, gen_c = load(args.target_baseline), load(args.cross_baseline)
    vtm_t, vtm_c = load_opt(args.target_vtm), load_opt(args.cross_vtm)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(gen_t["bpp"], gen_t["psnr"], "o-", color="tab:blue", label="Genérica (autores)", markersize=5, linewidth=1.4)
    ax[1].plot(gen_c["bpp"], gen_c["psnr"], "o-", color="tab:blue", label="Genérica (autores)", markersize=5, linewidth=1.4)
    for label, tj, cj, color, mk in variants:
        t, c = load_opt(tj), load_opt(cj)
        if t is None or c is None:
            print(f"  skip '{label}' (missing {tj} or {cj})"); continue
        ax[0].plot(t["bpp"], t["psnr"], mk, color=color, label=label, markersize=5, linewidth=1.4)
        ax[1].plot(c["bpp"], c["psnr"], mk, color=color, label=label, markersize=5, linewidth=1.4)
    if vtm_t is not None:
        ax[0].plot(vtm_t["bpp"], vtm_t["psnr"], "^--", color="tab:green", label="VTM (H.266)", markersize=5)
    if vtm_c is not None:
        ax[1].plot(vtm_c["bpp"], vtm_c["psnr"], "^--", color="tab:green", label="VTM (H.266)", markersize=5)

    ax[0].set_title(f"{args.target_name} (domínio alvo)")
    ax[0].set_xlim(0, args.target_xlim) if args.target_xlim > 0 else ax[0].set_xlim(left=0)
    ax[1].set_title(f"{args.cross_name} (domínio de origem)"); ax[1].set_xlim(left=0)
    for a in ax:
        a.set_xlabel("bpp"); a.set_ylabel("PSNR (dB)"); a.grid(True, alpha=0.3); a.legend(fontsize=7.5)
    fig.tight_layout()
    out = args.out or os.path.join("results/plots", f"spectrum_{args.domain}.png")
    out = out if os.path.isabs(out) else os.path.join(ROOT, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
