"""
Consolidated 'adapter spectrum' figure for the final paper.

Overlays all four fine-tuning variants (quantizer-only v4, decoder v7, encoder v8,
full v6) against the authors' generic STanH and VTM, on the target domain (X-ray)
and cross-domain (Kodak). One figure tells the whole story:
  - left  (X-ray): where the domain gain lives + the low-rate penalty + rate range;
  - right (Kodak): cross-domain forgetting grows with the decoder, not the encoder.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(n): return json.load(open(os.path.join(ROOT, n)))

# (label, xray_json, kodak_json, color, marker)
VARIANTS = [
    ("Genérica (autores)",      "results/xray_stanh_rd.json",        "results/kodak_rd.json",            "tab:blue",   "o-"),
    ("Só quantizador (320)",    "results/v4_finetuned_on_xray_rd.json","results/v4_finetuned_on_kodak_rd.json","tab:purple","s-"),
    ("Decoder (6,9M)",          "results/v7_decoder_on_xray_rd.json", "results/v7_decoder_on_kodak_rd.json","tab:orange", "D-"),
    ("Encoder (6,9M)",          "results/v8_encoder_on_xray_rd.json", "results/v8_encoder_on_kodak_rd.json","tab:red",    "v-"),
    ("Full backbone (75M)",     "results/v6_fullft_on_xray_rd.json",  "results/v6_fullft_on_kodak_rd.json","black",      "*-"),
]
VTM_X = load("results/xray_vtm_rd.json")
VTM_K = load("results/vtm_kodak_rd.json")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for label, xj, kj, color, mk in VARIANTS:
    x = load(xj); k = load(kj)
    ax[0].plot(x["bpp"], x["psnr"], mk, color=color, label=label, markersize=5, linewidth=1.4)
    ax[1].plot(k["bpp"], k["psnr"], mk, color=color, label=label, markersize=5, linewidth=1.4)
ax[0].plot(VTM_X["bpp"], VTM_X["psnr"], "^--", color="tab:green", label="VTM (H.266)", markersize=5)
ax[1].plot(VTM_K["bpp"], VTM_K["psnr"], "^--", color="tab:green", label="VTM (H.266)", markersize=5)

ax[0].set_title("Raio-X (domínio alvo)"); ax[0].set_xlim(left=0); ax[0].set_xlim(right=0.22)
ax[1].set_title("Kodak (domínio de origem)"); ax[1].set_xlim(left=0)
for a in ax:
    a.set_xlabel("bpp"); a.set_ylabel("PSNR (dB)"); a.grid(True, alpha=0.3); a.legend(fontsize=7.5)
fig.tight_layout()
out = os.path.join(ROOT, "results/plots/spectrum_rd.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)
