"""Saturation index computed from the GENERIC curve alone.

Motivacao: os rotulos "limitado por taxa" / "limitado por reconstrucao" da regra 2x2
sao hoje atribuidos DEPOIS de conhecer o vencedor, o que torna a 2x2 uma descricao
circular. Um indice calculado antes de qualquer adaptacao,
a partir apenas da curva generica, e o que permitiria PREVER o vencedor.

Indice primario -- inclinacao terminal normalizada:

    L_i = log2(bpp_i)                       (pontos pos-Pareto da generica)
    S   = [(P_n - P_{n-1}) / (L_n - L_{n-1})] / [(P_n - P_1) / (L_n - L_1)]

isto e, a inclinacao no ponto de maior taxa dividida pela inclinacao media da curva.
S -> 0 curva saturada no topo (limitada por reconstrucao); S -> 1 curva ainda subindo
como o resto (limitada por taxa); S > 1 curva convexa, longe de saturar.

Indice secundario, so como robustez (declarado sem poder de reverter o veredito):
razao entre o ganho de PSNR no terco superior de bpp e o ganho no terco inferior.

O vencedor por dominio e lido dos JSONs de BD-Rate no alvo (encoder x decoder), com IC
bootstrap pareado por imagem, nunca da memoria.

Uso:
    PYTHONPATH=src python scripts/saturation_index.py \
        --out results/_exp_30jul/saturation_index.json
"""
import argparse
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generic curve + adapted curves per domain, each on its canonical ruler
# (disjunta onde ela existe).
DOMAINS = {
    "raio-X": {
        "generic": "results/xray_generic_disjoint_rd.json",
        "encoder": "results/v8_encoder_on_xray_disjoint_rd.json",
        "decoder": "results/v7_decoder_on_xray_disjoint_rd.json",
    },
    "OCT": {
        "generic": "results/oct_generic_disjoint_v2_rd.json",
        "encoder": "results/oct_encoder_on_oct_disjoint_v2_rd.json",
        "decoder": "results/oct_decoder_on_oct_disjoint_v2_rd.json",
    },
    "retina": {
        "generic": "results/retina_generic_disjoint_rd.json",
        "encoder": "results/retina_encoder_on_retina_disjoint_rd.json",
        "decoder": "results/retina_decoder_on_retina_disjoint_rd.json",
    },
    "documentos": {
        "generic": "results/documents_generic_rd.json",
        "encoder": "results/documents_encoder_on_documents_rd.json",
        "decoder": "results/documents_decoder_on_documents_rd.json",
    },
    "DIOR": {
        "generic": "results/dior_generic_rd.json",
        "encoder": "results/dior_encoder_on_dior_rd.json",
        "decoder": "results/dior_decoder_on_dior_rd.json",
    },
    "RICO": {
        "generic": "results/rico_generic_rd.json",
        "encoder": "results/rico_encoder_on_rico_rd.json",
        "decoder": "results/rico_decoder_on_rico_rd.json",
    },
}


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


def keys(js):
    return js.get("levels") or js.get("lambdas")


def stack(js, key):
    return np.array([js["per_image"][k][key] for k in keys(js)])


def drop_dominated(bpp, psnr):
    """plots/analyze_finetuned.py:48-55 — convencao unica do projeto."""
    pts = sorted(zip(bpp, psnr))
    keep, best = [], -1e9
    for b, p in pts:
        if p > best:
            keep.append((b, p))
            best = p
    return [x[0] for x in keep], [x[1] for x in keep]


def bd_rate(bpp_ref, psnr_ref, bpp_test, psnr_test):
    """plots/analyze_finetuned.py:58-74."""
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


def boot_bd(ref, test, B=1000, seed=42):
    """95% BD-Rate CI, per-image paired bootstrap, plus the window floor (first guard)."""
    if "per_image" not in ref or "per_image" not in test:
        return None
    if ref.get("files") != test.get("files"):
        return None
    rb, rp = stack(ref, "bpp"), stack(ref, "psnr")
    tb, tp = stack(test, "bpp"), stack(test, "psnr")
    n = rb.shape[1]
    rng = np.random.default_rng(seed)
    vals, wins = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bd, (lo, hi) = bd_rate(rb[:, i].mean(1), rp[:, i].mean(1),
                               tb[:, i].mean(1), tp[:, i].mean(1))
        wins.append(hi - lo)
        if np.isfinite(bd):
            vals.append(bd)
    if len(vals) < B // 2:
        return None
    v = np.array(vals)
    return {"ci_lo": float(np.percentile(v, 2.5)),
            "ci_hi": float(np.percentile(v, 97.5)),
            "n_resamples": int(v.size),
            "window_floor_db": float(np.percentile(np.array(wins), 2.5))}


def terminal_slope_index(bpp, psnr):
    """PRIMARY index: terminal slope normalized by the mean slope."""
    b, p = drop_dominated(bpp, psnr)
    if len(b) < 3:
        return None
    L = np.log2(np.array(b))
    P = np.array(p)
    term = (P[-1] - P[-2]) / (L[-1] - L[-2])
    mean = (P[-1] - P[0]) / (L[-1] - L[0])
    return {"S": float(term / mean),
            "terminal_slope_db_per_doubling": float(term),
            "mean_slope_db_per_doubling": float(mean),
            "n_pareto": len(b),
            "bpp_range": [float(min(b)), float(max(b))],
            "psnr_range": [float(min(p)), float(max(p))]}


def tercile_ratio(bpp, psnr):
    """SECONDARY index (robustness only): PSNR gain in the upper bpp tercile over the gain
    in the lower one, on the post-Pareto curve resampled in log-bpp."""
    b, p = drop_dominated(bpp, psnr)
    if len(b) < 3:
        return None
    L = np.log2(np.array(b))
    f = PchipInterpolator(L, np.array(p))
    edges = np.linspace(L[0], L[-1], 4)
    lower = float(f(edges[1]) - f(edges[0]))
    upper = float(f(edges[3]) - f(edges[2]))
    return {"R": float(upper / lower) if lower != 0 else float("nan"),
            "gain_lower_third_db": lower, "gain_upper_third_db": upper}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/_exp_30jul/saturation_index.json")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = {"generated_by": "scripts/saturation_index.py (A3/W8, 30/07/2026)",
           "primary_index": ("S = terminal slope / mean slope over the post-Pareto GENERIC "
                             "curve, in dB per bpp doubling. S->0 saturated "
                             "(reconstruction-limited); S->1 still climbing (rate-limited)."),
           "secondary_index": ("R = PSNR gain in the upper bpp tercile / gain in the lower "
                               "one. Robustness only; cannot overturn the verdict."),
           "preregistration": ("defined and written down BEFORE computing. PASSES if some "
                               "threshold in S perfectly separates encoder-wins from "
                               "decoder-wins, in the predicted direction (higher S => "
                               "encoder wins). Otherwise the 2x2 loses predictive status "
                               "and stays descriptive."),
           "convention": "pontos pos-Pareto (drop_dominated), como no resto da analise",
           "bootstrap": {"B": args.boot, "seed": args.seed},
           "domains": {}}

    for dom, paths in DOMAINS.items():
        gen = load(paths["generic"])
        rec = {"generic_json": paths["generic"],
               "index_primary": terminal_slope_index(gen["bpp"], gen["psnr"]),
               "index_secondary": tercile_ratio(gen["bpp"], gen["psnr"]),
               "winner": {}}
        for block in ("encoder", "decoder"):
            t = load(paths[block])
            bd, (lo, hi) = bd_rate(gen["bpp"], gen["psnr"], t["bpp"], t["psnr"])
            ci = boot_bd(gen, t, B=args.boot, seed=args.seed)
            rec["winner"][block] = {"json": paths[block], "bd_rate_pct": float(bd),
                                    "psnr_window_db": float(hi - lo), "ci": ci}
        out["domains"][dom] = rec

    dst = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        raise SystemExit(f"RECUSADO: {dst} ja existe.")
    with open(dst, "w") as f:
        json.dump(out, f, indent=4)
    print(f"escrito: {dst}\n")

    hdr = f"{'dominio':12s} {'S':>7s} {'R':>7s} | {'BD enc':>9s} {'BD dec':>9s} | vencedor"
    print(hdr); print("-" * len(hdr))
    for dom, rec in out["domains"].items():
        S = rec["index_primary"]["S"]
        R = rec["index_secondary"]["R"]
        e = rec["winner"]["encoder"]; d = rec["winner"]["decoder"]

        def verdict(x):
            if x["ci"] is None:
                return "sem IC"
            return "cruza0" if x["ci"]["ci_lo"] * x["ci"]["ci_hi"] < 0 else "ok"

        ve, vd = verdict(e), verdict(d)
        if ve == "cruza0" and vd == "cruza0":
            w = "INDEFINIDO (os dois cruzam zero)"
        elif ve == "cruza0":
            w = "decoder"
        elif vd == "cruza0":
            w = "encoder"
        else:
            w = "encoder" if e["bd_rate_pct"] < d["bd_rate_pct"] else "decoder"
        print(f"{dom:12s} {S:7.3f} {R:7.3f} | {e['bd_rate_pct']:+8.2f}% {d['bd_rate_pct']:+8.2f}% | {w}")


if __name__ == "__main__":
    main()
