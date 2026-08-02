"""Z1 (01/08) -- contrastes cross do C2 sob TRES definicoes de suporte.

O "dano cross" do projeto e o ΔPSNR medio casado por bpp no Kodak: para cada ponto da
curva testada cuja taxa cai dentro do suporte da generica, delta = PSNR(teste) -
PSNR(generica @ mesma bpp), e a celula reporta a media desses deltas.

Quando duas celulas do MESMO dominio sao comparadas (encoder x encoder_hyper), essa media
e tomada sobre conjuntos de pontos DIFERENTES, porque o encoder_hyper retreina o modelo de
entropia e desloca a taxa: o desencontro de suporte e causado pelo tratamento. Este script
recomputa os contrastes sob tres estimadores, no mesmo reamostreio bootstrap:

  (a) suporte proprio   -- cada curva sobre os seus pontos dentro do suporte da generica
                           (o que esta no ledger);
  (b) faixa restrita    -- o encoder restrito a faixa de bpp dos pontos in-suporte do
                           encoder_hyper; o encoder_hyper inalterado;
  (c) grade comum       -- as duas curvas interpoladas (PCHIP) numa grade comum de bpp,
                           dentro do suporte da generica e da faixa comum as duas.

Bootstrap pareado por imagem: um unico sorteio de indices por reamostra, aplicado as TRES
curvas (generica, encoder, encoder_hyper) -- e a mesma convencao do A1/B2.

Uso:
  python scripts/cross_matched_support.py --out results/_exp_01ago/c2_cross_matched_support.json
"""
import argparse
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dominio -> (curva do encoder, curva do encoder_hyper); referencia comum = kodak_rd.json
DOMAINS = {
    "documentos": ("results/documents_encoder_on_cross_rd.json",
                   "results/documents_encoder_hyper_on_cross_rd.json"),
    "oct": ("results/oct_encoder_on_cross_rd.json",
            "results/oct_encoder_hyper_on_cross_rd.json"),
    "rico": ("results/rico_encoder_on_cross_rd.json",
             "results/rico_encoder_hyper_on_cross_rd.json"),
}
REF = "results/kodak_rd.json"


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


def drop_dominated(bpp, psnr):
    """Pareto-eficientes, monotonos em bpp (mesma funcao de plots/analyze_finetuned.py)."""
    pts = sorted(zip(bpp, psnr))
    keep, best = [], -1e9
    for b, p in pts:
        if p > best:
            keep.append((b, p))
            best = p
    return np.array([x[0] for x in keep]), np.array([x[1] for x in keep])


def per_level(curve):
    """Matriz (n_niveis, n_imagens) de bpp e psnr a partir de per_image."""
    keys = curve.get("levels") or curve.get("lambdas")
    b = np.array([curve["per_image"][k]["bpp"] for k in keys])
    p = np.array([curve["per_image"][k]["psnr"] for k in keys])
    return b, p


def matched_mean(bt, pt, br, pr, lo=None, hi=None):
    """Media dos deltas casados dos pontos de (bt,pt) dentro do suporte da referencia.

    lo/hi restringem adicionalmente a faixa de bpp considerada (estimador (b)).
    Devolve (media, mascara dos pontos usados).
    """
    gl, gh = br.min(), br.max()
    m = (bt >= gl) & (bt <= gh)
    if lo is not None:
        m &= (bt >= lo) & (bt <= hi)
    if not m.any():
        return np.nan, m
    return float(np.mean(pt[m] - np.interp(bt[m], br, pr)), ), m


def pchip_mean(bt, pt, br, pr, lo, hi, n=200):
    """Media do delta sobre uma grade comum [lo,hi], as duas curvas por PCHIP."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    ob, op = drop_dominated(bt, pt)          # PCHIP exige x estritamente crescente
    if len(ob) < 2 or len(br) < 2:
        return np.nan
    x = np.linspace(lo, hi, n)
    try:
        ft = PchipInterpolator(ob, op)
        fr = PchipInterpolator(br, pr)
    except ValueError:
        return np.nan
    return float(np.mean(ft(x) - fr(x)))


def estimators(gb, gp, eb, ep, hb, hp):
    """Os tres estimadores para uma reamostra. Devolve dict de escalares + diagnostico."""
    br, pr = drop_dominated(gb, gp)
    out = {}

    # (a) suporte proprio
    a_enc, m_enc = matched_mean(eb, ep, br, pr)
    a_hyp, m_hyp = matched_mean(hb, hp, br, pr)
    out["a_encoder"], out["a_hyper"] = a_enc, a_hyp
    out["a_contrast"] = a_hyp - a_enc

    # (b) encoder restrito a faixa de bpp dos pontos in-suporte do encoder_hyper
    if m_hyp.any():
        lo, hi = hb[m_hyp].min(), hb[m_hyp].max()
        b_enc, m_enc_b = matched_mean(eb, ep, br, pr, lo, hi)
    else:
        b_enc, m_enc_b = np.nan, m_hyp
    out["b_encoder"], out["b_hyper"] = b_enc, a_hyp
    out["b_contrast"] = a_hyp - b_enc

    # (c) grade PCHIP comum, dentro do suporte da generica e da faixa comum as duas curvas
    lo = max(br.min(), eb.min(), hb.min())
    hi = min(br.max(), eb.max(), hb.max())
    c_enc = pchip_mean(eb, ep, br, pr, lo, hi)
    c_hyp = pchip_mean(hb, hp, br, pr, lo, hi)
    out["c_encoder"], out["c_hyper"] = c_enc, c_hyp
    out["c_contrast"] = c_hyp - c_enc

    diag = {
        "generic_pareto_bpp": [float(br.min()), float(br.max())],
        "n_generic_pareto": int(len(br)),
        "n_support_encoder": int(m_enc.sum()),
        "n_support_hyper": int(m_hyp.sum()),
        "n_support_encoder_restricted": int(m_enc_b.sum()),
        "encoder_support_bpp": [float(x) for x in eb[m_enc]],
        "hyper_support_bpp": [float(x) for x in hb[m_hyp]],
        "encoder_restricted_bpp": [float(x) for x in eb[m_enc_b]],
        "common_grid_bpp": [float(lo), float(hi)],
    }
    return out, diag


def run_domain(name, enc_path, hyp_path, B=1000, seed=42):
    ref, enc, hyp = load(REF), load(enc_path), load(hyp_path)
    for c, p in ((enc, enc_path), (hyp, hyp_path)):
        if ref["files"] != c["files"]:
            raise SystemExit(f"conjuntos de imagens diferentes: {REF} x {p}")

    gb_i, gp_i = per_level(ref)
    eb_i, ep_i = per_level(enc)
    hb_i, hp_i = per_level(hyp)
    n = gb_i.shape[1]

    point, diag = estimators(gb_i.mean(1), gp_i.mean(1), eb_i.mean(1), ep_i.mean(1),
                             hb_i.mean(1), hp_i.mean(1))

    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    n_support_hyper = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals, d = estimators(gb_i[:, idx].mean(1), gp_i[:, idx].mean(1),
                             eb_i[:, idx].mean(1), ep_i[:, idx].mean(1),
                             hb_i[:, idx].mean(1), hp_i[:, idx].mean(1))
        for k, v in vals.items():
            if np.isfinite(v):
                boots[k].append(v)
        n_support_hyper.append(d["n_support_hyper"])

    ci = {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], len(v))
          for k, v in boots.items()}
    return {
        "domain": name,
        "encoder_json": enc_path,
        "hyper_json": hyp_path,
        "reference_json": REF,
        "n_images": int(n),
        "B": B,
        "seed": seed,
        "point": {k: (float(v) if np.isfinite(v) else None) for k, v in point.items()},
        "ci95": {k: c[0] for k, c in ci.items()},
        "n_resamples_valid": {k: c[1] for k, c in ci.items()},
        "support": diag,
        "n_support_hyper_over_resamples": {
            "min": int(np.min(n_support_hyper)),
            "max": int(np.max(n_support_hyper)),
            "mean": float(np.mean(n_support_hyper)),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/_exp_01ago/c2_cross_matched_support.json")
    ap.add_argument("-B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = {d: run_domain(d, e, h, args.B, args.seed) for d, (e, h) in DOMAINS.items()}
    out = {
        "what": "contrastes cross (encoder_hyper - encoder) do C2 sob tres definicoes de suporte",
        "unit": "ΔPSNR medio casado por bpp no Kodak, dB (negativo = o encoder_hyper danifica mais)",
        "estimators": {
            "a": "suporte proprio de cada curva dentro do suporte da generica (o do ledger)",
            "b": "encoder restrito a faixa de bpp dos pontos in-suporte do encoder_hyper",
            "c": "grade PCHIP comum as duas curvas, dentro do suporte da generica",
        },
        "preferred": "c",
        "why_preferred": (
            "(a) compara medias tomadas em faixas de bpp diferentes, e o desencontro e "
            "CAUSADO pelo tratamento (o encoder_hyper retreina o modelo de entropia e desloca "
            "a taxa), logo mistura dano com colocacao de lambda. (b) corrige isso mas depende "
            "de quais lambdas a celula por acaso tem, e e assimetrico por construcao (restringe "
            "so o encoder). (c) nao depende da grade de lambda de nenhuma das duas: avalia as "
            "duas curvas nos MESMOS bpps. O preco de (c) e interpolar entre pontos medidos; por "
            "isso os tres sao reportados lado a lado e a leitura exige que (b) e (c) concordem."
        ),
        "domains": res,
    }
    path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise SystemExit(f"recusando sobrescrever {path}")
    json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)

    print(f"{'dominio':<12} {'estimador':<4} {'encoder':>9} {'enc_hyper':>10} {'contraste':>10}  IC95 do contraste")
    for d, r in res.items():
        for e in "abc":
            p, c = r["point"], r["ci95"][f"{e}_contrast"]
            print(f"{d:<12} ({e})  {p[f'{e}_encoder']:>+9.3f} {p[f'{e}_hyper']:>+10.3f} "
                  f"{p[f'{e}_contrast']:>+10.3f}  [{c[0]:+.3f}, {c[1]:+.3f}]"
                  f"{'  EXCLUI ZERO' if c[0] * c[1] > 0 else '  cruza zero'}")
        s = r["support"]
        print(f"{'':<12}  suporte: generica {s['n_generic_pareto']} pts em bpp "
              f"[{s['generic_pareto_bpp'][0]:.4f}, {s['generic_pareto_bpp'][1]:.4f}] | "
              f"encoder {s['n_support_encoder']}/8, hyper {s['n_support_hyper']}/8, "
              f"encoder restrito {s['n_support_encoder_restricted']}/8")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
