"""ΔPSNR medio casado por bpp -- a "unidade unica" do projeto -- com contraste pareado.

Para cada ponto da curva testada cuja taxa cai dentro do suporte pos-Pareto da curva de
referencia, delta = PSNR(teste) - PSNR(referencia @ mesma bpp); a celula reporta a media.
E a unidade que o projeto usa no Kodak (nunca BD-Rate, que exige janela larga) e que e
robusta a janela estreita -- por isso e a leitura correta das celulas de 3 lambda.

Compara DUAS curvas contra a mesma referencia e devolve o contraste no MESMO reamostreio
bootstrap pareado por imagem (convencao do A1, B1 e B2).

`--lambdas` restringe as duas curvas a uma grade comum: uma celula de 3 lambda so pode ser
contrastada com os MESMOS 3 lambda da celula de 8 (comparacao like-for-like do B2).

⚠ `--common_grid` acrescenta o estimador (c) do Z1 (achado X01-1): as duas curvas
interpoladas (PCHIP) nos MESMOS bpps. E necessario sempre que o tratamento **desloca a
taxa**, porque ai o desencontro de suporte e causado pelo proprio tratamento e a diferenca
de duas medias tomadas em faixas diferentes mistura dano com colocacao de lambda.

Uso:
  python scripts/matched_delta_contrast.py --ref results/kodak_rd.json \
      --a sem_replay=results/v7_decoder_on_kodak_rd.json \
      --b com_replay=results/_exp_01ago/xray_decoder_replay_on_cross_rd.json \
      --lambdas lambda_0.02,lambda_0.06305,lambda_0.25 --common_grid --out results/_exp_XX/s.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"))
from analyze_finetuned import bd_rate, drop_dominated  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    return json.load(open(os.path.join(ROOT, name)))


def per_level(curve, keep=None):
    keys = curve.get("levels") or curve.get("lambdas")
    idx = [i for i, k in enumerate(keys) if keep is None or k in keep]
    if keep is not None and len(idx) != len(keep):
        raise SystemExit(f"lambdas pedidos ausentes: {set(keep) - set(keys)}")
    b = np.array([curve["per_image"][keys[i]]["bpp"] for i in idx])
    p = np.array([curve["per_image"][keys[i]]["psnr"] for i in idx])
    return b, p, [keys[i] for i in idx]


def matched(bt, pt, gb, gp):
    """Media dos deltas casados + nº de pontos dentro do suporte pos-Pareto da referencia."""
    br, pr = drop_dominated(gb, gp)  # devolve listas
    br, pr = np.asarray(br), np.asarray(pr)
    m = (bt >= br.min()) & (bt <= br.max())
    if not m.any():
        return np.nan, 0
    return float(np.mean(pt[m] - np.interp(bt[m], br, pr))), int(m.sum())


def pchip_mean(bt, pt, gb, gp, lo, hi, n=200):
    """Media do delta sobre a grade comum [lo,hi] -- estimador (c) do Z1 (X01-1)."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    ob, op = drop_dominated(bt, pt)
    br, pr = drop_dominated(gb, gp)
    if len(ob) < 2 or len(br) < 2:
        return np.nan
    x = np.linspace(lo, hi, n)
    try:
        return float(np.mean(PchipInterpolator(ob, op)(x) - PchipInterpolator(br, pr)(x)))
    except ValueError:
        return np.nan


def grid_range(gb, gp, ab, bb):
    """Faixa comum: suporte pos-Pareto da generica interceptado com as duas curvas."""
    br, _ = drop_dominated(gb, gp)
    return max(min(br), ab.min(), bb.min()), min(max(br), ab.max(), bb.max())


def run(ref_path, a_spec, b_spec, lambdas, B, seed, bd_ref_path=None, do_grid=False):
    a_name, a_path = a_spec.split("=", 1)
    b_name, b_path = b_spec.split("=", 1)
    ref, A, Bc = load(ref_path), load(a_path), load(b_path)
    for c, p in ((A, a_path), (Bc, b_path)):
        if ref.get("files") != c.get("files"):
            raise SystemExit(f"conjuntos de imagens diferentes: {ref_path} x {p}")

    keep_b = set(lambdas) if lambdas else None
    gb, gp, _ = per_level(ref)
    ab, ap, a_keys = per_level(A, keep_b)
    bb, bp, b_keys = per_level(Bc, keep_b)
    n = gb.shape[1]

    pt_a, n_a = matched(ab.mean(1), ap.mean(1), gb.mean(1), gp.mean(1))
    pt_b, n_b = matched(bb.mean(1), bp.mean(1), gb.mean(1), gp.mean(1))
    lo, hi = grid_range(gb.mean(1), gp.mean(1), ab.mean(1), bb.mean(1))
    gpt_a = pchip_mean(ab.mean(1), ap.mean(1), gb.mean(1), gp.mean(1), lo, hi) if do_grid else None
    gpt_b = pchip_mean(bb.mean(1), bp.mean(1), gb.mean(1), gp.mean(1), lo, hi) if do_grid else None

    rng = np.random.default_rng(seed)
    va, vb, vc, ga, gbo, gc = [], [], [], [], [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        gbm, gpm = gb[:, i].mean(1), gp[:, i].mean(1)
        abm, apm = ab[:, i].mean(1), ap[:, i].mean(1)
        bbm, bpm = bb[:, i].mean(1), bp[:, i].mean(1)
        x, _ = matched(abm, apm, gbm, gpm)
        y, _ = matched(bbm, bpm, gbm, gpm)
        if np.isfinite(x) and np.isfinite(y):
            va.append(x); vb.append(y); vc.append(y - x)
        if do_grid:
            l, h = grid_range(gbm, gpm, abm, bbm)
            u = pchip_mean(abm, apm, gbm, gpm, l, h)
            v = pchip_mean(bbm, bpm, gbm, gpm, l, h)
            if np.isfinite(u) and np.isfinite(v):
                ga.append(u); gbo.append(v); gc.append(v - u)

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {
        "reference": ref_path, "B": B, "seed": seed,
        "lambdas_used": {a_name: a_keys, b_name: b_keys},
        a_name: {"json": a_path, "matched_dpsnr": pt_a, "ci95": ci(va), "n_in_support": n_a},
        b_name: {"json": b_path, "matched_dpsnr": pt_b, "ci95": ci(vb), "n_in_support": n_b},
        "contrast": {"what": f"{b_name} - {a_name} (mesmo reamostreio)",
                     "value": pt_b - pt_a, "ci95": ci(vc),
                     "excludes_zero": bool(ci(vc)[0] * ci(vc)[1] > 0)},
        "n_resamples_valid": len(vc),
    }
    if do_grid:
        out["common_grid"] = {
            "why": ("estimador (c) do Z1/X01-1: as duas curvas nos MESMOS bpps. Obrigatorio "
                    "quando o tratamento desloca a taxa e os suportes proprios diferem."),
            "bpp_range": [float(lo), float(hi)],
            a_name: {"dpsnr": gpt_a, "ci95": ci(ga)},
            b_name: {"dpsnr": gpt_b, "ci95": ci(gbo)},
            "contrast": {"value": gpt_b - gpt_a, "ci95": ci(gc),
                         "excludes_zero": bool(ci(gc)[0] * ci(gc)[1] > 0)},
        }

    if bd_ref_path:  # BD-Rate da curva "b" no alvo, com as duas guardas do W15
        gen = load(bd_ref_path)
        bd, (blo, bhi) = bd_rate(gen["bpp"], gen["psnr"], Bc["bpp"], Bc["psnr"])
        gkeys = gen.get("levels") or gen.get("lambdas")
        rb = np.array([gen["per_image"][k]["bpp"] for k in gkeys])
        rp = np.array([gen["per_image"][k]["psnr"] for k in gkeys])
        tb, tp, _ = per_level(Bc)
        rng2 = np.random.default_rng(seed)
        vals, widths = [], []
        for _ in range(B):
            i = rng2.integers(0, n, n)
            v, (l, h) = bd_rate(rb[:, i].mean(1), rp[:, i].mean(1), tb[:, i].mean(1), tp[:, i].mean(1))
            if np.isfinite(v):
                vals.append(v); widths.append(h - l)
        floor = float(np.percentile(widths, 2.5))
        out["bd_rate_target"] = {
            "reference": bd_ref_path, "bd_rate_pct": float(bd), "ci95": ci(vals),
            "window_width": float(bhi - blo), "window_bootstrap_floor": floor,
            "n_pareto": len(drop_dominated(Bc["bpp"], Bc["psnr"])[0]),
            "n_points": len(Bc["bpp"]),
            "guard1_W15_pass": bool(floor >= 1.0),
            "reportable": bool(floor >= 1.0),
            "note": ("1a guarda do W15: piso bootstrap da janela (percentil 2,5) >= 1 dB. "
                     "Reprovada => o BD NAO e reportavel; usar o ΔPSNR casado."),
        }
    return out, a_name, b_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="curva de referencia (generica) no MESMO dominio")
    ap.add_argument("--a", required=True, help="nome=caminho.json (curva base)")
    ap.add_argument("--b", required=True, help="nome=caminho.json (curva tratada)")
    ap.add_argument("--lambdas", default="", help="grade comum, separada por virgula")
    ap.add_argument("--common_grid", action="store_true", help="acrescenta o estimador (c) do Z1")
    ap.add_argument("--bd_ref", default="", help="se dado, calcula tambem o BD-Rate da curva b")
    ap.add_argument("-B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lam = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    res, a_name, b_name = run(args.ref, args.a, args.b, lam, args.B, args.seed,
                              args.bd_ref or None, args.common_grid)

    print(f"referencia: {args.ref}   lambdas: {lam or 'todos'}\n")
    print("  (a) suporte proprio de cada curva:")
    for name in (a_name, b_name):
        r = res[name]
        print(f"    {name:<26} {r['matched_dpsnr']:+8.3f} dB  "
              f"IC [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  ({r['n_in_support']} pts no suporte)")
    c = res["contrast"]
    print(f"    {'CONTRASTE':<26} {c['value']:+8.3f} dB  IC [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  "
          f"{'EXCLUI ZERO' if c['excludes_zero'] else 'cruza zero'}")
    if "common_grid" in res:
        g = res["common_grid"]
        print(f"\n  (c) grade comum, bpp [{g['bpp_range'][0]:.4f}, {g['bpp_range'][1]:.4f}]:")
        for name in (a_name, b_name):
            print(f"    {name:<26} {g[name]['dpsnr']:+8.3f} dB  "
                  f"IC [{g[name]['ci95'][0]:+.3f}, {g[name]['ci95'][1]:+.3f}]")
        gc = g["contrast"]
        print(f"    {'CONTRASTE':<26} {gc['value']:+8.3f} dB  IC [{gc['ci95'][0]:+.3f}, {gc['ci95'][1]:+.3f}]  "
              f"{'EXCLUI ZERO' if gc['excludes_zero'] else 'cruza zero'}")
    if "bd_rate_target" in res:
        b = res["bd_rate_target"]
        print(f"\n  BD-Rate alvo {b['bd_rate_pct']:+.2f}% IC [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}] | "
              f"janela {b['window_width']:.3f} piso {b['window_bootstrap_floor']:.3f} "
              f"Pareto {b['n_pareto']}/{b['n_points']} -> "
              f"{'REPORTAVEL' if b['reportable'] else 'NAO REPORTAVEL (reprova 1a guarda do W15)'}")

    if args.out:
        path = os.path.join(ROOT, args.out)
        if os.path.exists(path):
            raise SystemExit(f"recusando sobrescrever {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(res, open(path, "w"), indent=2, ensure_ascii=False)
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
