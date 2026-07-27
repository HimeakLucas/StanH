"""
Convert the FULL decoder checkpoints (mode=decoder, ~301 MB each) into fp16 DELTA
checkpoints over the shared anchor, and report the sizes.

Why this exists: `train_xray_full.py --save_delta` was added after the v7 decoder run,
so `models/xray_decoder_finetuning_v7/*.pth.tar` are full 301 MB state_dicts with no
`is_delta` key. The `.tex` claims a "~14 MB delta (encoder OR decoder)"; the decoder
half of that claim had no artifact (finding N1c). This script produces it.

The trainable set is derived the same way the trainer derives it, by applying the
mode=decoder freeze pattern to a fresh model, so the delta contains exactly the tensors
that training could have moved. Output is compatible with eval/eval_full.py, which
reconstructs full = anchor + delta.

Usage:
    python scripts/make_decoder_delta.py \
        --src models/xray_decoder_finetuning_v7 \
        --out models/xray_decoder_v7_delta \
        --json results/a3_delta_sizes_decoder.json
"""
import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from compress.models.cnn_multiStanh import WACNNMultiSTanH


def trainable_names(anchor_path, mode, device="cpu"):
    """Names of the tensors `--mode <mode>` would train, from the trainer's own recipe."""
    ck = torch.load(anchor_path, map_location=device, weights_only=False)
    fact = dict(ck["factorized_configuration"][0]); gauss = dict(ck["gaussian_configuration"][0])
    for cfg in (fact, gauss):
        cfg["beta"] = 10; cfg["trainable"] = True; cfg["annealing"] = "gap_stoc"; cfg["gap_factor"] = 15
    model = WACNNMultiSTanH(N=192, M=320, num_stanh=1,
                            factorized_configuration=[fact], gaussian_configuration=[gauss])
    model.update(device=torch.device(device))
    model.load_state_dict(ck["state_dict"], state_dicts_stanh=None)
    if mode == "decoder":
        model.freeze_net(); model.unfreeze_decoder(); model.unfreeze_quantizer()
    elif mode == "encoder":
        model.freeze_net(); model.unfreeze_encoder(); model.unfreeze_quantizer()
    else:
        raise SystemExit(f"unsupported mode {mode!r}")
    return {k for k, v in model.named_parameters() if v.requires_grad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with full *_best.pth.tar checkpoints")
    ap.add_argument("--out", required=True, help="dir to write the delta checkpoints into")
    ap.add_argument("--mode", default="decoder", choices=["decoder", "encoder"])
    ap.add_argument("--anchor", default="models/original_paper/STanH/anchor/0728_last_.pth.tar")
    ap.add_argument("--json", default=None, help="write the size table here")
    args = ap.parse_args()

    names = trainable_names(args.anchor, args.mode)
    anchor_MB = os.path.getsize(args.anchor) / 1e6
    print(f"mode={args.mode}: {len(names)} trainable tensors; anchor {anchor_MB:.1f} MB")

    os.makedirs(args.out, exist_ok=True)
    table = {}
    for f in sorted(os.listdir(args.src)):
        if not f.endswith("_best.pth.tar"):
            continue
        src = os.path.join(args.src, f)
        ck = torch.load(src, map_location="cpu", weights_only=False)
        if ck.get("is_delta"):
            print(f"  SKIP {f}: already a delta"); continue
        sd = ck["state_dict"]
        missing = [k for k in names if k not in sd]
        if missing:
            raise SystemExit(f"{f}: {len(missing)} trainable tensors absent from the "
                             f"checkpoint, e.g. {missing[:3]}")
        state = {k: ck[k] for k in ("epoch", "best_val", "val_loss", "lmbda", "beta_y",
                                    "beta_z", "beta_max_y", "beta_max_z",
                                    "factorized_configuration", "gaussian_configuration")
                 if k in ck}
        state["is_delta"] = True
        state["anchor"] = args.anchor
        state["mode"] = args.mode
        state["delta"] = {k: sd[k].detach().cpu().half() for k in sorted(names)}
        dst = os.path.join(args.out, f)
        torch.save(state, dst)
        fp32_MB = os.path.getsize(src) / 1e6
        fp16_MB = os.path.getsize(dst) / 1e6
        n = sum(v.numel() for v in state["delta"].values())
        table[f] = {"fp32_full_MB": round(fp32_MB, 2), "fp16_delta_MB": round(fp16_MB, 2),
                    "params_M": round(n / 1e6, 3), "anchor_MB": round(anchor_MB, 1),
                    "razao_sobre_ancora": round(anchor_MB / fp16_MB, 1)}
        print(f"  {f}: {fp32_MB:.1f} MB full -> {fp16_MB:.2f} MB delta "
              f"({n/1e6:.2f}M params, {anchor_MB/fp16_MB:.1f}x over anchor)")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump(table, open(args.json, "w"), indent=1)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
