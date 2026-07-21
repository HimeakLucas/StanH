"""
Upper-bound diagnostic: FULL fine-tune on X-ray for a single lambda.

Unlike train_xray_stanh.py (which freezes the backbone and trains only the ~320
STanH params), this UNFREEZES the whole WACNN backbone and fine-tunes everything,
to measure the MAXIMUM domain gain available on X-ray. Interpretation:
  - if this beats the generic/v4 curve substantially -> "domain gain exists, the
    quantizer alone can't capture it";
  - if it also ties -> "no domain gain to capture with this backbone".

Design:
  - warm-start the STanH from the matching authors' generic derivation (--init_stanh),
    so we measure the gain ADDED by unfreezing the backbone on top of v4;
  - two LRs: small for the backbone (don't destroy the pretrained net), larger for
    the STanH quantizer (param group on '.sos.' params);
  - keep the beta annealing (eq. 8) and validation-based best selection;
  - save the FULL model checkpoint (~300 MB) + configs, for eval/eval_full.py.

Note: aux/quantile loss is omitted; we evaluate with entropy estimation, whose bpp
comes from the learned likelihoods (unaffected by the range-coder CDF tables).
"""
import os, argparse, math, sys, shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from compress.models.cnn_multiStanh import WACNNMultiSTanH
from compress.datasets import ImageFolder
from compress.utils.annealings import configure_annealings


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["mse_loss"] = self.mse(output["x_hat"], target)
        distortion = 255 ** 2 * out["mse_loss"]
        out["bpp_loss"] = sum(
            (torch.log(l).sum() / (-math.log(2) * num_pixels))
            for l in output["likelihoods"].values()
        )
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        return out


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.update(device=torch.device(device))
    model.eval()
    tot = tot_mse = tot_bpp = 0.0
    n = 0
    for d in val_loader:
        d = d.to(device)
        oc = criterion(model(d, training=False, stanh_level=0), d)
        tot += oc["loss"].item(); tot_mse += oc["mse_loss"].item(); tot_bpp += oc["bpp_loss"].item()
        n += 1
        if device == "cuda":
            torch.cuda.empty_cache()
    n = max(n, 1)
    return tot / n, tot_mse / n, tot_bpp / n


def main():
    p = argparse.ArgumentParser(description="FULL fine-tune (upper-bound) on X-ray")
    p.add_argument("--lmbda", type=float, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_stanh", type=float, default=1e-5)
    p.add_argument("--patch_size", type=int, nargs=2, default=(256, 256))
    p.add_argument("--clip_max_norm", type=float, default=1.0)
    p.add_argument("--gap_factor", type=int, default=15)
    # Beta schedule. The default 'gap_stoc' stochastic annealing samples
    # beta ~ U(1, beta_max); with an UNFROZEN backbone the low-beta (soft) samples
    # are exploited (encoder games the soft quantizer -> train collapses to a soft
    # optimum that explodes under the hard eval). For full fine-tune use a
    # DETERMINISTIC, always-hard beta ramp so train ~ eval.
    p.add_argument("--stoch_anneal", action="store_true", help="Use the (unstable for full-ft) stochastic annealing")
    p.add_argument("--beta_min", type=float, default=30.0)
    p.add_argument("--beta_step", type=float, default=20.0, help="beta increase per epoch")
    p.add_argument("--beta_max", type=float, default=170.0)
    p.add_argument("--dataset", type=str, default="datasets/xrays")
    p.add_argument("--anchor", type=str, default="models/original_paper/STanH/anchor/0728_last_.pth.tar")
    p.add_argument("--init_stanh", type=str, default=None, help="Generic derivation to warm-start STanH from")
    p.add_argument("--mode", choices=["full", "decoder", "encoder"], default="full",
                   help="full = whole backbone (v6); decoder = only g_s + STanH (v7); "
                        "encoder = only g_a + STanH (v8, isolates the latent-defining transform)")
    p.add_argument("--save_delta", action="store_true",
                   help="Save ONLY the trainable (requires_grad) params as a small delta over the "
                        "frozen anchor, instead of the full ~301 MB checkpoint. The frozen backbone "
                        "is shared (one anchor copy); eval reconstructs the full model = anchor + delta. "
                        "No-op savings for --mode full (everything is trainable).")
    p.add_argument("--replay_dataset", type=str, default="",
                   help="Natural-image dataset dir (ImageFolder layout) for knowledge replay. "
                        "Each step adds a replay batch to the loss, anchoring the source domain "
                        "(anti-forgetting, cf. Duan et al. CVPR'24). MUST NOT be the cross-domain "
                        "eval set (Kodak) — that would leak into training.")
    p.add_argument("--replay_alpha", type=float, default=0.8,
                   help="Weight of the DOMAIN loss; the replay loss gets (1 - alpha).")
    p.add_argument("--save_dir", type=str, default="models/xray_full_finetuning")
    p.add_argument("--wandb_project", type=str, default="PIBIC_StanH_XRay_v5_fullft")
    p.add_argument("--val_images", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    wandb.init(project=args.wandb_project, config=vars(args), name=f"full_lambda_{args.lmbda}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = transforms.Compose([transforms.RandomCrop(tuple(args.patch_size)), transforms.ToTensor()])
    val_tf = transforms.Compose([transforms.CenterCrop(tuple(args.patch_size)), transforms.ToTensor()])
    train_ds = ImageFolder(args.dataset, split="train", transform=train_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_ds = ImageFolder(args.dataset, num_images=args.val_images, split="val", transform=val_tf)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers, shuffle=False)

    replay_dl = None
    if args.replay_dataset:
        replay_ds = ImageFolder(args.replay_dataset, split="train", transform=train_tf)
        replay_dl = DataLoader(replay_ds, batch_size=args.batch_size, num_workers=2,
                               shuffle=True, pin_memory=True)
        print(f"Replay: {len(replay_ds)} images from {args.replay_dataset}, "
              f"loss = {args.replay_alpha}*domain + {1-args.replay_alpha:.2f}*replay")

    checkpoint = torch.load(args.anchor, map_location=device, weights_only=False)
    fact_cfg = dict(checkpoint["factorized_configuration"][0])
    gauss_cfg = dict(checkpoint["gaussian_configuration"][0])
    for cfg in (fact_cfg, gauss_cfg):
        cfg["beta"] = 10; cfg["trainable"] = True; cfg["annealing"] = "gap_stoc"; cfg["gap_factor"] = args.gap_factor

    model = WACNNMultiSTanH(N=192, M=320, num_stanh=1,
                            factorized_configuration=[fact_cfg], gaussian_configuration=[gauss_cfg]).to(device)
    model.update(device=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"], state_dicts_stanh=None)

    if args.init_stanh and os.path.exists(args.init_stanh):
        init = torch.load(args.init_stanh, map_location=device, weights_only=False)
        model.upload_stanh_values(init["state_dict"], index=0)
        print(f"Warm-started STanH from {args.init_stanh}")

    # What to fine-tune:
    #   full    -> everything (upper bound, v6);
    #   decoder -> only the synthesis transform g_s + STanH (middle ground): encoder
    #              and hyperprior stay frozen, so the rate is ~fixed (= generic) and
    #              only the reconstruction adapts. The lightest adapter that could
    #              still capture the mid/high-rate gains.
    if args.mode == "full":
        for prm in model.parameters():
            prm.requires_grad = True
    elif args.mode == "decoder":
        model.freeze_net()
        model.unfreeze_decoder()
        model.unfreeze_quantizer()
    elif args.mode == "encoder":
        # Complement of v7: adapt only g_a (the transform that DEFINES the latent)
        # + STanH; decoder and hyperprior stay frozen. Tests whether the big
        # mid/high-rate gain and the rate-range extension come from the encoder.
        model.freeze_net()
        model.unfreeze_encoder()
        model.unfreeze_quantizer()
    else:
        raise ValueError(f"unknown --mode {args.mode}")
    sos_params = [v for k, v in model.named_parameters() if v.requires_grad and ".sos." in k]
    base_params = [v for k, v in model.named_parameters() if v.requires_grad and ".sos." not in k]
    print(f"[mode={args.mode}] Trainable: base {sum(x.numel() for x in base_params)/1e6:.2f}M @ lr {args.lr_backbone} | "
          f"STanH {sum(x.numel() for x in sos_params)} @ lr {args.lr_stanh}")
    optimizer = optim.Adam([
        {"params": base_params, "lr": args.lr_backbone},
        {"params": sos_params, "lr": args.lr_stanh},
    ])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    annealing_z, annealing_y = configure_annealings(fact_cfg, gauss_cfg)

    os.makedirs(args.save_dir, exist_ok=True)
    last_path = os.path.join(args.save_dir, f"lambda_{args.lmbda}_last.pth.tar")
    best_path = os.path.join(args.save_dir, f"lambda_{args.lmbda}_best.pth.tar")
    best_val = float("inf"); start_epoch = 0

    if os.path.exists(last_path):
        try:
            print(f"Resuming from {last_path}")
            st = torch.load(last_path, map_location=device, weights_only=False)
            if st.get("is_delta"):
                # model already holds the anchor + warm-started STanH; overlay the delta.
                merged = model.state_dict()
                merged.update({k: v.to(device=device, dtype=merged[k].dtype)
                               for k, v in st["delta"].items() if k in merged})
                model.load_state_dict(merged, state_dicts_stanh=None)
            else:
                model.load_state_dict(st["state_dict"], state_dicts_stanh=None)
            start_epoch = st["epoch"] + 1
            if annealing_z is not None and st.get("beta_max_z") is not None:
                annealing_z.beta_max = annealing_z.beta = st["beta_max_z"]
            if annealing_y is not None and st.get("beta_max_y") is not None:
                annealing_y.beta_max = annealing_y.beta = st["beta_max_y"]
            best_val = st.get("best_val", best_val)
            print(f"Started from epoch {start_epoch}")
        except Exception as e:
            # A GPU crash mid-write can truncate the checkpoint; don't loop forever.
            print(f"WARN: could not resume from {last_path} ({type(e).__name__}: {e}). "
                  f"Re-warm-starting and training from scratch.")
            start_epoch = 0
            if args.init_stanh and os.path.exists(args.init_stanh):
                init = torch.load(args.init_stanh, map_location=device, weights_only=False)
                model.upload_stanh_values(init["state_dict"], index=0)

    # Names of the trainable params -> what a delta checkpoint needs to store.
    trainable_names = {k for k, v in model.named_parameters() if v.requires_grad}

    def save(epoch, val_loss):
        full_sd = model.state_dict()
        state = {
            "epoch": epoch, "best_val": min(best_val, val_loss), "val_loss": val_loss, "lmbda": args.lmbda,
            "beta_y": float(model.gaussian_conditional[0].sos.beta), "beta_z": float(model.entropy_bottleneck[0].sos.beta),
            "beta_max_y": float(annealing_y.beta_max) if annealing_y is not None else None,
            "beta_max_z": float(annealing_z.beta_max) if annealing_z is not None else None,
            "factorized_configuration": [fact_cfg], "gaussian_configuration": [gauss_cfg],
        }
        if args.save_delta and args.mode != "full":
            # Store only the trainable tensors; the frozen backbone is the shared anchor.
            # fp16 halves the delta (~28 MB -> ~14 MB); loaders cast back to the model dtype.
            state["is_delta"] = True
            state["anchor"] = args.anchor
            state["mode"] = args.mode
            state["delta"] = {k: full_sd[k].detach().cpu().half() for k in trainable_names if k in full_sd}
        else:
            state["state_dict"] = full_sd
        # Atomic write: a crash mid-save leaves the previous valid checkpoint intact.
        tmp = last_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, last_path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Deterministic, always-hard beta for full fine-tune (train ~ hard eval).
        if not args.stoch_anneal:
            cur_beta = min(args.beta_min + args.beta_step * epoch, args.beta_max)
            model.gaussian_conditional[0].sos.beta = cur_beta
            model.entropy_bottleneck[0].sos.beta = cur_beta
        ep_loss = ep_bpp = ep_mse = ep_replay = 0.0
        replay_iter = iter(replay_dl) if replay_dl is not None else None
        for i, d in enumerate(train_dl):
            d = d.to(device)
            optimizer.zero_grad()
            out_net = model(d, training=True, stanh_level=0)
            oc = criterion(out_net, d)
            if replay_iter is not None:
                try:
                    r = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_dl); r = next(replay_iter)
                r = r.to(device)
                oc_r = criterion(model(r, training=True, stanh_level=0), r)
                total = args.replay_alpha * oc["loss"] + (1.0 - args.replay_alpha) * oc_r["loss"]
                ep_replay += oc_r["loss"].item()
            else:
                total = oc["loss"]
            total.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            if args.stoch_anneal:
                gap = out_net["gap"]; lss = oc["loss"].detach().item()
                if annealing_z is not None:
                    annealing_z.step(gap[0], epoch, lss); model.entropy_bottleneck[0].sos.beta = annealing_z.beta
                if annealing_y is not None:
                    annealing_y.step(gap[1], epoch, lss); model.gaussian_conditional[0].sos.beta = annealing_y.beta

            ep_loss += oc["loss"].item(); ep_bpp += oc["bpp_loss"].item(); ep_mse += oc["mse_loss"].item()
            if i % 20 == 0:
                wandb.log({"batch_loss": oc["loss"].item(), "batch_bpp": oc["bpp_loss"].item(),
                           "batch_mse": oc["mse_loss"].item()})

        nb = len(train_dl); ep_loss, ep_bpp, ep_mse = ep_loss/nb, ep_bpp/nb, ep_mse/nb
        val_loss, val_mse, val_bpp = validate(model, val_dl, criterion, device)
        lr_scheduler.step(val_loss)
        train_psnr = -10*math.log10(ep_mse) if ep_mse > 0 else float("inf")
        val_psnr = -10*math.log10(val_mse) if val_mse > 0 else float("inf")
        bmy = float(model.gaussian_conditional[0].sos.beta) if not args.stoch_anneal else (
            float(annealing_y.beta_max) if annealing_y is not None else 0.0)
        print(f"==== Epoch {epoch} | train {ep_loss:.4f} | VAL {val_loss:.4f} | bpp {ep_bpp:.4f} | "
              f"val_psnr {val_psnr:.2f} | beta {bmy:.0f} | lr_bb {optimizer.param_groups[0]['lr']:.1e}")
        log = {"epoch": epoch, "train_loss": ep_loss, "val_loss": val_loss, "train_mse": ep_mse,
               "val_mse": val_mse, "train_psnr": train_psnr, "val_psnr": val_psnr, "bpp": ep_bpp,
               "val_bpp": val_bpp, "beta_max_y": bmy, "lr_backbone": optimizer.param_groups[0]["lr"]}
        if replay_dl is not None:
            log["replay_loss"] = ep_replay / nb
        wandb.log(log)

        save(epoch, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            tmpb = best_path + ".tmp"
            shutil.copyfile(last_path, tmpb)
            os.replace(tmpb, best_path)
            print(f" -> New best (val {val_loss:.4f})")

    wandb.finish()


if __name__ == "__main__":
    main()
