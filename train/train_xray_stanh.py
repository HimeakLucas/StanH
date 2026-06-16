"""
Fine-tune ONE STanH derivation on a domain dataset (X-ray) for a given lambda.

Follows the paper's refinement procedure (Sec. III-C / III-D and the authors'
src/train_multiStanh.py):
  - freeze the anchor backbone, train ONLY the STanH layers (w, b);
  - RD loss L = lambda * 255^2 * MSE + bpp  (eq. 11);
  - anneal the inverse temperature beta with 'gap_stoc' (eq. 8, K = gap_factor).

Refinement-quality improvements over the first round:
  - --init_stanh: warm-start w,b from the NEAREST already-refined derivation
    (paper: refining from the nearest derivation yields better RD) instead of
    always starting from the anchor;
  - best checkpoint selected on a VALIDATION pass with training=False (hard
    quantizer). Selecting on the training loss is biased: under gap_stoc beta
    is a random sample, and low-beta (soft) steps yield artificially low loss,
    so the saved levels would not match the hard inference quantizer.

Saves a STanH-only checkpoint (a few KB) compatible with eval/eval_finetuned.py.
"""
import os
import argparse
import math
import sys
import shutil
import glob
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
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        return out


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validation with the HARD quantizer (training=False) — unbiased by beta.
    Returns (avg_loss, avg_mse, avg_bpp) so MSE/PSNR can be tracked."""
    model.update(device=torch.device(device))
    model.eval()
    tot = tot_mse = tot_bpp = 0.0
    n = 0
    for d in val_loader:
        d = d.to(device)
        out = model(d, training=False, stanh_level=0)
        oc = criterion(out, d)
        tot += oc["loss"].item()
        tot_mse += oc["mse_loss"].item()
        tot_bpp += oc["bpp_loss"].item()
        n += 1
        if device == "cuda":
            torch.cuda.empty_cache()
    n = max(n, 1)
    return tot / n, tot_mse / n, tot_bpp / n


def main():
    parser = argparse.ArgumentParser(description="STanH domain fine-tuning (single lambda)")
    parser.add_argument("--lmbda", type=float, default=0.013, help="RD trade-off lambda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--gap_factor", type=int, default=15, help="K in the beta annealing (eq. 8)")
    parser.add_argument("--dataset", type=str, default="datasets/xrays")
    parser.add_argument("--anchor", type=str, default="models/original_paper/STanH/anchor/0728_last_.pth.tar")
    parser.add_argument("--init_stanh", type=str, default=None,
                        help="STanH-only checkpoint to warm-start w,b from (nearest derivation). Default: anchor's STanH.")
    parser.add_argument("--save_dir", type=str, default="models/xray_stanh_finetuning_v3")
    parser.add_argument("--wandb_project", type=str, default="PIBIC_StanH_XRay_v3_refined")
    parser.add_argument("--val_images", type=int, default=24, help="Validation images (center-cropped 256)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, config=vars(args), name=f"lambda_{args.lmbda}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transforms = transforms.Compose([
        transforms.RandomCrop(tuple(args.patch_size)),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.CenterCrop(tuple(args.patch_size)),
        transforms.ToTensor(),
    ])
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_dataset = ImageFolder(args.dataset, num_images=args.val_images, split="val", transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    checkpoint = torch.load(args.anchor, map_location=device, weights_only=False)

    # Single-level quantizer config, annealing fresh from beta=10 (paper setup).
    fact_cfg = dict(checkpoint["factorized_configuration"][0])
    gauss_cfg = dict(checkpoint["gaussian_configuration"][0])
    for cfg in (fact_cfg, gauss_cfg):
        cfg["beta"] = 10
        cfg["trainable"] = True
        cfg["annealing"] = "gap_stoc"
        cfg["gap_factor"] = args.gap_factor

    model = WACNNMultiSTanH(N=192, M=320, num_stanh=1,
                            factorized_configuration=[fact_cfg],
                            gaussian_configuration=[gauss_cfg])
    model = model.to(device)
    model.update(device=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"], state_dicts_stanh=None)

    # Warm-start the STanH layer from the nearest derivation, if given.
    if args.init_stanh and os.path.exists(args.init_stanh):
        init = torch.load(args.init_stanh, map_location=device, weights_only=False)
        model.upload_stanh_values(init["state_dict"], index=0)
        print(f"Warm-started STanH from {args.init_stanh}")

    # Freeze backbone, train only the STanH quantizer layers.
    model.freeze_net()
    model.unfreeze_quantizer()

    annealing_z, annealing_y = configure_annealings(fact_cfg, gauss_cfg)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    os.makedirs(args.save_dir, exist_ok=True)
    last_path = os.path.join(args.save_dir, f"lambda_{args.lmbda}_last.pth.tar")
    best_path = os.path.join(args.save_dir, f"lambda_{args.lmbda}_best.pth.tar")
    best_val = float("inf")
    start_epoch = 0

    if os.path.exists(last_path):
        print(f"Resuming from {last_path}")
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.upload_stanh_values(st["state_dict"], index=0)
        start_epoch = st["epoch"] + 1
        if annealing_z is not None and st.get("beta_max_z") is not None:
            annealing_z.beta_max = annealing_z.beta = st["beta_max_z"]
        if annealing_y is not None and st.get("beta_max_y") is not None:
            annealing_y.beta_max = annealing_y.beta = st["beta_max_y"]
        best_val = st.get("best_val", best_val)
        print(f"Started from epoch {start_epoch}")

    def save(epoch, val_loss):
        state = {
            "epoch": epoch, "best_val": min(best_val, val_loss),
            "val_loss": val_loss, "lmbda": args.lmbda,
            "beta_y": float(model.gaussian_conditional[0].sos.beta),
            "beta_z": float(model.entropy_bottleneck[0].sos.beta),
            "beta_max_y": float(annealing_y.beta_max) if annealing_y is not None else None,
            "beta_max_z": float(annealing_z.beta_max) if annealing_z is not None else None,
            "state_dict": {
                "gaussian_conditional": {"w": model.gaussian_conditional[0].sos.w.detach().cpu(),
                                         "b": model.gaussian_conditional[0].sos.b.detach().cpu()},
                "entropy_bottleneck": {"w": model.entropy_bottleneck[0].sos.w.detach().cpu(),
                                       "b": model.entropy_bottleneck[0].sos.b.detach().cpu()},
            },
        }
        torch.save(state, last_path)
        return state

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = epoch_bpp = epoch_mse = 0.0
        for i, d in enumerate(train_dataloader):
            d = d.to(device)
            optimizer.zero_grad()
            out_net = model(d, training=True, stanh_level=0)
            out_criterion = criterion(out_net, d)
            out_criterion["loss"].backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            gap = out_net["gap"]  # [gap_z, gap_y]
            lss = out_criterion["loss"].detach().item()
            if annealing_z is not None:
                annealing_z.step(gap[0], epoch, lss)
                model.entropy_bottleneck[0].sos.beta = annealing_z.beta
            if annealing_y is not None:
                annealing_y.step(gap[1], epoch, lss)
                model.gaussian_conditional[0].sos.beta = annealing_y.beta

            epoch_loss += out_criterion["loss"].item()
            epoch_bpp += out_criterion["bpp_loss"].item()
            epoch_mse += out_criterion["mse_loss"].item()
            if i % 10 == 0:
                wandb.log({"batch_loss": out_criterion["loss"].item(),
                           "batch_bpp": out_criterion["bpp_loss"].item(),
                           "batch_mse": out_criterion["mse_loss"].item(),
                           "beta_y": float(model.gaussian_conditional[0].sos.beta)})

        nb = len(train_dataloader)
        epoch_loss, epoch_bpp, epoch_mse = epoch_loss / nb, epoch_bpp / nb, epoch_mse / nb

        val_loss, val_mse, val_bpp = validate(model, val_dataloader, criterion, device)
        lr_scheduler.step(val_loss)
        # PSNR from MSE on [0,1] images: PSNR = -10*log10(MSE).
        train_psnr = -10.0 * math.log10(epoch_mse) if epoch_mse > 0 else float("inf")
        val_psnr = -10.0 * math.log10(val_mse) if val_mse > 0 else float("inf")
        beta_max_y = float(annealing_y.beta_max) if annealing_y is not None else 0.0
        print(f"==== Epoch {epoch} | train {epoch_loss:.4f} | VAL {val_loss:.4f} | "
              f"bpp {epoch_bpp:.4f} | val_psnr {val_psnr:.2f} | "
              f"beta_max_y {beta_max_y:.0f} | lr {optimizer.param_groups[0]['lr']:.2e}")
        wandb.log({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss,
                   "train_mse": epoch_mse, "val_mse": val_mse,
                   "train_psnr": train_psnr, "val_psnr": val_psnr,
                   "bpp": epoch_bpp, "val_bpp": val_bpp,
                   "beta_max_y": beta_max_y, "lr": optimizer.param_groups[0]["lr"]})

        save(epoch, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            shutil.copyfile(last_path, best_path)
            print(f" -> New best (val {val_loss:.4f})")

    wandb.finish()


if __name__ == "__main__":
    main()
