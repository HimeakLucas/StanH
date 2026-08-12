#!/bin/bash
# Quick round (v4): warm-start each derivation from the AUTHORS' GENERIC
# derivation at the same rate (D01..D13), NOT chained from the anchor.
# Goal: keep the full bpp spread (each generic derivation already sits at its
# target rate) and isolate the domain-adaptation delta on X-ray.
# Per-point lambda is the RD-optimal lambda estimated from the generic X-ray
# RD curve slope (lambda = -dbpp / (255^2 * dMSE)).
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"

EPOCHS=20
BATCH_SIZE=8
DATASET="datasets/xrays"
SAVE_DIR="models/xray_stanh_finetuning_v4_gen"
WANDB_PROJ="PIBIC_StanH_XRay_v4_genwarmstart"
DERIV_DIR="models/original_paper/STanH/derivations"

# derivation file : matched lambda  (D10 is the anchor's own rate)
DERIVS=("D01-A040" "D02-A040" "D03-A040" "D10-A040" "D11-A040" "D12-A040" "D13-A040")
LAMBDAS=("0.00091" "0.00199" "0.00666" "0.01250" "0.06305" "0.16535" "0.44014")

mkdir -p "$SAVE_DIR"

for i in "${!DERIVS[@]}"; do
    d="${DERIVS[$i]}"; lmbda="${LAMBDAS[$i]}"
    init="$DERIV_DIR/$d.pth.tar"
    echo "=========================================================="
    echo "v4  deriv=$d  lambda=$lmbda  (warm-start from generic $d)"
    echo "=========================================================="
    python -u train/train_xray_stanh.py \
        --lmbda "$lmbda" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
        --dataset "$DATASET" --save_dir "$SAVE_DIR" --wandb_project "$WANDB_PROJ" \
        --init_stanh "$init"
done

echo "ALL DONE -> $SAVE_DIR"
