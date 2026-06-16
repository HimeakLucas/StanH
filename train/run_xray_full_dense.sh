#!/bin/bash
# Densify the v6 full fine-tune RD curve: add intermediate/low rate points to the
# SAME dir as the first 3 (models/xray_full_finetuning_v6), each warm-started from
# the nearest authors' generic derivation. Deterministic always-hard beta (default).
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"

EPOCHS=20
BATCH_SIZE=8
DATASET="datasets/xrays"
SAVE_DIR="models/xray_full_finetuning_v6"   # same dir -> accumulates with the first 3
WANDB_PROJ="PIBIC_StanH_XRay_v6_fullft_fixed"
DERIV_DIR="models/original_paper/STanH/derivations"

# new lambda : nearest generic derivation to warm-start from
DERIVS=("D02-A040" "D10-A040" "D11-A040" "D12-A040" "D13-A040")
LAMBDAS=("0.003"   "0.02"     "0.04"     "0.13"     "0.25")

mkdir -p "$SAVE_DIR"
for i in "${!DERIVS[@]}"; do
    d="${DERIVS[$i]}"; lmbda="${LAMBDAS[$i]}"
    echo "=========================================================="
    echo "DENSIFY full-ft  lambda=$lmbda  (warm-start from generic $d)"
    echo "=========================================================="
    python -u train/train_xray_full.py \
        --lmbda "$lmbda" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
        --dataset "$DATASET" --save_dir "$SAVE_DIR" --wandb_project "$WANDB_PROJ" \
        --init_stanh "$DERIV_DIR/$d.pth.tar"
done

echo "ALL DONE -> $SAVE_DIR"
