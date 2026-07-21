#!/bin/bash
# Upper-bound: FULL fine-tune on X-ray for 2 bracketing rate points,
# each warm-started from the matching authors' generic derivation. Measures the
# maximum domain gain available when the backbone is also unfrozen.
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"

EPOCHS=20
BATCH_SIZE=8
DATASET="datasets/xrays"
SAVE_DIR="models/xray_full_finetuning_v6"
WANDB_PROJ="PIBIC_StanH_XRay_v6_fullft_fixed"
DERIV_DIR="models/original_paper/STanH/derivations"
# v6: deterministic always-hard beta ramp (default) fixes the soft-quantizer
# gaming that made the v5 stochastic full fine-tune diverge.

# low / mid / high rate points (D03 ~0.042, D11 ~0.058, D13 ~0.128 bpp)
DERIVS=("D03-A040" "D11-A040" "D13-A040")
LAMBDAS=("0.00666" "0.06305" "0.44014")

mkdir -p "$SAVE_DIR"
for i in "${!DERIVS[@]}"; do
    d="${DERIVS[$i]}"; lmbda="${LAMBDAS[$i]}"
    echo "=========================================================="
    echo "FULL fine-tune  lambda=$lmbda  (warm-start from generic $d)"
    echo "=========================================================="
    python -u train/train_xray_full.py \
        --lmbda "$lmbda" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
        --dataset "$DATASET" --save_dir "$SAVE_DIR" --wandb_project "$WANDB_PROJ" \
        --init_stanh "$DERIV_DIR/$d.pth.tar"
done

echo "ALL DONE -> $SAVE_DIR"
