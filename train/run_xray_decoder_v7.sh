#!/bin/bash
# Middle ground (v7): fine-tune ONLY the decoder g_s + STanH (mode=decoder).
# Encoder + hyperprior stay frozen -> rate ~fixed (= generic), only reconstruction
# adapts. ~6.9M trainable (9% of full). Same 8 rate points as v6 for direct compare.
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
EPOCHS=20; BATCH_SIZE=8; DATASET="datasets/xrays"
SAVE_DIR="models/xray_decoder_finetuning_v7"
WANDB_PROJ="PIBIC_StanH_XRay_v7_decoder"
DERIV_DIR="models/original_paper/STanH/derivations"
DERIVS=("D02-A040" "D03-A040" "D10-A040" "D11-A040" "D11-A040" "D12-A040" "D13-A040" "D13-A040")
LAMBDAS=("0.003"   "0.00666"  "0.02"     "0.04"     "0.06305"  "0.13"     "0.25"     "0.44014")
mkdir -p "$SAVE_DIR"
for i in "${!DERIVS[@]}"; do
    d="${DERIVS[$i]}"; lmbda="${LAMBDAS[$i]}"
    echo "=========================================================="
    echo "DECODER v7  lambda=$lmbda  (warm-start from generic $d)"
    echo "=========================================================="
    python -u train/train_xray_full.py --mode decoder \
        --lmbda "$lmbda" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
        --dataset "$DATASET" --save_dir "$SAVE_DIR" --wandb_project "$WANDB_PROJ" \
        --init_stanh "$DERIV_DIR/$d.pth.tar"
done
echo "ALL DONE -> $SAVE_DIR"
