#!/bin/bash
# v8 (encoder-only): fine-tune ONLY the encoder g_a + STanH (mode=encoder).
# Decoder + hyperprior stay frozen. Complement of v7 (decoder-only): isolates the
# transform that DEFINES the latent -> tests whether the big mid/high-rate gain and
# the rate-range EXTENSION (which v7/decoder could not reach) come from the encoder.
# Saves a small DELTA per derivation (--save_delta) instead of a 301 MB full model:
# the frozen anchor backbone is shared, each rate point stores only g_a + STanH.
# Same 8 rate points as v6/v7 for a direct three-way comparison.
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
EPOCHS=20; BATCH_SIZE=8; DATASET="datasets/xrays"
SAVE_DIR="models/xray_encoder_finetuning_v8"
WANDB_PROJ="PIBIC_StanH_XRay_v8_encoder"
DERIV_DIR="models/original_paper/STanH/derivations"
DERIVS=("D02-A040" "D03-A040" "D10-A040" "D11-A040" "D11-A040" "D12-A040" "D13-A040" "D13-A040")
LAMBDAS=("0.003"   "0.00666"  "0.02"     "0.04"     "0.06305"  "0.13"     "0.25"     "0.44014")
mkdir -p "$SAVE_DIR"
for i in "${!DERIVS[@]}"; do
    d="${DERIVS[$i]}"; lmbda="${LAMBDAS[$i]}"
    echo "=========================================================="
    echo "ENCODER v8  lambda=$lmbda  (warm-start from generic $d)"
    echo "=========================================================="
    python -u train/train_xray_full.py --mode encoder --save_delta \
        --lmbda "$lmbda" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
        --dataset "$DATASET" --save_dir "$SAVE_DIR" --wandb_project "$WANDB_PROJ" \
        --init_stanh "$DERIV_DIR/$d.pth.tar"
done
echo "ALL DONE -> $SAVE_DIR"
