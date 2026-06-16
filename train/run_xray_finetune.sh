#!/bin/bash
# Run from the repository root:  bash train/run_xray_finetune.sh
# Refined round (v3): warm-start each derivation from the NEAREST one (paper
# recommendation), extend lambdas above the 0.0483 anchor for high-rate coverage,
# and select best by validation (training=False) inside train_xray_stanh.py.
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"

EPOCHS=30
BATCH_SIZE=8
DATASET="datasets/xrays"
SAVE_DIR="models/xray_stanh_finetuning_v3"
WANDB_PROJ="PIBIC_StanH_XRay_v3_refined"
ANCHOR_LMBDA="0.0483"
# Descending chain from the anchor rate, then ascending chain for higher rates.
DESC=("0.025" "0.013" "0.0067" "0.0035" "0.0018")
ASC=("0.08" "0.13")

mkdir -p "$SAVE_DIR"

run() {  # $1 = lambda, $2 = optional warm-start checkpoint
    local lmbda="$1"; local init="$2"
    echo "=========================================================="
    echo "FINETUNING LAMBDA: $lmbda  (warm-start: ${init:-anchor})"
    echo "=========================================================="
    local extra=""
    [ -n "$init" ] && extra="--init_stanh $init"
    python -u train/train_xray_stanh.py \
        --lmbda "$lmbda" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
        --dataset "$DATASET" --save_dir "$SAVE_DIR" --wandb_project "$WANDB_PROJ" $extra
}

best() { echo "$SAVE_DIR/lambda_$1_best.pth.tar"; }

# 1) anchor rate, refined from the anchor's own STanH
run "$ANCHOR_LMBDA" ""

# 2) descending: each from the previous (nearest) derivation
prev="$ANCHOR_LMBDA"
for l in "${DESC[@]}"; do
    run "$l" "$(best "$prev")"
    prev="$l"
done

# 3) ascending (higher rate than anchor): each from the previous
prev="$ANCHOR_LMBDA"
for l in "${ASC[@]}"; do
    run "$l" "$(best "$prev")"
    prev="$l"
done

echo "ALL DONE -> $SAVE_DIR"
