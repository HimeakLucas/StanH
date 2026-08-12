#!/bin/bash
# 8-lambda spectrum of `full` mode on x-ray, trained in a single run.
#
# The lambda=0.003 point of an earlier run came out Pareto-inverted (more rate, less PSNR)
# and was retrained separately, leaving the curve mixing points from two runs. This script
# trains the remaining lambdas into the SAME save_dir, producing a homogeneous curve.
#
# Hyperparameters identical to the original run (see train/run_xray_full.sh and
# run_xray_full_dense.sh): batch 8, 20 epochs, patch 256, deterministic beta and the same
# warm-starts. The only difference is the de facto seed: the trainer exposes no --seed.
#
# lambda=0.003 already exists in models/xray_full_v6_runB and is skipped.
#
# Uso:  nohup bash train/run_xray_full_8lam.sh > logs/xray_full_8lam.log 2>&1 &
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_DIR="models/xray_full_v6_runB"
DERIV_DIR="models/original_paper/STanH/derivations"
DATASET="datasets/xrays"
EPOCHS=20
BATCH_SIZE=8
WANDB_PROJ="PIBIC_StanH_XRay_v6_fullft_runB"

# lambda : generic warm-start derivation (same map as the original v6 run)
LAMBDAS=("0.00666" "0.02"     "0.04"     "0.06305" "0.13"     "0.25"     "0.44014")
DERIVS=( "D03-A040" "D10-A040" "D11-A040" "D11-A040" "D12-A040" "D13-A040" "D13-A040")

mkdir -p "$SAVE_DIR" logs
echo "[$(date '+%F %T')] START — lambda restantes em $SAVE_DIR"

for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"; warm="${DERIVS[$i]}"
  if [ -f "$SAVE_DIR/lambda_${lam}_best.pth.tar" ]; then
    echo "[$(date '+%F %T')] SKIP lambda=$lam (já existe)"; continue
  fi
  echo "===== [$(date '+%F %T')] full lambda=$lam (warm $warm) ====="
  python -u train/train_xray_full.py --mode full \
      --lmbda "$lam" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
      --dataset "$DATASET" --patch_size 256 256 \
      --save_dir "$SAVE_DIR" --wandb_project "$WANDB_PROJ" \
      --init_stanh "$DERIV_DIR/${warm}.pth.tar"
done

echo "[$(date '+%F %T')] TRAIN DONE - avaliando na amostra disjunta"
python -u eval/eval_full.py --models_dir "$SAVE_DIR" \
    --dataset datasets/xrays/test/sample_eval_disjoint --limit 150 \
    --entropy_estimation --out_json results/v6_full_runB8_on_xray_disjoint_rd.json

echo "[$(date '+%F %T')] DONE -> results/v6_full_runB8_on_xray_disjoint_rd.json"
