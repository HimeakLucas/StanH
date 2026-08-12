#!/bin/bash
# QUANTIZER arm, retrained under the spectrum convention.
#
# The original quantizer arm came from train/train_xray_stanh.py, on a 7-lambda grid (only 3
# shared) with its own batch and lr. That forced the report to declare that the curves are
# not paired point by point and that the comparison is only legitimate against the shared
# generic reference. This script rebuilds the arm with --mode quantizer in the SAME trainer
# as the other three (train_xray_full.py), same 8-lambda grid, same warm-starts, same
# epochs/batch/lr_stanh.
#
# This is NOT just a re-grid: lr_stanh goes from 1e-4 (old script) to 1e-5 (spectrum
# convention) and epochs from 30 to 20. It is a genuinely different training run, and the
# result must be reported whatever it is -- including if it contradicts the published
# +1.23% BD-Rate.
#
# Usage: nohup bash train/run_xray_quantizer_v9.sh > logs/v9_quantizer.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=20; BATCH=8; DATASET="datasets/xrays"
SAVE_DIR="models/xray_quantizer_v9"
WANDB_PROJ="PIBIC_StanH_XRay_v9_quantizer"
DERIV_DIR="models/original_paper/STanH/derivations"
DERIVS=( "D02-A040" "D03-A040" "D10-A040" "D11-A040" "D11-A040" "D12-A040" "D13-A040" "D13-A040")
LAMBDAS=("0.003"    "0.00666"  "0.02"     "0.04"     "0.06305"  "0.13"     "0.25"     "0.44014")

mkdir -p "$SAVE_DIR" logs
echo "[$(date '+%F %T')] v9 quantizador — início (8 lambdas, epochs=$EPOCHS, batch=$BATCH)"
for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"; warm="${DERIVS[$i]}"
  if [ -f "$SAVE_DIR/lambda_${lam}_best.pth.tar" ]; then
    echo "[$(date '+%F %T')] SKIP lambda=$lam (já existe)"; continue
  fi
  echo "===== [$(date '+%F %T')] quantizer lambda=$lam (warm $warm) ====="
  python -u train/train_xray_full.py --mode quantizer --save_delta \
      --lmbda "$lam" --epochs "$EPOCHS" --batch_size "$BATCH" \
      --dataset "$DATASET" --save_dir "$SAVE_DIR" --wandb_project "$WANDB_PROJ" \
      --init_stanh "$DERIV_DIR/${warm}.pth.tar" || echo "FALHOU lambda=$lam"
done
echo "[$(date '+%F %T')] TREINO CONCLUÍDO — avaliando"
python -u eval/eval_full.py --models_dir "$SAVE_DIR" \
    --dataset datasets/xrays/test/sample_eval_disjoint --limit 150 --entropy_estimation \
    --out_json results/v9_quantizer_on_xray_disjoint_rd.json
python -u eval/eval_full.py --models_dir "$SAVE_DIR" \
    --dataset datasets/kodak --limit 24 --entropy_estimation \
    --out_json results/v9_quantizer_on_kodak_rd.json
echo "[$(date '+%F %T')] ALL DONE"
