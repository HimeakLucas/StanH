#!/bin/bash
# v9 — braço do QUANTIZADOR, re-treinado na convenção do espectro.
#
# MOTIVO. O braço original do quantizador veio de train/train_xray_stanh.py, com
# grade de 7 lambdas (só 3 coincidentes), batch e lr próprios. Isso obrigava o
# relatório a declarar que as curvas não são pareadas ponto a ponto e que a
# comparação só é legítima contra a referência genérica comum. Este script refaz o
# braço com --mode quantizer no MESMO treinador dos outros três (train_xray_full.py),
# mesma grade de 8 lambdas, mesmos warm-starts, mesmo epochs/batch/lr_stanh.
#
# ⚠ NÃO é só uma re-grade: lr_stanh passa de 1e-4 (script antigo) para 1e-5 (convenção
# do espectro) e epochs de 30 para 20. É um treino genuinamente diferente, e o
# resultado tem de ser reportado seja qual for — inclusive se contradisser o
# +1,23% de BD-Rate hoje publicado.
#
# Uso: nohup bash train/run_xray_quantizer_v9.sh > logs/v9_quantizer.log 2>&1 &
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
