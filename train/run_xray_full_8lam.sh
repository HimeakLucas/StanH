#!/bin/bash
# Espectro de 8 lambda do modo `full` em raio-X, treinado numa corrida unica.
#
# Motivacao: o ponto lambda=0,003 de uma corrida anterior ficou Pareto-invertido
# (mais taxa, menos PSNR) e foi re-treinado a parte, o que deixou a curva
# misturando pontos de duas corridas. Este script treina os lambda restantes no
# MESMO save_dir, produzindo uma curva homogenea.
#
# Hiperparametros identicos aos da corrida original (ver train/run_xray_full.sh e
# run_xray_full_dense.sh): batch 8, 20 epocas, patch 256, beta deterministico e os
# mesmos warm-starts. A unica diferenca e a semente de facto: o treinador nao
# expoe --seed.
#
# lambda=0.003 ja existe em models/xray_full_v6_runB e e pulado.
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

# lambda : derivação genérica de warm-start (mesmo mapa do v6 original)
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
