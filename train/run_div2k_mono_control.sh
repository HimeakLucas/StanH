#!/bin/bash
# Monochrome control: is the cross-domain collapse of the decoder caused by the distant
# domain STATISTICS, or simply by training on monochrome images?
#
# In the six real domains the two are perfectly confounded -- the three that collapse are
# exactly the three monochrome ones -- and with n=6 they cannot be separated.
#
# Design: two arms over THE SAME 800 DIV2K images (natural content the backbone never saw;
# WACNN and STanH were trained on OpenImages), differing in a single variable:
#     div2k_color   original RGB
#     div2k_gray    same content with R=G=B
# Everything else is identical: same split, same schedule, same warm-starts.
#
#   gray collapses and color does not -> monochrome is the cause
#   both behave alike                 -> monochrome is not enough; distance is back in play
#
# Hyperparameters match train/run_spectrum.sh (batch 16, patch 256, --save_delta, same
# lambda->derivation map). The only difference is the epoch count: 229 instead of 20, to
# match the ~160k image passes of the real domains, which have 8000 images against 700 here.
# Data diversity is still lower and must be declared when reporting.
#
# Execution order alternates by lambda, both arms side by side, so an interruption still
# leaves complete pairs -- the pair is the comparison that matters.
#
# Usage:  nohup bash train/run_div2k_mono_control.sh > logs/div2k_mono_control.log 2>&1 &
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=229
BATCH=16
PATCH=256
DERIV_DIR="models/original_paper/STanH/derivations"
LAMBDAS=("0.06305"  "0.02"      "0.25")
DERIVS=( "D11-A040" "D10-A040"  "D13-A040")
ARMS=("div2k_gray" "div2k_color")

mkdir -p logs
echo "[$(date '+%F %T')] CONTROLE DE MONOCROMIA — inicio"

for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"; warm="${DERIVS[$i]}"
  for arm in "${ARMS[@]}"; do
    save="models/${arm}_decoder"
    if [ -f "$save/lambda_${lam}_best.pth.tar" ]; then
      echo "[$(date '+%F %T')] SKIP $arm lambda=$lam"; continue
    fi
    mkdir -p "$save"
    echo "===== [$(date '+%F %T')] $arm  decoder  lambda=$lam  (warm $warm) ====="
    python -u train/train_xray_full.py --mode decoder --save_delta \
        --lmbda "$lam" --epochs "$EPOCHS" --batch_size "$BATCH" \
        --dataset "datasets/${arm}" --patch_size "$PATCH" "$PATCH" \
        --save_dir "$save" --wandb_project "PIBIC_StanH_${arm}_decoder" \
        --init_stanh "$DERIV_DIR/${warm}.pth.tar"
  done
done

echo "[$(date '+%F %T')] TREINO CONCLUIDO — avaliando no Kodak (dominio cross)"
for arm in "${ARMS[@]}"; do
  python -u eval/eval_full.py --models_dir "models/${arm}_decoder" \
      --dataset datasets/kodak --limit 24 --entropy_estimation \
      --out_json "results/${arm}_decoder_on_cross_rd.json"
done

echo "[$(date '+%F %T')] CONCLUIDO"
echo "Leitura: delta casado por bpp contra results/kodak_rd.json nos dois bracos."
echo "O contraste gray-color e a medicao; o valor absoluto de cada um nao e"
echo "comparavel aos seis dominios por causa da diferenca de diversidade de dados."
