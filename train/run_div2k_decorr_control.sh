#!/bin/bash
# THIRD arm of the color control: `div2k_decorr`.
#
# A copy of train/run_div2k_mono_control.sh changing ONLY the dataset, so the new arm is
# comparable to the two already run: same 3 lambdas, same warm-starts, same 229 epochs, same
# batch, same patch, same --save_delta.
#
# The `decorr` arm keeps chroma and displaces the color statistics (fixed KLT rotation,
# decorrelated channels). It is the arm that separates "the cause is the absence of chroma"
# from "the cause is the misaligned color statistics". Note this narrows the reading: the
# three arms share identical spatial content and differ only in the color map, so "distance"
# here is COLOR distance, not domain distance in general.
#
# Usage:  nohup bash train/run_div2k_decorr_control.sh > logs/b1_div2k_decorr.log 2>&1 &
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
ARM="div2k_decorr"

mkdir -p logs
echo "[$(date '+%F %T')] B1 -- braco $ARM -- inicio"

for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"; warm="${DERIVS[$i]}"
  save="models/${ARM}_decoder"
  if [ -f "$save/lambda_${lam}_best.pth.tar" ]; then
    echo "[$(date '+%F %T')] SKIP $ARM lambda=$lam (ja existe)"; continue
  fi
  mkdir -p "$save"
  echo "===== [$(date '+%F %T')] $ARM  decoder  lambda=$lam  (warm $warm) ====="
  python -u train/train_xray_full.py --mode decoder --save_delta \
      --lmbda "$lam" --epochs "$EPOCHS" --batch_size "$BATCH" \
      --dataset "datasets/${ARM}" --patch_size "$PATCH" "$PATCH" \
      --save_dir "$save" --wandb_project "PIBIC_StanH_${ARM}_decoder" \
      --init_stanh "$DERIV_DIR/${warm}.pth.tar"
done

echo "[$(date '+%F %T')] TREINO CONCLUIDO -- avaliando no Kodak (dominio cross)"
python -u eval/eval_full.py --models_dir "models/${ARM}_decoder" \
    --dataset datasets/kodak --limit 24 --entropy_estimation \
    --out_json "results/${ARM}_decoder_on_cross_rd.json"

echo "[$(date '+%F %T')] B1 CONCLUIDO"
echo "Leitura: SO o contraste entre bracos e medicao (700 imgs x 229 epocas contra"
echo "8000 x 20 dos dominios reais: mesmos passes, diversidade 11x menor)."
