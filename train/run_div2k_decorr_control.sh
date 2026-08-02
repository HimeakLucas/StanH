#!/bin/bash
# B1 -- TERCEIRO braco do controle de monocromia: `div2k_decorr`.
#
# Copia de train/run_div2k_mono_control.sh trocando SO o dataset, para que o braco novo
# seja comparavel aos dois que ja rodaram: mesmos 3 lambda, mesmos warm-starts, mesmas
# 229 epocas, mesmo batch, mesmo patch, mesmo --save_delta.
#
# O braco `decorr` preserva croma e desloca a estatistica de cor (rotacao KLT fixa,
# canais descorrelacionados). E o braco que discrimina "a causa e a ausencia de croma"
# de "a causa e o desalinhamento da estatistica de cor" -- ver o pre-registro, que
# ESTREITA a leitura da tabela: os tres bracos tem conteudo espacial identico e
# diferem apenas pelo mapa de cor, entao a "distancia" aqui e distancia DE COR,
# nao distancia de dominio em geral.
#
# Uso:  nohup bash train/run_div2k_decorr_control.sh > logs/b1_div2k_decorr.log 2>&1 &
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
