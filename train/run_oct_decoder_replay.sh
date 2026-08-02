#!/bin/bash
# B2 -- replay no OCT: o antidoto do esquecimento generaliza para o PIOR colapso do estudo?
#
# O artigo afirma que o esquecimento do decodificador "nao e intrinseco" e que um lote de
# imagens naturais na perda o elimina. A evidencia e de UM unico dominio (documentos,
# -6,01 -> -0,02 dB com alfa=0,8). O OCT e o pior colapso medido (-7,19 dB) e e onde a
# afirmacao e mais forte e menos testada.
#
# 3 lambda (os mesmos do controle de monocromia), alfa=0,8, replay em DIV2K.
# INVARIANTE DO PROJETO: replay NUNCA usa Kodak -- Kodak e so avaliacao cross-domain, e
# usa-lo como replay seria vazamento direto para a metrica que reporta o esquecimento.
#
# Hiperparametros identicos a train/run_spectrum.sh (EPOCHS=20, batch 16, patch 256,
# --save_delta, mesmo mapa lambda->derivacao), para que a celula seja comparavel ao
# `oct_decoder` sem replay que ja existe.
#
# Uso:  nohup bash train/run_oct_decoder_replay.sh > logs/b2_oct_replay.log 2>&1 &
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=20
BATCH=16
PATCH=256
ALPHA=0.8
DERIV_DIR="models/original_paper/STanH/derivations"
LAMBDAS=("0.02"     "0.06305"  "0.25")
DERIVS=( "D10-A040" "D11-A040" "D13-A040")
SAVE="models/oct_decoder_replay"

mkdir -p logs "$SAVE"
echo "[$(date '+%F %T')] B2 -- OCT decoder + replay(DIV2K, alpha=$ALPHA) -- inicio"

for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"; warm="${DERIVS[$i]}"
  if [ -f "$SAVE/lambda_${lam}_best.pth.tar" ]; then
    echo "[$(date '+%F %T')] SKIP lambda=$lam (ja existe)"; continue
  fi
  echo "===== [$(date '+%F %T')] oct decoder_replay  lambda=$lam  (warm $warm) ====="
  python -u train/train_xray_full.py --mode decoder --save_delta \
      --lmbda "$lam" --epochs "$EPOCHS" --batch_size "$BATCH" \
      --dataset datasets/oct --patch_size "$PATCH" "$PATCH" \
      --save_dir "$SAVE" --wandb_project "PIBIC_StanH_oct_decoder_replay" \
      --init_stanh "$DERIV_DIR/${warm}.pth.tar" \
      --replay_dataset datasets/div2k --replay_alpha "$ALPHA"
done

echo "[$(date '+%F %T')] TREINO CONCLUIDO -- avaliacoes"

# Alvo: amostra disjunta por PACIENTE, v2 (a antiga tinha chave de grupo errada -- R28-1)
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/oct/test/sample_eval_disjoint_v2 --limit 150 --entropy_estimation \
    --out_json results/oct_decoder_replay_on_oct_disjoint_v2_rd.json

# Cross: Kodak, 24 imagens
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/kodak --limit 24 --entropy_estimation \
    --out_json results/oct_decoder_replay_on_cross_rd.json

echo "[$(date '+%F %T')] B2 CONCLUIDO"
