#!/bin/bash
# E2/G2 -- replay no decodificador do RAIO-X: o antidoto do esquecimento chega ao dominio
# que carrega a Tabela I, o espectro completo e a manchete do artigo.
#
# O artigo reporta replay em documentos (n=1); o B2 acrescentou o OCT (n=2). Faltava o
# raio-X, e sem ele o antidoto parece testado so onde deu certo.
#
# 3 lambda (os mesmos do B2/OCT), alfa=0,8, replay em DIV2K.
# INVARIANTE DO PROJETO: replay NUNCA usa Kodak -- Kodak e so avaliacao cross-domain.
#
# ⚠ HIPERPARAMETROS: identicos aos do `v7_decoder` SEM replay, que e a celula de
# comparacao -- e NAO aos do runner do OCT. Conferido na fonte
# (train/run_xray_decoder_v7.sh:7): BATCH=8 (o OCT usou 16), patch 256 (default do
# treinador), 20 epocas, lr 1e-5, mesmo mapa lambda->warm-start.
# ⚠ SEM --save_delta, tambem para casar com o v7: o delta e gravado em fp16
# (train_xray_full.py:245) e o fp16 sozinho move o BD agregado ~0,5 p.p. (achado N1b/X30-7).
# O v7 tem checkpoints fp32 completos; a celula com replay tem de ter os mesmos.
#
# Uso:  nohup bash train/run_xray_decoder_replay.sh > logs/e2_xray_replay.log 2>&1 &
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=20
BATCH=8
ALPHA=0.8
DERIV_DIR="models/original_paper/STanH/derivations"
LAMBDAS=("0.02"     "0.06305"  "0.25")
DERIVS=( "D10-A040" "D11-A040" "D13-A040")
SAVE="models/xray_decoder_replay"

mkdir -p logs "$SAVE"
echo "[$(date '+%F %T')] E2 -- raio-X decoder + replay(DIV2K, alpha=$ALPHA) -- inicio"

for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"; warm="${DERIVS[$i]}"
  if [ -f "$SAVE/lambda_${lam}_best.pth.tar" ]; then
    echo "[$(date '+%F %T')] SKIP lambda=$lam (ja existe)"; continue
  fi
  echo "===== [$(date '+%F %T')] xray decoder_replay  lambda=$lam  (warm $warm) ====="
  python -u train/train_xray_full.py --mode decoder \
      --lmbda "$lam" --epochs "$EPOCHS" --batch_size "$BATCH" \
      --dataset datasets/xrays \
      --save_dir "$SAVE" --wandb_project "PIBIC_StanH_xray_decoder_replay" \
      --init_stanh "$DERIV_DIR/${warm}.pth.tar" \
      --replay_dataset datasets/div2k --replay_alpha "$ALPHA"
done

echo "[$(date '+%F %T')] TREINO CONCLUIDO -- avaliacoes"

# Alvo: amostra disjunta por PACIENTE (150 imgs / 144 pacientes)
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/xrays/test/sample_eval_disjoint --limit 150 --entropy_estimation \
    --out_json results/_exp_01ago/xray_decoder_replay_on_xray_disjoint_rd.json

# Cross: Kodak, 24 imagens
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/kodak --limit 24 --entropy_estimation \
    --out_json results/_exp_01ago/xray_decoder_replay_on_cross_rd.json

echo "[$(date '+%F %T')] E2 CONCLUIDO"
