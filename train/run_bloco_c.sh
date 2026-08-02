#!/bin/bash
# BLOCO C -- aprovado pelo usuario em 30/07 (C1 sim; C2 nos tres dominios).
#
# C1: replica de semente numa SEGUNDA celula (xray_encoder, 8 lambda).
#     A contribuicao (3) apoia-se hoje na replica de UMA celula (o `full`). O A4b de
#     30/07 mostrou que a selecao de checkpoint NAO explica a instabilidade, o que
#     deixa a evidencia mais dependente da replica empirica -- dai o valor de uma
#     segunda celula independente.
#
#     ⚠ CONFIG MEDIDA, NAO ASSUMIDA: a corrida original do v8 usou BATCH_SIZE=8
#     (train/run_xray_encoder_v8.sh), nao 16. Para ser REPLICA e nao outro
#     experimento, esta corrida repete batch 8, 20 epocas, mesmos warm-starts e
#     mesmo dataset. Custo medido nos timestamps do v8: ~62 min/lambda => ~8,3 h.
#
#     ⚠ O treinador NAO expoe --seed e nao chama manual_seed: a "semente nova" e a
#     nao-determinacao de ordem de dados e de cuDNN. O warm-start e FIXO, entao esta
#     variancia EXCLUI inicializacao por construcao -- mesma ressalva do N13.
#
# C2: encoder_hyper nos tres dominios que faltam (documentos, OCT, RICO).
#     Config identica a das tres celulas que ja existem (xray, dior, retina), que
#     usaram run_spectrum.sh com patch 256 / batch 16 -- verificado no cabecalho de
#     logs/train_{xray,dior}_encoder_hyper.log. Custo medido: ~29 min/lambda => ~3,9 h
#     por dominio, ~11,7 h nos tres.
#
# Uso:  nohup bash train/run_bloco_c.sh > logs/bloco_c.log 2>&1 &
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DERIV_DIR="models/original_paper/STanH/derivations"
DERIVS=("D02-A040" "D03-A040" "D10-A040" "D11-A040" "D11-A040" "D12-A040" "D13-A040" "D13-A040")
LAMBDAS=("0.003"   "0.00666"  "0.02"     "0.04"     "0.06305"  "0.13"     "0.25"     "0.44014")

say() { echo "[$(date '+%F %T')] $*"; }

# ------------------------------------------------------------------ C1 ---
say "===== C1 -- xray_encoder, 2a semente (batch 8, como o v8 original) ====="
SAVE="models/xray_encoder_v8_runB"
mkdir -p "$SAVE"
for i in "${!DERIVS[@]}"; do
  lam="${LAMBDAS[$i]}"; warm="${DERIVS[$i]}"
  if [ -f "$SAVE/lambda_${lam}_best.pth.tar" ]; then
    say "SKIP C1 lambda=$lam (ja existe)"; continue
  fi
  say "C1 encoder lambda=$lam (warm $warm)"
  python -u train/train_xray_full.py --mode encoder --save_delta \
      --lmbda "$lam" --epochs 20 --batch_size 8 \
      --dataset datasets/xrays --patch_size 256 256 \
      --save_dir "$SAVE" --wandb_project "PIBIC_StanH_XRay_v8_encoder_runB" \
      --init_stanh "$DERIV_DIR/${warm}.pth.tar"
done

say "C1 -- avaliacoes"
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/xrays/test/sample_eval_disjoint --limit 150 --entropy_estimation \
    --out_json results/v8_encoder_runB_on_xray_disjoint_rd.json
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/kodak --limit 24 --entropy_estimation \
    --out_json results/v8_encoder_runB_on_kodak_rd.json
say "C1 CONCLUIDO"

# ------------------------------------------------------------------ C2 ---
# dominio : dataset : amostra de avaliacao : sufixo do JSON alvo
C2="documents:datasets/documents:datasets/documents/test/sample_eval:documents
oct:datasets/oct:datasets/oct/test/sample_eval_disjoint_v2:oct_disjoint_v2
rico:datasets/rico:datasets/rico/test/sample_eval:rico"

while IFS=: read -r dom ds sample tag; do
  [ -z "$dom" ] && continue
  say "===== C2 -- encoder_hyper em $dom ====="
  bash train/run_spectrum.sh "$dom" "$ds" 256 16 encoder_hyper
  MDIR="models/${dom}_encoder_hyper"
  say "C2 $dom -- avaliacoes"
  python -u eval/eval_full.py --models_dir "$MDIR" --dataset "$sample" --limit 150 \
      --entropy_estimation --out_json "results/${dom}_encoder_hyper_on_${tag}_rd.json"
  python -u eval/eval_full.py --models_dir "$MDIR" --dataset datasets/kodak --limit 24 \
      --entropy_estimation --out_json "results/${dom}_encoder_hyper_on_cross_rd.json"
  say "C2 $dom CONCLUIDO"
done <<< "$C2"

say "===== BLOCO C CONCLUIDO ====="
