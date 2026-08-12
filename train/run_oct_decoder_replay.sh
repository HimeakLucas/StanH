#!/bin/bash
# Replay on OCT: does the forgetting antidote generalize to the WORST collapse of the study?
#
# The claim is that decoder forgetting is not intrinsic and that a batch of natural images in
# the loss removes it. The evidence covers ONE domain (documents, -6.01 -> -0.02 dB with
# alpha=0.8). OCT is the worst collapse measured (-7.19 dB), where the claim is strongest and
# least tested.
#
# 3 lambdas (the same as the color control), alpha=0.8, replay on DIV2K.
# PROJECT INVARIANT: replay NEVER uses Kodak -- Kodak is cross-domain evaluation only, and
# using it as replay would leak straight into the metric that reports forgetting.
#
# Hyperparameters identical to train/run_spectrum.sh (EPOCHS=20, batch 16, patch 256,
# --save_delta, same lambda->derivation map), so the cell is comparable to the existing
# `oct_decoder` without replay.
#
# Usage:  nohup bash train/run_oct_decoder_replay.sh > logs/b2_oct_replay.log 2>&1 &
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

# Target: patient-disjoint sample v2 (the older one used the wrong group key)
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/oct/test/sample_eval_disjoint_v2 --limit 150 --entropy_estimation \
    --out_json results/oct_decoder_replay_on_oct_disjoint_v2_rd.json

# Cross: Kodak, 24 imagens
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/kodak --limit 24 --entropy_estimation \
    --out_json results/oct_decoder_replay_on_cross_rd.json

echo "[$(date '+%F %T')] B2 CONCLUIDO"
