#!/bin/bash
# Replay on the X-RAY decoder: the antidote reaches the domain that carries the main table,
# the full spectrum and the headline claim.
#
# Replay was reported on documents (n=1), then OCT (n=2). X-ray was missing, and without it
# the antidote looks tested only where it worked.
#
# 3 lambdas (the same as the OCT run), alpha=0.8, replay on DIV2K.
# PROJECT INVARIANT: replay NEVER uses Kodak -- Kodak is cross-domain evaluation only.
#
# HYPERPARAMETERS: identical to `v7_decoder` WITHOUT replay, which is the comparison cell --
# and NOT to the OCT runner. Checked at the source (train/run_xray_decoder_v7.sh:7): BATCH=8
# (OCT used 16), patch 256 (trainer default), 20 epochs, lr 1e-5, same lambda->warm-start map.
# No --save_delta, again to match v7: the delta is stored in fp16 (train_xray_full.py:245)
# and fp16 alone moves the aggregate BD by ~0.5 p.p. v7 has full fp32 checkpoints, so the
# replay cell must have them too.
#
# Usage:  nohup bash train/run_xray_decoder_replay.sh > logs/e2_xray_replay.log 2>&1 &
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

# Target: patient-disjoint sample (150 imgs / 144 patients)
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/xrays/test/sample_eval_disjoint --limit 150 --entropy_estimation \
    --out_json results/_exp_01ago/xray_decoder_replay_on_xray_disjoint_rd.json

# Cross: Kodak, 24 imagens
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/kodak --limit 24 --entropy_estimation \
    --out_json results/_exp_01ago/xray_decoder_replay_on_cross_rd.json

echo "[$(date '+%F %T')] E2 CONCLUIDO"
