#!/bin/bash
# Chained eval for the X-ray encoder+hyperprior (E1) sweep. Waits for the training
# to print ALL DONE, then evaluates encoder_hyper on the SAME 150-image X-ray sample
# and Kodak (24) used by the rest of the spectrum, and prints its BD-Rate (+CI) next
# to the encoder-only (v8) numbers so we can see whether adding the hyperprior closes
# the encoder's low-rate penalty. All entropy estimation, per-image metrics.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_LOG="logs/train_xray_encoder_hyper.log"
SAMPLE="datasets/xrays/test/sample_consolidation"
KODAK="datasets/kodak"
N=150
MDIR="models/xray_encoder_hyper"
XJ="results/xray_encoder_hyper_on_xray_rd.json"
KJ="results/xray_encoder_hyper_on_kodak_rd.json"
LOG="logs/eval_xray_encoder_hyper.log"
mkdir -p logs results/plots

echo "[$(date '+%F %T')] waiting for training ALL DONE in $TRAIN_LOG ..." | tee "$LOG"
waited=0
until grep -q "ALL DONE" "$TRAIN_LOG" 2>/dev/null; do
  if grep -qE "Traceback|CUDA out of memory" "$TRAIN_LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR detected in training log; aborting eval." | tee -a "$LOG"; exit 1
  fi
  sleep 120; waited=$((waited + 120))
  [ "$waited" -gt 28800 ] && { echo "timeout (8h) waiting for training" | tee -a "$LOG"; exit 1; }
done
sleep 20
echo "[$(date '+%F %T')] training done. Evaluating encoder_hyper." | tee -a "$LOG"

python -u eval/eval_full.py --models_dir "$MDIR" --dataset "$SAMPLE" --limit "$N" \
    --entropy_estimation --out_json "$XJ" 2>&1 | tee -a "$LOG"
python -u eval/eval_full.py --models_dir "$MDIR" --dataset "$KODAK" --limit 0 \
    --entropy_estimation --out_json "$KJ" 2>&1 | tee -a "$LOG"

echo "===== E1 BD-Rate: encoder+hyperprior (N=$N) =====" | tee -a "$LOG"
python -u plots/analyze_finetuned.py --tag encoder_hyper --label "encoder+hyper" \
    --xray_json "$XJ" --kodak_json "$KJ" 2>&1 | tee -a "$LOG"

echo "===== comparação: encoder-only v8 (mesma amostra/baselines) =====" | tee -a "$LOG"
python -u plots/analyze_finetuned.py --tag v8 --label "encoder-only" \
    --xray_json results/v8_encoder_on_xray_rd.json \
    --kodak_json results/v8_encoder_on_kodak_rd.json 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] E1 EVAL DONE -> $XJ , $KJ" | tee -a "$LOG"
