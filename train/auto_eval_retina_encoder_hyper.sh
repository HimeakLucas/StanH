#!/bin/bash
# Chained eval for the retina encoder+hyperprior (E1) sweep. Waits for training to
# finish, evaluates encoder_hyper on the same 150-image retina sample + Kodak (24)
# used by the retina spectrum, and prints BD-Rate (+CI) next to the encoder-only
# numbers, to check whether the X-ray finding (the hyperprior closes the encoder's
# low-rate penalty) generalizes to a second rate-limited domain.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_LOG="logs/train_retina_encoder_hyper.log"
SAMPLE="datasets/retina/test/sample_eval"
KODAK="datasets/kodak"
N=150
MDIR="models/retina_encoder_hyper"
TB="results/retina_generic_rd.json"
XJ="results/retina_encoder_hyper_on_retina_rd.json"
KJ="results/retina_encoder_hyper_on_cross_rd.json"
LOG="logs/eval_retina_encoder_hyper.log"
mkdir -p logs results/plots

echo "[$(date '+%F %T')] waiting for training ALL DONE in $TRAIN_LOG ..." | tee "$LOG"
waited=0
until grep -q "ALL DONE" "$TRAIN_LOG" 2>/dev/null; do
  if grep -qE "Traceback|CUDA out of memory" "$TRAIN_LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR in training log; aborting." | tee -a "$LOG"; exit 1
  fi
  sleep 120; waited=$((waited + 120))
  [ "$waited" -gt 28800 ] && { echo "timeout (8h)" | tee -a "$LOG"; exit 1; }
done
sleep 20
echo "[$(date '+%F %T')] training done. Evaluating encoder_hyper (retina)." | tee -a "$LOG"

python -u eval/eval_full.py --models_dir "$MDIR" --dataset "$SAMPLE" --limit "$N" \
    --entropy_estimation --out_json "$XJ" 2>&1 | tee -a "$LOG"
python -u eval/eval_full.py --models_dir "$MDIR" --dataset "$KODAK" --limit 0 \
    --entropy_estimation --out_json "$KJ" 2>&1 | tee -a "$LOG"

COMMON=(--target_name retina --target_baseline "$TB" --cross_baseline results/kodak_rd.json
        --target_vtm "" --cross_vtm "" --anchor_level "")
echo "===== E1 BD-Rate: retina encoder+hyperprior (N=$N) =====" | tee -a "$LOG"
python -u plots/analyze_finetuned.py --tag retina_encoder_hyper --label "encoder+hyper" \
    --target_json "$XJ" --cross_json "$KJ" "${COMMON[@]}" 2>&1 | tee -a "$LOG"

echo "===== comparacao: retina encoder-only (mesma amostra/baselines) =====" | tee -a "$LOG"
python -u plots/analyze_finetuned.py --tag retina_encoder --label "encoder-only" \
    --target_json results/retina_encoder_on_retina_rd.json \
    --cross_json results/retina_encoder_on_cross_rd.json "${COMMON[@]}" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] E1 RETINA EVAL DONE -> $XJ , $KJ" | tee -a "$LOG"
