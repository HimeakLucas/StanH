#!/bin/bash
# Waits for the v8 encoder sweep to finish, then evaluates on X-ray + Kodak and
# runs the BD-Rate analysis + plot (encoder-only adaptation vs generic).
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"

TAG="v8"
SWEEP_LOG="train/logs/v8_resilient.log"
MODELS_DIR="models/xray_encoder_finetuning_v8"
XRAY_JSON="results/${TAG}_encoder_on_xray_rd.json"
KODAK_JSON="results/${TAG}_encoder_on_kodak_rd.json"
EVAL_LOG="train/logs/v8_eval.log"
LIMIT=24

echo "[$(date '+%F %T')] auto-eval waiting for '$SWEEP_LOG' to print ALL DONE..." | tee "$EVAL_LOG"
until grep -q "ALL DONE" "$SWEEP_LOG" 2>/dev/null; do sleep 120; done
sleep 30
echo "[$(date '+%F %T')] training finished. Evaluating." | tee -a "$EVAL_LOG"

python -u eval/eval_full.py --models_dir "$MODELS_DIR" \
    --dataset datasets/xrays/test/data --limit "$LIMIT" --entropy_estimation \
    --out_json "$XRAY_JSON" 2>&1 | tee -a "$EVAL_LOG"
python -u eval/eval_full.py --models_dir "$MODELS_DIR" \
    --dataset datasets/kodak --limit "$LIMIT" --entropy_estimation \
    --out_json "$KODAK_JSON" 2>&1 | tee -a "$EVAL_LOG"

python -u plots/analyze_finetuned.py --tag "$TAG" --label "STanH encoder+quant fine-tune (X-ray)" \
    --xray_json "$XRAY_JSON" --kodak_json "$KODAK_JSON" 2>&1 | tee -a "$EVAL_LOG"

echo "[$(date '+%F %T')] AUTO-EVAL DONE. Plot: results/plots/${TAG}_rd_comparison.png" | tee -a "$EVAL_LOG"
