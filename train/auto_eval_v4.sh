#!/bin/bash
# Waits for the v4 sweep to finish ("ALL DONE" in its resilient log), then
# evaluates the v4 derivations on X-ray (target) and Kodak (cross-domain) and
# runs the BD-Rate analysis + comparison plot. Safe to launch right after the
# training: it only starts evaluating once training is truly done and the GPU
# is free. Generic and reusable per round via the variables below.
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"

TAG="v4"
SWEEP_LOG="train/logs/v4_resilient.log"
MODELS_DIR="models/xray_stanh_finetuning_v4_gen"
XRAY_JSON="results/${TAG}_finetuned_on_xray_rd.json"
KODAK_JSON="results/${TAG}_finetuned_on_kodak_rd.json"
EVAL_LOG="train/logs/${TAG}_eval.log"
LIMIT=24

echo "[$(date '+%F %T')] auto-eval waiting for '$SWEEP_LOG' to print ALL DONE..." | tee "$EVAL_LOG"
until grep -q "ALL DONE" "$SWEEP_LOG" 2>/dev/null; do
    sleep 120
done
# let the training process release the GPU
sleep 30
echo "[$(date '+%F %T')] training finished. Starting evaluation." | tee -a "$EVAL_LOG"

echo "[$(date '+%F %T')] eval on X-ray (target)..." | tee -a "$EVAL_LOG"
python -u eval/eval_finetuned.py --models_dir "$MODELS_DIR" \
    --dataset datasets/xrays/test/data --limit "$LIMIT" --entropy_estimation \
    --out_json "$XRAY_JSON" 2>&1 | tee -a "$EVAL_LOG"

echo "[$(date '+%F %T')] eval on Kodak (cross-domain)..." | tee -a "$EVAL_LOG"
python -u eval/eval_finetuned.py --models_dir "$MODELS_DIR" \
    --dataset datasets/kodak --limit "$LIMIT" --entropy_estimation \
    --out_json "$KODAK_JSON" 2>&1 | tee -a "$EVAL_LOG"

echo "[$(date '+%F %T')] BD-Rate analysis + plot..." | tee -a "$EVAL_LOG"
python -u plots/analyze_finetuned.py --tag "$TAG" \
    --xray_json "$XRAY_JSON" --kodak_json "$KODAK_JSON" 2>&1 | tee -a "$EVAL_LOG"

echo "[$(date '+%F %T')] AUTO-EVAL DONE. Plot: results/plots/${TAG}_rd_comparison.png" | tee -a "$EVAL_LOG"
