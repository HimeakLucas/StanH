#!/bin/bash
# Waits for the full fine-tune sweep to finish, then evaluates the
# full models on X-ray + Kodak and runs the BD-Rate analysis + plot.
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"

TAG="v5"
SWEEP_LOG="train/logs/v5_resilient.log"
MODELS_DIR="models/xray_full_finetuning"
XRAY_JSON="results/${TAG}_fullft_on_xray_rd.json"
KODAK_JSON="results/${TAG}_fullft_on_kodak_rd.json"
EVAL_LOG="train/logs/${TAG}_eval.log"
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

# Compare against generic; label makes the curve clear in the plot.
python -u plots/analyze_finetuned.py --tag "$TAG" --label "STanH FULL fine-tune (X-ray)" \
    --xray_json "$XRAY_JSON" --kodak_json "$KODAK_JSON" 2>&1 | tee -a "$EVAL_LOG"

echo "[$(date '+%F %T')] AUTO-EVAL DONE. Plot: results/plots/${TAG}_rd_comparison.png" | tee -a "$EVAL_LOG"
