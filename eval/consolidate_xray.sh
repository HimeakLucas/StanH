#!/bin/bash
# Consolidation round: re-evaluate the whole STanH adapter spectrum on a FIXED
# random sample of 150 X-ray test images (seed 42, see sample_consolidation_manifest.txt)
# AND on Kodak (24 images), saving per-image metrics so BD-Rates come with
# bootstrapped 95% CIs. All entropy estimation (gap vs real coding <0.02%).
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAMPLE="datasets/xrays/test/sample_consolidation"
KODAK="datasets/kodak"
N=150
LOG="train/logs/consolidate_xray.log"
mkdir -p train/logs results/plots
echo "[$(date '+%F %T')] CONSOLIDATION start (N=$N images + Kodak, per-image metrics)" | tee "$LOG"

run() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; "$@" 2>&1 | grep -E "Average|Level|->|BD-Rate|raio|Kodak|delta|Error|error" | tee -a "$LOG"; }

# 1) generic STanH baselines (target sample + Kodak)
run python -u eval/evaluate_xray.py --dataset "$SAMPLE" --limit "$N" --entropy_estimation \
    --out_json results/xray_stanh_rd.json
run python -u eval/evaluate_xray.py --dataset "$KODAK" --limit 0 --entropy_estimation \
    --out_json results/kodak_rd.json
# 2) v4 quantizer-only
run python -u eval/eval_finetuned.py --models_dir models/xray_stanh_finetuning_v4_gen \
    --dataset "$SAMPLE" --limit "$N" --entropy_estimation \
    --out_json results/v4_finetuned_on_xray_rd.json
run python -u eval/eval_finetuned.py --models_dir models/xray_stanh_finetuning_v4_gen \
    --dataset "$KODAK" --limit 0 --entropy_estimation \
    --out_json results/v4_finetuned_on_kodak_rd.json
# 3) v6 full / v7 decoder / v8 encoder (full-model evals; v8 = anchor + delta)
for v in "xray_full_finetuning_v6:v6_fullft" \
         "xray_decoder_finetuning_v7:v7_decoder" \
         "xray_encoder_finetuning_v8:v8_encoder"; do
    mdir="models/${v%%:*}"; tag="${v##*:}"
    run python -u eval/eval_full.py --models_dir "$mdir" --dataset "$SAMPLE" --limit "$N" \
        --entropy_estimation --out_json "results/${tag}_on_xray_rd.json"
    run python -u eval/eval_full.py --models_dir "$mdir" --dataset "$KODAK" --limit 0 \
        --entropy_estimation --out_json "results/${tag}_on_kodak_rd.json"
done

echo "[$(date '+%F %T')] regenerating figure + BD-Rates (PCHIP + bootstrap CI)" | tee -a "$LOG"
python -u plots/plot_spectrum.py 2>&1 | grep -E "saved" | tee -a "$LOG"
cp results/plots/spectrum_xray.png "pibic-paper/fig_spectrum.png"

for v in "v4:results/v4_finetuned_on_xray_rd.json:results/v4_finetuned_on_kodak_rd.json" \
         "v6:results/v6_fullft_on_xray_rd.json:results/v6_fullft_on_kodak_rd.json" \
         "v7:results/v7_decoder_on_xray_rd.json:results/v7_decoder_on_kodak_rd.json" \
         "v8:results/v8_encoder_on_xray_rd.json:results/v8_encoder_on_kodak_rd.json"; do
    tag="${v%%:*}"; rest="${v#*:}"; xj="${rest%%:*}"; kj="${rest#*:}"
    echo "===== BD-Rate $tag (sample N=$N) =====" | tee -a "$LOG"
    python -u plots/analyze_finetuned.py --tag "$tag" --label "$tag" \
        --xray_json "$xj" --kodak_json "$kj" 2>&1 | grep -E "BD-Rate|X-ray|Kodak|bpp " | tee -a "$LOG"
done

echo "[$(date '+%F %T')] CONSOLIDATION DONE" | tee -a "$LOG"
