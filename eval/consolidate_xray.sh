#!/bin/bash
# Consolidation round: re-evaluate the whole STanH adapter spectrum on a FIXED
# random sample of 150 X-ray test images (seed 42, see sample_consolidation_manifest.txt),
# replacing the earlier 24-first-image numbers. All entropy estimation (gap vs real
# coding shown to be <0.02%). Kodak is untouched (it is exactly 24 images by design).
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAMPLE="datasets/xrays/test/sample_consolidation"
N=150
LOG="train/logs/consolidate_xray.log"
mkdir -p train/logs results/plots
echo "[$(date '+%F %T')] CONSOLIDATION start (N=$N images)" | tee "$LOG"

run() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; "$@" 2>&1 | grep -E "Average|Level|->|BD-Rate|raio|Kodak|delta|Error|error" | tee -a "$LOG"; }

# 1) generic STanH baseline
run python -u eval/evaluate_xray.py --dataset "$SAMPLE" --limit "$N" --entropy_estimation \
    --out_json results/xray_stanh_rd.json
# 2) v4 quantizer-only
run python -u eval/eval_finetuned.py --models_dir models/xray_stanh_finetuning_v4_gen \
    --dataset "$SAMPLE" --limit "$N" --entropy_estimation \
    --out_json results/v4_finetuned_on_xray_rd.json
# 3) v6 full
run python -u eval/eval_full.py --models_dir models/xray_full_finetuning_v6 \
    --dataset "$SAMPLE" --limit "$N" --entropy_estimation \
    --out_json results/v6_fullft_on_xray_rd.json
# 4) v7 decoder
run python -u eval/eval_full.py --models_dir models/xray_decoder_finetuning_v7 \
    --dataset "$SAMPLE" --limit "$N" --entropy_estimation \
    --out_json results/v7_decoder_on_xray_rd.json
# 5) v8 encoder (delta checkpoints -> anchor + delta)
run python -u eval/eval_full.py --models_dir models/xray_encoder_finetuning_v8 \
    --dataset "$SAMPLE" --limit "$N" --entropy_estimation \
    --out_json results/v8_encoder_on_xray_rd.json

echo "[$(date '+%F %T')] regenerating figure + BD-Rates" | tee -a "$LOG"
python -u plots/plot_spectrum.py 2>&1 | grep -E "saved" | tee -a "$LOG"
cp results/plots/spectrum_rd.png "IEEE_Conference_Template__1_/fig_spectrum.png"

# BD-Rates (xray re-evaluated on sample; kodak unchanged) for each variant
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
