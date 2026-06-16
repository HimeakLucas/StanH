#!/bin/bash
# VTM consolidation: re-evaluate VTM/H.266 on the SAME fixed 150-image random sample
# as the STanH spectrum, then regenerate the spectrum figure so the VTM curve matches.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
SAMPLE="datasets/xrays/test/sample_consolidation"
N=150
LOG="train/logs/consolidate_vtm.log"
mkdir -p train/logs
echo "[$(date '+%F %T')] VTM consolidation start (N=$N, 6 QPs)" | tee "$LOG"

python -u eval/eval_vtm_xray.py --dataset "$SAMPLE" --limit "$N" \
    --qps 22 27 32 37 42 47 --out_json results/xray_vtm_rd.json 2>&1 \
    | grep -E "QP|Average|Saved" | tee -a "$LOG"

echo "[$(date '+%F %T')] regenerating spectrum figure with new VTM curve" | tee -a "$LOG"
python -u plots/plot_spectrum.py 2>&1 | grep -E "saved" | tee -a "$LOG"
cp results/plots/spectrum_rd.png "IEEE_Conference_Template__1_/fig_spectrum.png"
echo "[$(date '+%F %T')] VTM CONSOLIDATION DONE" | tee -a "$LOG"
