#!/bin/bash
# E1 tiebreaker on DIOR (aerial): runs after the E3 RICO chain finishes and the GPU
# frees. Trains the DIOR encoder+hyperprior spectrum, then evaluates it on the same
# 150-image DIOR sample + Kodak used by the DIOR spectrum and prints BD-Rate (+CI)
# next to the encoder-only numbers.
#
# Why DIOR: on X-ray the hyperprior removed the encoder's low-rate penalty, on retina
# it did not, but retina's generic curve saturates (~0.4 dB overlap window) so its
# BD-Rate is unreliable. DIOR is rate-limited AND its generic curve does not saturate,
# so it decides whether the mechanism is general or X-ray-specific.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RICO_CHAIN_LOG="logs/chain_e3_rico.log"
SAMPLE="datasets/dior/test/sample_eval"
KODAK="datasets/kodak"
N=150
MDIR="models/dior_encoder_hyper"
TB="results/dior_generic_rd.json"
TJ="results/dior_encoder_hyper_on_dior_rd.json"
CJ="results/dior_encoder_hyper_on_cross_rd.json"
LOG="logs/chain_e1_dior.log"
mkdir -p logs results/plots
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

# 1) wait for the RICO chain (training + eval) to finish
say "E1-DIOR chain armed; waiting for 'E3 RICO DONE' in $RICO_CHAIN_LOG ..."
w=0
until grep -q "E3 RICO DONE" "$RICO_CHAIN_LOG" 2>/dev/null; do
  sleep 120; w=$((w + 120)); [ "$w" -gt 43200 ] && { say "timeout (12h) waiting for RICO chain"; exit 1; }
done
say "RICO chain done."

# 2) make sure the GPU is free before grabbing it
until [ "$(gpu_used)" -lt 2000 ]; do sleep 30; done
sleep 15
say "GPU free (used=$(gpu_used) MiB). Training DIOR encoder+hyperprior spectrum."

# 3) train
bash train/run_spectrum.sh dior datasets/dior 256 16 encoder_hyper > logs/train_dior_encoder_hyper.log 2>&1
say "DIOR encoder_hyper training done -> $MDIR."

# 4) evaluate on target sample + Kodak
python -u eval/eval_full.py --models_dir "$MDIR" --dataset "$SAMPLE" --limit "$N" \
    --entropy_estimation --out_json "$TJ" >> "$LOG" 2>&1
python -u eval/eval_full.py --models_dir "$MDIR" --dataset "$KODAK" --limit 0 \
    --entropy_estimation --out_json "$CJ" >> "$LOG" 2>&1

COMMON=(--target_name dior --target_baseline "$TB" --cross_baseline results/kodak_rd.json
        --target_vtm "" --cross_vtm "" --anchor_level "")
echo "===== E1 BD-Rate: DIOR encoder+hyperprior (N=$N) =====" | tee -a "$LOG"
python -u plots/analyze_finetuned.py --tag dior_encoder_hyper --label "encoder+hyper" \
    --target_json "$TJ" --cross_json "$CJ" "${COMMON[@]}" 2>&1 | tee -a "$LOG"

echo "===== comparacao: DIOR encoder-only (mesma amostra/baselines) =====" | tee -a "$LOG"
python -u plots/analyze_finetuned.py --tag dior_encoder --label "encoder-only" \
    --target_json results/dior_encoder_on_dior_rd.json \
    --cross_json results/dior_encoder_on_cross_rd.json "${COMMON[@]}" 2>&1 | tee -a "$LOG"

say "E1 DIOR DONE -> $TJ , $CJ"
