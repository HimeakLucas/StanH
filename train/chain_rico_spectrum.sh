#!/bin/bash
# Chains the RICO run: once the retina encoder+hyperprior training finishes AND its chained
# eval releases the GPU, train the RICO (screen content) adapter spectrum (encoder + decoder)
# and then evaluate it (150-image sample + Kodak) with the consolidated spectrum figure.
# Keeps the GPU busy with no manual step.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RETINA_TRAIN_LOG="logs/train_retina_encoder_hyper.log"
RETINA_EVAL_LOG="logs/eval_retina_encoder_hyper.log"
LOG="logs/chain_e3_rico.log"
mkdir -p logs results/plots
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

# 1) wait for retina TRAINING to finish
say "E3 chain armed; waiting for retina training ALL DONE ..."
w=0
until grep -q "ALL DONE" "$RETINA_TRAIN_LOG" 2>/dev/null; do
  sleep 120; w=$((w + 120)); [ "$w" -gt 43200 ] && { say "timeout (12h) waiting retina train"; exit 1; }
done
say "retina training done."

# 2) wait for retina EVAL to finish (bounded), so we don't collide on the GPU
say "waiting for retina chained eval to finish ..."
w=0
until grep -q "E1 RETINA EVAL DONE" "$RETINA_EVAL_LOG" 2>/dev/null; do
  sleep 30; w=$((w + 30)); [ "$w" -gt 1800 ] && { say "eval marker not seen in 30min; proceeding by GPU state"; break; }
done

# 3) make sure the GPU is actually free before we grab it
until [ "$(gpu_used)" -lt 2000 ]; do sleep 30; done
sleep 15
say "GPU free (used=$(gpu_used) MiB). Launching RICO spectrum (encoder + decoder)."

# 4) train RICO adapter spectrum
bash train/run_spectrum.sh rico datasets/rico 256 16 encoder decoder > logs/train_rico_spectrum.log 2>&1
say "RICO spectrum training done -> models/rico_{encoder,decoder}."

# 5) evaluate RICO spectrum (target sample + Kodak) + consolidated figure
bash eval/run_eval_spectrum.sh rico datasets/rico/test/sample_eval datasets/kodak 150 encoder decoder \
    > logs/eval_rico_spectrum.log 2>&1
say "E3 RICO DONE -> results/plots/spectrum_rico.png (log: logs/eval_rico_spectrum.log)"
