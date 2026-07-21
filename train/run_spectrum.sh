#!/bin/bash
# Generic adapter-spectrum trainer for ANY domain (replaces per-domain run_*.sh).
# Trains the requested adapter modes (default: encoder then decoder), 8 rate points
# each, delta checkpoints, warm-started from the matching generic derivation.
#
# Usage:
#   bash train/run_spectrum.sh <domain> <dataset_dir> [patch] [batch] [modes...]
# Examples:
#   bash train/run_spectrum.sh documents datasets/documents 256 16
#   bash train/run_spectrum.sh retina    datasets/retina    256 16 encoder decoder full
# Optional env vars:
#   SUFFIX=_replay      appended to save dir + wandb project (variant experiments)
#   EXTRA_ARGS="..."    extra flags passed to train_xray_full.py (e.g. --replay_dataset ...)
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DOMAIN="${1:?domain name required}"
DATASET="${2:?dataset dir required}"
PATCH="${3:-256}"
BATCH="${4:-16}"
shift $(( $# < 4 ? $# : 4 ))
MODES=("$@"); [ ${#MODES[@]} -eq 0 ] && MODES=(encoder decoder)
EPOCHS=20
SUFFIX="${SUFFIX:-}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

DERIV_DIR="models/original_paper/STanH/derivations"
DERIVS=("D02-A040" "D03-A040" "D10-A040" "D11-A040" "D11-A040" "D12-A040" "D13-A040" "D13-A040")
LAMBDAS=("0.003"   "0.00666"  "0.02"     "0.04"     "0.06305"  "0.13"     "0.25"     "0.44014")

echo "[run_spectrum] domain=$DOMAIN dataset=$DATASET patch=$PATCH batch=$BATCH modes=${MODES[*]}"
for mode in "${MODES[@]}"; do
  savedir="models/${DOMAIN}_${mode}${SUFFIX}"; proj="PIBIC_StanH_${DOMAIN}_${mode}${SUFFIX}"
  mkdir -p "$savedir"
  for i in "${!DERIVS[@]}"; do
    echo "===== ${DOMAIN} ${mode}${SUFFIX}  lambda=${LAMBDAS[$i]}  (warm ${DERIVS[$i]}) ====="
    python -u train/train_xray_full.py --mode "$mode" --save_delta \
        --lmbda "${LAMBDAS[$i]}" --epochs "$EPOCHS" --batch_size "$BATCH" \
        --dataset "$DATASET" --patch_size "$PATCH" "$PATCH" \
        --save_dir "$savedir" --wandb_project "$proj" \
        --init_stanh "$DERIV_DIR/${DERIVS[$i]}.pth.tar" "${EXTRA[@]}"
  done
done
echo "ALL DONE -> ${DOMAIN} spectrum (${MODES[*]})"
