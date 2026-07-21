#!/bin/bash
# Documents domain spectrum: encoder-only then decoder-only adaptation (the two most
# informative variants), 8 rate points each, delta checkpoints. Grayscale ~1000px,
# patch 256. Warm-start STanH from the matching generic derivation, deterministic beta.
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPOCHS=20; BATCH=16; DATASET="datasets/documents"; PATCH=256  # batch 16 on RTX 4090 (24GB)
DERIV_DIR="models/original_paper/STanH/derivations"
DERIVS=("D02-A040" "D03-A040" "D10-A040" "D11-A040" "D11-A040" "D12-A040" "D13-A040" "D13-A040")
LAMBDAS=("0.003"   "0.00666"  "0.02"     "0.04"     "0.06305"  "0.13"     "0.25"     "0.44014")

run_mode() {
  local mode="$1" savedir="$2" proj="$3"
  mkdir -p "$savedir"
  for i in "${!DERIVS[@]}"; do
    echo "===== DOCUMENTS $mode  lambda=${LAMBDAS[$i]}  (warm ${DERIVS[$i]}) ====="
    python -u train/train_xray_full.py --mode "$mode" --save_delta \
        --lmbda "${LAMBDAS[$i]}" --epochs "$EPOCHS" --batch_size "$BATCH" \
        --dataset "$DATASET" --patch_size "$PATCH" "$PATCH" \
        --save_dir "$savedir" --wandb_project "$proj" \
        --init_stanh "$DERIV_DIR/${DERIVS[$i]}.pth.tar"
  done
}

run_mode encoder models/documents_encoder PIBIC_StanH_Docs_encoder
run_mode decoder models/documents_decoder PIBIC_StanH_Docs_decoder
echo "ALL DONE -> documents spectrum"
