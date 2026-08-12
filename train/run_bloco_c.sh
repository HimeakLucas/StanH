#!/bin/bash
# Two blocks in one run.
#
# 1) Seed replica on a SECOND cell (xray_encoder, 8 lambdas). The instability claim rests on
#    the replica of a single cell (`full`), and checkpoint selection was measured NOT to
#    explain it, which makes an independent second cell the evidence that matters.
#
#    Config is MEASURED, not assumed: the original v8 run used BATCH_SIZE=8
#    (train/run_xray_encoder_v8.sh), not 16. To be a REPLICA rather than another experiment,
#    this run repeats batch 8, 20 epochs, same warm-starts and same dataset. Measured cost
#    from the v8 timestamps: ~62 min/lambda => ~8.3 h.
#
#    The trainer exposes no --seed and never calls manual_seed: the "new seed" is the
#    nondeterminism of data order and cuDNN. The warm-start is FIXED, so this variance
#    excludes initialization by construction.
#
# 2) encoder_hyper on the three remaining domains (documents, OCT, RICO). Config identical to
#    the three existing cells (xray, dior, retina), which used run_spectrum.sh with patch 256
#    / batch 16. Measured cost: ~29 min/lambda => ~3.9 h per domain, ~11.7 h for the three.
#
# Usage:  nohup bash train/run_bloco_c.sh > logs/bloco_c.log 2>&1 &
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DERIV_DIR="models/original_paper/STanH/derivations"
DERIVS=("D02-A040" "D03-A040" "D10-A040" "D11-A040" "D11-A040" "D12-A040" "D13-A040" "D13-A040")
LAMBDAS=("0.003"   "0.00666"  "0.02"     "0.04"     "0.06305"  "0.13"     "0.25"     "0.44014")

say() { echo "[$(date '+%F %T')] $*"; }

# ------------------------------------------------------------------ C1 ---
say "===== C1 -- xray_encoder, 2a semente (batch 8, como o v8 original) ====="
SAVE="models/xray_encoder_v8_runB"
mkdir -p "$SAVE"
for i in "${!DERIVS[@]}"; do
  lam="${LAMBDAS[$i]}"; warm="${DERIVS[$i]}"
  if [ -f "$SAVE/lambda_${lam}_best.pth.tar" ]; then
    say "SKIP C1 lambda=$lam (ja existe)"; continue
  fi
  say "C1 encoder lambda=$lam (warm $warm)"
  python -u train/train_xray_full.py --mode encoder --save_delta \
      --lmbda "$lam" --epochs 20 --batch_size 8 \
      --dataset datasets/xrays --patch_size 256 256 \
      --save_dir "$SAVE" --wandb_project "PIBIC_StanH_XRay_v8_encoder_runB" \
      --init_stanh "$DERIV_DIR/${warm}.pth.tar"
done

say "C1 -- avaliacoes"
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/xrays/test/sample_eval_disjoint --limit 150 --entropy_estimation \
    --out_json results/v8_encoder_runB_on_xray_disjoint_rd.json
python -u eval/eval_full.py --models_dir "$SAVE" \
    --dataset datasets/kodak --limit 24 --entropy_estimation \
    --out_json results/v8_encoder_runB_on_kodak_rd.json
say "C1 CONCLUIDO"

# ------------------------------------------------------------------ C2 ---
# domain : dataset : eval sample : target JSON suffix
C2="documents:datasets/documents:datasets/documents/test/sample_eval:documents
oct:datasets/oct:datasets/oct/test/sample_eval_disjoint_v2:oct_disjoint_v2
rico:datasets/rico:datasets/rico/test/sample_eval:rico"

while IFS=: read -r dom ds sample tag; do
  [ -z "$dom" ] && continue
  say "===== C2 -- encoder_hyper em $dom ====="
  bash train/run_spectrum.sh "$dom" "$ds" 256 16 encoder_hyper
  MDIR="models/${dom}_encoder_hyper"
  say "C2 $dom -- avaliacoes"
  python -u eval/eval_full.py --models_dir "$MDIR" --dataset "$sample" --limit 150 \
      --entropy_estimation --out_json "results/${dom}_encoder_hyper_on_${tag}_rd.json"
  python -u eval/eval_full.py --models_dir "$MDIR" --dataset datasets/kodak --limit 24 \
      --entropy_estimation --out_json "results/${dom}_encoder_hyper_on_cross_rd.json"
  say "C2 $dom CONCLUIDO"
done <<< "$C2"

say "===== BLOCO C CONCLUIDO ====="
