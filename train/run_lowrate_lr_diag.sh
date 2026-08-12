#!/bin/bash
# Diagnostic with a single variable = learning rate. Question: is the low-rate decline
# undertraining (lr too low to climb) or the objective (lambda=0.003 optimum sits
# below the generic curve)?
#
# Config = the baseline enc_hyper lambda=0.003 point (warm-start D02, the good anchor
# that landed at -0.09 dB), changing ONLY lr 1e-5 -> 5e-5. Beta ramp 30->170 kept.
#   climbs above warm-start -> undertraining, fixable.
#   flat / declines         -> objective-driven, scoped reading confirmed.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DERIV="models/original_paper/STanH/derivations"
SAMPLE="datasets/xrays/test/sample_consolidation"
MDIR="models/g1_diag_enchyper_lr5"
LOG="logs/g1_diag.log"
mkdir -p logs results
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "DIAG start: enc_hyper lambda=0.003 warm=D02 lr=5e-5 (5x) beta 30->170"
python -u train/train_xray_full.py --mode encoder_hyper --save_delta \
    --lmbda 0.003 --epochs 20 --batch_size 16 --patch_size 256 256 \
    --lr_backbone 5e-5 --lr_stanh 5e-5 --beta_min 30 \
    --dataset datasets/xrays --save_dir "$MDIR" \
    --wandb_project PIBIC_StanH_g1_diag \
    --init_stanh "$DERIV/D02-A040.pth.tar" >> "$LOG" 2>&1 \
  || { say "DIAG FAILED"; exit 1; }
say "DIAG TRAIN DONE"

python -u eval/eval_full.py --models_dir "$MDIR" --dataset "$SAMPLE" --limit 150 \
    --entropy_estimation --out_json results/g1_diag_enchyper_lr5_rd.json >> "$LOG" 2>&1

python -u plots/lowlam_penalty.py --json results/g1_diag_enchyper_lr5_rd.json \
    --label "enc_hyper lambda=0.003 lr=5e-5 (diag)" 2>&1 | tee -a "$LOG"
say "DIAG DONE -> results/g1_diag_enchyper_lr5_rd.json"
