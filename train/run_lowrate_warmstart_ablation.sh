#!/bin/bash
# Isolates the cause of the low-rate penalty of the X-ray encoder and
# encoder+hyperprior adapters by separating the two variables that earlier runs
# confounded: the warm-start derivation and beta_min.
#
#   baseline (already trained)  lambda 0.003->D02, 0.00666->D03, beta_min 30
#   arm A                       warm-start shifted down (D01/D02), beta_min 30
#   arm B                       same shift,                        beta_min 100
#
# Reading (arm A vs baseline isolates the warm-start; A vs B isolates beta_min).
# Everything else matches the baseline run: 20 epochs, batch 16, patch 256, lr 1e-5.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DERIV="models/original_paper/STanH/derivations"
SAMPLE="datasets/xrays/test/sample_consolidation"
N=150
LOG="logs/g1_lowlam.log"
mkdir -p logs results
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "G1 start: 8 trainings (2 modes x 2 lambdas x 2 arms)"

for mode in encoder encoder_hyper; do
  for arm in A B; do
    bmin=30; [ "$arm" = B ] && bmin=100
    savedir="models/g1_xray_${mode}_arm${arm}"
    for pair in "0.003 D01-A040" "0.00666 D02-A040"; do
      set -- $pair; lam=$1; warm=$2
      say "train ${mode} arm${arm} lambda=${lam} warm=${warm} beta_min=${bmin}"
      python -u train/train_xray_full.py --mode "$mode" --save_delta \
          --lmbda "$lam" --epochs 20 --batch_size 16 --patch_size 256 256 \
          --beta_min "$bmin" --dataset datasets/xrays \
          --save_dir "$savedir" \
          --wandb_project "PIBIC_StanH_g1_lowlam" \
          --init_stanh "$DERIV/${warm}.pth.tar" >> "$LOG" 2>&1 \
        || { say "FAILED: ${mode} arm${arm} lambda=${lam}"; exit 1; }
    done
  done
done
say "G1 TRAIN DONE"

# ---------- eval on the same 150-image sample as the existing curves ----------
for mode in encoder encoder_hyper; do
  for arm in A B; do
    say "eval ${mode} arm${arm}"
    python -u eval/eval_full.py --models_dir "models/g1_xray_${mode}_arm${arm}" \
        --dataset "$SAMPLE" --limit "$N" --entropy_estimation \
        --out_json "results/g1_xray_${mode}_arm${arm}_rd.json" >> "$LOG" 2>&1
  done
done

# ---------- penalty table (dB below the generic curve at matched bpp) ----------
{
  echo ""
  echo "########## G1: penalidade de baixa taxa (dB vs curva generica) ##########"
  for mode in encoder encoder_hyper; do
    for arm in A B; do
      python -u plots/lowlam_penalty.py --json "results/g1_xray_${mode}_arm${arm}_rd.json" \
          --label "${mode} arm${arm}"
    done
  done
  echo ""
  echo "Referencia (baseline ja medido): encoder lambda=0.003 -1.36 dB | enc_hyper -0.09 dB"
  echo "Criterio (Portao A): <=0.3 dB sumiu | >=1.0 dB persiste | intermediario = ambiguo = NAO"
} 2>&1 | tee -a "$LOG"

say "G1 DONE -> results/g1_xray_*_rd.json (log: $LOG)"
