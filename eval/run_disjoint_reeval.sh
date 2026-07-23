#!/bin/bash
# P1: re-evaluate OCT and retina on LEAKAGE-FREE samples (groups disjoint from the
# training split) and print the new BD-Rates next to the old ones.
#
# Only the TARGET curves are recomputed: cross-domain (Kodak) never enters training,
# so those JSONs stay valid and are reused as-is. The generic baseline must also be
# recomputed on the same disjoint images, otherwise BD-Rate compares curves measured
# on different sets.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N=150
LOG="logs/p1_disjoint_reeval.log"
mkdir -p logs results/plots
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "P1 start: leakage-free re-eval (OCT volume-disjoint, retina patient-disjoint)"

# ---------- OCT ----------
OCT_S="datasets/oct/test/sample_eval_disjoint"
say "OCT generic baseline on disjoint sample"
python -u eval/evaluate_xray.py --dataset "$OCT_S" --limit "$N" --entropy_estimation \
    --out_json results/oct_generic_disjoint_rd.json >> "$LOG" 2>&1
for m in encoder decoder; do
  say "OCT $m on disjoint sample"
  python -u eval/eval_full.py --models_dir "models/oct_${m}" --dataset "$OCT_S" --limit "$N" \
      --entropy_estimation --out_json "results/oct_${m}_on_oct_disjoint_rd.json" >> "$LOG" 2>&1
done

# ---------- RETINA ----------
RET_S="datasets/retina/test/sample_eval_disjoint"
say "retina generic baseline on disjoint sample"
python -u eval/evaluate_xray.py --dataset "$RET_S" --limit "$N" --entropy_estimation \
    --out_json results/retina_generic_disjoint_rd.json >> "$LOG" 2>&1
for m in encoder decoder encoder_hyper; do
  say "retina $m on disjoint sample"
  python -u eval/eval_full.py --models_dir "models/retina_${m}" --dataset "$RET_S" --limit "$N" \
      --entropy_estimation --out_json "results/retina_${m}_on_retina_disjoint_rd.json" >> "$LOG" 2>&1
done

# ---------- comparison ----------
cmp_one() {  # domain tag mode old_target new_target cross old_baseline new_baseline
  local dom=$1 mode=$2 oldt=$3 newt=$4 cross=$5 oldb=$6 newb=$7
  echo "" | tee -a "$LOG"
  echo "########## $dom / $mode ##########" | tee -a "$LOG"
  echo "--- ANTES (amostra com vazamento) ---" | tee -a "$LOG"
  python -u plots/analyze_finetuned.py --tag "${dom}_${mode}_leaky" --label "$mode (leaky)" \
      --target_json "$oldt" --cross_json "$cross" --target_name "$dom" \
      --target_baseline "$oldb" --cross_baseline results/kodak_rd.json \
      --target_vtm "" --cross_vtm "" --anchor_level "" 2>&1 | grep -E "target|Kodak|BD-Rate" | tee -a "$LOG"
  echo "--- DEPOIS (amostra disjunta) ---" | tee -a "$LOG"
  python -u plots/analyze_finetuned.py --tag "${dom}_${mode}_disjoint" --label "$mode (disjoint)" \
      --target_json "$newt" --cross_json "$cross" --target_name "$dom" \
      --target_baseline "$newb" --cross_baseline results/kodak_rd.json \
      --target_vtm "" --cross_vtm "" --anchor_level "" 2>&1 | grep -E "target|Kodak|BD-Rate" | tee -a "$LOG"
}

say "=== COMPARACAO ANTES x DEPOIS ==="
for m in encoder decoder; do
  cmp_one oct "$m" "results/oct_${m}_on_oct_rd.json" "results/oct_${m}_on_oct_disjoint_rd.json" \
      "results/oct_${m}_on_cross_rd.json" results/oct_generic_rd.json results/oct_generic_disjoint_rd.json
done
for m in encoder decoder encoder_hyper; do
  case "$m" in
    encoder_hyper) oldt="results/retina_encoder_hyper_on_retina_rd.json"; cross="results/retina_encoder_hyper_on_cross_rd.json" ;;
    *)             oldt="results/retina_${m}_on_retina_rd.json";           cross="results/retina_${m}_on_cross_rd.json" ;;
  esac
  cmp_one retina "$m" "$oldt" "results/retina_${m}_on_retina_disjoint_rd.json" \
      "$cross" results/retina_generic_rd.json results/retina_generic_disjoint_rd.json
done

say "P1 DONE -> results/*_disjoint_rd.json (log: $LOG)"
