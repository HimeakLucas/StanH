#!/bin/bash
# Generic eval+analysis for an adapter spectrum on ANY domain (replaces auto_eval_*.sh).
# For each trained mode dir (models/<domain>_<mode>): evaluates on the TARGET test set
# and on the CROSS domain (default Kodak), then runs BD-Rate analysis per mode and the
# consolidated spectrum figure. The generic baseline on the target is computed if missing.
#
# Usage:
#   bash eval/run_eval_spectrum.sh <domain> <target_test_dir> [cross_test_dir] [limit] [modes...]
# Example:
#   bash eval/run_eval_spectrum.sh documents datasets/documents/test/data datasets/kodak 150 encoder decoder
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DOMAIN="${1:?domain required}"
TARGET_DIR="${2:?target test dir required}"
CROSS_DIR="${3:-datasets/kodak}"
LIMIT="${4:-150}"
shift $(( $# < 4 ? $# : 4 ))
MODES=("$@"); [ ${#MODES[@]} -eq 0 ] && MODES=(encoder decoder)

GEN_BASELINE="results/${DOMAIN}_generic_rd.json"
CROSS_BASELINE="results/kodak_rd.json"   # generic STanH on Kodak (already computed)

# 1) generic baseline on the target domain (authors' derivations), if not present
if [ ! -f "$GEN_BASELINE" ]; then
  echo "[eval] computing generic baseline on $DOMAIN ..."
  python -u eval/evaluate_xray.py --dataset "$TARGET_DIR" --limit "$LIMIT" --entropy_estimation \
      --out_json "$GEN_BASELINE"
fi

# 2) per-mode eval on target + cross
VARIANT_ARGS=()
declare -A COLOR=( [encoder]="tab:red" [decoder]="tab:orange" [full]="black" [quantizer]="tab:purple" )
declare -A MARK=(  [encoder]="v-"      [decoder]="D-"          [full]="*-"     [quantizer]="s-" )
for mode in "${MODES[@]}"; do
  mdir="models/${DOMAIN}_${mode}"
  [ -d "$mdir" ] || { echo "[eval] skip $mode (no $mdir)"; continue; }
  tj="results/${DOMAIN}_${mode}_on_${DOMAIN}_rd.json"
  cj="results/${DOMAIN}_${mode}_on_cross_rd.json"
  echo "[eval] $mode on target ($DOMAIN)";  python -u eval/eval_full.py --models_dir "$mdir" --dataset "$TARGET_DIR" --limit "$LIMIT" --entropy_estimation --out_json "$tj"
  echo "[eval] $mode on cross ($CROSS_DIR)"; python -u eval/eval_full.py --models_dir "$mdir" --dataset "$CROSS_DIR"  --limit "$LIMIT" --entropy_estimation --out_json "$cj"
  python -u plots/analyze_finetuned.py --tag "${DOMAIN}_${mode}" --target_name "$DOMAIN" \
      --target_json "$tj" --cross_json "$cj" \
      --target_baseline "$GEN_BASELINE" --cross_baseline "$CROSS_BASELINE" \
      --target_vtm "" --cross_vtm "" --anchor_level ""
  VARIANT_ARGS+=( --variant "${mode}:${tj}:${cj}:${COLOR[$mode]:-tab:gray}:${MARK[$mode]:-o-}" )
done

# 3) consolidated spectrum figure
python -u plots/plot_spectrum.py --domain "$DOMAIN" --target_name "$DOMAIN" \
    --target_baseline "$GEN_BASELINE" --cross_baseline "$CROSS_BASELINE" \
    --target_vtm "" --cross_vtm "" "${VARIANT_ARGS[@]}"
echo "[eval] DONE -> results/plots/spectrum_${DOMAIN}.png"
