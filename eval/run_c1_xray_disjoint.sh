#!/usr/bin/env bash
# C1 -- reavaliacao do raio-X na amostra disjunta por paciente (train+val, pos-N6).
# Nenhum treino: so reavaliacao dos checkpoints existentes na nova amostra.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S=datasets/xrays/test/sample_eval_disjoint
L=150

echo "=== C1 inicio: $(date '+%F %T') | amostra $S ($(ls $S | wc -l) imgs) ==="

echo "--- [1/6] generica (ancora + derivacoes) ---"
python -u eval/evaluate_xray.py --dataset "$S" --limit $L --entropy_estimation \
    --out_json results/xray_generic_disjoint_rd.json

echo "--- [2/6] v4 quantizador (STanH-only) ---"
python -u eval/eval_finetuned.py --models_dir models/xray_stanh_finetuning_v4_gen \
    --dataset "$S" --limit $L --entropy_estimation \
    --out_json results/v4_finetuned_on_xray_disjoint_rd.json

i=3
for v in "xray_full_finetuning_v6:v6_fullft" \
         "xray_decoder_finetuning_v7:v7_decoder" \
         "xray_encoder_finetuning_v8:v8_encoder" \
         "xray_encoder_hyper:xray_encoder_hyper"; do
  echo "--- [$i/6] ${v%%:*} ---"
  python -u eval/eval_full.py --models_dir "models/${v%%:*}" \
      --dataset "$S" --limit $L --entropy_estimation \
      --out_json "results/${v##*:}_on_xray_disjoint_rd.json"
  i=$((i+1))
done

echo "=== C1 fim: $(date '+%F %T') ==="
