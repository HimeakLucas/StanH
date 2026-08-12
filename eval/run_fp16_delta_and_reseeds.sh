#!/usr/bin/env bash
# Dois lotes independentes:
#   1) converts the encoder deltas to fp16, measures the real on-disk size against the
#      shared anchor and re-evaluates RD to measure the conversion loss;
#   2) retrains two checkpoints with a fresh seed (the lowest-lambda `full`, which came out
#      Pareto-inverted, and the RICO encoder), to separate seed instability from real
#      effect.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S=datasets/xrays/test/sample_eval_disjoint
DERIV=models/original_paper/STanH/derivations

echo "=== inicio: $(date '+%F %T') ==="

# ------------- 1. delta fp16 -------------
echo "--- 1.1: converter deltas do v8 (encoder) para fp16 ---"
python - <<'EOF'
import torch, os, glob, json
src="models/xray_encoder_finetuning_v8"; dst="models/xray_encoder_v8_fp16"
os.makedirs(dst, exist_ok=True); rel={}
for p in sorted(glob.glob(f"{src}/*_best.pth.tar")):
    ck=torch.load(p, map_location="cpu")
    assert ck.get("is_delta"), p
    ck["delta"]={k:v.half() for k,v in ck["delta"].items()}
    q=os.path.join(dst, os.path.basename(p)); torch.save(ck, q)
    a,b=os.path.getsize(p)/1e6, os.path.getsize(q)/1e6
    rel[os.path.basename(p)]={"fp32_MB":round(a,2),"fp16_MB":round(b,2),"razao":round(a/b,2)}
    print(f"  {os.path.basename(p)}: {a:.2f} MB -> {b:.2f} MB  ({a/b:.2f}x)")
json.dump(rel, open("results/a3_delta_sizes.json","w"), indent=2)
anchor=os.path.getsize("models/original_paper/STanH/anchor/0728_last_.pth.tar")/1e6
m=sum(v["fp16_MB"] for v in rel.values())/len(rel)
print(f"  ancora {anchor:.1f} MB | delta fp16 medio {m:.2f} MB | compressao {anchor/m:.1f}x")
EOF

echo "--- 1.2: avaliar o v8 em fp16 na amostra disjunta ---"
python -u eval/eval_full.py --models_dir models/xray_encoder_v8_fp16 \
    --dataset "$S" --limit 150 --entropy_estimation \
    --out_json results/v8_encoder_fp16_on_xray_disjoint_rd.json

# ------------- 2. re-treinos -------------
echo "--- 2.1: re-treino full lambda=0.003 (semente nova, warm-start D02) ---"
python -u train/train_xray_full.py --lmbda 0.003 --epochs 20 --batch_size 8 \
    --dataset datasets/xrays --save_dir models/xray_full_v6_runB \
    --wandb_project PIBIC_StanH_XRay_v6_fullft_runB \
    --init_stanh "$DERIV/D02-A040.pth.tar"

echo "--- 2.2: re-treino rico encoder lambda=0.003 (semente nova) ---"
python -u train/train_xray_full.py --lmbda 0.003 --epochs 20 --batch_size 16 \
    --mode encoder --save_delta --dataset datasets/rico \
    --save_dir models/rico_encoder_runB \
    --wandb_project PIBIC_StanH_RICO_encoder_runB \
    --init_stanh "$DERIV/D02-A040.pth.tar"

echo "=== fim: $(date '+%F %T') ==="
