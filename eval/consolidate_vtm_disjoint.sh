#!/bin/bash
# VTM/H.266 on the patient-DISJOINT x-ray sample.
# Same protocol as consolidate_vtm.sh: chunks x QPs in parallel, per-image merge.
# Differences: disjoint sample, its own JSON, and it does NOT regenerate the figure, which
# still mixes curves from an earlier sample and has to be rebuilt separately.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
SAMPLE="datasets/xrays/test/sample_eval_disjoint"
CHUNKS=3
QPS=(22 27 32 37 42 47)
WORK="/tmp/vtm_x150_disjoint"
LOG="logs/consolidate_vtm_disjoint.log"
OUT="results/xray_vtm_disjoint_rd.json"
mkdir -p logs "$WORK/out"

python - <<EOF
import os, glob
files = sorted(glob.glob("$SAMPLE/*.png"))
for c in range($CHUNKS):
    d = f"$WORK/chunk{c}"
    os.makedirs(d, exist_ok=True)
    for f in files[c::$CHUNKS]:
        dst = os.path.join(d, os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(f), dst)
print(f"{len(files)} images -> $CHUNKS chunks")
EOF

echo "[$(date '+%F %T')] VTM disjunto start ($CHUNKS chunks x ${#QPS[@]} QPs)" | tee "$LOG"
pids=()
for qp in "${QPS[@]}"; do
  for c in $(seq 0 $((CHUNKS-1))); do
    python -u eval/eval_vtm_xray.py --dataset "$WORK/chunk$c" --limit 0 \
        --qps "$qp" --out_json "$WORK/out/qp${qp}_c${c}.json" \
        > "$WORK/out/qp${qp}_c${c}.log" 2>&1 &
    pids+=($!)
  done
done
wait "${pids[@]}"

python - <<EOF
import json, glob
res = {"qp": [], "bpp": [], "psnr": [], "files": [], "per_image": {}}
for qp in (${QPS[@]/%/,}):
    bpp, psnr, files = [], [], []
    for p in sorted(glob.glob(f"$WORK/out/qp{qp}_c*.json")):
        d = json.load(open(p))
        pi = d["per_image"][str(qp)]
        bpp += pi["bpp"]; psnr += pi["psnr"]; files += d["files"]
    assert len(files) == 150, f"QP {qp}: {len(files)} images (expected 150)"
    order = sorted(range(len(files)), key=lambda i: files[i])
    res["per_image"][str(qp)] = {"bpp": [bpp[i] for i in order], "psnr": [psnr[i] for i in order]}
    res["files"] = [files[i] for i in order]
    res["qp"].append(qp); res["bpp"].append(sum(bpp)/len(bpp)); res["psnr"].append(sum(psnr)/len(psnr))
    print(f"QP {qp}: bpp {res['bpp'][-1]:.4f}, psnr {res['psnr'][-1]:.2f}")
json.dump(res, open("$OUT", "w"), indent=4)
print("merged -> $OUT")
EOF

echo "[$(date '+%F %T')] VTM DISJUNTO DONE" | tee -a "$LOG"
