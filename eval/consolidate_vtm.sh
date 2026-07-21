#!/bin/bash
# VTM consolidation: re-evaluate VTM/H.266 on the SAME fixed 150-image random sample
# as the STanH spectrum, then regenerate the spectrum figure so the VTM curve matches.
# VTM intra is single-threaded and slow (~minutes/image), so the sample is split into
# chunks and one process runs per (chunk, QP); per-image results are merged at the end.
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
SAMPLE="datasets/xrays/test/sample_consolidation"
CHUNKS=3
QPS=(22 27 32 37 42 47)
WORK="/tmp/vtm_x150"
LOG="train/logs/consolidate_vtm.log"
mkdir -p train/logs "$WORK/out"

# Split the sample into equal chunk dirs (round-robin keeps chunks balanced).
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

echo "[$(date '+%F %T')] VTM consolidation start ($CHUNKS chunks x ${#QPS[@]} QPs in parallel)" | tee "$LOG"
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

# Merge chunk results: concatenate per-image lists per QP, recompute the averages.
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
json.dump(res, open("results/xray_vtm_rd.json", "w"), indent=4)
print("merged -> results/xray_vtm_rd.json")
EOF

echo "[$(date '+%F %T')] regenerating spectrum figure with new VTM curve" | tee -a "$LOG"
python -u plots/plot_spectrum.py 2>&1 | grep -E "saved" | tee -a "$LOG"
cp results/plots/spectrum_xray.png "pibic-paper/fig_spectrum.png"
echo "[$(date '+%F %T')] VTM CONSOLIDATION DONE" | tee -a "$LOG"
