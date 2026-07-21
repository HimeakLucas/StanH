#!/bin/bash
# Finalize retina once you've manually downloaded the Kaggle zip.
# Usage: bash scripts/finalize_retina_from_zip.sh [path/to/zip]
# Default zip path: scripts/staging/retina.zip
set -u
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
cd "$(dirname "$0")/.."
ZIP="${1:-scripts/staging/retina.zip}"
RAW="scripts/staging/retina_raw"
LOG="train/logs/finalize_retina_zip.log"

if [ ! -f "$ZIP" ]; then echo "ZIP não encontrado: $ZIP" | tee "$LOG"; exit 1; fi
echo "[$(date '+%F %T')] unzip $ZIP" | tee "$LOG"
rm -rf "$RAW"; mkdir -p "$RAW"
unzip -q "$ZIP" -d "$RAW" 2>&1 | tail -2 | tee -a "$LOG"

echo "[$(date '+%F %T')] checando resolução (amostra)" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import glob, numpy as np
from PIL import Image
fs = glob.glob("scripts/staging/retina_raw/**/*", recursive=True)
fs = [f for f in fs if f.lower().endswith((".jpg",".jpeg",".png"))][:200]
ws=[Image.open(f).size[0] for f in fs]; hs=[Image.open(f).size[1] for f in fs]
print(f"imgs amostradas={len(fs)} | W med={int(np.median(ws))} min={min(ws)} | H med={int(np.median(hs))} min={min(hs)}")
print("ALERTA: imagens <400px (mesmo problema dos minúsculos)" if np.median(ws)<400 else "resolução OK (>=400px)")
PY

echo "[$(date '+%F %T')] split 8000/1000/2000" | tee -a "$LOG"
python -u scripts/prepare_domain_split.py --src "$RAW" --name retina \
    --n_train 8000 --n_val 1000 --n_test 2000 --seed 42 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] FINALIZE RETINA (zip) DONE" | tee -a "$LOG"
