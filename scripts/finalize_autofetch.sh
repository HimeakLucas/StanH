#!/bin/bash
# Waits for the two auth-free fetches (documents stream, histopath zip) and builds
# the 8000/val/test ImageFolder split for each.
set -u
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
cd "$(dirname "$0")/.."
LOG="train/logs/finalize_autofetch.log"
echo "[$(date '+%F %T')] finalize start" | tee "$LOG"

# --- Documents ---
until grep -q "DONE:" train/logs/fetch_documents.log 2>/dev/null; do sleep 20; done
echo "[$(date '+%F %T')] documents fetched -> splitting" | tee -a "$LOG"
python -u scripts/prepare_domain_split.py --src scripts/staging/documents --name documents \
    --n_train 8000 --n_val 1000 --n_test 2000 --seed 42 2>&1 | tee -a "$LOG"

# --- Histopath: wait for wget to finish (no wget process touching the zip) ---
until ! pgrep -f "NCT-CRC-HE-100K.zip" >/dev/null; do sleep 30; done
sleep 5
echo "[$(date '+%F %T')] histopath zip downloaded -> sampling 11k" | tee -a "$LOG"
python -u - <<'PY' 2>&1 | tee -a "$LOG"
import zipfile, random, os
zp = "scripts/staging/NCT-CRC-HE-100K.zip"
out = "scripts/staging/histopath"
os.makedirs(out, exist_ok=True)
zf = zipfile.ZipFile(zp)
names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff", ".png"))]
print("entries in zip:", len(names))
random.seed(42)
sample = random.sample(names, min(11500, len(names)))
for i, n in enumerate(sample):
    data = zf.read(n)
    with open(os.path.join(out, f"hist_{i:06d}.tif"), "wb") as f:
        f.write(data)
    if i % 2000 == 0:
        print("extracted", i, flush=True)
print("extracted total", len(sample))
PY
python -u scripts/prepare_domain_split.py --src scripts/staging/histopath --name histopath \
    --n_train 8000 --n_val 1000 --n_test 2000 --seed 42 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] FINALIZE AUTOFETCH DONE" | tee -a "$LOG"
