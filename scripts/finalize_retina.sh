#!/bin/bash
# Patiently download EyePACS (Kaggle generates the zip server-side -> slow start),
# with a disk guard, then build the 8000/1000/2000 split. Retina images are
# augmented_resized_V2/{train,test}/{0..4}/<id>-600.jpg (~600px RGB).
set -u
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
cd "$(dirname "$0")/.."
LOG="train/logs/fetch_retina.log"
RAW="scripts/staging/retina_raw"
GUARD_GB=50
rm -rf "$RAW"; mkdir -p "$RAW"
echo "[$(date '+%F %T')] retina download start (guard ${GUARD_GB}GB)" | tee "$LOG"

python -u -c "
import kaggle; kaggle.api.authenticate()
kaggle.api.dataset_download_files('tanlikesmath/diabetic-retinopathy-resized', path='$RAW', unzip=True, quiet=False)
print('RETINA DOWNLOAD DONE', flush=True)
" >> "$LOG" 2>&1 &
DLPID=$!

# disk guard
while kill -0 "$DLPID" 2>/dev/null; do
  used=$(du -sg "$RAW" 2>/dev/null | awk '{print $1}')
  if [ "${used:-0}" -gt "$GUARD_GB" ]; then
    echo "[$(date '+%F %T')] ABORT: staging > ${GUARD_GB}GB" | tee -a "$LOG"
    kill "$DLPID" 2>/dev/null
    exit 1
  fi
  sleep 30
done

if grep -q "RETINA DOWNLOAD DONE" "$LOG"; then
  echo "[$(date '+%F %T')] download ok -> splitting" | tee -a "$LOG"
  python -u scripts/prepare_domain_split.py --src "$RAW" --name retina \
      --n_train 8000 --n_val 1000 --n_test 2000 --seed 42 2>&1 | tee -a "$LOG"
  echo "[$(date '+%F %T')] FINALIZE RETINA DONE" | tee -a "$LOG"
else
  echo "[$(date '+%F %T')] download did not finish cleanly; check log" | tee -a "$LOG"
fi
