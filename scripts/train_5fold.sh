#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/config_TRIAGE_MIL_CLAM_UNI_5fold.json"

for FOLD in 0 1 2 3 4
do
  echo "Running TRIAGE-MIL fold ${FOLD}"
  python src/train_triage_mil.py \
    --config "${CONFIG}" \
    --fold "${FOLD}"
done
