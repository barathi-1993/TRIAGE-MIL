#!/usr/bin/env bash
set -e

for FOLD in 0 1 2 3 4
do
  python src/train_triage_mil.py \
    --config configs/config_TRIAGE_MIL_CLAM_UNI_5fold.json \
    --fold ${FOLD}
done
