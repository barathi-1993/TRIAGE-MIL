#!/usr/bin/env bash
set -e

python src/mass_selector.py \
  --feature-root ./data/features \
  --encoder UNI/pt_files \
  --out-cache ./cache/MASS_TOPK_TILES/CLAM_UNI \
  --topk 4096
