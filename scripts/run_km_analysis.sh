#!/usr/bin/env bash
set -e

python src/km_analysis.py \
  --predictions-dir ./results/TRIAGE_MIL_CLAM_UNI \
  --output-dir ./results/TRIAGE_MIL_CLAM_UNI/km_results \
  --method median
