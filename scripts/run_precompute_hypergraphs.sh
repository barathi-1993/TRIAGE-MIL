#!/usr/bin/env bash
set -euo pipefail

python src/precompute_hypergraphs.py \
  --config configs/config_TRIAGE_MIL_CLAM_UNI_5fold.json
