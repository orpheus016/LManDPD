#!/usr/bin/env bash
set -euo pipefail

python utils/compare_dpd_outputs.py \
  --inputs "dpd_out/DPA_200MHz/*.csv" "dpd_out/DPA_100MHz/*.csv" "dpd_out/DPA_160MHz/*.csv" \
  --output_dir "dpd_out/analysis/dpa_after_dpd_transfer_200_to_100_160" \
  --PA_backbone dgru \
  --PA_hidden_size 23 \
  --PA_num_layers 1
