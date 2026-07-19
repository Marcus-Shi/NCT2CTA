#!/usr/bin/env bash
set -e

# Run from the root directory of the original med-ddpm repository.
# Replace results_nct2cta/model-50.pt with your actual checkpoint.

python sample_ct.py \
  -i dataset/nct \
  -e exports_nct2cta \
  -w results_nct2cta/model-50.pt \
  --input_size 128 \
  --depth_size 128 \
  --hu_min -200 \
  --hu_max 1000 \
  --num_channels 64 \
  --num_res_blocks 1 \
  --timesteps 250 \
  --resize_back
