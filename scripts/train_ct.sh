#!/usr/bin/env bash
set -e

# Run from the root directory of the original med-ddpm repository.

python train_ct.py \
  -i dataset/nct \
  -t dataset/cta \
  --results_folder results_nct2cta \
  --input_size 128 \
  --depth_size 128 \
  --hu_min -200 \
  --hu_max 1000 \
  --num_channels 64 \
  --num_res_blocks 1 \
  --batchsize 1 \
  --timesteps 250 \
  --epochs 50000 \
  --save_and_sample_every 1000 \
  --train_lr 1e-5 \
  --gradient_accumulate_every 1 \
  -r ""
