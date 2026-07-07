# -*- coding: utf-8 -*-
"""
Train Med-DDPM for paired NCT -> CTA synthesis.

Example:
python train_ct.py \
  --nct_dir /home/maia-user/scy/Sample-Combine-preprocessed/nct \
  --cta_dir /home/maia-user/scy/Sample-Combine-preprocessed/cta \
  --results_folder ./results_nct_cta \
  --input_size 128 \
  --depth_size 64 \
  --batchsize 1 \
  --gradient_accumulate_every 2 \
  --timesteps 250 \
  --train_num_steps 100000
"""

import argparse
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset_ct_patch import PairedCTPatchDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nct_dir", type=str, default="/home/maia-user/scy/Sample-Combine-preprocessed/nct")
    parser.add_argument("--cta_dir", type=str, default="/home/maia-user/scy/Sample-Combine-preprocessed/cta")
    parser.add_argument("--samples_per_epoch", type=int, default=2000)
    parser.add_argument("--cache_size", type=int, default=2)

    parser.add_argument("--input_size", type=int, default=128, help="Patch H/W")
    parser.add_argument("--depth_size", type=int, default=64, help="Patch D")

    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=1)

    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--train_lr", type=float, default=1e-5)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--train_num_steps", type=int, default=100000)
    parser.add_argument("--gradient_accumulate_every", type=int, default=2)
    parser.add_argument("--save_and_sample_every", type=int, default=1000)
    parser.add_argument("--results_folder", type=str, default="./results_nct_cta")

    parser.add_argument("--resume_weight", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="ema", choices=["ema", "model"])

    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_size = args.input_size
    depth_size = args.depth_size

    dataset = PairedCTPatchDataset(
        nct_dir=args.nct_dir,
        cta_dir=args.cta_dir,
        patch_size=(depth_size, input_size, input_size),
        samples_per_epoch=args.samples_per_epoch,
    )

    # U-Net input = noisy CTA + NCT condition = 2 channels.
    # U-Net output = predicted CTA noise = 1 channel.
    model = create_model(
        input_size,
        args.num_channels,
        args.num_res_blocks,
        in_channels=2,
        out_channels=1,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        depth_size=depth_size,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
        with_condition=True,
        channels=1,
    ).cuda()

    if args.resume_weight:
        ckpt = torch.load(args.resume_weight, map_location="cuda")
        diffusion.load_state_dict(ckpt[args.resume_from])
        print(f"Loaded checkpoint: {args.resume_weight}, key={args.resume_from}")

    trainer = Trainer(
        diffusion,
        dataset,
        image_size=input_size,
        depth_size=depth_size,
        train_batch_size=args.batchsize,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=0.995,
        fp16=False,
        with_condition=True,
        save_and_sample_every=args.save_and_sample_every,
        results_folder=args.results_folder,
    )

    trainer.train()


if __name__ == "__main__":
    main()
