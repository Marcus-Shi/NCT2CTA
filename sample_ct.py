# -*- coding: utf-8 -*-
"""
Sample CTA-like image from preprocessed NCT using trained Med-DDPM.

Modes:
  center  : generate one center patch, fast sanity check.
  sliding : sliding-window generation for the full volume. This can be very slow.

Center patch example:
python sample_ct.py \
  --nct_path /home/maia-user/scy/Sample-Combine-preprocessed/nct/PAT_000.nii.gz \
  --weight_path ./results_nct_cta/model-10.pt \
  --out_path ./PAT_000_cta_like_center_patch.nii.gz \
  --mode center \
  --save_hu
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm

from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet import create_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nct_path", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)

    parser.add_argument("--mode", type=str, default="center", choices=["center", "sliding"])

    # Must match training.
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--depth_size", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=250)

    parser.add_argument("--stride_d", type=int, default=32)
    parser.add_argument("--stride_hw", type=int, default=64)

    parser.add_argument("--ckpt_key", type=str, default="ema", choices=["ema", "model"])

    parser.add_argument("--save_hu", action="store_true", help="Denormalize output from [-1,1] to HU before saving.")
    parser.add_argument("--hu_min", type=float, default=-1000)
    parser.add_argument("--hu_max", type=float, default=1500)

    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def read_sitk_array(path):
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D,H,W]
    arr = np.clip(arr, -1.0, 1.0).astype(np.float32)
    return img, arr


def denormalize_ct(x, hu_min=-1000, hu_max=1500):
    x = (x + 1.0) / 2.0
    x = x * (hu_max - hu_min) + hu_min
    return x


def to_condition_tensor(patch):
    # patch [D,H,W] -> [1,1,D,H,W]
    return torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()


def build_diffusion(args):
    model = create_model(
        args.input_size,
        args.num_channels,
        args.num_res_blocks,
        in_channels=2,
        out_channels=1,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=args.input_size,
        depth_size=args.depth_size,
        timesteps=args.timesteps,
        loss_type="l1",
        with_condition=True,
        channels=1,
    ).cuda()

    ckpt = torch.load(args.weight_path, map_location="cuda")
    diffusion.load_state_dict(ckpt[args.ckpt_key])
    diffusion.eval()
    return diffusion


def compute_starts(size, patch, stride):
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def center_crop_coords(shape, patch_size):
    D, H, W = shape
    pd, ph, pw = patch_size

    if D < pd or H < ph or W < pw:
        raise RuntimeError(f"Volume too small: volume={shape}, patch={patch_size}")

    z0 = (D - pd) // 2
    y0 = (H - ph) // 2
    x0 = (W - pw) // 2
    return z0, y0, x0


def make_patch_image(arr_zyx, ref_img, z0, y0, x0):
    """
    Save a patch with correct spacing/direction and adjusted origin.
    arr_zyx shape: [D,H,W]
    SimpleITK index order is [x,y,z].
    """
    out = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    out.SetSpacing(ref_img.GetSpacing())
    out.SetDirection(ref_img.GetDirection())
    patch_origin = ref_img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0)))
    out.SetOrigin(patch_origin)
    return out


@torch.no_grad()
def sample_one_patch(diffusion, nct_patch):
    condition = to_condition_tensor(nct_patch)
    pred = diffusion.sample(batch_size=1, condition_tensors=condition)
    pred = pred[0, 0].detach().cpu().numpy().astype(np.float32)  # [D,H,W]
    pred = np.clip(pred, -1.0, 1.0)
    return pred


@torch.no_grad()
def sample_center_patch(diffusion, ref_img, nct_arr, args):
    patch_size = (args.depth_size, args.input_size, args.input_size)
    z0, y0, x0 = center_crop_coords(nct_arr.shape, patch_size)

    pd, ph, pw = patch_size
    nct_patch = nct_arr[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    pred = sample_one_patch(diffusion, nct_patch)

    if args.save_hu:
        pred = denormalize_ct(pred, args.hu_min, args.hu_max)

    out_img = make_patch_image(pred, ref_img, z0, y0, x0)
    return out_img


@torch.no_grad()
def sample_sliding_window(diffusion, ref_img, nct_arr, args):
    patch_size = (args.depth_size, args.input_size, args.input_size)
    pd, ph, pw = patch_size
    D, H, W = nct_arr.shape

    if D < pd or H < ph or W < pw:
        raise RuntimeError(f"Volume too small: volume={nct_arr.shape}, patch={patch_size}")

    z_starts = compute_starts(D, pd, args.stride_d)
    y_starts = compute_starts(H, ph, args.stride_hw)
    x_starts = compute_starts(W, pw, args.stride_hw)

    total = len(z_starts) * len(y_starts) * len(x_starts)
    print(f"Sliding-window patches: {total}")
    print(f"z starts: {len(z_starts)}, y starts: {len(y_starts)}, x starts: {len(x_starts)}")

    acc = np.zeros((D, H, W), dtype=np.float32)
    weight = np.zeros((D, H, W), dtype=np.float32)

    for z0 in tqdm(z_starts, desc="z"):
        for y0 in y_starts:
            for x0 in x_starts:
                nct_patch = nct_arr[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
                pred = sample_one_patch(diffusion, nct_patch)
                acc[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += pred
                weight[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += 1.0

    pred_full = acc / np.maximum(weight, 1e-6)
    pred_full = np.clip(pred_full, -1.0, 1.0)

    if args.save_hu:
        pred_full = denormalize_ct(pred_full, args.hu_min, args.hu_max)

    out_img = sitk.GetImageFromArray(pred_full.astype(np.float32))
    out_img.CopyInformation(ref_img)
    return out_img


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    ref_img, nct_arr = read_sitk_array(args.nct_path)

    print("NCT path:", args.nct_path)
    print("NCT shape [D,H,W]:", nct_arr.shape)
    print("NCT spacing [x,y,z]:", ref_img.GetSpacing())
    print("Mode:", args.mode)

    diffusion = build_diffusion(args)

    if args.mode == "center":
        out_img = sample_center_patch(diffusion, ref_img, nct_arr, args)
    else:
        out_img = sample_sliding_window(diffusion, ref_img, nct_arr, args)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
