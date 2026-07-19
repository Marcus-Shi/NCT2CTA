# -*- coding: utf-8 -*-
"""
Sample CTA-like image from preprocessed NCT using trained Med-DDPM.

Modes:
  center  : generate one center patch, fast sanity check.
  sliding : sliding-window generation for the full volume.

Center patch example:
python sample_ct.py \
  --nct_path /home/maia-user/scy/Sample-Combine-preprocessed/nct/PAT_000.nii.gz \
  --weight_path ./results_nct_cta/model-10.pt \
  --out_path ./PAT_000_cta_like_center_patch.nii.gz \
  --mode center \
  --save_hu

Sliding-window example, recommended:
python sample_ct.py \
  --nct_path /home/maia-user/scy/Sample-Combine-preprocessed/nct/PAT_000.nii.gz \
  --weight_path ./results_nct_cta/model-10.pt \
  --out_path ./PAT_000_cta_like_full_gaussian.nii.gz \
  --mode sliding \
  --input_size 128 \
  --depth_size 64 \
  --stride_d 16 \
  --stride_hw 32 \
  --blend gaussian \
  --valid_margin_d 8 \
  --valid_margin_hw 16 \
  --sampler ddim \
  --ddim_timesteps 50 \
  --ddim_eta 0.0 \
  --save_hu

Seam reduction:
  Sliding-window stitching used to show visible "checkerboard" seams at
  patch boundaries. The root cause was that every patch was generated with
  GaussianDiffusion.sample() (full stochastic DDPM), which draws an
  independent random noise trajectory per patch -- so even patches whose
  NCT condition overlaps almost entirely could produce very different
  overall brightness/contrast.
  --sampler ddim (the default) switches to GaussianDiffusion.ddim_sample()
  with eta=0.0 (fully deterministic reverse process) and a single shared
  initial noise tensor reused for every patch, removing that per-patch
  randomness. This is combined with the existing Gaussian blend + valid
  margin fusion below. Use --sampler ddpm to fall back to the old
  behavior for comparison/debugging.
"""

import argparse
import contextlib
import io
from itertools import product
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

    parser.add_argument(
        "--mode",
        type=str,
        default="center",
        choices=["center", "sliding"]
    )

    # Must match training.
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--depth_size", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=250)

    # Sampler used for each individual patch.
    # ddim (recommended): deterministic (eta=0) reverse process with a
    # single initial noise tensor shared across every patch. This removes
    # the per-patch random sampling trajectory that otherwise causes
    # visible intensity/style jumps ("checkerboard" seams) between
    # overlapping patches during sliding-window stitching.
    # ddpm: original stochastic full-length reverse sampling, independent
    # random noise per patch. Kept only for comparison/debugging.
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help=(
            "ddim = deterministic sampling + shared noise across patches "
            "(recommended, reduces stitching seams). "
            "ddpm = original stochastic sampler (visible seams)."
        )
    )
    parser.add_argument(
        "--ddim_timesteps",
        type=int,
        default=50,
        help="Number of DDIM steps (subsequence of --timesteps). Fewer = faster."
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="0.0 = fully deterministic DDIM (required for shared-noise seam fix). >0 reintroduces per-step randomness."
    )

    # Sliding-window parameters.
    # Smaller stride = more overlap = smoother stitching, but slower inference.
    parser.add_argument("--stride_d", type=int, default=16)
    parser.add_argument("--stride_hw", type=int, default=32)

    # Patch fusion parameters.
    parser.add_argument(
        "--blend",
        type=str,
        default="gaussian",
        choices=["gaussian", "constant"],
        help="Patch fusion weight. gaussian is recommended."
    )
    parser.add_argument(
        "--gaussian_sigma_scale",
        type=float,
        default=0.25,
        help=(
            "Gaussian sigma relative to normalized patch coordinates. "
            "Recommended range: 0.20-0.35. "
            "Larger value means weaker center weighting."
        )
    )
    parser.add_argument(
        "--valid_margin_d",
        type=int,
        default=8,
        help=(
            "Discard this many voxels from patch front/back borders during fusion, "
            "except at volume boundary."
        )
    )
    parser.add_argument(
        "--valid_margin_hw",
        type=int,
        default=16,
        help=(
            "Discard this many voxels from patch H/W borders during fusion, "
            "except at volume boundary."
        )
    )

    parser.add_argument(
        "--ckpt_key",
        type=str,
        default="ema",
        choices=["ema", "model"]
    )

    parser.add_argument(
        "--save_hu",
        action="store_true",
        help="Denormalize output from [-1,1] to HU before saving."
    )
    parser.add_argument("--hu_min", type=float, default=-1000)
    parser.add_argument("--hu_max", type=float, default=1500)

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--show_inner_tqdm",
        action="store_true",
        help=(
            "Show the original diffusion timestep progress bar for every patch. "
            "Default is False because it makes sliding-window output messy."
        )
    )

    parser.add_argument(
        "--empty_cache_every",
        type=int,
        default=0,
        help="If >0, call torch.cuda.empty_cache() every N patches."
    )

    return parser.parse_args()


def read_sitk_array(path):
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D,H,W]

    # Preprocessed NCT should already be in [-1, 1].
    arr = np.clip(arr, -1.0, 1.0).astype(np.float32)

    return img, arr


def denormalize_ct(x, hu_min=-1000, hu_max=1500):
    x = (x + 1.0) / 2.0
    x = x * (hu_max - hu_min) + hu_min
    return x


def to_condition_tensor(patch):
    # patch [D,H,W] -> [1,1,D,H,W]
    return (
        torch.from_numpy(patch.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
    )


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
    """
    Compute sliding-window start indices and force the last patch to touch the end.
    """
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
        raise RuntimeError(
            f"Volume too small: volume={shape}, patch={patch_size}"
        )

    z0 = (D - pd) // 2
    y0 = (H - ph) // 2
    x0 = (W - pw) // 2

    return z0, y0, x0


def make_patch_image(arr_zyx, ref_img, z0, y0, x0):
    """
    Save a patch with correct spacing/direction and adjusted origin.

    arr_zyx shape: [D,H,W]
    SimpleITK image index order is [x,y,z].
    """
    out = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    out.SetSpacing(ref_img.GetSpacing())
    out.SetDirection(ref_img.GetDirection())

    patch_origin = ref_img.TransformIndexToPhysicalPoint(
        (int(x0), int(y0), int(z0))
    )
    out.SetOrigin(patch_origin)

    return out


def make_gaussian_weight(shape, sigma_scale=0.25):
    """
    Create a 3D Gaussian blending weight for one patch.

    shape: [D,H,W]

    Center has high weight, borders have low weight.
    This reduces patch boundary artifacts during stitching.
    """
    dz, dy, dx = shape

    z = np.linspace(-1.0, 1.0, dz, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, dy, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, dx, dtype=np.float32)

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    sigma = float(sigma_scale)
    weight = np.exp(
        -(zz * zz + yy * yy + xx * xx) / (2.0 * sigma * sigma)
    )

    weight = weight.astype(np.float32)
    weight = np.maximum(weight, 1e-6)
    weight = weight / np.max(weight)

    return weight


def make_constant_weight(shape):
    return np.ones(shape, dtype=np.float32)


def get_valid_slices(
    z0,
    y0,
    x0,
    pd,
    ph,
    pw,
    D,
    H,
    W,
    margin_d,
    margin_hw,
):
    """
    Return volume slices and corresponding patch slices for center-valid fusion.

    For internal patches:
      discard patch borders.

    For patches touching volume boundary:
      keep the outer boundary so the whole volume is covered.

    This reduces the influence of patch-border predictions.
    """

    # Patch-local valid region.
    pz0 = margin_d if z0 > 0 else 0
    pz1 = pd - margin_d if (z0 + pd) < D else pd

    py0 = margin_hw if y0 > 0 else 0
    py1 = ph - margin_hw if (y0 + ph) < H else ph

    px0 = margin_hw if x0 > 0 else 0
    px1 = pw - margin_hw if (x0 + pw) < W else pw

    if pz1 <= pz0 or py1 <= py0 or px1 <= px0:
        raise RuntimeError(
            "Invalid valid crop. Reduce valid margins or increase patch size. "
            f"patch={(pd, ph, pw)}, margins={(margin_d, margin_hw)}"
        )

    # Corresponding volume region.
    vz0 = z0 + pz0
    vz1 = z0 + pz1

    vy0 = y0 + py0
    vy1 = y0 + py1

    vx0 = x0 + px0
    vx1 = x0 + px1

    vol_slices = (
        slice(vz0, vz1),
        slice(vy0, vy1),
        slice(vx0, vx1),
    )

    patch_slices = (
        slice(pz0, pz1),
        slice(py0, py1),
        slice(px0, px1),
    )

    return vol_slices, patch_slices


def build_shared_noise(diffusion, args):
    """
    One fixed initial noise tensor, reused for every patch when
    --sampler ddim.

    Combined with eta=0 (fully deterministic DDIM reverse process), this
    removes the independent-random-trajectory source of inter-patch
    intensity/style jumps that cause visible seams in sliding-window
    stitching. See GaussianDiffusion.ddim_sample() in
    diffusion_model/trainer.py for the underlying mechanism.
    """
    shape = (1, diffusion.channels, args.depth_size, args.input_size, args.input_size)
    return torch.randn(shape, device="cuda")


@torch.no_grad()
def sample_one_patch(
    diffusion,
    nct_patch,
    show_inner_tqdm=False,
    sampler="ddim",
    ddim_timesteps=50,
    eta=0.0,
    noise=None,
):
    """
    Generate one CTA-like patch from one NCT patch.

    nct_patch: numpy array [D,H,W]
    return: numpy array [D,H,W], normalized to [-1,1]

    sampler="ddim" with `noise` set to the SAME tensor for every patch call
    removes the per-patch random sampling trajectory that otherwise causes
    visible intensity/style jumps at patch boundaries when stitching a
    sliding-window volume together.
    """
    condition = to_condition_tensor(nct_patch)

    def _run():
        if sampler == "ddim":
            return diffusion.ddim_sample(
                batch_size=1,
                condition_tensors=condition,
                ddim_timesteps=ddim_timesteps,
                eta=eta,
                noise=noise,
            )
        return diffusion.sample(batch_size=1, condition_tensors=condition)

    if show_inner_tqdm:
        pred = _run()
    else:
        # The original GaussianDiffusion sampling loops use tqdm internally.
        # During sliding-window inference this becomes extremely noisy.
        # Suppress inner tqdm and keep only the outer patch-level progress bar.
        with contextlib.redirect_stderr(io.StringIO()):
            pred = _run()

    pred = pred[0, 0].detach().cpu().numpy().astype(np.float32)  # [D,H,W]
    pred = np.clip(pred, -1.0, 1.0)

    return pred


@torch.no_grad()
def sample_center_patch(diffusion, ref_img, nct_arr, args, shared_noise=None):
    patch_size = (args.depth_size, args.input_size, args.input_size)
    z0, y0, x0 = center_crop_coords(nct_arr.shape, patch_size)

    pd, ph, pw = patch_size

    nct_patch = nct_arr[
        z0:z0 + pd,
        y0:y0 + ph,
        x0:x0 + pw,
    ]

    print("Center patch location:")
    print(f"  z: {z0} -> {z0 + pd}")
    print(f"  y: {y0} -> {y0 + ph}")
    print(f"  x: {x0} -> {x0 + pw}")

    pred = sample_one_patch(
        diffusion,
        nct_patch,
        show_inner_tqdm=args.show_inner_tqdm,
        sampler=args.sampler,
        ddim_timesteps=args.ddim_timesteps,
        eta=args.ddim_eta,
        noise=shared_noise,
    )

    if args.save_hu:
        pred = denormalize_ct(pred, args.hu_min, args.hu_max)

    out_img = make_patch_image(pred, ref_img, z0, y0, x0)

    return out_img


@torch.no_grad()
def sample_sliding_window(diffusion, ref_img, nct_arr, args, shared_noise=None):
    patch_size = (args.depth_size, args.input_size, args.input_size)
    pd, ph, pw = patch_size
    D, H, W = nct_arr.shape

    if D < pd or H < ph or W < pw:
        raise RuntimeError(
            f"Volume too small: volume={nct_arr.shape}, patch={patch_size}"
        )

    # Check valid region size.
    valid_d = pd - 2 * args.valid_margin_d
    valid_h = ph - 2 * args.valid_margin_hw
    valid_w = pw - 2 * args.valid_margin_hw

    if valid_d <= 0 or valid_h <= 0 or valid_w <= 0:
        raise RuntimeError(
            "Margins are too large. Need patch size larger than 2 * margin. "
            f"patch={(pd, ph, pw)}, margins={(args.valid_margin_d, args.valid_margin_hw)}"
        )

    if args.stride_d > valid_d:
        print(
            f"Warning: stride_d={args.stride_d} > valid_d={valid_d}. "
            "This may create uncovered voxels when using valid crop."
        )

    if args.stride_hw > valid_h or args.stride_hw > valid_w:
        print(
            f"Warning: stride_hw={args.stride_hw} > valid_hw={min(valid_h, valid_w)}. "
            "This may create uncovered voxels when using valid crop."
        )

    z_starts = compute_starts(D, pd, args.stride_d)
    y_starts = compute_starts(H, ph, args.stride_hw)
    x_starts = compute_starts(W, pw, args.stride_hw)

    coords = list(product(z_starts, y_starts, x_starts))
    total = len(coords)

    print("Sliding-window setup:")
    print(f"  volume shape [D,H,W]: {nct_arr.shape}")
    print(f"  patch size   [D,H,W]: {patch_size}")
    print(f"  stride_d: {args.stride_d}")
    print(f"  stride_hw: {args.stride_hw}")
    print(f"  z starts: {len(z_starts)}")
    print(f"  y starts: {len(y_starts)}")
    print(f"  x starts: {len(x_starts)}")
    print(f"  total patches: {total}")
    print(f"  fusion: {args.blend}")
    print(f"  valid_margin_d: {args.valid_margin_d}")
    print(f"  valid_margin_hw: {args.valid_margin_hw}")
    print(f"  show_inner_tqdm: {args.show_inner_tqdm}")
    print(f"  sampler: {args.sampler}")
    if args.sampler == "ddim":
        print(f"  ddim_timesteps: {args.ddim_timesteps}")
        print(f"  ddim_eta: {args.ddim_eta}")
        print(f"  shared noise across patches: {shared_noise is not None}")

    if args.blend == "gaussian":
        patch_weight = make_gaussian_weight(
            (pd, ph, pw),
            sigma_scale=args.gaussian_sigma_scale,
        )
    else:
        patch_weight = make_constant_weight((pd, ph, pw))

    acc = np.zeros((D, H, W), dtype=np.float32)
    weight = np.zeros((D, H, W), dtype=np.float32)

    pbar = tqdm(
        coords,
        total=total,
        desc="Sliding patches",
        dynamic_ncols=True,
        leave=True,
    )

    for patch_counter, (z0, y0, x0) in enumerate(pbar, start=1):
        nct_patch = nct_arr[
            z0:z0 + pd,
            y0:y0 + ph,
            x0:x0 + pw,
        ]

        pred = sample_one_patch(
            diffusion,
            nct_patch,
            show_inner_tqdm=args.show_inner_tqdm,
            sampler=args.sampler,
            ddim_timesteps=args.ddim_timesteps,
            eta=args.ddim_eta,
            noise=shared_noise,
        )

        vol_slices, patch_slices = get_valid_slices(
            z0,
            y0,
            x0,
            pd,
            ph,
            pw,
            D,
            H,
            W,
            args.valid_margin_d,
            args.valid_margin_hw,
        )

        pred_valid = pred[patch_slices]
        weight_valid = patch_weight[patch_slices]

        acc[vol_slices] += pred_valid * weight_valid
        weight[vol_slices] += weight_valid

        pbar.set_postfix(
            {
                "patch": f"{patch_counter}/{total}",
                "z": z0,
                "y": y0,
                "x": x0,
            }
        )

        if args.empty_cache_every > 0:
            if patch_counter % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    # Check uncovered voxels.
    uncovered = weight <= 1e-8

    if np.any(uncovered):
        n_uncovered = int(np.sum(uncovered))
        ratio = n_uncovered / float(D * H * W)

        print(
            f"Warning: uncovered voxels detected: {n_uncovered} "
            f"({ratio:.6f}). Filling them with NCT input values."
        )

        # This should not happen with the recommended stride/margins.
        # Use NCT as a conservative fallback for uncovered voxels.
        acc[uncovered] = nct_arr[uncovered]
        weight[uncovered] = 1.0

    pred_full = acc / np.maximum(weight, 1e-6)
    pred_full = np.clip(pred_full, -1.0, 1.0)

    if args.save_hu:
        pred_full = denormalize_ct(
            pred_full,
            args.hu_min,
            args.hu_max,
        )

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

    shared_noise = None
    if args.sampler == "ddim":
        shared_noise = build_shared_noise(diffusion, args)
        print(
            f"Sampler: ddim (ddim_timesteps={args.ddim_timesteps}, "
            f"eta={args.ddim_eta}), noise shared across all patches: True"
        )
    else:
        print("Sampler: ddpm (original stochastic sampler, independent noise per patch)")

    if args.mode == "center":
        out_img = sample_center_patch(diffusion, ref_img, nct_arr, args, shared_noise=shared_noise)
    else:
        out_img = sample_sliding_window(diffusion, ref_img, nct_arr, args, shared_noise=shared_noise)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(out_img, str(out_path))

    print("Saved:", out_path)


if __name__ == "__main__":
    main()