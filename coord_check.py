# -*- coding: utf-8 -*-
"""
Diagnostic: check whether the trained model actually enhances a given point
(e.g. a ventricle location) to CTA-like HU, on a TRAINING-set patient.

Key fixes vs. the first version:
  * Seed fixing so DDPM output is reproducible.
  * Optional DDIM (deterministic) sampling for a stable second opinion.
  * Explicit range checks on stored NCT/CTA (are they [-1,1] or raw HU?).
  * Whole-patch enhancement statistics, not just a single voxel.
"""

import argparse
import numpy as np
import torch
import SimpleITK as sitk

from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet import create_model


def read_arr(path):
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]
    return img, arr


def denormalize_ct(x, hu_min=-1000, hu_max=1500):
    return (x + 1.0) / 2.0 * (hu_max - hu_min) + hu_min


def build_model(weight_path, input_size=128, depth_size=64, timesteps=250):
    model = create_model(
        input_size, 64, 1, in_channels=2, out_channels=1,
    ).cuda()
    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        depth_size=depth_size,
        timesteps=timesteps,
        loss_type="l1",
        with_condition=True,
        channels=1,
    ).cuda()
    ckpt = torch.load(weight_path, map_location="cuda")
    diffusion.load_state_dict(ckpt["ema"])
    diffusion.eval()
    return diffusion


def crop_centered(arr, x, y, z, patch_size=(64, 128, 128)):
    pd, ph, pw = patch_size
    D, H, W = arr.shape  # arr is [z, y, x] -> D=z, H=y, W=x
    z0 = max(0, min(z - pd // 2, D - pd))
    y0 = max(0, min(y - ph // 2, H - ph))
    x0 = max(0, min(x - pw // 2, W - pw))
    patch = arr[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    return patch, (z0, y0, x0), (z - z0, y - y0, x - x0)


@torch.no_grad()
def sample_patch(diffusion, nct_patch, sampler="ddim", ddim_timesteps=50, seed=0):
    if seed is not None:
        torch.manual_seed(seed)
    cond = torch.from_numpy(nct_patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    if sampler == "ddim":
        pred = diffusion.ddim_sample(
            batch_size=1, condition_tensors=cond,
            ddim_timesteps=ddim_timesteps, eta=0.0,
        )
    else:
        pred = diffusion.sample(batch_size=1, condition_tensors=cond)
    pred = pred[0, 0].detach().cpu().numpy().astype(np.float32)
    return np.clip(pred, -1, 1)


def local_stats(arr_hu, lz, ly, lx, r=2):
    cube = arr_hu[max(0, lz - r):lz + r + 1,
                  max(0, ly - r):ly + r + 1,
                  max(0, lx - r):lx + r + 1]
    return float(cube.mean()), float(np.median(cube))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=int, default=348)
    p.add_argument("--y", type=int, default=290)
    p.add_argument("--z", type=int, default=191)
    p.add_argument("--nct_path", type=str,
                   default="/home/maia-user/scy/Sample-Combine-preprocessed/nct/PAT_000.nii.gz")
    p.add_argument("--cta_path", type=str,
                   default="/home/maia-user/scy/Sample-Combine-preprocessed/cta/PAT_000.nii.gz")
    p.add_argument("--weight_path", type=str, default="./results_nct_cta/model-93.pt")
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm", "both"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    nct_img, nct = read_arr(args.nct_path)
    cta_img, cta = read_arr(args.cta_path)

    # ---- CRITICAL: confirm what is actually stored on disk ----
    print("=== Stored value ranges (MUST be ~[-1, 1] if preprocessing normalized) ===")
    print(f"NCT range: [{nct.min():.3f}, {nct.max():.3f}]")
    print(f"CTA range: [{cta.min():.3f}, {cta.max():.3f}]")
    if nct.min() < -1.5 or nct.max() > 1.5:
        print("!! WARNING: NCT is NOT in [-1,1]. It looks like raw HU. "
              "Feeding it to the model as condition is WRONG, and denormalize_ct on it is meaningless.")
    print()

    diffusion = build_model(args.weight_path)

    nct_patch, origin, local = crop_centered(nct, args.x, args.y, args.z)
    cta_patch, _, _ = crop_centered(cta, args.x, args.y, args.z)
    lz, ly, lx = local

    nct_hu = denormalize_ct(nct_patch)
    cta_hu = denormalize_ct(cta_patch)

    print("Patch origin (z,y,x):", origin)
    print("Local coord  (z,y,x):", local)
    print(f"\nReal CTA patch HU range: [{cta_hu.min():.1f}, {cta_hu.max():.1f}]")
    print(f"Real CTA voxels >200HU: {(cta_hu > 200).mean() * 100:.2f}%")

    samplers = ["ddpm", "ddim"] if args.sampler == "both" else [args.sampler]
    for s in samplers:
        pred_norm = sample_patch(diffusion, nct_patch, sampler=s, seed=args.seed)
        pred_hu = denormalize_ct(pred_norm)
        print(f"\n----- Sampler: {s.upper()} -----")
        print(f"Generated norm range: [{pred_norm.min():.3f}, {pred_norm.max():.3f}]")
        print(f"Generated HU range:   [{pred_hu.min():.1f}, {pred_hu.max():.1f}]")
        print(f"Generated voxels >200HU: {(pred_hu > 200).mean() * 100:.2f}%")
        print(f"Point  NCT HU: {nct_hu[lz, ly, lx]:.1f} | "
              f"Real CTA HU: {cta_hu[lz, ly, lx]:.1f} | "
              f"Gen CTA HU: {pred_hu[lz, ly, lx]:.1f}")
        print(f"Local(5^3) mean/median  NCT: {local_stats(nct_hu, lz, ly, lx)} | "
              f"Real: {local_stats(cta_hu, lz, ly, lx)} | "
              f"Gen: {local_stats(pred_hu, lz, ly, lx)}")

        out = sitk.GetImageFromArray(pred_hu.astype(np.float32))
        out.SetSpacing(nct_img.GetSpacing())
        out.SetDirection(nct_img.GetDirection())
        z0, y0, x0 = origin
        out.SetOrigin(nct_img.TransformIndexToPhysicalPoint((x0, y0, z0)))
        sitk.WriteImage(out, f"coord_check_generated_{s}.nii.gz")
        print(f"Saved: coord_check_generated_{s}.nii.gz")


if __name__ == "__main__":
    main()