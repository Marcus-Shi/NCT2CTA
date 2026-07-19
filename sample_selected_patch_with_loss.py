# -*- coding: utf-8 -*-
"""
根据 ITK-SNAP 中选中的 x, y, z 坐标，
从预处理后的 NCT 中找到对应 sliding-window patch，
生成一个 CTA-like patch，并计算 DDPM noise prediction loss。

功能：
1. 坐标参数按照 ITK-SNAP 顺序输入：x, y, z
2. NumPy 内部自动使用 z, y, x
3. patch 起点与 full-volume sliding-window 网格保持一致
4. 如果一个点被多个重叠 patch 覆盖，选择让该点最接近 patch 中心的 patch
5. 保存：
   - NCT patch，归一化范围 [-1,1]
   - 真实 CTA patch，归一化范围 [-1,1]
   - 生成 CTA-like patch，归一化范围 [-1,1]
6. 使用真实 CTA patch 计算 DDPM noise prediction loss
7. 保存 noise prediction loss 的 JSON 结果

注意：
- 输出 NIfTI 不进行反归一化，灰度值仍然是 [-1,1]
- --loss_type 必须和训练时使用的 loss 一致
- 你的当前模型如果用 L2 训练，运行时使用 --loss_type l2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk

from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet import create_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nct_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cta_path",
        type=str,
        default="",
        help=(
            "真实 CTA 路径。"
            "计算 DDPM noise prediction loss 时必须提供。"
        ),
    )

    parser.add_argument(
        "--weight_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./selected_patch_result",
    )

    # ITK-SNAP 中显示的 voxel index 坐标顺序
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--z", type=int, required=True)

    # 必须与训练模型一致
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--depth_size", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=250)

    # 与 full-volume sliding-window 保持一致
    parser.add_argument("--stride_d", type=int, default=16)
    parser.add_argument("--stride_hw", type=int, default=32)

    parser.add_argument(
        "--ckpt_key",
        type=str,
        default="ema",
        choices=["ema", "model"],
    )

    # 必须和训练时一致
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1", "l2"],
        help=(
            "DDPM noise prediction loss 类型。"
            "必须和训练时一致。"
        ),
    )

    parser.add_argument(
        "--loss_repeats",
        type=int,
        default=20,
        help=(
            "同一个 patch 重复计算 noise prediction loss 的次数。"
            "每次会随机采样 timestep 和 Gaussian noise。"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="生成 CTA-like patch 使用的随机种子。",
    )

    parser.add_argument(
        "--loss_seed",
        type=int,
        default=1234,
        help="计算 noise prediction loss 使用的随机种子。",
    )

    # 这里只用于打印对应 HU，不影响保存的 NIfTI
    parser.add_argument("--hu_min", type=float, default=-1000)
    parser.add_argument("--hu_max", type=float, default=1500)

    return parser.parse_args()


def read_image(path):
    """
    读取 NIfTI。

    SimpleITK GetArrayFromImage 返回：
        [z, y, x]
    """
    img = sitk.ReadImage(
        str(path),
        sitk.sitkFloat32,
    )

    arr = sitk.GetArrayFromImage(
        img
    ).astype(np.float32)

    return img, arr


def normalized_to_hu(
    x,
    hu_min=-1000,
    hu_max=1500,
):
    """
    仅用于打印对应的 HU。

    不用于保存 patch。
    """
    return (
        (x + 1.0)
        / 2.0
        * (hu_max - hu_min)
        + hu_min
    )


def compute_starts(
    size,
    patch,
    stride,
):
    """
    生成与 full-volume sliding-window 相同的 patch 起点。
    """
    if size <= patch:
        return [0]

    starts = list(
        range(
            0,
            size - patch + 1,
            stride,
        )
    )

    # 保证最后一个 patch 覆盖 volume 尾部
    last_start = size - patch

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def select_start_for_coordinate(
    coord,
    starts,
    patch_size,
):
    """
    从所有包含 coord 的 patch 中，
    选择让 coord 最接近 patch 中心的 patch。
    """
    candidates = [
        start
        for start in starts
        if start <= coord < start + patch_size
    ]

    if not candidates:
        raise RuntimeError(
            f"坐标 {coord} 没有被任何 patch 覆盖。"
        )

    best_start = min(
        candidates,
        key=lambda start: abs(
            coord
            - (
                start
                + (patch_size - 1) / 2.0
            )
        ),
    )

    return best_start


def select_patch_origin(
    volume_shape,
    selected_xyz,
    patch_size,
    stride,
):
    """
    参数：
        volume_shape:
            NumPy shape [D,H,W]

        selected_xyz:
            ITK-SNAP 坐标 [x,y,z]

        patch_size:
            [D,H,W]

        stride:
            [stride_d,stride_h,stride_w]

    返回：
        patch origin [z0,y0,x0]
    """
    D, H, W = volume_shape
    x, y, z = selected_xyz

    pd, ph, pw = patch_size
    sd, sh, sw = stride

    if not 0 <= z < D:
        raise ValueError(
            f"z={z} 超出范围 [0,{D - 1}]"
        )

    if not 0 <= y < H:
        raise ValueError(
            f"y={y} 超出范围 [0,{H - 1}]"
        )

    if not 0 <= x < W:
        raise ValueError(
            f"x={x} 超出范围 [0,{W - 1}]"
        )

    z_starts = compute_starts(
        D,
        pd,
        sd,
    )

    y_starts = compute_starts(
        H,
        ph,
        sh,
    )

    x_starts = compute_starts(
        W,
        pw,
        sw,
    )

    z0 = select_start_for_coordinate(
        z,
        z_starts,
        pd,
    )

    y0 = select_start_for_coordinate(
        y,
        y_starts,
        ph,
    )

    x0 = select_start_for_coordinate(
        x,
        x_starts,
        pw,
    )

    return z0, y0, x0


def crop_patch(
    arr,
    origin_zyx,
    patch_size,
):
    """
    从 NumPy volume [D,H,W] 裁 patch。
    """
    z0, y0, x0 = origin_zyx
    pd, ph, pw = patch_size

    patch = arr[
        z0:z0 + pd,
        y0:y0 + ph,
        x0:x0 + pw,
    ]

    expected_shape = (
        pd,
        ph,
        pw,
    )

    if patch.shape != expected_shape:
        raise RuntimeError(
            f"Patch shape 错误："
            f"得到 {patch.shape}，"
            f"期望 {expected_shape}"
        )

    return patch


def make_patch_image(
    arr_zyx,
    reference_img,
    origin_zyx,
):
    """
    给 patch 设置正确的 spacing、direction 和物理 origin。

    arr_zyx 保持原始传入数值。
    本脚本中传入的是归一化后的 [-1,1] 数值。
    """
    z0, y0, x0 = origin_zyx

    out = sitk.GetImageFromArray(
        arr_zyx.astype(np.float32)
    )

    out.SetSpacing(
        reference_img.GetSpacing()
    )

    out.SetDirection(
        reference_img.GetDirection()
    )

    physical_origin = (
        reference_img.TransformIndexToPhysicalPoint(
            (
                int(x0),
                int(y0),
                int(z0),
            )
        )
    )

    out.SetOrigin(
        physical_origin
    )

    return out


def build_diffusion(args):
    """
    构建与训练时相同的 U-Net 和 GaussianDiffusion。
    """
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
        loss_type=args.loss_type,
        with_condition=True,
        channels=1,
    ).cuda()

    checkpoint = torch.load(
        args.weight_path,
        map_location="cuda",
    )

    if args.ckpt_key not in checkpoint:
        raise KeyError(
            f"Checkpoint 中没有 key='{args.ckpt_key}'。"
            f"现有 keys: {list(checkpoint.keys())}"
        )

    diffusion.load_state_dict(
        checkpoint[args.ckpt_key]
    )

    diffusion.eval()

    return diffusion


def numpy_patch_to_tensor(patch):
    """
    NumPy:
        [D,H,W]

    Tensor:
        [1,1,D,H,W]
    """
    return (
        torch.from_numpy(
            patch.astype(np.float32)
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
    )


@torch.no_grad()
def generate_patch(
    diffusion,
    nct_patch_norm,
):
    """
    使用一个 NCT patch 生成一个 CTA-like patch。

    输入：
        nct_patch_norm:
            NumPy [D,H,W]，范围 [-1,1]

    输出：
        generated_patch_norm:
            NumPy [D,H,W]，范围 [-1,1]
    """
    condition = numpy_patch_to_tensor(
        nct_patch_norm
    )

    generated = diffusion.sample(
        batch_size=1,
        condition_tensors=condition,
    )

    generated = (
        generated[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return np.clip(
        generated,
        -1.0,
        1.0,
    )


@torch.no_grad()
def calculate_ddpm_noise_prediction_loss(
    diffusion,
    nct_patch_norm,
    cta_patch_norm,
    loss_type="l2",
    repeats=20,
    seed=1234,
):
    """
    计算 DDPM 训练形式的 noise prediction loss。

    对每次重复：
    1. 随机采样 timestep t
    2. 随机生成真实 Gaussian noise epsilon
    3. 使用真实 CTA x0 和 epsilon 构造 x_t
    4. 将 [x_t, NCT condition] 输入 U-Net
    5. 得到预测噪声 epsilon_hat
    6. 计算 epsilon 与 epsilon_hat 的 L1 或 L2 loss

    注意：
    这个 loss 不是 generated CTA-like patch 与真实 CTA 的图像误差。
    """
    if repeats <= 0:
        raise ValueError(
            f"loss_repeats 必须大于 0，当前为 {repeats}"
        )

    if loss_type not in {"l1", "l2"}:
        raise ValueError(
            f"不支持的 loss_type: {loss_type}"
        )

    # 这里单独设置 loss 随机种子。
    # CTA-like patch 已经在此前生成，不会受到这里的影响。
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    condition = numpy_patch_to_tensor(
        nct_patch_norm
    )

    x_start = numpy_patch_to_tensor(
        cta_patch_norm
    )

    batch_size = x_start.shape[0]
    device = x_start.device

    loss_values = []
    timestep_values = []

    for _ in range(repeats):
        # 有效 timestep 索引为：
        # 0, 1, ..., num_timesteps - 1
        t = torch.randint(
            low=0,
            high=diffusion.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

        # 真实累计等效 Gaussian noise
        noise = torch.randn_like(
            x_start
        )

        # 真实 CTA x0 -> noisy CTA x_t
        x_noisy = diffusion.q_sample(
            x_start=x_start,
            t=t,
            noise=noise,
        )

        # U-Net 输入：
        # channel 0 = noisy CTA
        # channel 1 = NCT condition
        model_input = torch.cat(
            [
                x_noisy,
                condition,
            ],
            dim=1,
        )

        # 模型预测累计等效噪声
        predicted_noise = diffusion.denoise_fn(
            model_input,
            t,
        )

        error = (
            noise
            - predicted_noise
        )

        if loss_type == "l1":
            loss = (
                error
                .abs()
                .mean()
            )
        else:
            loss = (
                error
                .pow(2)
                .mean()
            )

        loss_values.append(
            float(loss.item())
        )

        timestep_values.append(
            int(t[0].item())
        )

    loss_array = np.asarray(
        loss_values,
        dtype=np.float64,
    )

    return {
        "loss_type": loss_type,
        "repeats": int(repeats),
        "seed": int(seed),
        "mean": float(loss_array.mean()),
        "std": float(loss_array.std()),
        "min": float(loss_array.min()),
        "max": float(loss_array.max()),
        "median": float(np.median(loss_array)),
        "losses": [
            float(v)
            for v in loss_values
        ],
        "timesteps": [
            int(v)
            for v in timestep_values
        ],
    }


def local_statistics(
    arr_norm,
    local_zyx,
    radius=2,
):
    """
    计算归一化空间中的点值和局部统计。
    """
    z, y, x = local_zyx

    region = arr_norm[
        max(0, z - radius):
        min(arr_norm.shape[0], z + radius + 1),

        max(0, y - radius):
        min(arr_norm.shape[1], y + radius + 1),

        max(0, x - radius):
        min(arr_norm.shape[2], x + radius + 1),
    ]

    return {
        "point_norm": float(
            arr_norm[z, y, x]
        ),
        "mean_norm": float(
            region.mean()
        ),
        "median_norm": float(
            np.median(region)
        ),
        "min_norm": float(
            region.min()
        ),
        "max_norm": float(
            region.max()
        ),
    }


def print_statistics(
    name,
    stats,
    hu_min,
    hu_max,
):
    """
    patch 文件仍然保存归一化值。

    为方便理解，这里同时打印：
    - normalized value
    - 对应 HU
    """
    print(f"\n{name}:")

    point_hu = normalized_to_hu(
        stats["point_norm"],
        hu_min,
        hu_max,
    )

    mean_hu = normalized_to_hu(
        stats["mean_norm"],
        hu_min,
        hu_max,
    )

    median_hu = normalized_to_hu(
        stats["median_norm"],
        hu_min,
        hu_max,
    )

    min_hu = normalized_to_hu(
        stats["min_norm"],
        hu_min,
        hu_max,
    )

    max_hu = normalized_to_hu(
        stats["max_norm"],
        hu_min,
        hu_max,
    )

    print(
        f"  point normalized : "
        f"{stats['point_norm']:.6f}"
    )
    print(
        f"  point equivalent HU : "
        f"{point_hu:.2f}"
    )

    print(
        f"  local mean normalized : "
        f"{stats['mean_norm']:.6f}"
    )
    print(
        f"  local mean equivalent HU : "
        f"{mean_hu:.2f}"
    )

    print(
        f"  local median normalized : "
        f"{stats['median_norm']:.6f}"
    )
    print(
        f"  local median equivalent HU : "
        f"{median_hu:.2f}"
    )

    print(
        f"  local min normalized : "
        f"{stats['min_norm']:.6f}"
    )
    print(
        f"  local min equivalent HU : "
        f"{min_hu:.2f}"
    )

    print(
        f"  local max normalized : "
        f"{stats['max_norm']:.6f}"
    )
    print(
        f"  local max equivalent HU : "
        f"{max_hu:.2f}"
    )


def print_noise_loss_stats(stats):
    print(
        "\n===== DDPM Noise Prediction Loss ====="
    )

    print(
        f"Loss type : {stats['loss_type']}"
    )
    print(
        f"Repeats   : {stats['repeats']}"
    )
    print(
        f"Loss seed : {stats['seed']}"
    )
    print(
        f"Mean      : {stats['mean']:.8f}"
    )
    print(
        f"Std       : {stats['std']:.8f}"
    )
    print(
        f"Median    : {stats['median']:.8f}"
    )
    print(
        f"Min       : {stats['min']:.8f}"
    )
    print(
        f"Max       : {stats['max']:.8f}"
    )

    print("\nIndividual results:")

    for index, (t, loss) in enumerate(
        zip(
            stats["timesteps"],
            stats["losses"],
        ),
        start=1,
    ):
        print(
            f"  repeat {index:02d}: "
            f"t={t:3d}, "
            f"loss={loss:.8f}"
        )


def main():
    args = parse_args()

    # 这个 seed 用于 CTA-like patch 的 DDPM sampling
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    nct_img, nct_arr = read_image(
        args.nct_path
    )

    # 预处理后的数据应当是 [-1,1]
    nct_arr = np.clip(
        nct_arr,
        -1.0,
        1.0,
    ).astype(np.float32)

    patch_size = (
        args.depth_size,
        args.input_size,
        args.input_size,
    )

    stride = (
        args.stride_d,
        args.stride_hw,
        args.stride_hw,
    )

    selected_xyz = (
        args.x,
        args.y,
        args.z,
    )

    origin_zyx = select_patch_origin(
        volume_shape=nct_arr.shape,
        selected_xyz=selected_xyz,
        patch_size=patch_size,
        stride=stride,
    )

    z0, y0, x0 = origin_zyx
    pd, ph, pw = patch_size

    # 选中点在 patch 内部的局部坐标
    local_zyx = (
        args.z - z0,
        args.y - y0,
        args.x - x0,
    )

    print(
        "Volume shape [D,H,W]:",
        nct_arr.shape,
    )

    print(
        "Selected ITK-SNAP coordinate [x,y,z]:",
        selected_xyz,
    )

    print(
        "Patch size [D,H,W]:",
        patch_size,
    )

    print(
        "Stride [D,H,W]:",
        stride,
    )

    print("\nSelected patch:")
    print(
        f"  z range: [{z0}, {z0 + pd})"
    )
    print(
        f"  y range: [{y0}, {y0 + ph})"
    )
    print(
        f"  x range: [{x0}, {x0 + pw})"
    )

    print(
        "Selected point local coordinate [z,y,x]:",
        local_zyx,
    )

    nct_patch_norm = crop_patch(
        nct_arr,
        origin_zyx,
        patch_size,
    )

    diffusion = build_diffusion(
        args
    )

    # 生成 CTA-like patch，输出仍为 [-1,1]
    generated_patch_norm = generate_patch(
        diffusion,
        nct_patch_norm,
    )

    # 直接保存归一化 patch，不反归一化为 HU
    nct_patch_img = make_patch_image(
        nct_patch_norm,
        nct_img,
        origin_zyx,
    )

    generated_patch_img = make_patch_image(
        generated_patch_norm,
        nct_img,
        origin_zyx,
    )

    nct_out_path = (
        out_dir
        / "selected_nct_patch_normalized.nii.gz"
    )

    generated_out_path = (
        out_dir
        / "selected_cta_like_patch_normalized.nii.gz"
    )

    sitk.WriteImage(
        nct_patch_img,
        str(nct_out_path),
    )

    sitk.WriteImage(
        generated_patch_img,
        str(generated_out_path),
    )

    nct_stats = local_statistics(
        nct_patch_norm,
        local_zyx,
    )

    generated_stats = local_statistics(
        generated_patch_norm,
        local_zyx,
    )

    print_statistics(
        "NCT patch",
        nct_stats,
        args.hu_min,
        args.hu_max,
    )

    print_statistics(
        "Generated CTA-like patch",
        generated_stats,
        args.hu_min,
        args.hu_max,
    )

    if not args.cta_path:
        print(
            "\nWarning: 没有提供 --cta_path。"
        )
        print(
            "无法保存真实 CTA patch，"
            "也无法计算 DDPM noise prediction loss。"
        )

    else:
        cta_img, cta_arr = read_image(
            args.cta_path
        )

        if cta_arr.shape != nct_arr.shape:
            raise RuntimeError(
                f"NCT/CTA shape 不一致："
                f"NCT={nct_arr.shape}, "
                f"CTA={cta_arr.shape}"
            )

        cta_arr = np.clip(
            cta_arr,
            -1.0,
            1.0,
        ).astype(np.float32)

        cta_patch_norm = crop_patch(
            cta_arr,
            origin_zyx,
            patch_size,
        )

        # 保存归一化真实 CTA patch
        cta_patch_img = make_patch_image(
            cta_patch_norm,
            cta_img,
            origin_zyx,
        )

        cta_out_path = (
            out_dir
            / "selected_real_cta_patch_normalized.nii.gz"
        )

        sitk.WriteImage(
            cta_patch_img,
            str(cta_out_path),
        )

        cta_stats = local_statistics(
            cta_patch_norm,
            local_zyx,
        )

        print_statistics(
            "Real CTA patch",
            cta_stats,
            args.hu_min,
            args.hu_max,
        )

        # 计算 DDPM noise prediction loss
        noise_loss_stats = (
            calculate_ddpm_noise_prediction_loss(
                diffusion=diffusion,
                nct_patch_norm=nct_patch_norm,
                cta_patch_norm=cta_patch_norm,
                loss_type=args.loss_type,
                repeats=args.loss_repeats,
                seed=args.loss_seed,
            )
        )

        print_noise_loss_stats(
            noise_loss_stats
        )

        loss_json_path = (
            out_dir
            / "ddpm_noise_prediction_loss.json"
        )

        with open(
            loss_json_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                noise_loss_stats,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\nSaved real CTA patch:",
            cta_out_path,
        )

        print(
            "Saved noise loss JSON:",
            loss_json_path,
        )

    print(
        "\nSaved normalized NCT patch:",
        nct_out_path,
    )

    print(
        "Saved normalized CTA-like patch:",
        generated_out_path,
    )

    print(
        "\n注意：上述 NIfTI patch 均保持 [-1,1]，"
        "没有反归一化成 HU。"
    )


if __name__ == "__main__":
    main()