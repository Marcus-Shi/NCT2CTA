# -*- coding: utf-8 -*-
"""
Paired NCT-to-CTA dataset for adapting Med-DDPM.

Place this file under the root directory of the original med-ddpm repository.

Expected directory structure:

dataset/
├── nct/
│   ├── case001.nii.gz
│   ├── case002.nii.gz
│   └── ...
└── cta/
    ├── case001.nii.gz
    ├── case002.nii.gz
    └── ...

Each NCT file must have a paired CTA file with the same basename.
For example:
    dataset/nct/case001.nii.gz
    dataset/cta/case001.nii.gz

The dataset returns:
    input  = NCT tensor, shape [1, D, H, W]
    target = CTA tensor, shape [1, D, H, W]

Both NCT and CTA are normalized from HU to [-1, 1] using the same fixed HU window.
"""

import os
from glob import glob
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset


def ct_window_norm(img: np.ndarray, hu_min: float = -200.0, hu_max: float = 1000.0) -> np.ndarray:
    """Convert CT HU values to [-1, 1] using a fixed HU window."""
    img = img.astype(np.float32)
    img = np.clip(img, hu_min, hu_max)
    img = (img - hu_min) / (hu_max - hu_min)  # [0, 1]
    img = img * 2.0 - 1.0                    # [-1, 1]
    return img.astype(np.float32)


def ct_window_denorm(img: np.ndarray, hu_min: float = -200.0, hu_max: float = 1000.0) -> np.ndarray:
    """Convert normalized CT image from [-1, 1] back to HU."""
    img = img.astype(np.float32)
    img = (img + 1.0) / 2.0
    img = img * (hu_max - hu_min) + hu_min
    return img.astype(np.float32)


def to_medddpm_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert image from [H, W, D] numpy array to [1, D, H, W] tensor.

    Original med-ddpm expects model input as [B, C, D, H, W].
    Dataset item should therefore be [C, D, H, W].
    """
    tensor = torch.from_numpy(img.astype(np.float32))  # [H, W, D]
    tensor = tensor.unsqueeze(0)                       # [1, H, W, D]
    tensor = tensor.transpose(3, 1)                    # [1, D, W, H]
    return tensor.contiguous()


def from_medddpm_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert med-ddpm tensor from [1, D, W, H] or [D, W, H] back to [H, W, D].
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().numpy().astype(np.float32)  # [D, W, H]
    arr = np.transpose(arr, (2, 1, 0))                      # [H, W, D]
    return arr


class NiftiCTPairImageGenerator(Dataset):
    """
    Paired continuous CT dataset for NCT-conditioned CTA generation.

    This replaces the original mask-conditioned dataset in med-ddpm.
    It does NOT call label2masks(), one-hot encode the condition, or use per-volume MinMaxScaler().
    """

    def __init__(
        self,
        input_folder: str,
        target_folder: str,
        input_size: int = 128,
        depth_size: int = 128,
        hu_min: float = -200.0,
        hu_max: float = 1000.0,
        strict_pairing: bool = True,
    ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.input_size = int(input_size)
        self.depth_size = int(depth_size)
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.strict_pairing = bool(strict_pairing)

        self.pair_files = self._pair_files()
        if len(self.pair_files) == 0:
            raise RuntimeError(
                f"No paired NCT/CTA NIfTI files found.\n"
                f"input_folder={self.input_folder}\n"
                f"target_folder={self.target_folder}"
            )

    @staticmethod
    def _case_id(path: str) -> str:
        name = os.path.basename(path)
        if name.endswith('.nii.gz'):
            return name[:-7]
        if name.endswith('.nii'):
            return name[:-4]
        return os.path.splitext(name)[0]

    def _pair_files(self) -> List[Tuple[str, str]]:
        nct_files = sorted(glob(os.path.join(self.input_folder, '*.nii')) +
                           glob(os.path.join(self.input_folder, '*.nii.gz')))
        cta_files = sorted(glob(os.path.join(self.target_folder, '*.nii')) +
                           glob(os.path.join(self.target_folder, '*.nii.gz')))

        nct_map = {self._case_id(p): p for p in nct_files}
        cta_map = {self._case_id(p): p for p in cta_files}
        common_ids = sorted(set(nct_map.keys()) & set(cta_map.keys()))

        if self.strict_pairing:
            missing_cta = sorted(set(nct_map.keys()) - set(cta_map.keys()))
            missing_nct = sorted(set(cta_map.keys()) - set(nct_map.keys()))
            if missing_cta or missing_nct:
                raise AssertionError(
                    'NCT and CTA filenames do not match.\n'
                    f'Missing CTA for NCT cases: {missing_cta[:10]}\n'
                    f'Missing NCT for CTA cases: {missing_nct[:10]}'
                )

        return [(nct_map[k], cta_map[k]) for k in common_ids]

    def _read_nifti(self, file_path: str) -> np.ndarray:
        img = nib.load(file_path).get_fdata(dtype=np.float32)
        return img.astype(np.float32)

    def _resize_img(self, img: np.ndarray) -> np.ndarray:
        """Resize image to [input_size, input_size, depth_size]. Input/output: [H, W, D]."""
        h, w, d = img.shape
        if h == self.input_size and w == self.input_size and d == self.depth_size:
            return img.astype(np.float32)
        scalar = tio.ScalarImage(tensor=img[np.newaxis, ...])
        resize = tio.Resize((self.input_size, self.input_size, self.depth_size))
        resized = np.asarray(resize(scalar))[0]
        return resized.astype(np.float32)

    def _load_process_ct(self, file_path: str) -> np.ndarray:
        img = self._read_nifti(file_path)
        img = ct_window_norm(img, self.hu_min, self.hu_max)
        img = self._resize_img(img)
        return img

    def sample_conditions(self, batch_size: int) -> torch.Tensor:
        """
        Used by original med-ddpm Trainer during periodic sampling.
        Returns NCT condition tensor with shape [B, 1, D, H, W].
        """
        indices = np.random.randint(0, len(self), int(batch_size))
        tensors = []
        for idx in indices:
            nct_path, _ = self.pair_files[idx]
            nct = self._load_process_ct(nct_path)
            tensors.append(to_medddpm_tensor(nct))
        return torch.stack(tensors, dim=0).cuda()

    def __len__(self) -> int:
        return len(self.pair_files)

    def __getitem__(self, index: int):
        nct_path, cta_path = self.pair_files[index]
        nct = self._load_process_ct(nct_path)
        cta = self._load_process_ct(cta_path)
        return {
            'input': to_medddpm_tensor(nct),
            'target': to_medddpm_tensor(cta),
            'input_path': nct_path,
            'target_path': cta_path,
        }
