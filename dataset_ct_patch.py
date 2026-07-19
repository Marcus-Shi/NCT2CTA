import random
from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset


class PairedCTPatchDataset(Dataset):
    def __init__(
        self,
        nct_dir,
        cta_dir,
        patch_size=(64, 128, 128),   # D, H, W
        samples_per_epoch=2000,
        case_list=None,
    ):
        """
        case_list: optional list of case IDs (e.g. "PAT_000", no extension)
            to restrict this dataset to. Used for train/val/test splits --
            the same case_list can be reused across differently resampled
            preprocessed folders (orig/2x/3x) since they share the same
            case IDs. If None, uses every matched case in nct_dir/cta_dir
            (original behavior).
        """
        self.nct_dir = Path(nct_dir)
        self.cta_dir = Path(cta_dir)

        self.nct_paths = sorted(self.nct_dir.glob("*.nii.gz"))
        self.cta_paths = sorted(self.cta_dir.glob("*.nii.gz"))

        self.nct_map = {p.name: p for p in self.nct_paths}
        self.cta_map = {p.name: p for p in self.cta_paths}

        self.case_names = sorted(set(self.nct_map.keys()) & set(self.cta_map.keys()))

        assert len(self.case_names) > 0, "没有找到匹配的 NCT/CTA 文件"
        assert len(self.case_names) == len(self.nct_paths) == len(self.cta_paths), \
            "NCT 和 CTA 文件数量或命名不一致"

        if case_list is not None:
            case_list_set = set(case_list)
            self.case_names = [c for c in self.case_names if self._case_id(c) in case_list_set]
            assert len(self.case_names) > 0, \
                f"case_list 过滤后没有匹配的病例，case_list={sorted(case_list_set)[:10]}..."

        self.patch_d, self.patch_h, self.patch_w = patch_size
        self.samples_per_epoch = samples_per_epoch

        print(f"Loaded {len(self.case_names)} paired NCT-CTA cases")
        print(f"Patch size: D={self.patch_d}, H={self.patch_h}, W={self.patch_w}")

    @staticmethod
    def _case_id(filename):
        if filename.endswith(".nii.gz"):
            return filename[:-7]
        if filename.endswith(".nii"):
            return filename[:-4]
        return filename

    def __len__(self):
        return self.samples_per_epoch

    def read_nii(self, path):
        img = sitk.ReadImage(str(path), sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        # SimpleITK 输出 shape 是 [D, H, W]
        return arr

    def random_crop_pair(self, nct, cta):
        """
        nct, cta shape: [D, H, W]
        return: [D_patch, H_patch, W_patch]
        """
        assert nct.shape == cta.shape, f"NCT/CTA shape 不一致: {nct.shape}, {cta.shape}"

        D, H, W = nct.shape

        pd, ph, pw = self.patch_d, self.patch_h, self.patch_w

        assert D >= pd, f"Depth 不够裁 patch: D={D}, patch_d={pd}"
        assert H >= ph, f"Height 不够裁 patch: H={H}, patch_h={ph}"
        assert W >= pw, f"Width 不够裁 patch: W={W}, patch_w={pw}"

        z0 = random.randint(0, D - pd)
        y0 = random.randint(0, H - ph)
        x0 = random.randint(0, W - pw)

        nct_patch = nct[z0:z0+pd, y0:y0+ph, x0:x0+pw]
        cta_patch = cta[z0:z0+pd, y0:y0+ph, x0:x0+pw]

        return nct_patch, cta_patch

    def to_tensor(self, arr):
        """
        arr: [D, H, W]
        return: [1, D, H, W]
        """
        return torch.from_numpy(arr).float().unsqueeze(0)

    def __getitem__(self, idx):
        case_name = random.choice(self.case_names)

        nct = self.read_nii(self.nct_map[case_name])
        cta = self.read_nii(self.cta_map[case_name])

        nct_patch, cta_patch = self.random_crop_pair(nct, cta)

        nct_patch = self.to_tensor(nct_patch)
        cta_patch = self.to_tensor(cta_patch)

        return {
            "input": nct_patch,    # NCT condition
            "target": cta_patch,   # CTA target
            "case_name": case_name,
        }

    def sample_conditions(self, batch_size=1):
        """
        给 Trainer 里定期 sample 用。
        返回 NCT patch condition: [B, 1, D, H, W]
        """
        conditions = []

        for _ in range(batch_size):
            case_name = random.choice(self.case_names)

            nct = self.read_nii(self.nct_map[case_name])
            cta = self.read_nii(self.cta_map[case_name])

            nct_patch, _ = self.random_crop_pair(nct, cta)

            nct_patch = self.to_tensor(nct_patch)
            conditions.append(nct_patch.unsqueeze(0))

        return torch.cat(conditions, dim=0).cuda()