# Med-DDPM adaptation for NCT-to-CTA translation

This patch adapts the original Med-DDPM project from:

```text
segmentation mask -> MRI
```

to:

```text
NCT -> CTA-like CTA
```

The original repository is `mobaidoctor/med-ddpm`. This patch only adds new files and reuses the original `diffusion_model/` directory.

## Files included

```text
dataset_ct.py
train_ct.py
sample_ct.py
scripts/train_ct.sh
scripts/sample_ct.sh
README_NCT2CTA.md
```

## Put these files here

Place all files under the root directory of the original med-ddpm repository:

```text
med-ddpm/
├── diffusion_model/
├── dataset_ct.py
├── train_ct.py
├── sample_ct.py
└── scripts/
    ├── train_ct.sh
    └── sample_ct.sh
```

## Dataset structure

Prepare paired NCT and CTA NIfTI files:

```text
dataset/
├── nct/
│   ├── case001.nii.gz
│   ├── case002.nii.gz
│   └── ...
└── cta/
    ├── case001.nii.gz
    ├── case002.nii.gz
    └── ...
```

The basenames must match.

Correct:

```text
dataset/nct/case001.nii.gz
dataset/cta/case001.nii.gz
```

## Training

From the med-ddpm root directory:

```bash
bash scripts/train_ct.sh
```

or directly:

```bash
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
  -r ""
```

## Sampling

After training, use a checkpoint such as:

```text
results_nct2cta/model-50.pt
```

Then run:

```bash
bash scripts/sample_ct.sh
```

or directly:

```bash
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
```

## What changed from original Med-DDPM?

Original Med-DDPM:

```text
condition = segmentation mask
target    = MRI
model input = concat(noisy MRI, mask)
model output = predicted noise
```

Modified NCT-to-CTA version:

```text
condition = NCT
target    = CTA
model input = concat(noisy CTA, NCT)
model output = predicted noise
```

## Important preprocessing assumptions

Before training, each NCT-CTA pair should ideally already be:

```text
1. registered
2. resampled to the same spacing
3. reoriented consistently
4. cropped to the same cardiac region
5. matched in shape or safely resizable
```

If NCT and CTA are not spatially aligned, the diffusion model may learn blurred or anatomically inconsistent mappings.

## Suggested first-run settings for limited GPU memory

If `128x128x128` is too large, try:

```bash
python train_ct.py \
  -i dataset/nct \
  -t dataset/cta \
  --results_folder results_nct2cta_96 \
  --input_size 96 \
  --depth_size 96 \
  --num_channels 32 \
  --num_res_blocks 1 \
  --batchsize 1 \
  --timesteps 250 \
  --epochs 50000 \
  --train_lr 1e-5 \
  -r ""
```

## Notes

This is a baseline adaptation. It does not yet include residual CTA-NCT generation, vessel ROI loss, heart-mask guided training, registration, DDIM/DEIS fast sampling, or 2.5D slice consistency.
