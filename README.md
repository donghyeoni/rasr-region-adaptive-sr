# RASR: Region-Adaptive Super-Resolution

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

**RASR** reconstructs a **512² image from only 3 × 128² captured pixels**
(~19% of a full scan): **IMCNN** learns *where* a capture-limited UAV
should look next, and **UUDCNN** reconstructs the image around what it
captured — **+3.4 dB** over spending the same capture budget at random.

## Overview

A reconnaissance UAV can capture only **128² pixels per shot**, at any
altitude — flying high covers the whole scene coarsely, flying low sees a
small area sharply. RASR pairs the UAV with a server that has no such limit
and learns **where each lower, sharper pass should look**:

1. **Highest pass** — the UAV captures the whole scene in one 128² shot and
   sends it to the server.
2. The server upscales the shot 2x with a CNN and predicts a per-patch
   **importance map** over the upscaled view, then replies with the
   coordinates of the **top-K most important patches** — K sized so the
   patches fill exactly one more 128² shot (`K = 128² / patch_size²`).
3. **Mid pass** — the UAV descends and captures just those regions at twice
   the ground resolution; the server blends them into its upscaled base:

   ```
   reconstructed = upscaled + (hr - upscaled) * mask
   ```

4. **Lowest pass** — the same loop repeats on the 256² reconstruction,
   yielding a 512² image.

The server ends up with a 512² reconstruction while the UAV captured and
transmitted only 3 × 128² pixels — about 19% of a full-resolution scan.
Evaluation simulates the UAV's shots by bicubic-resizing a source image (see
Dataset).

## Pipeline

![RASR two-stage pipeline](assets/pipeline.svg)

Each stage upscales 2x (Reconstruction) and then blends in the real patches
captured by the next, lower pass (Sensing), each pass spending the same 128²
budget: stage (a) is the mid-altitude pass reconstructing 128 → 256, and
stage (b) is the lowest pass repeating the composition at 256 → 512.

## Models

Each model lives in its own module under `models/`. The final pipeline uses
**UUDCNN** for reconstruction and **IMCNN** for patch selection. UUDCNN is the
best upscaler in the measurements below. IMCNN is *not* the highest-scoring
mask — MRIMCNN edges it by 0.04 dB — but it reaches that with 5.5K parameters
against MRIMCNN's 16.6K, so under a parameter budget IMCNN is the pick. The
rest are compared against these two.

- **Upscalers** (all net 2x)
  - `TransConv` — single transposed-conv upscaler
  - `UDUCNN` — up / down / up CNN
  - `UUDCNN` — up / up / down CNN (refines features at 4x)
- **Region selection / importance masks**
  - `IMCNN` — single-resolution importance-mask CNN
  - `MRIMCNN` — multi-resolution importance-mask CNN (fuses full/½/¼ scales)

Models are trained with MSE loss, Adam, and a StepLR schedule, and evaluated
by PSNR, including a two-stage `128 -> 256 -> 512` reconstruction.

## Structure

```
RASR/
├── models/            # one module per model
│   ├── blocks.py      # ResidualBlock, ResidualBlock2
│   ├── transconv.py   # TransConv
│   ├── uducnn.py      # UDUCNN
│   ├── uudcnn.py      # UUDCNN
│   ├── imcnn.py       # IMCNN
│   └── mrimcnn.py     # MRIMCNN
├── train_upscale.py   # train an upscaler
├── train_region.py    # train region selection through a frozen upscaler
├── test.py            # PSNR evaluation of the pipelines
├── assets/            # result figures (shown below)
├── requirements.txt
└── LICENSE            # MIT
```

## Dataset

Training and evaluation use **COCO** images as high-resolution ground truth;
LR/HR pairs are generated on the fly by bicubic downsampling, so all you need
is a folder of images.

The dataset is **not included**. Download images from
[cocodataset.org](http://cocodataset.org) (any split works — only the images
are used, no annotations) and place them in two folders, e.g.:

```
data/COCO/Train/   # training images
data/COCO/Test/    # held-out test images
```

## Usage

```bash
pip install -r requirements.txt

# 1. Train an upscaler (transconv / uducnn / uudcnn)
python train_upscale.py --data-dir data/COCO/Train --model uudcnn --seed 0

# 2. Train region selection (imcnn / mrimcnn) through the frozen upscaler
python train_region.py --data-dir data/COCO/Train --model mrimcnn \
    --upscaler uudcnn --upscaler-ckpt weights/uudcnn.pt --seed 0

# 3. For the second stage (256 -> 512), train the masks at the 512 scale
python train_region.py --data-dir data/COCO/Train --model mrimcnn \
    --upscaler uudcnn --upscaler-ckpt weights/uudcnn.pt --seed 0 \
    --lr-size 256 --checkpoint weights/mrimcnn_512.pt

# 4. Evaluate PSNR across the pipelines
python test.py --data-dir data/COCO/Test --weights-dir weights
```

Region selection trains end-to-end through the frozen upscaler: the soft mask
blends true HR patches into the upscaled image and the MSE loss teaches it to
spend the patch budget where the upscaler's error is largest (at eval time the
mask switches to a hard top-K selection).

Run `python train_upscale.py -h` / `python train_region.py -h` for all
hyper-parameters (LR size, batch size, learning rate, scheduler, patch size,
temperature). `test.py` expects the
weight file names listed in `WEIGHT_FILES` at the top of the script inside
`--weights-dir`, and skips any pipeline whose weights are missing.
`--viz-image <path>` additionally renders a qualitative side-by-side
comparison.

## Results

Measured with this repository's code: every model trained for 50 epochs
(batch 4, Adam 1e-3, StepLR, seed 0) on **3,000 COCO train2017 images** and
evaluated as the average PSNR over **500 COCO val2017 images**.

Upscalers:

| Model | SRCNN (vanilla) | TransConv | UDUCNN | UUDCNN |
| --- | --- | --- | --- | --- |
| Parameters | 56K | 209K | 264K | 376K |
| Avg PSNR (dB) | 28.21 | 30.78 | 30.99 | **31.10** |

Region selection on top of the best upscaler (UUDCNN), sweeping the mask
patch size under the fixed capture budget (`K = 128² / patch_size²`, hard
top-K at evaluation):

![patch-size sweep](assets/region_sweep.png)

Both learned masks peak at patch size 2 and beat random selection at every
patch size (IMCNN: 5.5K parameters, MRIMCNN: 16.6K). The random baseline is
flat — under a fixed budget, *where* the patches go is what matters, not how
finely they are cut.

Two-stage reconstruction (128 → 256 → 512, masks applied at both passes,
PSNR on the 512² output):

| Pipeline | UUDCNN cascade | + Random | + IMCNN | + MRIMCNN |
| --- | --- | --- | --- | --- |
| Avg PSNR (dB) | 27.15 | 28.15 | **31.53** | 31.57 |
| Mask parameters | — | — | **5.5K** | 16.6K |

**IMCNN is the selected mask.** MRIMCNN is 0.04 dB higher, but spends 3x the
parameters to get there — inside the measurement spread and not worth the
cost. The gap that matters is learned vs. random (+3.4 dB), not IMCNN vs.
MRIMCNN.

Both passes on a COCO val2017 image — the mask spends each pass's budget on
the detail-heavy regions (face, hat boundary) and skips the flat background.
The mid pass lifts this image from 23.61 dB to 39.48 dB; the lowest pass,
whose fixed budget now covers only 6.25% of the 512² frame, still adds
+2.5 dB (29.39 → 31.88):

![stage-by-stage result](assets/result_stages.png)
