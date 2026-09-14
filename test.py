"""Evaluate the reconstruction + region-sensing pipelines by average PSNR.

Loads trained weights from --weights-dir (files listed in WEIGHT_FILES below;
any pipeline whose weights are missing is skipped) and reports the average
PSNR over the test images for each pipeline. Region sensing follows the RASR
composition: the mask keeps the top-K most important patches, which are
blended into the upscaled image as

    reconstructed = upscaled + (hr - upscaled) * mask

Usage:
    python test.py --data-dir data/COCO/Test
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from models import IMCNN, MRIMCNN, TransConv, UDUCNN, UUDCNN

# Expected file names inside --weights-dir. The @256 names match the training
# scripts' default output paths; the *_512 variants are the optional second
# stage (train with --lr-size 256 and a matching --checkpoint name).
WEIGHT_FILES = {
    "transconv1": "transconv.pt",
    "transconv2": "transconv_512.pt",
    "uducnn": "uducnn.pt",
    "uudcnn": "uudcnn.pt",
    "imcnn1": "imcnn.pt",
    "imcnn2": "imcnn_512.pt",
    "mrimcnn1": "mrimcnn.pt",
    "mrimcnn2": "mrimcnn_512.pt",
}

# Region-sensing model hyper-parameters matching those weights.
SENSING_MODELS = {
    "imcnn1": (IMCNN, dict(image_size=256, patch_size=2, temperature=0.05)),
    "imcnn2": (IMCNN, dict(image_size=512, patch_size=2, temperature=0.05)),
    "mrimcnn1": (MRIMCNN, dict(image_size=256, patch_size=2, temperature=0.05)),
    "mrimcnn2": (MRIMCNN, dict(image_size=512, patch_size=2, temperature=0.05)),
}


def compute_psnr(x1, x2):
    """PSNR in dB for tensors in the [0, 1] range."""
    mse = torch.mean((x1 - x2) ** 2)
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def blend(upscaled, hr, mask):
    """Blend the mask-selected high-resolution patches into the upscaled image."""
    return upscaled + (hr - upscaled) * mask


def _load(model, path, device):
    """Load a state dict into ``model``; return None if the file is missing."""
    if not os.path.isfile(path):
        print(f"[skip] weights not found: {path}")
        return None
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _evaluate(image_paths, fn):
    """Average PSNR of ``fn(hr_pil) -> (target, output)`` over all images."""
    total_psnr, total = 0.0, 0
    with torch.no_grad():
        for path in tqdm(image_paths):
            hr = Image.open(path).convert("RGB")
            target, output = fn(hr)
            total_psnr += compute_psnr(target, output).item()
            total += 1
    return total_psnr / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Folder of high-resolution test images (e.g. COCO)")
    parser.add_argument("--weights-dir", default="weights",
                        help="Folder containing the trained weight files")
    parser.add_argument("--viz-image", default=None,
                        help="Optional image path for a qualitative side-by-side")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def wpath(key):
        return os.path.join(args.weights_dir, WEIGHT_FILES[key])

    # Reconstruction models
    transconv1 = _load(TransConv().to(device), wpath("transconv1"), device)
    transconv2 = _load(TransConv().to(device), wpath("transconv2"), device)
    uducnn = _load(UDUCNN().to(device), wpath("uducnn"), device)
    uudcnn = _load(UUDCNN().to(device), wpath("uudcnn"), device)

    # Region-sensing models (hard top-K mask at eval time)
    def _sensing(key):
        cls, kwargs = SENSING_MODELS[key]
        m = _load(cls(**kwargs).to(device), wpath(key), device)
        if m is not None:
            m.use_hard_mask = True
        return m

    imcnn1 = _sensing("imcnn1")
    imcnn2 = _sensing("imcnn2")
    mrimcnn1 = _sensing("mrimcnn1")
    mrimcnn2 = _sensing("mrimcnn2")

    image_paths = sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for p in glob.glob(os.path.join(args.data_dir, ext))
    )
    if not image_paths:
        raise RuntimeError(
            f"No images found in {args.data_dir}. "
            "Point --data-dir at your COCO test folder (see the README)."
        )

    # --- Pipeline 1: TransConv plain upscaling 128 -> 256 -------------------
    if transconv1 is not None:
        def fn(hr):
            hr256 = hr.resize((256, 256), Image.BICUBIC)
            lr = hr256.resize((128, 128), Image.BICUBIC)
            lr_t = to_tensor(lr).unsqueeze(0).to(device)
            out = transconv1(lr_t).squeeze(0).cpu()
            return to_tensor(hr256), out

        print(f"[TransConv 128->256] Average PSNR: {_evaluate(image_paths, fn):.4f}")

    # --- Pipeline 2: UUDCNN + MRIMCNN mask blend at 256 (comparison) --------
    if uudcnn is not None and mrimcnn1 is not None:
        def fn(hr):
            hr256 = hr.resize((256, 256), Image.BICUBIC)
            lr = hr256.resize((128, 128), Image.BICUBIC)
            hr_t = to_tensor(hr256).unsqueeze(0).to(device)
            lr_t = to_tensor(lr).unsqueeze(0).to(device)
            up = uudcnn(lr_t)
            out = blend(up, hr_t, mrimcnn1(up)).squeeze(0).cpu()
            return to_tensor(hr256), out

        print(f"[UUDCNN + MRIMCNN @256] Average PSNR: {_evaluate(image_paths, fn):.4f}")

    # --- Pipeline 3: two-stage TransConv 128 -> 256 -> 512 ------------------
    if transconv1 is not None and transconv2 is not None and mrimcnn1 is not None:
        def fn(hr):
            hr512 = hr.resize((512, 512), Image.BICUBIC)
            hr256 = hr512.resize((256, 256), Image.BICUBIC)
            hr128 = hr512.resize((128, 128), Image.BICUBIC)
            hr_mid = to_tensor(hr256).unsqueeze(0).to(device)
            lr_t = to_tensor(hr128).unsqueeze(0).to(device)
            up = transconv1(lr_t)
            mid = blend(up, hr_mid, mrimcnn1(up))
            out = transconv2(mid).squeeze(0).cpu()
            return to_tensor(hr512), out

        print(f"[TransConv two-stage 128->256->512] Average PSNR: {_evaluate(image_paths, fn):.4f}")

    # --- Pipeline 4: two-stage UUDCNN + IMCNN 128 -> 256 -> 512 (final) -----
    if uudcnn is not None and imcnn1 is not None and imcnn2 is not None:
        def fn(hr):
            hr512 = hr.resize((512, 512), Image.BICUBIC)
            hr256 = hr512.resize((256, 256), Image.BICUBIC)
            hr128 = hr512.resize((128, 128), Image.BICUBIC)
            hr_full = to_tensor(hr512).unsqueeze(0).to(device)
            hr_mid = to_tensor(hr256).unsqueeze(0).to(device)
            lr_t = to_tensor(hr128).unsqueeze(0).to(device)
            step1 = uudcnn(lr_t)
            step2 = blend(step1, hr_mid, imcnn1(step1))
            step3 = uudcnn(step2)
            out = blend(step3, hr_full, imcnn2(step3)).squeeze(0).cpu()
            return to_tensor(hr512), out

        print(f"[UUDCNN + IMCNN two-stage 128->256->512] Average PSNR: {_evaluate(image_paths, fn):.4f}")

    # --- Pipeline 5: two-stage UUDCNN + MRIMCNN 128 -> 256 -> 512 (compare) -
    if uudcnn is not None and mrimcnn1 is not None and mrimcnn2 is not None:
        def fn(hr):
            hr512 = hr.resize((512, 512), Image.BICUBIC)
            hr256 = hr512.resize((256, 256), Image.BICUBIC)
            hr128 = hr512.resize((128, 128), Image.BICUBIC)
            hr_full = to_tensor(hr512).unsqueeze(0).to(device)
            hr_mid = to_tensor(hr256).unsqueeze(0).to(device)
            lr_t = to_tensor(hr128).unsqueeze(0).to(device)
            step1 = uudcnn(lr_t)
            step2 = blend(step1, hr_mid, mrimcnn1(step1))
            step3 = uudcnn(step2)
            out = blend(step3, hr_full, mrimcnn2(step3)).squeeze(0).cpu()
            return to_tensor(hr512), out

        print(f"[UUDCNN + MRIMCNN two-stage 128->256->512] Average PSNR: {_evaluate(image_paths, fn):.4f}")

    # --- Qualitative visualization -----------------------------------------
    if args.viz_image and os.path.isfile(args.viz_image) and uudcnn is not None:
        hr_512 = Image.open(args.viz_image).convert("RGB")
        hr_256 = hr_512.resize((256, 256), Image.BICUBIC)
        hr_128 = hr_512.resize((128, 128), Image.BICUBIC)
        lr_t = to_tensor(hr_128).unsqueeze(0).to(device)

        with torch.no_grad():
            step1 = uudcnn(lr_t)

        psnr_val = compute_psnr(to_tensor(hr_256), step1.squeeze(0).cpu())

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(hr_512)
        plt.title("Original", fontsize=15)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(step1.squeeze(0).cpu().permute(1, 2, 0))
        plt.title(f"Output (PSNR: {psnr_val:.2f} dB)", fontsize=15)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
