import argparse
import glob
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from models import IMCNN, MRIMCNN, TransConv, UUDCNN

WEIGHT_FILES = {
    "transconv1": "transconv.pt",
    "transconv2": "transconv_512.pt",
    "uudcnn": "uudcnn.pt",
    "imcnn1": "imcnn.pt",
    "imcnn2": "imcnn_512.pt",
    "mrimcnn1": "mrimcnn.pt",
    "mrimcnn2": "mrimcnn_512.pt",
}

SENSING_MODELS = {
    "imcnn1": (IMCNN, dict(image_size=256, patch_size=2, temperature=0.05)),
    "imcnn2": (IMCNN, dict(image_size=512, patch_size=2, temperature=0.05)),
    "mrimcnn1": (MRIMCNN, dict(image_size=256, patch_size=2, temperature=0.05)),
    "mrimcnn2": (MRIMCNN, dict(image_size=512, patch_size=2, temperature=0.05)),
}


def compute_psnr(x1, x2):
    mse = torch.mean((x1 - x2) ** 2)
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def blend(upscaled, hr, mask):
    return upscaled + (hr - upscaled) * mask


def _load(model, path, device):
    if not os.path.isfile(path):
        print(f"[skip] weights not found: {path}")
        return None
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


class Views:
    def __init__(self, hr, device):
        self.hr = hr
        self.device = device
        self._memo = {}

    def get(self, key, make):
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    def pil(self, key):
        if key == "direct256":
            return self.get(key, lambda: self.hr.resize((256, 256), Image.BICUBIC))
        if key == "direct128":
            return self.get(key, lambda: self.pil("direct256").resize((128, 128), Image.BICUBIC))
        if key == "hr512":
            return self.get(key, lambda: self.hr.resize((512, 512), Image.BICUBIC))
        return self.get(key, lambda: self.pil("hr512").resize(
            (256, 256) if key == "hr256" else (128, 128), Image.BICUBIC))

    def cpu(self, key):
        return self.get(("cpu", key), lambda: to_tensor(self.pil(key)))

    def dev(self, key):
        return self.get(("dev", key), lambda: self.cpu(key).unsqueeze(0).to(self.device))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the reconstruction and region-sensing pipelines by average PSNR.")
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

    transconv1 = _load(TransConv().to(device), wpath("transconv1"), device)
    transconv2 = _load(TransConv().to(device), wpath("transconv2"), device)
    uudcnn = _load(UUDCNN().to(device), wpath("uudcnn"), device)

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

    def uudcnn_step1(v):
        return v.get("uudcnn_step1", lambda: uudcnn(v.dev("hr128")))

    def two_stage(v, mask1, mask2):
        step1 = uudcnn_step1(v)
        step2 = blend(step1, v.dev("hr256"), mask1(step1))
        step3 = uudcnn(step2)
        return blend(step3, v.dev("hr512"), mask2(step3))

    pipelines = []
    if transconv1 is not None:
        pipelines.append(("[TransConv 128->256]", lambda v: (
            v.cpu("direct256"), transconv1(v.dev("direct128")))))
    if uudcnn is not None and mrimcnn1 is not None:
        def uudcnn_mrimcnn_256(v):
            up = uudcnn(v.dev("direct128"))
            return v.cpu("direct256"), blend(up, v.dev("direct256"), mrimcnn1(up))
        pipelines.append(("[UUDCNN + MRIMCNN @256]", uudcnn_mrimcnn_256))
    if transconv1 is not None and transconv2 is not None and mrimcnn1 is not None:
        def transconv_two_stage(v):
            up = transconv1(v.dev("hr128"))
            return v.cpu("hr512"), transconv2(blend(up, v.dev("hr256"), mrimcnn1(up)))
        pipelines.append(("[TransConv two-stage 128->256->512]", transconv_two_stage))
    if uudcnn is not None and imcnn1 is not None and imcnn2 is not None:
        pipelines.append(("[UUDCNN + IMCNN two-stage 128->256->512]", lambda v: (
            v.cpu("hr512"), two_stage(v, imcnn1, imcnn2))))
    if uudcnn is not None and mrimcnn1 is not None and mrimcnn2 is not None:
        pipelines.append(("[UUDCNN + MRIMCNN two-stage 128->256->512]", lambda v: (
            v.cpu("hr512"), two_stage(v, mrimcnn1, mrimcnn2))))

    if pipelines:
        totals = [0.0] * len(pipelines)
        with torch.no_grad():
            for path in tqdm(image_paths):
                views = Views(Image.open(path).convert("RGB"), device)
                for i, (_, fn) in enumerate(pipelines):
                    target, output = fn(views)
                    totals[i] += compute_psnr(target, output.squeeze(0).cpu()).item()
        for (label, _), total in zip(pipelines, totals):
            print(f"{label} Average PSNR: {total / len(image_paths):.4f}")

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
