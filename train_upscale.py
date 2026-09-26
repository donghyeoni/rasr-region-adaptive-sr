import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor
from tqdm import tqdm

from models import TransConv, UDUCNN, UUDCNN

UPSCALERS = {
    "transconv": TransConv,
    "uducnn": UDUCNN,
    "uudcnn": UUDCNN,
}


class LoadDataset(Dataset):
    def __init__(self, img_dir, lr_size=128, max_cache_size=1000):
        self.lr_size = int(lr_size)
        self.hr_size = 2 * int(lr_size)
        self.max_cache_size = int(max_cache_size)
        self._cache = {}

        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            paths.extend(glob.glob(os.path.join(img_dir, ext)))
        self.paths = sorted(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pair = self._cache.get(idx)
        if pair is None:
            img = Image.open(self.paths[idx]).convert("RGB")
            hr = resize(img, [self.hr_size, self.hr_size],
                        interpolation=InterpolationMode.BICUBIC)
            lr = resize(hr, [self.lr_size, self.lr_size],
                        interpolation=InterpolationMode.BICUBIC)
            pair = (lr, hr)
            if len(self._cache) < self.max_cache_size:
                self._cache[idx] = pair
        return to_tensor(pair[0]), to_tensor(pair[1])


def make_loader(dataset, batch_size, num_workers, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True,
                      pin_memory=device.type == "cuda")


def train(model, device, train_loader, optimizer, criterion, num_epochs, scheduler=None):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().double()

        avg_loss = epoch_loss.item() / len(train_loader)
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train an upscaler (TransConv / UDUCNN / UUDCNN).")
    parser.add_argument("--data-dir", required=True,
                        help="Folder of high-resolution training images (e.g. COCO)")
    parser.add_argument("--model", default="uudcnn", choices=sorted(UPSCALERS),
                        help="Upscaler to train")
    parser.add_argument("--lr-size", type=int, default=128,
                        help="Low-resolution input side; the HR target is 2x this")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-step", type=int, default=40,
                        help="StepLR: epochs between learning-rate decays")
    parser.add_argument("--scheduler-gamma", type=float, default=0.1,
                        help="StepLR: learning-rate decay factor")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint", default=None,
                        help="Output checkpoint path (default: weights/<model>.pt)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed torch/numpy RNG for reproducible training")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainset = LoadDataset(args.data_dir, lr_size=args.lr_size)
    print(f"# of trainset = {len(trainset)}")
    if len(trainset) == 0:
        raise RuntimeError(
            f"No images found in {args.data_dir}. "
            "Point --data-dir at your COCO image folder (see the README)."
        )

    trainloader = make_loader(trainset, args.batch_size, args.num_workers, device)

    model = UPSCALERS[args.model]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step,
                                          gamma=args.scheduler_gamma)
    criterion = nn.MSELoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs, scheduler)

    ckpt_path = args.checkpoint or os.path.join("weights", f"{args.model}.pt")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
