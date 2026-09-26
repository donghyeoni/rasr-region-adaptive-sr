import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import IMCNN, MRIMCNN
from train_upscale import UPSCALERS, LoadDataset, make_loader

SENSING = {
    "imcnn": IMCNN,
    "mrimcnn": MRIMCNN,
}


def train_region(mask_net, upscaler, device, train_loader, optimizer, criterion,
                 num_epochs, scheduler=None):
    mask_net.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.no_grad():
                upscaled = upscaler(inputs)
            mask = mask_net(upscaled)
            reconstructed = upscaled + (targets - upscaled) * mask
            loss = criterion(reconstructed, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().double()

        avg_loss = epoch_loss.item() / len(train_loader)
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a region-selection model (IMCNN / MRIMCNN) through a frozen upscaler.")
    parser.add_argument("--data-dir", required=True,
                        help="Folder of high-resolution training images (e.g. COCO)")
    parser.add_argument("--model", default="mrimcnn", choices=sorted(SENSING),
                        help="Region-selection model to train")
    parser.add_argument("--upscaler", default="uducnn", choices=sorted(UPSCALERS),
                        help="Frozen upscaler architecture")
    parser.add_argument("--upscaler-ckpt", required=True,
                        help="Pretrained upscaler checkpoint (train with train_upscale.py first)")
    parser.add_argument("--lr-size", type=int, default=128,
                        help="Low-resolution input side; the mask sees the 2x upscaled image")
    parser.add_argument("--patch-size", type=int, default=2,
                        help="Importance-mask patch size (2 was best in our sweep)")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Softmax temperature of the soft mask")
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

    upscaler = UPSCALERS[args.upscaler]().to(device)
    upscaler.load_state_dict(torch.load(args.upscaler_ckpt, map_location=device,
                                        weights_only=True))
    upscaler.eval()
    for p in upscaler.parameters():
        p.requires_grad_(False)

    mask_net = SENSING[args.model](image_size=2 * args.lr_size,
                                   patch_size=args.patch_size,
                                   temperature=args.temperature).to(device)

    optimizer = optim.Adam(mask_net.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step,
                                          gamma=args.scheduler_gamma)
    criterion = nn.MSELoss()

    train_region(mask_net, upscaler, device, trainloader, optimizer, criterion,
                 args.epochs, scheduler)

    ckpt_path = args.checkpoint or os.path.join("weights", f"{args.model}.pt")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save(mask_net.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
