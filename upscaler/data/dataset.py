"""Training tiles (random augmented crops) and paired LR/HR validation images."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from upscaler.imageio import read_rgb

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"directory not found: {directory}")
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


class TileDataset(Dataset):
    """Random `crop`x`crop` HR crops with flips / transposes. LR is made later, on GPU."""

    def __init__(self, tiles_dir: str | Path, crop: int = 256, max_tiles: int | None = None):
        self.files = list_images(tiles_dir)
        if max_tiles:
            self.files = self.files[:max_tiles]
        if not self.files:
            raise FileNotFoundError(f"no tiles found in {tiles_dir}")
        self.crop = crop

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = read_rgb(self.files[i])
        _, h, w = img.shape
        c = self.crop
        if h < c or w < c:
            raise ValueError(f"{self.files[i]} is {w}x{h}, smaller than crop {c}")
        top = int(torch.randint(0, h - c + 1, (1,)))
        left = int(torch.randint(0, w - c + 1, (1,)))
        img = img[:, top:top + c, left:left + c]
        if torch.rand(1) < 0.5:
            img = img.flip(-1)
        if torch.rand(1) < 0.5:
            img = img.flip(-2)
        if torch.rand(1) < 0.5:
            img = img.transpose(-1, -2)
        return img.contiguous()


class PairedImageDataset(Dataset):
    """`root/lr/<name>` and `root/hr/<name>` pairs (fixed validation set)."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.lr_files = list_images(root / "lr")
        if not self.lr_files:
            raise FileNotFoundError(f"no LR images in {root / 'lr'}")
        self.hr_dir = root / "hr"
        for p in self.lr_files:
            if not (self.hr_dir / p.name).is_file():
                raise FileNotFoundError(f"missing HR image {self.hr_dir / p.name}")

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, i: int):
        p = self.lr_files[i]
        return read_rgb(p), read_rgb(self.hr_dir / p.name), p.name
