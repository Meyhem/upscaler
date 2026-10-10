"""Conversions between PIL images, files and float RGB tensors in [0, 1]."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.detach().float().clamp(0, 1) * 255).round().to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy(), "RGB")


def read_rgb(path: str | Path) -> torch.Tensor:
    with Image.open(path) as img:
        return pil_to_tensor(img)


def write_rgb(path: str | Path, t: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(t).save(path)
