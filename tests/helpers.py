"""Shared synthetic fixtures for tests (no downloads)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from upscaler.models import build_model
from upscaler.utils import export_generator

TINY_CFG = {"type": "srvgg_compact", "num_feat": 32, "num_conv": 8, "scale": 4}


def make_tiny_checkpoint(path: Path, scale: int = 4) -> Path:
    torch.manual_seed(0)
    cfg = dict(TINY_CFG, scale=scale)
    export_generator(path, cfg, build_model(cfg).state_dict())
    return path


def random_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Smooth color blobs plus fine noise: has both structure and texture."""
    rng = np.random.default_rng(seed)
    base = rng.random((h // 8 + 2, w // 8 + 2, 3)).astype(np.float32)
    smooth = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    arr = np.clip(smooth * 0.8 + rng.random((h, w, 3)) * 0.2, 0, 1)
    return (arr * 255).astype(np.uint8)


def write_images(directory: Path, n: int, h: int, w: int, seed: int = 0) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = directory / f"img_{i:03d}.png"
        Image.fromarray(random_image(h, w, seed + i)).save(p)
        paths.append(p)
    return paths


def make_val_dir(root: Path, n: int = 2, lr_hw=(16, 16), scale: int = 4) -> Path:
    root = Path(root)
    (root / "lr").mkdir(parents=True, exist_ok=True)
    (root / "hr").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        hr = Image.fromarray(random_image(lr_hw[0] * scale, lr_hw[1] * scale, seed=100 + i))
        hr.save(root / "hr" / f"v{i}.png")
        hr.resize((lr_hw[1], lr_hw[0]), Image.BICUBIC).save(root / "lr" / f"v{i}.png")
    return root
