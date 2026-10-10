"""Score an upscaling function on the fixed paired validation set."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from upscaler.data.dataset import PairedImageDataset
from upscaler.metrics import LPIPSMetric, psnr_y, ssim_y


def bicubic_upscale(lr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)


@torch.no_grad()
def evaluate_pairs(upscale_fn: Callable[[torch.Tensor], torch.Tensor], val_dir, device,
                   max_images: int | None = None, lpips_metric: LPIPSMetric | None = None) -> dict:
    ds = PairedImageDataset(val_dir)
    n = len(ds) if max_images is None else min(len(ds), max_images)
    if n == 0:
        raise ValueError(f"no validation images in {val_dir}")
    totals = {"psnr": 0.0, "ssim": 0.0}
    if lpips_metric is not None:
        totals["lpips"] = 0.0
    for i in range(n):
        lr, hr, _ = ds[i]
        lr, hr = lr[None].to(device), hr[None].to(device)
        sr = upscale_fn(lr).float().clamp(0, 1)
        sr = (sr * 255).round() / 255
        totals["psnr"] += psnr_y(sr, hr)
        totals["ssim"] += ssim_y(sr, hr)
        if lpips_metric is not None:
            totals["lpips"] += lpips_metric(sr, hr)
    return {k: v / n for k, v in totals.items()}
