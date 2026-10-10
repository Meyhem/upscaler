"""Compact VGG-style SR generator (Real-ESRGAN "SRVGGNetCompact" design)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SRVGGCompact(nn.Module):
    """Plain conv body at LR resolution, PixelShuffle upsampling, global bilinear residual."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64,
                 num_conv: int = 16, scale: int = 4):
        super().__init__()
        self.scale = scale
        layers: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        layers += [nn.Conv2d(num_feat, num_out_ch * scale * scale, 3, 1, 1), nn.PixelShuffle(scale)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return self.body(x) + base
