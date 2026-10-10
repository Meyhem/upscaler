"""Tiled inference so large images fit in GPU memory without visible seams."""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def tiled_forward(model: nn.Module, x: torch.Tensor, scale: int, tile: int = 512, tile_pad: int = 16,
                  out_device=None) -> torch.Tensor:
    """Upscale `x` tile by tile. Each tile is processed with `tile_pad` px of context on
    every side; the context is cropped away from the output before stitching. `out_device`
    (e.g. "cpu") keeps the full-size output off the GPU for very large images."""
    b, c, h, w = x.shape
    if tile <= 0 or (h <= tile and w <= tile):
        out = model(x)
        return out if out_device is None else out.to(out_device)
    out = torch.zeros(b, c, h * scale, w * scale, dtype=x.dtype, device=out_device or x.device)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - tile_pad, 0), max(x0 - tile_pad, 0)
            py1, px1 = min(y1 + tile_pad, h), min(x1 + tile_pad, w)
            sr = model(x[..., py0:py1, px0:px1])
            oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
            out[..., y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                sr[..., oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale].to(out.device)
    return out
