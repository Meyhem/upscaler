"""PSNR / SSIM on the Y channel (standard SR convention) and LPIPS."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma of RGB in [0,1] -> Y in [16, 235]."""
    w = torch.tensor([65.481, 128.553, 24.966], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True) + 16.0


def _prep(sr: torch.Tensor, hr: torch.Tensor, crop: int):
    sr = rgb_to_y(sr.clamp(0, 1).double())
    hr = rgb_to_y(hr.clamp(0, 1).double())
    if crop > 0:
        sr = sr[..., crop:-crop, crop:-crop]
        hr = hr[..., crop:-crop, crop:-crop]
    return sr, hr


def psnr_y(sr: torch.Tensor, hr: torch.Tensor, crop: int = 4, max_psnr: float = 100.0) -> float:
    sr, hr = _prep(sr, hr, crop)
    mse = ((sr - hr) ** 2).mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-12))
    return psnr.clamp(max=max_psnr).mean().item()


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim_y(sr: torch.Tensor, hr: torch.Tensor, crop: int = 4) -> float:
    sr, hr = _prep(sr, hr, crop)
    win = _gaussian_window().to(sr.device)
    conv = lambda t: F.conv2d(t, win)
    mu1, mu2 = conv(sr), conv(hr)
    s11 = conv(sr * sr) - mu1 ** 2
    s22 = conv(hr * hr) - mu2 ** 2
    s12 = conv(sr * hr) - mu1 * mu2
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return m.mean(dim=(1, 2, 3)).mean().item()


class LPIPSMetric:
    """LPIPS (AlexNet). Pretrained weights are allowed for evaluation only."""

    def __init__(self, device):
        import lpips

        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> float:
        return self.net(sr.float() * 2 - 1, hr.float() * 2 - 1).mean().item()
