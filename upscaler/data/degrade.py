"""Real-ESRGAN-style degradations, batched on the device of the input tensor.

All randomness comes from an explicit torch.Generator that lives on the same
device as the images, so a given seed always produces the same LR images.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg, encode_jpeg

RESIZE_MODES = ("area", "bilinear", "bicubic")


# --- random helpers -------------------------------------------------------

def _uniform(gen: torch.Generator, n: int, lo: float, hi: float) -> torch.Tensor:
    return torch.rand(n, generator=gen, device=gen.device) * (hi - lo) + lo


def _bernoulli(gen: torch.Generator, n: int, p: float) -> torch.Tensor:
    return torch.rand(n, generator=gen, device=gen.device) < p


def _choice(gen: torch.Generator, probs) -> int:
    p = torch.tensor(probs, dtype=torch.float32, device=gen.device)
    return int(torch.multinomial(p, 1, generator=gen).item())


# --- kernels --------------------------------------------------------------

def _grid(size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    ax = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    return xx, yy


def gaussian_kernels(sigma_x, sigma_y, theta, beta, size: int) -> torch.Tensor:
    """(Generalized) anisotropic Gaussian kernels; beta=1 is a plain Gaussian."""
    xx, yy = _grid(size, sigma_x.device)
    c = theta.cos()[:, None, None]
    s = theta.sin()[:, None, None]
    u = c * xx + s * yy
    v = -s * xx + c * yy
    q = (u / sigma_x[:, None, None]) ** 2 + (v / sigma_y[:, None, None]) ** 2
    k = torch.exp(-0.5 * q.pow(beta[:, None, None]))
    return k / k.sum(dim=(1, 2), keepdim=True)


def sinc_kernels(omega: torch.Tensor, size: int) -> torch.Tensor:
    """Circular low-pass (2D sinc) kernels with cutoff `omega`; produce ringing artifacts."""
    xx, yy = _grid(size, omega.device)
    r = torch.sqrt(xx ** 2 + yy ** 2)[None]
    w = omega[:, None, None]
    safe_r = torch.where(r == 0, torch.ones_like(r), r)
    k = w * torch.special.bessel_j1(w * safe_r) / (2 * math.pi * safe_r)
    k = torch.where(r == 0, w ** 2 / (4 * math.pi), k)
    return k / k.sum(dim=(1, 2), keepdim=True)


def delta_kernels(n: int, size: int, device) -> torch.Tensor:
    k = torch.zeros(n, size, size, device=device)
    k[:, size // 2, size // 2] = 1.0
    return k


def random_kernels(gen, n, size, type_probs, sigma_range, betag_range, sinc_prob, omega_range):
    """Per-sample kernels. type_probs order: iso, aniso, generalized_iso, generalized_aniso."""
    dev = gen.device
    types = torch.multinomial(torch.tensor(type_probs, dtype=torch.float32, device=dev), n,
                              replacement=True, generator=gen)
    iso = (types == 0) | (types == 2)
    sx = _uniform(gen, n, *sigma_range)
    sy = torch.where(iso, sx, _uniform(gen, n, *sigma_range))
    theta = torch.where(iso, torch.zeros_like(sx), _uniform(gen, n, -math.pi, math.pi))
    beta = torch.where(types >= 2, _uniform(gen, n, *betag_range), torch.ones_like(sx))
    k = gaussian_kernels(sx, sy, theta, beta, size)
    use_sinc = _bernoulli(gen, n, sinc_prob)[:, None, None]
    sinc = sinc_kernels(_uniform(gen, n, *omega_range), size)
    return torch.where(use_sinc, sinc, k)


# --- ops ------------------------------------------------------------------

def _pad_mode(x: torch.Tensor, pad: int) -> str:
    return "reflect" if pad < min(x.shape[-2:]) else "replicate"


def filter2d(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Convolve each sample in the batch with its own kernel (same kernel for all channels)."""
    b, c, h, w = x.shape
    size = kernels.shape[-1]
    pad = size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode=_pad_mode(x, pad))
    weight = kernels.to(x.dtype).repeat_interleave(c, dim=0)[:, None]
    out = F.conv2d(x.reshape(1, b * c, h + 2 * pad, w + 2 * pad), weight, groups=b * c)
    return out.reshape(b, c, h, w)


def resize(x: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "area":
        return F.interpolate(x, size=size, mode="area")
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def random_resize(x, gen, probs, scale_range, base_hw):
    """Resize the batch relative to `base_hw`: probs = (up, down, keep)."""
    choice = _choice(gen, probs)
    if choice == 0:
        s = _uniform(gen, 1, 1.0, scale_range[1]).item()
    elif choice == 1:
        s = _uniform(gen, 1, scale_range[0], 1.0).item()
    else:
        s = 1.0
    mode = RESIZE_MODES[_choice(gen, (1 / 3, 1 / 3, 1 / 3))]
    size = (max(1, int(base_hw[0] * s)), max(1, int(base_hw[1] * s)))
    return resize(x, size, mode)


def add_noise(x, gen, gaussian_prob, sigma_range, poisson_range, gray_prob):
    """Per-sample Gaussian (sigma in 0-255 units) or Poisson noise, color or gray."""
    b, c, h, w = x.shape
    bshape = (b, 1, 1, 1)
    gray = _bernoulli(gen, b, gray_prob).view(bshape)
    use_gauss = _bernoulli(gen, b, gaussian_prob).view(bshape)

    sigma = (_uniform(gen, b, *sigma_range) / 255.0).view(bshape)
    n_color = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    n_gray = torch.randn((b, 1, h, w), generator=gen, device=x.device, dtype=x.dtype).expand_as(x)
    gauss = torch.where(gray, n_gray, n_color) * sigma

    scale = _uniform(gen, b, *poisson_range).view(bshape)
    img = x.clamp(0, 1)
    gray_img = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    vals = 256.0
    p_color = torch.poisson(img * vals, generator=gen) / vals - img
    p_gray = (torch.poisson(gray_img * vals, generator=gen) / vals - gray_img).expand_as(x)
    poisson = torch.where(gray, p_gray, p_color) * scale

    return x + torch.where(use_gauss, gauss, poisson)


def _encode(img_u8: torch.Tensor, quality: int) -> torch.Tensor:
    try:
        return encode_jpeg(img_u8, quality=quality).cpu()
    except (RuntimeError, NotImplementedError):  # no GPU JPEG encoder: fall back to CPU
        return encode_jpeg(img_u8.cpu(), quality=quality)


def jpeg(x, gen, quality_range):
    """Real JPEG round-trip per image (torchvision; nvjpeg on CUDA when available)."""
    qualities = _uniform(gen, x.shape[0], *quality_range).round().int().tolist()
    x8 = (x.clamp(0, 1) * 255).round().to(torch.uint8)
    data = [_encode(img.contiguous(), q) for img, q in zip(x8, qualities)]
    out = decode_jpeg(data, device=x.device)
    return torch.stack(out).to(x.dtype) / 255.0


def _separable_blur(x: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    b, c, h, w = x.shape
    pad = size // 2
    ax = torch.arange(size, device=x.device, dtype=x.dtype) - (size - 1) / 2
    k = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = k / k.sum()
    y = F.pad(x, (pad, pad, pad, pad), mode=_pad_mode(x, pad)).reshape(b * c, 1, h + 2 * pad, w + 2 * pad)
    y = F.conv2d(y, k.view(1, 1, 1, size))
    y = F.conv2d(y, k.view(1, 1, size, 1))
    return y.reshape(b, c, h, w)


def usm_sharp(x: torch.Tensor, weight: float = 0.5, radius: int = 50, threshold: int = 10) -> torch.Tensor:
    """Unsharp masking used on the training target (Real-ESRGAN 'USM ground truth')."""
    if radius % 2 == 0:
        radius += 1
    sigma = 0.3 * ((radius - 1) * 0.5 - 1) + 0.8  # OpenCV's sigma for a given kernel size
    residual = x - _separable_blur(x, radius, sigma)
    mask = (residual.abs() * 255 > threshold).to(x.dtype)
    soft = _separable_blur(mask, radius, sigma)
    sharp = (x + weight * residual).clamp(0, 1)
    return soft * sharp + (1 - soft) * x


# --- pipeline -------------------------------------------------------------

@dataclass
class PassConfig:
    blur_prob: float
    sigma_range: tuple[float, float]
    betag_range: tuple[float, float]
    sinc_prob: float
    resize_probs: tuple[float, float, float]  # up, down, keep
    resize_range: tuple[float, float]
    gaussian_noise_prob: float
    noise_range: tuple[float, float]
    poisson_range: tuple[float, float]
    gray_noise_prob: float
    jpeg_range: tuple[float, float]


def _first_pass() -> PassConfig:
    return PassConfig(1.0, (0.2, 3.0), (0.5, 4.0), 0.1, (0.2, 0.7, 0.1), (0.15, 1.5),
                      0.5, (1.0, 30.0), (0.05, 3.0), 0.4, (30.0, 95.0))


def _second_pass() -> PassConfig:
    return PassConfig(0.8, (0.2, 1.5), (0.5, 4.0), 0.1, (0.3, 0.4, 0.3), (0.3, 1.2),
                      0.5, (1.0, 25.0), (0.05, 2.5), 0.4, (30.0, 95.0))


def _as_value(v):
    return tuple(v) if isinstance(v, list) else v


@dataclass
class DegradeConfig:
    scale: int = 4
    kernel_size: int = 21
    kernel_type_probs: tuple[float, float, float, float] = (0.5, 0.3, 0.15, 0.05)
    pass1: PassConfig = field(default_factory=_first_pass)
    pass2: PassConfig = field(default_factory=_second_pass)
    final_sinc_prob: float = 0.8

    @classmethod
    def from_dict(cls, d: dict | None) -> "DegradeConfig":
        cfg = cls()
        for key, value in (d or {}).items():
            if key in ("pass1", "pass2"):
                p = getattr(cfg, key)
                for pk, pv in value.items():
                    if not hasattr(p, pk):
                        raise KeyError(f"unknown degradation key {key}.{pk}")
                    setattr(p, pk, _as_value(pv))
            elif hasattr(cfg, key):
                setattr(cfg, key, _as_value(value))
            else:
                raise KeyError(f"unknown degradation key {key}")
        return cfg


def _degrade_pass(x, gen, p: PassConfig, cfg: DegradeConfig, base_hw):
    n = x.shape[0]
    kernels = random_kernels(gen, n, cfg.kernel_size, cfg.kernel_type_probs, p.sigma_range,
                             p.betag_range, p.sinc_prob, (math.pi / 5, math.pi))
    use_blur = _bernoulli(gen, n, p.blur_prob)[:, None, None]
    kernels = torch.where(use_blur, kernels, delta_kernels(n, cfg.kernel_size, x.device))
    x = filter2d(x, kernels)
    x = random_resize(x, gen, p.resize_probs, p.resize_range, base_hw)
    return add_noise(x, gen, p.gaussian_noise_prob, p.noise_range, p.poisson_range, p.gray_noise_prob)


@torch.no_grad()
def degrade(hr: torch.Tensor, gen: torch.Generator, cfg: DegradeConfig | None = None) -> torch.Tensor:
    """HR batch (B,3,H,W) in [0,1] -> realistically degraded LR batch (B,3,H/s,W/s)."""
    cfg = cfg or DegradeConfig()
    b, _, h, w = hr.shape
    lr_hw = (h // cfg.scale, w // cfg.scale)

    x = _degrade_pass(hr, gen, cfg.pass1, cfg, (h, w))
    x = jpeg(x, gen, cfg.pass1.jpeg_range)
    x = _degrade_pass(x, gen, cfg.pass2, cfg, lr_hw)

    # final: resize to exact LR size + optional sinc, and JPEG — in random order
    sinc = sinc_kernels(_uniform(gen, b, math.pi / 3, math.pi), cfg.kernel_size)
    use_sinc = _bernoulli(gen, b, cfg.final_sinc_prob)[:, None, None]
    final_k = torch.where(use_sinc, sinc, delta_kernels(b, cfg.kernel_size, x.device))
    mode = RESIZE_MODES[_choice(gen, (1 / 3, 1 / 3, 1 / 3))]
    if _bernoulli(gen, 1, 0.5).item():
        x = filter2d(resize(x, lr_hw, mode), final_k)
        x = jpeg(x, gen, cfg.pass2.jpeg_range)
    else:
        x = jpeg(x, gen, cfg.pass2.jpeg_range)
        x = filter2d(resize(x, lr_hw, mode), final_k)
    return (x.clamp(0, 1) * 255).round() / 255
