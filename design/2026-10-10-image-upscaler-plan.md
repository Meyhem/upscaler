# Image Upscaler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train and ship a from-scratch ×4 real-world photo super-resolution model (SRVGGCompact generator, Real-ESRGAN-style degradations, L1 → GAN two-stage training) with an evaluation script and a tiled inference CLI.

**Architecture:** A small `upscaler/` library holds focused units (models, GPU degradation pipeline, datasets, losses, metrics, tiling, checkpoint utils, trainer). Three thin root-level CLIs (`train.py`, `evaluate.py`, `upscale.py`) wire them together. Training behaviour (stage 1 vs stage 2 vs smoke) is selected purely by YAML config with `_base_` inheritance.

**Tech Stack:** Python 3.12, PyTorch ≥ 2.7 + torchvision ≥ 0.22 (CUDA 12.8 wheels), numpy, Pillow, OpenCV (headless), PyYAML, TensorBoard, lpips, scikit-image (tests), tqdm, pytest, uv.

**Spec:** `design/2026-10-10-image-upscaler-design.md` — read it alongside this plan.

## Global Constraints

- Scale is ×4 only; `--outscale 2` is produced by Lanczos-downscaling the ×4 output.
- Generator is trained from random init. Only pretrained weights allowed: frozen VGG19 (perceptual loss, stage 2) and LPIPS (AlexNet, eval metric). Never initialize the generator from external weights.
- Generator defaults: `num_feat=64`, `num_conv=16`, `scale=4` → exactly 621,424 params. `tiny`: `num_feat=32`, `num_conv=8` → 89,040 params.
- Image tensors are float32 RGB in [0, 1], shape (B, 3, H, W) (or (3, H, W) for single images).
- All degradation randomness comes from an explicit `torch.Generator` on the images' device.
- PSNR/SSIM on Y channel (BT.601: `Y = 16 + 65.481R + 128.553G + 24.966B`), 4-px border crop; PSNR capped at 100 dB.
- Checkpoints embed `model_cfg`; shipped model = EMA generator weights.
- PyTorch must be a CUDA ≥ 12.8 build (RTX 5080 = sm_120).
- Tests never require downloaded datasets or pretrained-weight downloads.
- `datasets/`, `runs/`, `weights/`, `*.pt` are gitignored — never commit data or checkpoints.
- Python commands use the project venv: `.venv/bin/python …`.
- **Auto-push (CLAUDE.md):** every task ends with commit + `git push` to `origin main` once its tests pass.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Phone photos with EXIF orientation** → output must be upright (same orientation as an image viewer shows). Test in Task 15 (`test_exif_orientation_is_applied`).
2. **Non-RGB inputs (grayscale `L`, palette `P`, `RGBA`)** → CLI upscales them without crashing; alpha preserved for transparent ones. Tests in Task 15 (`test_grayscale_and_palette_inputs`, `test_rgba_keeps_alpha`).
3. **Tiny inputs (1×1, 3×5 px)** → model, tiling and CLI produce a 4× output instead of crashing. Tests in Task 2 (`test_tiny_input`) and Task 15 (`test_one_pixel_image`).
4. **Degradation on small crops / extreme random draws** (pass-2 downscale to a few px, reflect padding larger than the image) → no crash, output exactly HR/4. Test in Task 5 (`test_small_inputs_many_seeds`).
5. **Interrupted training resumed with `--resume`** → continues from the latest checkpoint iteration instead of restarting. Test in Task 13 (`test_resume_continues_from_latest_checkpoint`).

---

## File map

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml` | package + deps + pytest config | 1 |
| `upscaler/utils.py` | config loading (`_base_`), seeding; later EMA, checkpoint I/O, `load_generator` | 1, 6 |
| `upscaler/imageio.py` | PIL ⇄ tensor, read/write RGB | 1 |
| `upscaler/models/srvgg.py`, `upscaler/models/__init__.py` | generator + registry | 2 |
| `upscaler/models/discriminator.py` | U-Net SN discriminator | 3 |
| `upscaler/data/degrade.py` | degradation primitives (Task 4) + pipeline/config (Task 5) | 4, 5 |
| `upscaler/tiling.py` | tiled inference | 7 |
| `upscaler/data/dataset.py` | `TileDataset`, `PairedImageDataset`, `list_images` | 8 |
| `upscaler/metrics.py`, `upscaler/evaluation.py` | PSNR/SSIM/LPIPS; `evaluate_pairs`, `bicubic_upscale` | 9 |
| `upscaler/losses.py` | VGG perceptual loss, GAN loss | 10 |
| `upscaler/data/download.py` | dataset download/extract | 11 |
| `upscaler/data/prepare.py` | tile extraction + fixed val set | 12 |
| `upscaler/trainer.py`, `train.py`, `configs/*.yaml` | training loop + CLI + configs | 13 |
| `evaluate.py` | eval CLI: metrics table + comparison strips | 14 |
| `upscale.py` | inference CLI | 15 |
| `tests/helpers.py` | shared synthetic-data helpers | 6, 8 |

---

### Task 1: Project scaffold, environment, config & image I/O utils

**Files:**
- Create: `pyproject.toml`, `upscaler/__init__.py`, `upscaler/data/__init__.py`, `upscaler/models/__init__.py` (empty placeholder, filled in Task 2), `upscaler/utils.py`, `upscaler/imageio.py`, `tests/__init__.py`
- Test: `tests/test_utils.py`, `tests/test_imageio.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `upscaler.utils.deep_merge(base: dict, override: dict) -> dict` (returns new dict, inputs untouched)
  - `upscaler.utils.load_config(path: str | Path) -> dict` (resolves `_base_: <relative yaml>` recursively, key removed from result)
  - `upscaler.utils.set_seed(seed: int) -> None`
  - `upscaler.imageio.pil_to_tensor(img: PIL.Image) -> Tensor (3,H,W) float32 [0,1]`
  - `upscaler.imageio.tensor_to_pil(t: Tensor (3,H,W)) -> PIL.Image (RGB)` (clamps + rounds)
  - `upscaler.imageio.read_rgb(path) -> Tensor (3,H,W)`; `upscaler.imageio.write_rgb(path, t) -> None` (creates parent dirs)

- [ ] **Step 1: Create `pyproject.toml` and package skeleton**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "upscaler"
version = "0.1.0"
description = "x4 real-world photo super-resolution, trained from scratch"
requires-python = ">=3.12"
dependencies = [
  "torch>=2.7",
  "torchvision>=0.22",
  "numpy",
  "pillow",
  "opencv-python-headless",
  "pyyaml",
  "tensorboard",
  "lpips",
  "tqdm",
]

[project.optional-dependencies]
dev = ["pytest", "scikit-image"]

[tool.setuptools.packages.find]
include = ["upscaler*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Create empty files: `upscaler/__init__.py`, `upscaler/data/__init__.py`, `upscaler/models/__init__.py`, `tests/__init__.py`.

- [ ] **Step 2: Create the venv with CUDA 12.8 PyTorch**

```bash
cd /home/meyhem/dev/upscaler
command -v uv || pip install --user uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability() if torch.cuda.is_available() else None)"
```
Expected: torch ≥ 2.7, `True`, `(12, 0)`. If capability is not `(12, 0)` or CUDA is unavailable, stop and report — the wheel is wrong.

- [ ] **Step 3: Write the failing tests**

`tests/test_utils.py`:
```python
import random

import numpy as np
import torch

from upscaler.utils import deep_merge, load_config, set_seed


def test_deep_merge_nested_and_does_not_mutate():
    base = {"a": 1, "b": {"c": 2, "d": 3}, "g": None}
    over = {"b": {"c": 20}, "e": 5, "g": {"x": 1}}
    out = deep_merge(base, over)
    assert out == {"a": 1, "b": {"c": 20, "d": 3}, "e": 5, "g": {"x": 1}}
    assert base == {"a": 1, "b": {"c": 2, "d": 3}, "g": None}


def test_load_config_resolves_base(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\ntrain:\n  lr: 0.1\n  iters: 10\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\ntrain:\n  lr: 0.5\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"seed": 1, "train": {"lr": 0.5, "iters": 10}}


def test_set_seed_is_reproducible():
    set_seed(3)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(3)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b
```

`tests/test_imageio.py`:
```python
import torch

from upscaler.imageio import read_rgb, tensor_to_pil, write_rgb, pil_to_tensor


def test_roundtrip_png(tmp_path):
    t = torch.randint(0, 256, (3, 7, 5)).float() / 255
    write_rgb(tmp_path / "sub" / "x.png", t)
    back = read_rgb(tmp_path / "sub" / "x.png")
    assert back.shape == (3, 7, 5)
    assert back.dtype == torch.float32
    assert torch.allclose(back, t, atol=1e-6)


def test_tensor_to_pil_clamps():
    t = torch.full((3, 2, 2), 2.0)
    t[0] = -1.0
    img = tensor_to_pil(t)
    assert img.mode == "RGB"
    back = pil_to_tensor(img)
    assert back[0].max() == 0.0 and back[1].min() == 1.0
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_utils.py tests/test_imageio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upscaler.utils'` / `'upscaler.imageio'`.

- [ ] **Step 5: Implement `upscaler/utils.py`**

```python
"""Config loading, seeding, checkpoint I/O and EMA."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config; a `_base_` key names a parent config relative to this file."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

- [ ] **Step 6: Implement `upscaler/imageio.py`**

```python
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
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_utils.py tests/test_imageio.py -v`
Expected: 5 passed.

- [ ] **Step 8: Commit and push**

```bash
git add pyproject.toml upscaler tests
git commit -m "feat: project scaffold, config loading and image I/O

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 2: Generator `SRVGGCompact` + model registry

**Files:**
- Create: `upscaler/models/srvgg.py`
- Modify: `upscaler/models/__init__.py` (replace empty file)
- Test: `tests/test_srvgg.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `upscaler.models.srvgg.SRVGGCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, scale=4)`; attribute `.scale: int`; `forward(x: (B,3,H,W)) -> (B,3,sH,sW)` (not clamped).
  - `upscaler.models.build_model(cfg: dict) -> nn.Module` where `cfg = {"type": "srvgg_compact", "num_feat": ..., "num_conv": ..., "scale": ...}`; raises `ValueError` for unknown `type`; does not mutate `cfg`.
  - `upscaler.models.MODELS: dict[str, type]`

- [ ] **Step 1: Write the failing tests** — `tests/test_srvgg.py`

```python
import pytest
import torch

from upscaler.models import build_model
from upscaler.models.srvgg import SRVGGCompact


@pytest.mark.parametrize("shape", [(1, 3, 16, 16), (2, 3, 17, 23)])
def test_output_is_4x(shape):
    net = SRVGGCompact()
    out = net(torch.rand(shape))
    assert out.shape == (shape[0], 3, shape[2] * 4, shape[3] * 4)


def test_tiny_input():
    net = SRVGGCompact(num_feat=32, num_conv=8)
    assert net(torch.rand(1, 3, 1, 1)).shape == (1, 3, 4, 4)
    assert net(torch.rand(1, 3, 3, 5)).shape == (1, 3, 12, 20)


def test_param_counts():
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(SRVGGCompact()) == 621_424
    assert count(SRVGGCompact(num_feat=32, num_conv=8)) == 89_040


def test_gradients_reach_first_conv():
    net = SRVGGCompact(num_feat=32, num_conv=8)
    net(torch.rand(1, 3, 8, 8)).mean().backward()
    first = net.body[0].weight.grad
    assert first is not None and first.abs().sum() > 0


def test_registry_builds_and_does_not_mutate():
    cfg = {"type": "srvgg_compact", "num_feat": 32, "num_conv": 8, "scale": 4}
    net = build_model(cfg)
    assert isinstance(net, SRVGGCompact) and net.scale == 4
    assert cfg["type"] == "srvgg_compact"


def test_registry_unknown_type():
    with pytest.raises(ValueError, match="unknown model type"):
        build_model({"type": "nope"})
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_srvgg.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_model'`.

- [ ] **Step 3: Implement `upscaler/models/srvgg.py`**

```python
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
```

- [ ] **Step 4: Implement `upscaler/models/__init__.py`**

```python
"""Model registry: config `type` name -> generator class."""
from __future__ import annotations

from torch import nn

from upscaler.models.srvgg import SRVGGCompact

MODELS: dict[str, type[nn.Module]] = {
    "srvgg_compact": SRVGGCompact,
}


def build_model(cfg: dict) -> nn.Module:
    kwargs = dict(cfg)
    model_type = kwargs.pop("type", None)
    if model_type not in MODELS:
        raise ValueError(f"unknown model type {model_type!r}; known: {sorted(MODELS)}")
    return MODELS[model_type](**kwargs)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_srvgg.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit and push**

```bash
git add upscaler/models tests/test_srvgg.py
git commit -m "feat: SRVGGCompact generator and model registry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 3: U-Net discriminator with spectral norm

**Files:**
- Create: `upscaler/models/discriminator.py`
- Test: `tests/test_discriminator.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `upscaler.models.discriminator.UNetDiscriminatorSN(num_in_ch=3, num_feat=64)`; `forward(x: (B,3,H,W)) -> (B,1,H,W)` per-pixel logits. H and W must be divisible by 8.

- [ ] **Step 1: Write the failing tests** — `tests/test_discriminator.py`

```python
import torch

from upscaler.models.discriminator import UNetDiscriminatorSN


def test_per_pixel_logits_shape():
    d = UNetDiscriminatorSN(num_feat=16)
    assert d(torch.rand(2, 3, 64, 72)).shape == (2, 1, 64, 72)


def test_param_count_default():
    d = UNetDiscriminatorSN()
    assert sum(p.numel() for p in d.parameters()) == 4_376_897


def test_backward_runs():
    d = UNetDiscriminatorSN(num_feat=16)
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    d(x).mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_discriminator.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `upscaler/models/discriminator.py`**

```python
"""U-Net discriminator with spectral normalization (Real-ESRGAN). Training only."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_feat: int = 64):
        super().__init__()
        sn = spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # downsample
        self.conv1 = sn(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = sn(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convs
        self.conv7 = sn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = sn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = lambda t: F.leaky_relu(t, 0.2)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        x0 = act(self.conv0(x))
        x1 = act(self.conv1(x0))
        x2 = act(self.conv2(x1))
        x3 = act(self.conv3(x2))
        x4 = act(self.conv4(up(x3))) + x2
        x5 = act(self.conv5(up(x4))) + x1
        x6 = act(self.conv6(up(x5))) + x0
        out = act(self.conv7(x6))
        out = act(self.conv8(out))
        return self.conv9(out)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_discriminator.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
git add upscaler/models/discriminator.py tests/test_discriminator.py
git commit -m "feat: U-Net spectral-norm discriminator

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 4: Degradation primitives (kernels, blur, resize, noise, JPEG, USM)

**Files:**
- Create: `upscaler/data/degrade.py`
- Test: `tests/test_degrade_ops.py`

**Interfaces:**
- Consumes: nothing.
- Produces (all in `upscaler.data.degrade`; `gen` is a `torch.Generator` on the images' device):
  - `gaussian_kernels(sigma_x, sigma_y, theta, beta, size) -> (N,size,size)` normalized; inputs are (N,) tensors.
  - `sinc_kernels(omega: (N,), size) -> (N,size,size)` normalized.
  - `delta_kernels(n, size, device) -> (N,size,size)` identity kernels.
  - `random_kernels(gen, n, size, type_probs, sigma_range, betag_range, sinc_prob, omega_range) -> (N,size,size)`
  - `filter2d(x: (B,C,H,W), kernels: (B,k,k)) -> (B,C,H,W)` per-sample depthwise conv, reflect pad (replicate if image too small).
  - `RESIZE_MODES = ("area", "bilinear", "bicubic")`; `resize(x, size: tuple[int,int], mode) -> Tensor`
  - `random_resize(x, gen, probs: (up,down,keep), scale_range: (lo,hi), base_hw: (h,w)) -> Tensor` — output size `max(1, int(base*s))`.
  - `add_noise(x, gen, gaussian_prob, sigma_range, poisson_range, gray_prob) -> Tensor` (not clamped; sigma_range in 0–255 units).
  - `jpeg(x, gen, quality_range) -> Tensor` clamped, on the 1/255 grid; per-image quality.
  - `usm_sharp(x, weight=0.5, radius=50, threshold=10) -> Tensor`
  - private helpers `_uniform(gen, n, lo, hi)`, `_bernoulli(gen, n, p)`, `_choice(gen, probs) -> int` (used by Task 5).

- [ ] **Step 1: Write the failing tests** — `tests/test_degrade_ops.py`

```python
import math

import torch

from upscaler.data import degrade as D


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_gaussian_kernels_normalized_and_iso_symmetric():
    one = torch.ones(1)
    k = D.gaussian_kernels(one * 2.0, one * 2.0, one * 0.0, one, 21)
    assert k.shape == (1, 21, 21)
    assert torch.allclose(k.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(k[0], k[0].T, atol=1e-7)


def test_anisotropic_rotation_changes_kernel():
    one = torch.ones(1)
    a = D.gaussian_kernels(one * 3.0, one * 0.5, one * 0.0, one, 21)
    b = D.gaussian_kernels(one * 3.0, one * 0.5, one * (math.pi / 4), one, 21)
    assert not torch.allclose(a, b)


def test_sinc_kernels_finite_normalized():
    k = D.sinc_kernels(torch.tensor([math.pi / 5, math.pi / 2, math.pi]), 21)
    assert torch.isfinite(k).all()
    assert torch.allclose(k.sum(dim=(1, 2)), torch.ones(3), atol=1e-5)


def test_random_kernels_deterministic():
    args = (8, 21, (0.5, 0.3, 0.15, 0.05), (0.2, 3.0), (0.5, 4.0), 0.3, (math.pi / 5, math.pi))
    a = D.random_kernels(gen(1), *args)
    b = D.random_kernels(gen(1), *args)
    assert a.shape == (8, 21, 21)
    assert torch.isfinite(a).all()
    assert torch.allclose(a.sum(dim=(1, 2)), torch.ones(8), atol=1e-4)
    assert torch.equal(a, b)


def test_filter2d_delta_is_identity_and_blur_changes():
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(D.filter2d(x, D.delta_kernels(2, 21, x.device)), x, atol=1e-6)
    one = torch.ones(2)
    blurred = D.filter2d(x, D.gaussian_kernels(one * 2, one * 2, one * 0, one, 21))
    assert blurred.shape == x.shape
    assert (blurred - x).abs().mean() > 0.01


def test_filter2d_image_smaller_than_kernel():
    x = torch.rand(1, 3, 4, 4)
    assert D.filter2d(x, D.delta_kernels(1, 21, x.device)).shape == (1, 3, 4, 4)


def test_random_resize_ranges():
    x = torch.rand(1, 3, 40, 60)
    keep = D.random_resize(x, gen(), (0.0, 0.0, 1.0), (0.5, 1.5), (40, 60))
    assert keep.shape[-2:] == (40, 60)
    for s in range(10):
        down = D.random_resize(x, gen(s), (0.0, 1.0, 0.0), (0.5, 1.5), (40, 60))
        assert 20 <= down.shape[-2] <= 40 and 30 <= down.shape[-1] <= 60
    tiny = D.random_resize(torch.rand(1, 3, 1, 1), gen(), (0.0, 1.0, 0.0), (0.15, 1.5), (1, 1))
    assert tiny.shape[-2:] == (1, 1)  # never shrinks to 0 px


def test_add_noise_changes_and_zero_sigma_is_identity():
    x = torch.rand(4, 3, 16, 16)
    noisy = D.add_noise(x, gen(), 0.5, (1, 30), (0.05, 3.0), 0.4)
    assert (noisy - x).abs().mean() > 0.001
    same = D.add_noise(x, gen(), 1.0, (0, 0), (0.05, 3.0), 0.4)
    assert torch.allclose(same, x)
    assert torch.equal(D.add_noise(x, gen(5), 0.5, (1, 30), (0.05, 3.0), 0.4),
                       D.add_noise(x, gen(5), 0.5, (1, 30), (0.05, 3.0), 0.4))


def test_jpeg_quantizes_and_changes():
    x = torch.rand(2, 3, 24, 24)
    out = D.jpeg(x, gen(), (30, 30))
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1
    assert torch.allclose(out * 255, (out * 255).round(), atol=1e-4)
    assert (out - x).abs().mean() > 0.01
    hi = D.jpeg(x, gen(), (95, 95))
    assert (hi - x).abs().mean() < (out - x).abs().mean()


def test_usm_sharp_constant_unchanged_edges_sharpened():
    flat = torch.full((1, 3, 64, 64), 0.5)
    assert torch.allclose(D.usm_sharp(flat), flat, atol=1e-6)
    step = torch.zeros(1, 3, 64, 64)
    step[..., 32:] = 0.8
    step += 0.1
    out = D.usm_sharp(step)
    assert out.min() >= 0 and out.max() <= 1
    assert (out - step).abs().sum() > 0
    assert out[..., 33].mean() > step[..., 33].mean()  # bright side of edge gets brighter
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_degrade_ops.py -v`
Expected: FAIL with `ImportError` (module `upscaler.data.degrade` missing).

- [ ] **Step 3: Implement `upscaler/data/degrade.py` (primitives only)**

```python
"""Real-ESRGAN-style degradations, batched on the device of the input tensor.

All randomness comes from an explicit torch.Generator that lives on the same
device as the images, so a given seed always produces the same LR images.
"""
from __future__ import annotations

import math

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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_degrade_ops.py -v`
Expected: 10 passed.

- [ ] **Step 5: Verify GPU JPEG path works on the 5080**

Run:
```bash
.venv/bin/python -c "
import torch; from upscaler.data import degrade as D
g = torch.Generator(device='cuda').manual_seed(0)
x = torch.rand(4, 3, 64, 64, device='cuda')
y = D.jpeg(x, g, (30, 95)); print(y.device, y.shape)
y = D.add_noise(x, g, 0.5, (1, 30), (0.05, 3), 0.4); print('noise ok')
print(D.random_kernels(g, 4, 21, (0.5,0.3,0.15,0.05), (0.2,3), (0.5,4), 0.1, (0.6,3.14)).shape)"
```
Expected: `cuda:0 torch.Size([4, 3, 64, 64])`, `noise ok`, `torch.Size([4, 21, 21])`.

- [ ] **Step 6: Commit and push**

```bash
git add upscaler/data/degrade.py tests/test_degrade_ops.py
git commit -m "feat: GPU degradation primitives (blur kernels, resize, noise, JPEG, USM)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 5: Degradation pipeline `degrade()` + `DegradeConfig`

**Files:**
- Modify: `upscaler/data/degrade.py` (append below the primitives; add `from dataclasses import dataclass, field` to imports)
- Test: `tests/test_degrade_pipeline.py`

**Interfaces:**
- Consumes: Task 4 primitives.
- Produces:
  - `PassConfig` dataclass: `blur_prob, sigma_range, betag_range, sinc_prob, resize_probs, resize_range, gaussian_noise_prob, noise_range, poisson_range, gray_noise_prob, jpeg_range`.
  - `DegradeConfig` dataclass: `scale=4, kernel_size=21, kernel_type_probs=(0.5,0.3,0.15,0.05), pass1, pass2, final_sinc_prob=0.8`; `DegradeConfig.from_dict(d: dict | None) -> DegradeConfig` (unknown key → `KeyError`; YAML lists → tuples).
  - `degrade(hr: (B,3,H,W), gen: torch.Generator, cfg: DegradeConfig | None = None) -> (B,3,H//scale,W//scale)` in [0,1], quantized to 1/255. Runs under `torch.no_grad()`.

- [ ] **Step 1: Write the failing tests** — `tests/test_degrade_pipeline.py`

```python
import pytest
import torch
import torch.nn.functional as F

from upscaler.data.degrade import DegradeConfig, degrade


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_shape_range_quantization():
    hr = torch.rand(2, 3, 64, 64)
    lr = degrade(hr, gen())
    assert lr.shape == (2, 3, 16, 16)
    assert lr.min() >= 0 and lr.max() <= 1
    assert torch.allclose(lr * 255, (lr * 255).round(), atol=1e-4)


def test_non_square():
    assert degrade(torch.rand(1, 3, 64, 96), gen()).shape == (1, 3, 16, 24)


def test_seeded_determinism():
    hr = torch.rand(2, 3, 64, 64)
    assert torch.equal(degrade(hr, gen(7)), degrade(hr, gen(7)))
    assert not torch.equal(degrade(hr, gen(7)), degrade(hr, gen(8)))


def test_differs_from_plain_bicubic():
    hr = torch.rand(2, 3, 64, 64)
    bic = F.interpolate(hr, size=(16, 16), mode="bicubic", align_corners=False).clamp(0, 1)
    assert (degrade(hr, gen()) - bic).abs().mean() > 0.01


def test_small_inputs_many_seeds():
    hr = torch.rand(2, 3, 32, 32)
    for s in range(25):
        lr = degrade(hr, gen(s))
        assert lr.shape == (2, 3, 8, 8)
        assert torch.isfinite(lr).all()


def test_from_dict_overrides_and_rejects_unknown():
    cfg = DegradeConfig.from_dict({"final_sinc_prob": 0.0, "pass1": {"jpeg_range": [90, 95]}})
    assert cfg.final_sinc_prob == 0.0
    assert cfg.pass1.jpeg_range == (90, 95)
    assert cfg.pass2.jpeg_range == (30, 95)
    assert DegradeConfig.from_dict(None).scale == 4
    with pytest.raises(KeyError):
        DegradeConfig.from_dict({"nope": 1})
    with pytest.raises(KeyError):
        DegradeConfig.from_dict({"pass2": {"nope": 1}})
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_degrade_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'DegradeConfig'`.

- [ ] **Step 3: Implement — add to the imports of `upscaler/data/degrade.py`**

```python
from dataclasses import dataclass, field
```

and append at the end of the file:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_degrade_pipeline.py tests/test_degrade_ops.py -v`
Expected: 16 passed.

- [ ] **Step 5: Measure GPU throughput (informational)**

```bash
.venv/bin/python -c "
import time, torch; from upscaler.data.degrade import degrade
g = torch.Generator(device='cuda').manual_seed(0); hr = torch.rand(32, 3, 256, 256, device='cuda')
for _ in range(3): degrade(hr, g)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): degrade(hr, g)
torch.cuda.synchronize(); print(f'{(time.time()-t)/20*1000:.1f} ms / batch of 32')"
```
Expected: a number printed. Record it in the commit message. If > 100 ms, note it for the user (it would dominate training time) but do not optimize now.

- [ ] **Step 6: Commit and push**

```bash
git add upscaler/data/degrade.py tests/test_degrade_pipeline.py
git commit -m "feat: two-pass Real-ESRGAN-style degradation pipeline (<N> ms/batch32 on 5080)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 6: EMA, checkpoint I/O, generator export/load

**Files:**
- Modify: `upscaler/utils.py` (append; add `import os` and `from upscaler.models import build_model` to imports)
- Create: `tests/helpers.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `build_model` (Task 2).
- Produces (in `upscaler.utils`):
  - `class EMA(model, decay)`: `.module` (eval copy, no grad), `.update(model)`, `.state_dict()`, `.load_state_dict(sd)`.
  - `save_checkpoint(path, state: dict) -> None` (atomic via `.tmp` + `os.replace`, creates parent dirs).
  - `latest_checkpoint(ckpt_dir) -> Path | None` (highest N among `iter_<N>.pt`).
  - `export_generator(path, model_cfg: dict, state_dict) -> None` → file `{"model_cfg", "params"}`.
  - `load_generator(path, device="cpu", expected_scale: int | None = None) -> nn.Module` (eval mode). Accepts exported files (`params`) and training checkpoints (`ema` + `model_cfg`). Raises `FileNotFoundError` (missing), `ValueError` (not a checkpoint / scale mismatch / architecture mismatch).
- `tests/helpers.py`: `TINY_CFG: dict`, `make_tiny_checkpoint(path, scale=4) -> Path`.

- [ ] **Step 1: Create `tests/helpers.py`**

```python
"""Shared synthetic fixtures for tests (no downloads)."""
from __future__ import annotations

from pathlib import Path

import torch

from upscaler.models import build_model
from upscaler.utils import export_generator

TINY_CFG = {"type": "srvgg_compact", "num_feat": 32, "num_conv": 8, "scale": 4}


def make_tiny_checkpoint(path: Path, scale: int = 4) -> Path:
    torch.manual_seed(0)
    cfg = dict(TINY_CFG, scale=scale)
    export_generator(path, cfg, build_model(cfg).state_dict())
    return path
```

- [ ] **Step 2: Write the failing tests** — `tests/test_checkpoint.py`

```python
import pytest
import torch

from tests.helpers import TINY_CFG, make_tiny_checkpoint
from upscaler.models import build_model
from upscaler.utils import EMA, latest_checkpoint, load_generator, save_checkpoint


def test_ema_update_math():
    net = torch.nn.Linear(2, 2)
    ema = EMA(net, decay=0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    before = [p.clone() for p in ema.module.parameters()]
    ema.update(net)
    for b, e, p in zip(before, ema.module.parameters(), net.parameters()):
        assert torch.allclose(e, 0.5 * b + 0.5 * p)
        assert not e.requires_grad
    assert not ema.module.training


def test_save_load_roundtrip_with_optimizer(tmp_path):
    net = build_model(TINY_CFG)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net(torch.rand(1, 3, 8, 8)).mean().backward()
    opt.step()
    save_checkpoint(tmp_path / "ck" / "iter_5.pt", {"iter": 5, "net": net.state_dict(), "opt": opt.state_dict()})
    ck = torch.load(tmp_path / "ck" / "iter_5.pt", weights_only=True)
    net2 = build_model(TINY_CFG)
    net2.load_state_dict(ck["net"])
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    opt2.load_state_dict(ck["opt"])
    assert ck["iter"] == 5
    for a, b in zip(net.state_dict().values(), net2.state_dict().values()):
        assert torch.equal(a, b)
    assert opt2.state_dict()["state"][0]["exp_avg"].equal(opt.state_dict()["state"][0]["exp_avg"])
    assert not (tmp_path / "ck" / "iter_5.pt.tmp").exists()


def test_latest_checkpoint_numeric_order(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    for n in (9, 10, 2):
        (tmp_path / f"iter_{n}.pt").write_bytes(b"x")
    assert latest_checkpoint(tmp_path).name == "iter_10.pt"


def test_export_and_load_generator_same_output(tmp_path):
    path = make_tiny_checkpoint(tmp_path / "g.pt")
    torch.manual_seed(0)
    ref = build_model(TINY_CFG).eval()
    model = load_generator(path, expected_scale=4)
    x = torch.rand(1, 3, 6, 6)
    assert torch.allclose(model(x), ref(x))
    assert not model.training


def test_load_generator_accepts_training_checkpoint(tmp_path):
    net = build_model(TINY_CFG)
    save_checkpoint(tmp_path / "iter_1.pt", {"model_cfg": TINY_CFG, "ema": net.state_dict(), "iter": 1})
    assert load_generator(tmp_path / "iter_1.pt").scale == 4


def test_load_generator_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_generator(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="x2 model"):
        load_generator(make_tiny_checkpoint(tmp_path / "x2.pt", scale=2), expected_scale=4)
    bad = tmp_path / "bad.pt"
    save_checkpoint(bad, {"model_cfg": dict(TINY_CFG, num_feat=16), "params": build_model(TINY_CFG).state_dict()})
    with pytest.raises(ValueError, match="do not match"):
        load_generator(bad)
    junk = tmp_path / "junk.pt"
    torch.save({"hello": 1}, junk)
    with pytest.raises(ValueError, match="not an upscaler checkpoint"):
        load_generator(junk)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL with `ImportError: cannot import name 'EMA'`.

- [ ] **Step 4: Implement — add to `upscaler/utils.py` imports**

```python
import os

from torch import nn

from upscaler.models import build_model
```

and append:

```python
class EMA:
    """Exponential moving average of a model's weights (the shipped generator)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1.0 - self.decay)
        for e, b in zip(self.module.buffers(), model.buffers()):
            e.copy_(b)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd)


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    files = list(Path(ckpt_dir).glob("iter_*.pt"))
    if not files:
        return None
    return max(files, key=lambda p: int(p.stem.split("_")[1]))


def export_generator(path: str | Path, model_cfg: dict, state_dict: dict) -> None:
    params = {k: v.detach().cpu() for k, v in state_dict.items()}
    save_checkpoint(path, {"model_cfg": dict(model_cfg), "params": params})


def load_generator(path: str | Path, device="cpu", expected_scale: int | None = None) -> nn.Module:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or "model_cfg" not in ckpt:
        raise ValueError(f"{path} is not an upscaler checkpoint (missing 'model_cfg')")
    params = ckpt.get("params", ckpt.get("ema"))
    if params is None:
        raise ValueError(f"{path} has no 'params' or 'ema' weights")
    cfg = ckpt["model_cfg"]
    if expected_scale is not None and cfg.get("scale") != expected_scale:
        raise ValueError(f"{path} is a x{cfg.get('scale')} model, expected x{expected_scale}")
    try:
        model = build_model(cfg)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{path}: cannot build model from {cfg}: {e}") from e
    try:
        model.load_state_dict(params)
    except RuntimeError as e:
        raise ValueError(f"{path}: weights do not match architecture {cfg}: {e}") from e
    return model.to(device).eval()
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py tests/test_utils.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit and push**

```bash
git add upscaler/utils.py tests/helpers.py tests/test_checkpoint.py
git commit -m "feat: EMA, atomic checkpoints, generator export/load

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 7: Tiled inference

**Files:**
- Create: `upscaler/tiling.py`
- Test: `tests/test_tiling.py`

**Interfaces:**
- Consumes: `SRVGGCompact` (tests only).
- Produces: `upscaler.tiling.tiled_forward(model, x: (B,3,H,W), scale: int, tile: int = 512, tile_pad: int = 16) -> (B,3,sH,sW)`; runs under `no_grad`; `tile <= 0` or image ≤ tile → single `model(x)` call. Caller sets eval mode / autocast.

- [ ] **Step 1: Write the failing tests** — `tests/test_tiling.py`

```python
import torch

from upscaler.models.srvgg import SRVGGCompact
from upscaler.tiling import tiled_forward


def _model():
    torch.manual_seed(0)
    return SRVGGCompact().eval()


def test_tiled_matches_full_image():
    model = _model()
    x = torch.rand(1, 3, 200, 300)
    with torch.no_grad():
        full = model(x)
    tiled = tiled_forward(model, x, 4, tile=64, tile_pad=24)
    assert tiled.shape == (1, 3, 800, 1200)
    assert (tiled - full).abs().max() < 1e-4


def test_odd_sizes_match():
    model = _model()
    x = torch.rand(1, 3, 45, 70)
    with torch.no_grad():
        full = model(x)
    assert (tiled_forward(model, x, 4, tile=32, tile_pad=24) - full).abs().max() < 1e-4


def test_disabled_or_large_tile_is_single_call():
    model = _model()
    x = torch.rand(1, 3, 20, 30)
    with torch.no_grad():
        full = model(x)
    assert torch.equal(tiled_forward(model, x, 4, tile=0), full)
    assert torch.equal(tiled_forward(model, x, 4, tile=512), full)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_tiling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upscaler.tiling'`.

- [ ] **Step 3: Implement `upscaler/tiling.py`**

```python
"""Tiled inference so large images fit in GPU memory without visible seams."""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def tiled_forward(model: nn.Module, x: torch.Tensor, scale: int, tile: int = 512, tile_pad: int = 16) -> torch.Tensor:
    """Upscale `x` tile by tile. Each tile is processed with `tile_pad` px of context on
    every side; the context is cropped away from the output before stitching."""
    b, c, h, w = x.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return model(x)
    out = x.new_zeros(b, c, h * scale, w * scale)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - tile_pad, 0), max(x0 - tile_pad, 0)
            py1, px1 = min(y1 + tile_pad, h), min(x1 + tile_pad, w)
            sr = model(x[..., py0:py1, px0:px1])
            oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
            out[..., y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                sr[..., oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_tiling.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
git add upscaler/tiling.py tests/test_tiling.py
git commit -m "feat: seamless tiled inference

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 8: Datasets (`TileDataset`, `PairedImageDataset`)

**Files:**
- Create: `upscaler/data/dataset.py`
- Modify: `tests/helpers.py` (append synthetic image helpers)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `read_rgb` (Task 1).
- Produces (in `upscaler.data.dataset`):
  - `IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}`; `list_images(directory) -> list[Path]` (sorted, non-recursive; `FileNotFoundError` if dir missing).
  - `TileDataset(tiles_dir, crop=256, max_tiles=None)`: `__getitem__ -> (3,crop,crop)` random crop + flips + transpose using the torch RNG. Raises `FileNotFoundError` when empty, `ValueError` when a tile is smaller than `crop`.
  - `PairedImageDataset(root)`: reads `root/lr/*` + `root/hr/<same name>`; `__getitem__ -> (lr (3,h,w), hr (3,H,W), name: str)`; `FileNotFoundError` when empty or an HR file is missing.
- `tests/helpers.py` additions: `random_image(h, w, seed=0) -> np.ndarray uint8 (h,w,3)`, `write_images(directory, n, h, w, seed=0) -> list[Path]`, `make_val_dir(root, n=2, lr_hw=(16,16), scale=4) -> Path`.

- [ ] **Step 1: Append to `tests/helpers.py`**

```python
import cv2
import numpy as np
from PIL import Image


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
```

(Place the new imports at the top of `tests/helpers.py` with the existing ones.)

- [ ] **Step 2: Write the failing tests** — `tests/test_dataset.py`

```python
import pytest
import torch
from PIL import Image

from tests.helpers import make_val_dir, write_images
from upscaler.data.dataset import PairedImageDataset, TileDataset, list_images


def test_tile_dataset_crops_and_limits(tmp_path):
    write_images(tmp_path, 5, 80, 96)
    ds = TileDataset(tmp_path, crop=64, max_tiles=3)
    assert len(ds) == 3
    x = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert 0 <= x.min() and x.max() <= 1


def test_tile_dataset_seeded_augmentation(tmp_path):
    write_images(tmp_path, 1, 80, 96)
    ds = TileDataset(tmp_path, crop=64)
    torch.manual_seed(1)
    a = ds[0]
    torch.manual_seed(1)
    assert torch.equal(a, ds[0])


def test_tile_dataset_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        TileDataset(tmp_path, crop=64)
    write_images(tmp_path, 1, 32, 32)
    with pytest.raises(ValueError, match="smaller than crop"):
        TileDataset(tmp_path, crop=64)[0]


def test_list_images_filters_extensions(tmp_path):
    write_images(tmp_path, 2, 8, 8)
    (tmp_path / "notes.txt").write_text("x")
    Image.new("RGB", (4, 4)).save(tmp_path / "a.JPG")
    assert [p.name for p in list_images(tmp_path)] == ["a.JPG", "img_000.png", "img_001.png"]


def test_paired_dataset(tmp_path):
    make_val_dir(tmp_path, n=2, lr_hw=(8, 12))
    ds = PairedImageDataset(tmp_path)
    lr, hr, name = ds[1]
    assert len(ds) == 2 and name == "v1.png"
    assert lr.shape == (3, 8, 12) and hr.shape == (3, 32, 48)


def test_paired_dataset_missing_hr(tmp_path):
    make_val_dir(tmp_path, n=1)
    (tmp_path / "hr" / "v0.png").unlink()
    with pytest.raises(FileNotFoundError, match="v0.png"):
        PairedImageDataset(tmp_path)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upscaler.data.dataset'`.

- [ ] **Step 4: Implement `upscaler/data/dataset.py`**

```python
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
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit and push**

```bash
git add upscaler/data/dataset.py tests/helpers.py tests/test_dataset.py
git commit -m "feat: tile and paired validation datasets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 9: Metrics (PSNR/SSIM/LPIPS) + evaluation helper

**Files:**
- Create: `upscaler/metrics.py`, `upscaler/evaluation.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `PairedImageDataset` (Task 8).
- Produces:
  - `upscaler.metrics.rgb_to_y(x: (B,3,H,W)) -> (B,1,H,W)` in [16, 235].
  - `upscaler.metrics.psnr_y(sr, hr, crop=4, max_psnr=100.0) -> float` (batch mean).
  - `upscaler.metrics.ssim_y(sr, hr, crop=4) -> float` (batch mean; 11×11 Gaussian σ=1.5, valid region).
  - `upscaler.metrics.LPIPSMetric(device)`; `__call__(sr, hr) -> float` (AlexNet LPIPS, inputs in [0,1]).
  - `upscaler.evaluation.bicubic_upscale(lr, scale=4) -> Tensor` clamped.
  - `upscaler.evaluation.evaluate_pairs(upscale_fn, val_dir, device, max_images=None, lpips_metric=None) -> dict` with keys `psnr`, `ssim` (+ `lpips` when metric given). Output of `upscale_fn` is clamped and quantized to 8 bit before scoring. `ValueError` if no images.

- [ ] **Step 1: Write the failing tests** — `tests/test_metrics.py`

```python
import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from tests.helpers import make_val_dir
from upscaler.evaluation import bicubic_upscale, evaluate_pairs
from upscaler.metrics import psnr_y, rgb_to_y, ssim_y


def _pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    hr = torch.rand(1, 3, 48, 40, generator=g)
    sr = (hr + 0.05 * torch.randn(hr.shape, generator=g)).clamp(0, 1)
    return sr, hr


def _y_np(x, crop=4):
    return rgb_to_y(x.double())[0, 0, crop:-crop, crop:-crop].numpy()


def test_rgb_to_y_range():
    assert rgb_to_y(torch.zeros(1, 3, 1, 1)).item() == pytest.approx(16.0)
    assert rgb_to_y(torch.ones(1, 3, 1, 1)).item() == pytest.approx(235.0)


def test_psnr_identical_is_capped():
    x = torch.rand(2, 3, 16, 16)
    assert psnr_y(x, x) == 100.0


def test_psnr_matches_skimage():
    sr, hr = _pair()
    ref = peak_signal_noise_ratio(_y_np(hr), _y_np(sr), data_range=255)
    assert psnr_y(sr, hr) == pytest.approx(ref, abs=1e-6)


def test_ssim_matches_skimage():
    sr, hr = _pair(1)
    ref = structural_similarity(_y_np(hr), _y_np(sr), data_range=255, gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False)
    assert ssim_y(sr, hr) == pytest.approx(ref, abs=1e-4)


def test_border_crop_ignores_border():
    hr = torch.rand(1, 3, 32, 32)
    sr = hr.clone()
    sr[..., :4, :] = 0
    assert psnr_y(sr, hr, crop=4) == 100.0
    assert psnr_y(sr, hr, crop=0) < 100.0


def test_evaluate_pairs(tmp_path):
    make_val_dir(tmp_path, n=3, lr_hw=(16, 16))
    calls = []

    def fn(lr):
        calls.append(lr.shape)
        return bicubic_upscale(lr)

    res = evaluate_pairs(fn, tmp_path, torch.device("cpu"), max_images=2)
    assert set(res) == {"psnr", "ssim"}
    assert len(calls) == 2
    assert 15 < res["psnr"] < 100 and 0 < res["ssim"] <= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upscaler.metrics'`.

- [ ] **Step 3: Implement `upscaler/metrics.py`**

```python
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
```

- [ ] **Step 4: Implement `upscaler/evaluation.py`**

```python
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
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit and push**

```bash
git add upscaler/metrics.py upscaler/evaluation.py tests/test_metrics.py
git commit -m "feat: Y-channel PSNR/SSIM, LPIPS and validation-set evaluation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 10: Losses (VGG perceptual, GAN)

**Files:**
- Create: `upscaler/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `upscaler.losses.PerceptualLoss(layer_weights: dict | None = None, pretrained: bool = True)`; `forward(x, target) -> scalar`. Default weights `{"conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1, "conv4_4": 1, "conv5_4": 1}`; features taken **before ReLU**; VGG frozen and always in eval mode; no grad through target.
  - `upscaler.losses.gan_loss(logits, target_is_real: bool) -> scalar` (BCE with logits, computed in fp32).

- [ ] **Step 1: Write the failing tests** — `tests/test_losses.py`

```python
import torch

from upscaler.losses import PerceptualLoss, gan_loss


def test_perceptual_zero_for_identical_and_positive_otherwise():
    loss = PerceptualLoss(pretrained=False)
    x = torch.rand(1, 3, 32, 32)
    assert loss(x, x.clone()).item() == 0.0
    assert loss(x, torch.rand(1, 3, 32, 32)).item() > 0.0


def test_perceptual_grad_flows_to_input_only():
    loss = PerceptualLoss(pretrained=False)
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    loss(x, torch.rand(1, 3, 32, 32)).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in loss.parameters())


def test_perceptual_stays_in_eval_mode_and_relu_not_inplace():
    loss = PerceptualLoss(pretrained=False)
    loss.train()
    assert not loss.training and not loss.vgg.training
    assert not any(getattr(m, "inplace", False) for m in loss.vgg)


def test_gan_loss_direction():
    big = torch.full((2, 1, 4, 4), 10.0)
    assert gan_loss(big, True).item() < 1e-3
    assert gan_loss(big, False).item() > 5.0
    assert gan_loss(-big, False).item() < 1e-3
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upscaler.losses'`.

- [ ] **Step 3: Implement `upscaler/losses.py`**

```python
"""Stage-2 losses: frozen-VGG19 perceptual loss and per-pixel GAN loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class PerceptualLoss(nn.Module):
    """L1 between VGG19 features (before ReLU). VGG is frozen and never shipped."""

    LAYER_IDS = {"conv1_2": 2, "conv2_2": 7, "conv3_4": 16, "conv4_4": 25, "conv5_4": 34}
    DEFAULT_WEIGHTS = {"conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1.0, "conv4_4": 1.0, "conv5_4": 1.0}

    def __init__(self, layer_weights: dict | None = None, pretrained: bool = True):
        super().__init__()
        self.layer_weights = dict(layer_weights or self.DEFAULT_WEIGHTS)
        last = max(self.LAYER_IDS[name] for name in self.layer_weights)
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        self.vgg = vgg19(weights=weights).features[: last + 1]
        for m in self.vgg:
            if isinstance(m, nn.ReLU):
                m.inplace = False  # in-place ReLU would overwrite the pre-activation features we keep
        self.vgg.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.train(False)

    def train(self, mode: bool = True):
        return super().train(False)  # always eval

    def _features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        wanted = {self.LAYER_IDS[name]: name for name in self.layer_weights}
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in wanted:
                feats[wanted[i]] = x
        return feats

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fx = self._features(x)
        with torch.no_grad():
            ft = self._features(target)
        return sum(self.layer_weights[n] * F.l1_loss(fx[n].float(), ft[n].float()) for n in fx)


def gan_loss(logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    logits = logits.float()
    target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_losses.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit and push**

```bash
git add upscaler/losses.py tests/test_losses.py
git commit -m "feat: VGG19 perceptual loss and GAN loss

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 11: Dataset download

**Files:**
- Create: `upscaler/data/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `URLS: dict[str, str]` — `DIV2K_train_HR`, `DIV2K_valid_HR`, `Flickr2K`.
  - `download_and_extract(name, url, raw_dir, keep_archive=False) -> Path` → extracts into `raw_dir/<name>/`, writes marker `raw_dir/<name>/.done`, skips if marker exists. Supports `.zip` and `.tar`/`.tar.gz`; other → `ValueError`.
  - `main(argv=None) -> int` CLI: `python -m upscaler.data.download [--raw-dir datasets/raw] [--only NAME ...] [--url NAME=URL ...] [--keep-archives]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_download.py`

```python
import tarfile
import zipfile

import pytest

from upscaler.data.download import download_and_extract


def _zip(tmp_path):
    src = tmp_path / "src.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("DIV2K_train_HR/0001.png", b"png-bytes")
    return src


def test_zip_download_extract_and_skip(tmp_path):
    src = _zip(tmp_path)
    raw = tmp_path / "raw"
    out = download_and_extract("DIV2K_train_HR", src.as_uri(), raw)
    assert (out / "DIV2K_train_HR" / "0001.png").read_bytes() == b"png-bytes"
    assert (out / ".done").exists()
    assert not (raw / "src.zip").exists()  # archive removed by default
    src.unlink()  # second call must not touch the URL
    assert download_and_extract("DIV2K_train_HR", src.as_uri(), raw) == out


def test_tar_extract(tmp_path):
    payload = tmp_path / "f.png"
    payload.write_bytes(b"abc")
    src = tmp_path / "Flickr2K.tar"
    with tarfile.open(src, "w") as t:
        t.add(payload, arcname="Flickr2K/Flickr2K_HR/000001.png")
    out = download_and_extract("Flickr2K", src.as_uri(), tmp_path / "raw", keep_archive=True)
    assert (out / "Flickr2K" / "Flickr2K_HR" / "000001.png").read_bytes() == b"abc"
    assert (tmp_path / "raw" / "Flickr2K.tar").exists()


def test_unknown_archive_type(tmp_path):
    src = tmp_path / "x.rar"
    src.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported archive"):
        download_and_extract("X", src.as_uri(), tmp_path / "raw")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `upscaler/data/download.py`**

```python
"""Download and extract the DF2K training data and DIV2K validation data.

python -m upscaler.data.download [--raw-dir datasets/raw] [--only DIV2K_valid_HR] [--url NAME=URL]
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

URLS = {
    "DIV2K_train_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "Flickr2K": "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar",
}


def _fetch(url: str, out: Path) -> None:
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=out.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def _extract(archive: Path, dest: Path) -> None:
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)  # zipfile strips absolute paths and '..'
    elif name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")


def download_and_extract(name: str, url: str, raw_dir: str | Path, keep_archive: bool = False) -> Path:
    raw_dir = Path(raw_dir)
    dest = raw_dir / name
    marker = dest / ".done"
    if marker.exists():
        print(f"{name}: already present, skipping")
        return dest
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / url.rsplit("/", 1)[-1]
    if not archive.exists():
        part = archive.with_name(archive.name + ".part")
        _fetch(url, part)
        part.rename(archive)
    dest.mkdir(parents=True, exist_ok=True)
    _extract(archive, dest)
    marker.touch()
    if not keep_archive:
        archive.unlink()
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="datasets/raw")
    ap.add_argument("--only", nargs="*", choices=sorted(URLS), help="subset of datasets")
    ap.add_argument("--url", action="append", default=[], metavar="NAME=URL", help="override a mirror URL")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args(argv)
    urls = dict(URLS)
    for item in args.url:
        key, _, value = item.partition("=")
        urls[key] = value
    failed = []
    for name in args.only or list(URLS):
        try:
            download_and_extract(name, urls[name], args.raw_dir, args.keep_archives)
        except OSError as e:
            print(f"{name}: download failed ({e}). Retry later or pass --url {name}=<mirror>", file=sys.stderr)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_download.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
git add upscaler/data/download.py tests/test_download.py
git commit -m "feat: DF2K / DIV2K download and extraction

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 12: Data preparation (tiles + fixed validation set)

**Files:**
- Create: `upscaler/data/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `IMAGE_EXTS` (Task 8), `degrade`, `DegradeConfig` (Task 5), `read_rgb`, `write_rgb` (Task 1).
- Produces:
  - `find_hr_images(root) -> list[Path]` — recursive, skips any path component containing `"LR"`.
  - `tile_positions(length, tile, stride) -> list[int]` — includes the last position flush with the edge; `[]` if `length < tile`.
  - `texture_score(tile_rgb_u8: np.ndarray) -> float` — std of grayscale Laplacian.
  - `extract_tiles(images, out_dir, tile=400, stride=200, min_texture=8.0) -> tuple[int, int]` (kept, dropped); files named `<stem>_<y:04d>_<x:04d>.png`.
  - `build_val_set(hr_images, out_dir, scale=4, seed=1234, degrade_cfg=None, device="cpu") -> int` — center-crop to multiple of `scale`, degrade with one seeded generator, write `out_dir/{hr,lr}/<stem>.png`.
  - `main(argv=None) -> int` CLI: `python -m upscaler.data.prepare [--raw-dir datasets/raw] [--out-dir datasets] [--tile 400] [--stride 200] [--min-texture 8.0] [--skip-tiles] [--skip-val]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_prepare.py`

```python
import numpy as np
from PIL import Image

from tests.helpers import random_image
from upscaler.data.prepare import (build_val_set, extract_tiles, find_hr_images, texture_score,
                                   tile_positions)


def test_tile_positions():
    assert tile_positions(1000, 400, 200) == [0, 200, 400, 600]
    assert tile_positions(450, 400, 200) == [0, 50]
    assert tile_positions(400, 400, 200) == [0]
    assert tile_positions(300, 400, 200) == []


def test_texture_score_flat_vs_textured():
    flat = np.full((64, 64, 3), 128, np.uint8)
    assert texture_score(flat) == 0.0
    assert texture_score(random_image(64, 64)) > 8.0


def test_find_hr_images_skips_lr(tmp_path):
    for rel in ["DIV2K/0001.png", "Flickr2K/Flickr2K_HR/000001.png", "Flickr2K/Flickr2K_LR_bicubic/X4/000001x4.png"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(p)
    names = [p.name for p in find_hr_images(tmp_path)]
    assert names == ["0001.png", "000001.png"]


def test_extract_tiles_drops_flat(tmp_path):
    tex = tmp_path / "tex.png"
    flat = tmp_path / "flat.png"
    Image.fromarray(random_image(120, 160)).save(tex)
    Image.fromarray(np.full((120, 160, 3), 90, np.uint8)).save(flat)
    kept, dropped = extract_tiles([tex, flat], tmp_path / "tiles", tile=64, stride=32)
    # positions: y in [0, 32, 56] x in [0, 32, 64, 96] -> 12 tiles per image
    assert (kept, dropped) == (12, 12)
    files = sorted(p.name for p in (tmp_path / "tiles").iterdir())
    assert files[0] == "tex_0000_0000.png" and len(files) == 12
    assert Image.open(tmp_path / "tiles" / files[0]).size == (64, 64)


def test_build_val_set_crops_and_is_deterministic(tmp_path):
    src = tmp_path / "0801.png"
    Image.fromarray(random_image(37, 50)).save(src)
    assert build_val_set([src], tmp_path / "a") == 1
    build_val_set([src], tmp_path / "b")
    assert Image.open(tmp_path / "a" / "hr" / "0801.png").size == (48, 36)
    assert Image.open(tmp_path / "a" / "lr" / "0801.png").size == (12, 9)
    assert (tmp_path / "a" / "lr" / "0801.png").read_bytes() == (tmp_path / "b" / "lr" / "0801.png").read_bytes()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `upscaler/data/prepare.py`**

```python
"""One-time data preparation: HR training tiles + fixed degraded validation set.

python -m upscaler.data.prepare [--raw-dir datasets/raw] [--out-dir datasets]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from upscaler.data.dataset import IMAGE_EXTS
from upscaler.data.degrade import DegradeConfig, degrade
from upscaler.imageio import read_rgb, write_rgb

TRAIN_SOURCES = ("DIV2K_train_HR", "Flickr2K")
VAL_SOURCE = "DIV2K_valid_HR"


def find_hr_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and not any("LR" in part for part in p.relative_to(root).parts)
    )


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    if length < tile:
        return []
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def texture_score(tile_rgb_u8: np.ndarray) -> float:
    gray = cv2.cvtColor(tile_rgb_u8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).std())


def extract_tiles(images, out_dir, tile: int = 400, stride: int = 200, min_texture: float = 8.0) -> tuple[int, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    for path in tqdm(images, desc="tiles"):
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"))
        h, w = img.shape[:2]
        for y in tile_positions(h, tile, stride):
            for x in tile_positions(w, tile, stride):
                t = img[y:y + tile, x:x + tile]
                if texture_score(t) < min_texture:
                    dropped += 1
                    continue
                cv2.imwrite(str(out_dir / f"{Path(path).stem}_{y:04d}_{x:04d}.png"),
                            cv2.cvtColor(t, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
                kept += 1
    return kept, dropped


def build_val_set(hr_images, out_dir, scale: int = 4, seed: int = 1234,
                  degrade_cfg: DegradeConfig | None = None, device="cpu") -> int:
    out_dir = Path(out_dir)
    gen = torch.Generator(device=device).manual_seed(seed)
    cfg = degrade_cfg or DegradeConfig(scale=scale)
    count = 0
    for path in tqdm(hr_images, desc="val"):
        hr = read_rgb(path)
        _, h, w = hr.shape
        th, tw = h - h % scale, w - w % scale
        top, left = (h - th) // 2, (w - tw) // 2
        hr = hr[:, top:top + th, left:left + tw]
        lr = degrade(hr[None].to(device), gen, cfg)[0].cpu()
        write_rgb(out_dir / "hr" / f"{Path(path).stem}.png", hr)
        write_rgb(out_dir / "lr" / f"{Path(path).stem}.png", lr)
        count += 1
    return count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="datasets/raw")
    ap.add_argument("--out-dir", default="datasets")
    ap.add_argument("--tile", type=int, default=400)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--min-texture", type=float, default=8.0)
    ap.add_argument("--skip-tiles", action="store_true")
    ap.add_argument("--skip-val", action="store_true")
    args = ap.parse_args(argv)
    raw, out = Path(args.raw_dir), Path(args.out_dir)

    if not args.skip_tiles:
        images = [p for src in TRAIN_SOURCES if (raw / src).is_dir() for p in find_hr_images(raw / src)]
        if not images:
            print(f"no training images under {raw}/{{{','.join(TRAIN_SOURCES)}}}", file=sys.stderr)
            return 1
        kept, dropped = extract_tiles(images, out / "df2k_tiles", args.tile, args.stride, args.min_texture)
        total = kept + dropped
        print(f"tiles: kept {kept}, dropped {dropped} ({100 * dropped / max(total, 1):.1f}% low-texture) "
              f"from {len(images)} images")

    if not args.skip_val:
        val_images = find_hr_images(raw / VAL_SOURCE) if (raw / VAL_SOURCE).is_dir() else []
        if not val_images:
            print(f"no validation images under {raw / VAL_SOURCE}", file=sys.stderr)
            return 1
        print(f"val: wrote {build_val_set(val_images, out / 'val_x4')} pairs to {out / 'val_x4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_prepare.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit and push**

```bash
git add upscaler/data/prepare.py tests/test_prepare.py
git commit -m "feat: tile extraction with texture filter and fixed validation set

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 13: Trainer, `train.py`, configs

**Files:**
- Create: `upscaler/trainer.py`, `train.py`, `configs/stage1_psnr.yaml`, `configs/stage2_gan.yaml`, `configs/smoke.yaml`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: everything from Tasks 1–10.
- Produces:
  - `upscaler.trainer.train(cfg: dict, resume: bool = False) -> dict` with keys `iter` (final iteration), `history` (list of `{"iter": int, "l_pix": float, ...}` logged every `log_every`; stage 2 adds `l_percep`, `l_gan`, `l_d`), `val` (last validation dict).
  - Run dir layout: `<run_dir>/config.yaml`, `<run_dir>/tb/`, `<run_dir>/ckpt/iter_<N>.pt` (keys `iter, net, ema, opt_g, model_cfg, torch_rng, degrade_rng`, optional `sched`, `net_d`, `opt_d`), `<run_dir>/ema.pt` (exported generator, latest).
  - `train.py` CLI: `python train.py --config PATH [--resume] [--run-dir DIR] [--device cpu|cuda]`.
  - Config keys (all required in `stage1_psnr.yaml`): `seed, run_dir, model{type,num_feat,num_conv,scale}, data{tiles_dir,max_tiles,crop,batch_size,num_workers}, degradation{}, train{total_iters,lr,scheduler(cosine|constant),eta_min,ema_decay,usm_gt,compile,init_from,pixel_weight,log_every,val_every,ckpt_every}, val{dir,max_images,lpips,tile}, gan(null | {d_num_feat,d_lr,perceptual_weight,gan_weight,vgg_pretrained})`; optional top-level `device`.

- [ ] **Step 1: Create the configs**

`configs/stage1_psnr.yaml`:
```yaml
# Stage 1: L1-only "PSNR" model, ~200k iters (target 2-4 h on RTX 5080).
seed: 0
run_dir: runs/stage1
model:
  type: srvgg_compact
  num_feat: 64
  num_conv: 16
  scale: 4
data:
  tiles_dir: datasets/df2k_tiles
  max_tiles: null
  crop: 256
  batch_size: 32
  num_workers: 8
degradation: {}          # empty = DegradeConfig defaults (Real-ESRGAN x4plus-like)
train:
  total_iters: 200000
  lr: 2.0e-4
  scheduler: cosine
  eta_min: 1.0e-6
  ema_decay: 0.999
  usm_gt: true
  compile: true
  init_from: null
  pixel_weight: 1.0
  log_every: 100
  val_every: 5000
  ckpt_every: 5000
val:
  dir: datasets/val_x4
  max_images: null
  lpips: true
  tile: 512
gan: null
```

`configs/stage2_gan.yaml`:
```yaml
# Stage 2: GAN fine-tune from the stage-1 EMA weights, ~100k iters (target 3-4 h).
_base_: stage1_psnr.yaml
run_dir: runs/stage2
train:
  total_iters: 100000
  lr: 1.0e-4
  scheduler: constant
  init_from: runs/stage1/ema.pt
gan:
  d_num_feat: 64
  d_lr: 1.0e-4
  perceptual_weight: 1.0
  gan_weight: 0.1
  vgg_pretrained: true
```

`configs/smoke.yaml`:
```yaml
# End-to-end sanity check on 50 real tiles: tiny model, 200 iters, < 1 min on GPU.
_base_: stage1_psnr.yaml
run_dir: runs/smoke
model:
  num_feat: 32
  num_conv: 8
data:
  max_tiles: 50
  batch_size: 4
  num_workers: 2
train:
  total_iters: 200
  compile: false
  log_every: 10
  val_every: 100
  ckpt_every: 100
val:
  max_images: 4
  lpips: false
```

- [ ] **Step 2: Write the failing tests** — `tests/test_trainer.py`

```python
from pathlib import Path

import torch

from tests.helpers import make_val_dir, write_images
from upscaler.trainer import train
from upscaler.utils import load_config, load_generator

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, config="smoke.yaml", run="run", **train_over):
    cfg = load_config(ROOT / "configs" / config)
    tiles, val = tmp_path / "tiles", tmp_path / "val"
    if not tiles.exists():
        write_images(tiles, 8, 96, 96)
        make_val_dir(val, n=2, lr_hw=(16, 16))
    cfg["device"] = "cpu"
    cfg["run_dir"] = str(tmp_path / run)
    cfg["model"].update(num_feat=32, num_conv=8)
    cfg["data"].update(tiles_dir=str(tiles), crop=64, batch_size=4, num_workers=0, max_tiles=None)
    cfg["val"].update(dir=str(val), max_images=2, lpips=False, tile=0)
    cfg["train"].update(total_iters=20, log_every=5, val_every=10, ckpt_every=10, compile=False)
    cfg["train"].update(train_over)
    return cfg


def test_stage1_runs_writes_outputs_and_learns(tmp_path):
    cfg = _cfg(tmp_path, total_iters=60, lr=1e-3, val_every=60, ckpt_every=30)
    res = train(cfg)
    run = Path(cfg["run_dir"])
    assert res["iter"] == 60
    assert {"psnr", "ssim"} <= set(res["val"])
    assert (run / "config.yaml").exists() and any((run / "tb").iterdir())
    assert sorted(p.name for p in (run / "ckpt").iterdir()) == ["iter_30.pt", "iter_60.pt"]
    losses = [h["l_pix"] for h in res["history"]]
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3
    model = load_generator(run / "ema.pt", expected_scale=4)
    assert model(torch.rand(1, 3, 8, 8)).shape == (1, 3, 32, 32)


def test_resume_continues_from_latest_checkpoint(tmp_path):
    cfg = _cfg(tmp_path)
    train(cfg)
    ckpt_dir = Path(cfg["run_dir"]) / "ckpt"
    (ckpt_dir / "iter_20.pt").unlink()  # simulate a crash after iter 10's checkpoint
    res = train(cfg, resume=True)
    assert res["iter"] == 20
    assert [h["iter"] for h in res["history"]] == [15, 20]
    assert (ckpt_dir / "iter_20.pt").exists()


def test_stage2_gan_from_stage1(tmp_path):
    train(_cfg(tmp_path, total_iters=10))
    cfg = _cfg(tmp_path, config="stage2_gan.yaml", run="gan", total_iters=4, log_every=2,
               val_every=4, ckpt_every=4, init_from=str(tmp_path / "run" / "ema.pt"))
    cfg["gan"]["vgg_pretrained"] = False
    res = train(cfg)
    assert {"l_pix", "l_percep", "l_gan", "l_d"} <= set(res["history"][-1])
    ck = torch.load(Path(cfg["run_dir"]) / "ckpt" / "iter_4.pt", weights_only=True)
    assert "net_d" in ck and "opt_d" in ck
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upscaler.trainer'`.

- [ ] **Step 4: Implement `upscaler/trainer.py`**

```python
"""Training loop for stage 1 (L1) and stage 2 (GAN); behaviour selected by config."""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from upscaler.data.dataset import PairedImageDataset, TileDataset
from upscaler.data.degrade import DegradeConfig, degrade, usm_sharp
from upscaler.evaluation import evaluate_pairs
from upscaler.losses import PerceptualLoss, gan_loss
from upscaler.metrics import LPIPSMetric
from upscaler.models import build_model
from upscaler.models.discriminator import UNetDiscriminatorSN
from upscaler.tiling import tiled_forward
from upscaler.utils import EMA, export_generator, latest_checkpoint, load_generator, save_checkpoint, set_seed


def _infinite(loader):
    while True:
        yield from loader


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _maybe_compile(net, enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return net
    try:
        compiled = torch.compile(net)
        with torch.no_grad(), _autocast(device):
            compiled(torch.zeros(1, 3, 32, 32, device=device))
        return compiled
    except Exception as e:  # compilation problems must not kill a training run
        print(f"warning: torch.compile failed ({e}); continuing without it", flush=True)
        return net


@torch.no_grad()
def _sample_grid(model, val_dir, device, scale: int, n: int = 4, crop: int = 64) -> torch.Tensor:
    """Rows of [bicubic | model | HR] center crops for TensorBoard."""
    ds = PairedImageDataset(val_dir)
    rows = []
    for i in range(min(n, len(ds))):
        lr, hr, _ = ds[i]
        _, h, w = lr.shape
        c = min(crop, h, w)
        t, l = (h - c) // 2, (w - c) // 2
        lr_c = lr[None, :, t:t + c, l:l + c].to(device)
        hr_c = hr[None, :, t * scale:(t + c) * scale, l * scale:(l + c) * scale].to(device)
        bic = F.interpolate(lr_c, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)
        sr = model(lr_c).float().clamp(0, 1)
        rows.append(torch.cat([bic, sr, hr_c], dim=-1))
    width = max(r.shape[-1] for r in rows)
    rows = [F.pad(r, (0, width - r.shape[-1])) for r in rows]
    return torch.cat(rows, dim=-2)[0]


def train(cfg: dict, resume: bool = False) -> dict:
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg["seed"])
    tcfg, dcfg, vcfg, gcfg = cfg["train"], cfg["data"], cfg["val"], cfg.get("gan")
    scale = cfg["model"]["scale"]
    run_dir = Path(cfg["run_dir"])
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # --- models / optimizers ---------------------------------------------
    net = build_model(cfg["model"]).to(device)
    if tcfg.get("init_from") and not resume:
        net.load_state_dict(load_generator(tcfg["init_from"], device).state_dict())
    ema = EMA(net, tcfg["ema_decay"])
    total = tcfg["total_iters"]
    opt_g = torch.optim.Adam(net.parameters(), lr=tcfg["lr"], betas=(0.9, 0.99))
    sched = None
    if tcfg["scheduler"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=total, eta_min=tcfg["eta_min"])
    net_d = opt_d = percep = None
    if gcfg:
        net_d = UNetDiscriminatorSN(num_feat=gcfg["d_num_feat"]).to(device)
        opt_d = torch.optim.Adam(net_d.parameters(), lr=gcfg["d_lr"], betas=(0.9, 0.99))
        percep = PerceptualLoss(pretrained=gcfg["vgg_pretrained"]).to(device)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])

    start = 0
    if resume:
        path = latest_checkpoint(ckpt_dir)
        if path is None:
            raise FileNotFoundError(f"--resume given but no checkpoint in {ckpt_dir}")
        ck = torch.load(path, map_location=device, weights_only=True)
        net.load_state_dict(ck["net"])
        ema.load_state_dict(ck["ema"])
        opt_g.load_state_dict(ck["opt_g"])
        if sched is not None:
            sched.load_state_dict(ck["sched"])
        if net_d is not None:
            net_d.load_state_dict(ck["net_d"])
            opt_d.load_state_dict(ck["opt_d"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        gen.set_state(ck["degrade_rng"].cpu())
        start = ck["iter"]
        print(f"resumed from {path} at iter {start}", flush=True)

    # --- data -------------------------------------------------------------
    ds = TileDataset(dcfg["tiles_dir"], crop=dcfg["crop"], max_tiles=dcfg.get("max_tiles"))
    if len(ds) < dcfg["batch_size"]:
        raise ValueError(f"{len(ds)} tiles < batch_size {dcfg['batch_size']}")
    nw = dcfg["num_workers"]
    loader = DataLoader(ds, batch_size=dcfg["batch_size"], shuffle=True, num_workers=nw,
                        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=nw > 0)
    batches = _infinite(loader)
    degrade_cfg = DegradeConfig.from_dict(cfg.get("degradation"))
    if degrade_cfg.scale != scale:
        raise ValueError(f"degradation scale {degrade_cfg.scale} != model scale {scale}")

    fwd = _maybe_compile(net, tcfg.get("compile", False), device)
    lpips_metric = LPIPSMetric(device) if vcfg["lpips"] else None
    writer = SummaryWriter(str(run_dir / "tb"))
    history: list[dict] = []
    last_val: dict = {}
    w_pix = tcfg.get("pixel_weight", 1.0)

    # --- loop -------------------------------------------------------------
    net.train()
    t0 = time.time()
    for it in range(start + 1, total + 1):
        hr = next(batches).to(device, non_blocking=True)
        lr_img = degrade(hr, gen, degrade_cfg)
        gt = usm_sharp(hr) if tcfg["usm_gt"] else hr

        if net_d is not None:
            net_d.requires_grad_(False)
        with _autocast(device):
            sr = fwd(lr_img)
            losses = {"l_pix": F.l1_loss(sr.float(), gt) * w_pix}
            if net_d is not None:
                losses["l_percep"] = percep(sr, gt) * gcfg["perceptual_weight"]
                losses["l_gan"] = gan_loss(net_d(sr), True) * gcfg["gan_weight"]
            loss_g = sum(losses.values())
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        if sched is not None:
            sched.step()

        if net_d is not None:
            net_d.requires_grad_(True)
            with _autocast(device):
                l_d = gan_loss(net_d(hr), True) + gan_loss(net_d(sr.detach()), False)
            opt_d.zero_grad(set_to_none=True)
            l_d.backward()
            opt_d.step()
            losses["l_d"] = l_d
        ema.update(net)

        if it % tcfg["log_every"] == 0:
            vals = {k: float(v) for k, v in losses.items()}
            its = tcfg["log_every"] / max(time.time() - t0, 1e-9)
            t0 = time.time()
            for k, v in vals.items():
                writer.add_scalar(f"train/{k}", v, it)
            writer.add_scalar("train/lr", opt_g.param_groups[0]["lr"], it)
            writer.add_scalar("train/it_per_s", its, it)
            print(f"[{it}/{total}] " + " ".join(f"{k}={v:.4f}" for k, v in vals.items()) + f" | {its:.1f} it/s",
                  flush=True)
            history.append({"iter": it, **vals})

        if it % tcfg["val_every"] == 0 or it == total:
            last_val = evaluate_pairs(lambda x: tiled_forward(ema.module, x, scale, vcfg["tile"], 16),
                                      vcfg["dir"], device, vcfg.get("max_images"), lpips_metric)
            for k, v in last_val.items():
                writer.add_scalar(f"val/{k}", v, it)
            writer.add_image("val/bicubic_sr_hr", _sample_grid(ema.module, vcfg["dir"], device, scale), it)
            print(f"[{it}/{total}] val " + " ".join(f"{k}={v:.4f}" for k, v in last_val.items()), flush=True)

        if it % tcfg["ckpt_every"] == 0 or it == total:
            state = {"iter": it, "net": net.state_dict(), "ema": ema.state_dict(), "opt_g": opt_g.state_dict(),
                     "model_cfg": cfg["model"], "torch_rng": torch.get_rng_state(),
                     "degrade_rng": gen.get_state()}
            if sched is not None:
                state["sched"] = sched.state_dict()
            if net_d is not None:
                state["net_d"] = net_d.state_dict()
                state["opt_d"] = opt_d.state_dict()
            save_checkpoint(ckpt_dir / f"iter_{it}.pt", state)
            export_generator(run_dir / "ema.pt", cfg["model"], ema.state_dict())

    writer.close()
    return {"iter": max(start, total), "history": history, "val": last_val}
```

- [ ] **Step 5: Implement `train.py`**

```python
"""Train the upscaler.

python train.py --config configs/stage1_psnr.yaml [--resume] [--run-dir runs/x] [--device cuda]
"""
from __future__ import annotations

import argparse
import sys

from upscaler.trainer import train
from upscaler.utils import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in run_dir")
    ap.add_argument("--run-dir")
    ap.add_argument("--device", choices=["cpu", "cuda"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.run_dir:
        cfg["run_dir"] = args.run_dir
    if args.device:
        cfg["device"] = args.device
    res = train(cfg, resume=args.resume)
    print(f"done at iter {res['iter']}; final val: {res['val']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_trainer.py -v`
Expected: 3 passed (CPU; may take ~30–60 s).

- [ ] **Step 7: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 8: Commit and push**

```bash
git add upscaler/trainer.py train.py configs tests/test_trainer.py
git commit -m "feat: two-stage trainer with EMA, bf16, resume, TensorBoard; configs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 14: `evaluate.py`

**Files:**
- Create: `evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `load_generator`, `tiled_forward`, `evaluate_pairs`, `bicubic_upscale`, `LPIPSMetric`, `PairedImageDataset`, `list_images`, `read_rgb`, `write_rgb`.
- Produces:
  - `evaluate.model_name(path) -> str` = `"<parent dir name>/<stem>"`.
  - `evaluate.run(ckpts, out_dir, val_dir, real_dir=None, device="cpu", max_images=None, use_lpips=True, crop=128, tile=512) -> dict` — writes `out_dir/results.json`, `out_dir/val/<stem>.png` strips (LR-nearest | bicubic | models… | HR), `out_dir/real/<stem>.png` strips (no HR). Strip columns are center crops of `crop` LR px (clipped to image size).
  - `evaluate.main(argv=None) -> int` CLI: `python evaluate.py --ckpt A [--ckpt B ...] --out DIR [--val-dir datasets/val_x4] [--real-dir datasets/real_test] [--max-images N] [--no-lpips] [--crop 128] [--tile 512] [--cpu]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_evaluate.py`

```python
import json

from PIL import Image

import evaluate
from tests.helpers import make_tiny_checkpoint, make_val_dir, write_images


def test_run_writes_results_and_strips(tmp_path):
    ckpt = make_tiny_checkpoint(tmp_path / "stage1" / "ema.pt")
    val = make_val_dir(tmp_path / "val", n=2, lr_hw=(16, 16))
    real = tmp_path / "real"
    write_images(real, 1, 20, 24)
    out = tmp_path / "eval"
    res = evaluate.run([ckpt], out, val, real_dir=real, use_lpips=False, crop=8, tile=0)
    assert set(res) == {"bicubic", "stage1/ema"}
    assert json.loads((out / "results.json").read_text()) == res
    # val strip: nearest | bicubic | model | HR = 4 columns of 8*4 px
    assert Image.open(out / "val" / "v0.png").size == (4 * 32, 32)
    # real strip: nearest | bicubic | model
    assert Image.open(out / "real" / "img_000.png").size == (3 * 32, 32)


def test_cli_bad_checkpoint_returns_1(tmp_path, capsys):
    val = make_val_dir(tmp_path / "val")
    rc = evaluate.main(["--ckpt", str(tmp_path / "missing.pt"), "--out", str(tmp_path / "o"),
                        "--val-dir", str(val), "--no-lpips", "--cpu"])
    assert rc == 1
    assert "checkpoint not found" in capsys.readouterr().err
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`.

- [ ] **Step 3: Implement `evaluate.py`**

```python
"""Score checkpoints vs bicubic on the fixed validation set and save comparison strips.

python evaluate.py --ckpt runs/stage1/ema.pt --ckpt runs/stage2/ema.pt --out runs/eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from upscaler.data.dataset import PairedImageDataset, list_images
from upscaler.evaluation import bicubic_upscale, evaluate_pairs
from upscaler.imageio import read_rgb, write_rgb
from upscaler.metrics import LPIPSMetric
from upscaler.tiling import tiled_forward
from upscaler.utils import load_generator

SCALE = 4


def model_name(path) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def _center_crop(lr: torch.Tensor, crop: int):
    _, h, w = lr.shape
    ch, cw = min(crop, h), min(crop, w)
    t, l = (h - ch) // 2, (w - cw) // 2
    return lr[:, t:t + ch, l:l + cw], (t, l, ch, cw)


@torch.no_grad()
def _strip(lr_crop: torch.Tensor, models: dict, device, tile: int, hr_crop: torch.Tensor | None):
    x = lr_crop[None].to(device)
    cols = [F.interpolate(x, scale_factor=SCALE, mode="nearest"), bicubic_upscale(x, SCALE)]
    cols += [tiled_forward(m, x, SCALE, tile, 16).float().clamp(0, 1) for m in models.values()]
    if hr_crop is not None:
        cols.append(hr_crop[None].to(device))
    return torch.cat(cols, dim=-1)[0].cpu()


def run(ckpts, out_dir, val_dir, real_dir=None, device="cpu", max_images=None, use_lpips=True,
        crop: int = 128, tile: int = 512) -> dict:
    device = torch.device(device)
    out_dir = Path(out_dir)
    models = {model_name(c): load_generator(c, device, expected_scale=SCALE) for c in ckpts}
    lp = LPIPSMetric(device) if use_lpips else None

    results = {"bicubic": evaluate_pairs(bicubic_upscale, val_dir, device, max_images, lp)}
    for name, m in models.items():
        results[name] = evaluate_pairs(lambda x, m=m: tiled_forward(m, x, SCALE, tile, 16),
                                       val_dir, device, max_images, lp)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    keys = list(next(iter(results.values())))
    print(f"{'model':<28}" + "".join(f"{k.upper():>10}" for k in keys))
    for name, r in results.items():
        print(f"{name:<28}" + "".join(f"{r[k]:>10.4f}" for k in keys))

    ds = PairedImageDataset(val_dir)
    n = len(ds) if max_images is None else min(len(ds), max_images)
    for i in range(n):
        lr, hr, name = ds[i]
        lr_c, (t, l, ch, cw) = _center_crop(lr, crop)
        hr_c = hr[:, t * SCALE:(t + ch) * SCALE, l * SCALE:(l + cw) * SCALE]
        write_rgb(out_dir / "val" / f"{Path(name).stem}.png", _strip(lr_c, models, device, tile, hr_c))

    if real_dir is not None and Path(real_dir).is_dir():
        for p in list_images(real_dir):
            lr_c, _ = _center_crop(read_rgb(p), crop)
            write_rgb(out_dir / "real" / f"{p.stem}.png", _strip(lr_c, models, device, tile, None))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-dir", default="datasets/val_x4")
    ap.add_argument("--real-dir", default="datasets/real_test")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--crop", type=int, default=128, help="LR crop size for comparison strips")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    try:
        run(args.ckpt, args.out, args.val_dir, args.real_dir, device, args.max_images,
            not args.no_lpips, args.crop, args.tile)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit and push**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluate.py metrics table and comparison strips

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 15: `upscale.py` inference CLI

**Files:**
- Create: `upscale.py`
- Test: `tests/test_upscale.py`

**Interfaces:**
- Consumes: `load_generator`, `tiled_forward`, `pil_to_tensor`, `tensor_to_pil`, `IMAGE_EXTS`.
- Produces:
  - `upscale.load_image(path) -> tuple[Tensor (3,H,W), PIL.Image | None]` — EXIF-transposed RGB tensor + alpha channel (if any).
  - `upscale.main(argv=None) -> int` CLI: `python upscale.py INPUT -o OUTPUT [--model weights/upscaler_x4.pt] [--outscale {2,4}] [--tile 512] [--tile-pad 16] [--fp16] [--cpu]`. Exit codes: 0 success (directory mode: ≥1 image succeeded), 1 otherwise. Directory output names: `<stem>_x<outscale>.png`.

- [ ] **Step 1: Write the failing tests** — `tests/test_upscale.py`

```python
import pytest
from PIL import Image

import upscale
from tests.helpers import make_tiny_checkpoint, random_image


@pytest.fixture
def model(tmp_path):
    return str(make_tiny_checkpoint(tmp_path / "m.pt"))


def _run(*args):
    return upscale.main([*map(str, args), "--cpu"])


def test_single_file(tmp_path, model):
    src = tmp_path / "a.png"
    Image.fromarray(random_image(16, 20)).save(src)
    assert _run(src, "-o", tmp_path / "out.png", "--model", model) == 0
    out = Image.open(tmp_path / "out.png")
    assert out.size == (80, 64) and out.mode == "RGB"


def test_outscale_2_and_tiling(tmp_path, model):
    src = tmp_path / "a.png"
    Image.fromarray(random_image(16, 20)).save(src)
    assert _run(src, "-o", tmp_path / "o.png", "--model", model, "--outscale", 2, "--tile", 8) == 0
    assert Image.open(tmp_path / "o.png").size == (40, 32)


def test_rgba_keeps_alpha(tmp_path, model):
    src = tmp_path / "a.png"
    Image.new("RGBA", (10, 6), (200, 10, 10, 77)).save(src)
    assert _run(src, "-o", tmp_path / "o.png", "--model", model) == 0
    out = Image.open(tmp_path / "o.png")
    assert out.mode == "RGBA" and out.size == (40, 24)
    assert out.getchannel("A").getextrema() == (77, 77)


def test_grayscale_and_palette_inputs(tmp_path, model):
    Image.new("L", (7, 5), 120).save(tmp_path / "g.png")
    Image.new("RGB", (7, 5), (1, 2, 3)).convert("P").save(tmp_path / "p.png")
    for name in ("g", "p"):
        assert _run(tmp_path / f"{name}.png", "-o", tmp_path / f"{name}_o.png", "--model", model) == 0
        assert Image.open(tmp_path / f"{name}_o.png").size == (28, 20)


def test_one_pixel_image(tmp_path, model):
    Image.new("RGB", (1, 1), (9, 9, 9)).save(tmp_path / "px.png")
    assert _run(tmp_path / "px.png", "-o", tmp_path / "px_o.png", "--model", model) == 0
    assert Image.open(tmp_path / "px_o.png").size == (4, 4)


def test_exif_orientation_is_applied(tmp_path, model):
    img = Image.fromarray(random_image(10, 20))  # 20 wide x 10 tall as stored
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW when displayed
    img.save(tmp_path / "phone.jpg", exif=exif)
    assert _run(tmp_path / "phone.jpg", "-o", tmp_path / "o.png", "--model", model) == 0
    assert Image.open(tmp_path / "o.png").size == (40, 80)  # displayed as 10 wide x 20 tall


def test_directory_mode_skips_bad_files(tmp_path, model, capsys):
    d = tmp_path / "in"
    d.mkdir()
    Image.fromarray(random_image(8, 8)).save(d / "ok.jpg")
    (d / "broken.png").write_bytes(b"not an image")
    (d / "notes.txt").write_text("ignored")
    assert _run(d, "-o", tmp_path / "out", "--model", model) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["ok_x4.png"]
    assert "skipping" in capsys.readouterr().err


def test_errors(tmp_path, model, capsys):
    src = tmp_path / "a.png"
    Image.fromarray(random_image(8, 8)).save(src)
    assert _run(src, "-o", tmp_path / "o.png", "--model", tmp_path / "nope.pt") == 1
    x2 = make_tiny_checkpoint(tmp_path / "x2.pt", scale=2)
    assert _run(src, "-o", tmp_path / "o.png", "--model", x2) == 1
    (tmp_path / "bad.png").write_bytes(b"junk")
    assert _run(tmp_path / "bad.png", "-o", tmp_path / "o.png", "--model", model) == 1
    assert _run(tmp_path / "missing.png", "-o", tmp_path / "o.png", "--model", model) == 1
    (tmp_path / "empty").mkdir()
    assert _run(tmp_path / "empty", "-o", tmp_path / "o", "--model", model) == 1
    err = capsys.readouterr().err
    assert "checkpoint not found" in err and "x2 model" in err
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_upscale.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upscale'`.

- [ ] **Step 3: Implement `upscale.py`**

```python
"""Upscale images x4 (or x2) with a trained upscaler.

python upscale.py photo.jpg -o photo_x4.png
python upscale.py ./photos -o ./out --fp16
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from upscaler.data.dataset import IMAGE_EXTS
from upscaler.imageio import pil_to_tensor, tensor_to_pil
from upscaler.tiling import tiled_forward
from upscaler.utils import load_generator

SCALE = 4


def load_image(path) -> tuple[torch.Tensor, Image.Image | None]:
    """RGB tensor (EXIF orientation applied) and the alpha channel if the image has one."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        alpha = None
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            alpha = img.getchannel("A")
        return pil_to_tensor(img), alpha


@torch.no_grad()
def _upscale(model, rgb: torch.Tensor, device, tile: int, tile_pad: int, fp16: bool) -> torch.Tensor:
    x = rgb[None].to(device)
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 and device.type == "cuda" else contextlib.nullcontext()
    with ctx:
        y = tiled_forward(model, x, SCALE, tile, tile_pad)
    return y.float().clamp(0, 1)[0].cpu()


def _process(src: Path, dst: Path, model, args, device) -> None:
    rgb, alpha = load_image(src)
    out = tensor_to_pil(_upscale(model, rgb, device, args.tile, args.tile_pad, args.fp16))
    if args.outscale != SCALE:
        out = out.resize((rgb.shape[2] * args.outscale, rgb.shape[1] * args.outscale), Image.LANCZOS)
    if alpha is not None:
        out.putalpha(alpha.resize(out.size, Image.BICUBIC))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="image file or directory")
    ap.add_argument("-o", "--output", required=True, help="output file, or directory for directory input")
    ap.add_argument("--model", default="weights/upscaler_x4.pt")
    ap.add_argument("--outscale", type=int, choices=[2, 4], default=4)
    ap.add_argument("--tile", type=int, default=512, help="LR tile size; 0 disables tiling")
    ap.add_argument("--tile-pad", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    try:
        model = load_generator(args.model, device, expected_scale=SCALE)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    src = Path(args.input)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not files:
            print(f"error: no images found in {src}", file=sys.stderr)
            return 1
        ok = 0
        for p in tqdm(files, desc="upscaling"):
            try:
                _process(p, Path(args.output) / f"{p.stem}_x{args.outscale}.png", model, args, device)
                ok += 1
            except (OSError, ValueError) as e:
                print(f"warning: skipping {p}: {e}", file=sys.stderr)
        return 0 if ok else 1

    if not src.is_file():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 1
    dst = Path(args.output)
    if dst.is_dir():
        dst = dst / f"{src.stem}_x{args.outscale}.png"
    try:
        _process(src, dst, model, args, device)
    except (OSError, ValueError) as e:
        print(f"error: cannot upscale {src}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_upscale.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit and push**

```bash
git add upscale.py tests/test_upscale.py
git commit -m "feat: upscale.py CLI with tiling, alpha, EXIF orientation, x2 output

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 16: Real data, training runs, evaluation, shipping weights

This task is operational (long-running GPU jobs). **Ask the user before starting each multi-hour step** (download, stage 1, stage 2). Run long commands in the background with logs in `runs/`.

**Files:**
- Modify: `README.md` (status + results table)
- Create (gitignored, not committed): `datasets/…`, `runs/…`, `weights/upscaler_x4.pt`

**Interfaces:**
- Consumes: all CLIs.
- Produces: `weights/upscaler_x4.pt` (stage-2 EMA), `runs/eval/results.json`, comparison strips.

- [ ] **Step 1: Download data (~25 GB)**

```bash
.venv/bin/python -m upscaler.data.download --raw-dir datasets/raw > runs_download.log 2>&1
```
(Run in background; `mkdir -p runs` first and write the log to `runs/download.log`.) Expected exit code 0 and three `.done` markers in `datasets/raw/*/`. If Flickr2K's mirror fails, report to the user and continue with DIV2K only (`--only DIV2K_train_HR DIV2K_valid_HR`); do not invent a mirror URL.

- [ ] **Step 2: Prepare tiles and validation set**

```bash
.venv/bin/python -m upscaler.data.prepare --raw-dir datasets/raw --out-dir datasets
```
Expected: `tiles: kept ~100k, dropped …(X%)` and `val: wrote 100 pairs`. If X > 10 %, rerun tiles only with a lower threshold: `--skip-val --min-texture 5` (delete `datasets/df2k_tiles` first). Record final threshold and counts.

- [ ] **Step 3: Smoke run on real data**

```bash
.venv/bin/python train.py --config configs/smoke.yaml
```
Expected: completes in < 1 min, prints losses and a val line, writes `runs/smoke/ema.pt`.

- [ ] **Step 4: Measure stage-1 throughput**

```bash
timeout 180 .venv/bin/python train.py --config configs/stage1_psnr.yaml --run-dir runs/stage1_speedtest
```
Read the steady-state `it/s` from the last log lines. Hours for stage 1 = 200000 / it_s / 3600. If > 4 h, report the number to the user and propose a lower `train.total_iters`; do not change it without their answer. Delete `runs/stage1_speedtest` afterwards.

- [ ] **Step 5: Stage 1 training (ask first; background)**

```bash
.venv/bin/python train.py --config configs/stage1_psnr.yaml > runs/stage1.log 2>&1
```
Monitor via `runs/stage1.log` and `tensorboard --logdir runs`. If interrupted: same command plus `--resume`.

- [ ] **Step 6: Check success criterion 1**

```bash
.venv/bin/python evaluate.py --ckpt runs/stage1/ema.pt --out runs/eval_stage1
```
Expected: `stage1/ema` beats `bicubic` on PSNR and SSIM (higher) and LPIPS (lower). If not, stop and debug with superpowers:systematic-debugging before stage 2.

- [ ] **Step 7: Stage 2 training (ask first; background)**

```bash
.venv/bin/python train.py --config configs/stage2_gan.yaml > runs/stage2.log 2>&1
```

- [ ] **Step 8: Final evaluation (criteria 2 and 3)**

Ask the user to drop a few of their own photos into `datasets/real_test/`, then:
```bash
.venv/bin/python evaluate.py --ckpt runs/stage1/ema.pt --ckpt runs/stage2/ema.pt --out runs/eval
```
Expected: `stage2/ema` LPIPS < `stage1/ema` LPIPS. Send the user `runs/eval/real/*.png` and a few `runs/eval/val/*.png` strips for visual judgement.

- [ ] **Step 9: Ship weights and check criterion 4 (4K input)**

```bash
mkdir -p weights && cp runs/stage2/ema.pt weights/upscaler_x4.pt
.venv/bin/python -c "from PIL import Image; from tests.helpers import random_image; Image.fromarray(random_image(2160, 3840)).save('runs/test_4k.png')"
.venv/bin/python upscale.py runs/test_4k.png -o runs/test_4k_x4.png --fp16
```
Expected: exit 0, no CUDA OOM, output 15360×8640. Inspect a 512 px crop across a tile boundary (e.g. around x=2048 in output) for seams.

- [ ] **Step 10: Update README and push**

Replace the README "Status" line with a short results section: the `runs/eval/results.json` table (bicubic / stage1 / stage2 PSNR, SSIM, LPIPS), training times, and the tile threshold used. Then:
```bash
git add README.md
git commit -m "docs: training results and usage

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
