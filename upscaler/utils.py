"""Config loading, seeding, checkpoint I/O and EMA."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from upscaler.models import build_model


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
