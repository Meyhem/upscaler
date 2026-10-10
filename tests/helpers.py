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
