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
