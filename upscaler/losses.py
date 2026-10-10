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
