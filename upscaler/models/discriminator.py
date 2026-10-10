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
