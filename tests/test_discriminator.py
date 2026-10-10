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
