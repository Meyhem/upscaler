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
