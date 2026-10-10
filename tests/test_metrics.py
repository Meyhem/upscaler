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
