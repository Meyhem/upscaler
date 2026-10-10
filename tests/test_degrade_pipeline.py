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
