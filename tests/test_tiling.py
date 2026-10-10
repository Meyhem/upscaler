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


def test_output_can_live_on_another_device():
    model = _model()
    x = torch.rand(1, 3, 40, 40)
    out = tiled_forward(model, x, 4, tile=16, tile_pad=24, out_device="cpu")
    assert out.device.type == "cpu" and out.shape == (1, 3, 160, 160)
