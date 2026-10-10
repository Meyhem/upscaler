import pytest
import torch

from upscaler.models import build_model
from upscaler.models.srvgg import SRVGGCompact


@pytest.mark.parametrize("shape", [(1, 3, 16, 16), (2, 3, 17, 23)])
def test_output_is_4x(shape):
    net = SRVGGCompact()
    out = net(torch.rand(shape))
    assert out.shape == (shape[0], 3, shape[2] * 4, shape[3] * 4)


def test_tiny_input():
    net = SRVGGCompact(num_feat=32, num_conv=8)
    assert net(torch.rand(1, 3, 1, 1)).shape == (1, 3, 4, 4)
    assert net(torch.rand(1, 3, 3, 5)).shape == (1, 3, 12, 20)


def test_param_counts():
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(SRVGGCompact()) == 621_424
    assert count(SRVGGCompact(num_feat=32, num_conv=8)) == 89_040


def test_gradients_reach_first_conv():
    net = SRVGGCompact(num_feat=32, num_conv=8)
    net(torch.rand(1, 3, 8, 8)).mean().backward()
    first = net.body[0].weight.grad
    assert first is not None and first.abs().sum() > 0


def test_registry_builds_and_does_not_mutate():
    cfg = {"type": "srvgg_compact", "num_feat": 32, "num_conv": 8, "scale": 4}
    net = build_model(cfg)
    assert isinstance(net, SRVGGCompact) and net.scale == 4
    assert cfg["type"] == "srvgg_compact"


def test_registry_unknown_type():
    with pytest.raises(ValueError, match="unknown model type"):
        build_model({"type": "nope"})
