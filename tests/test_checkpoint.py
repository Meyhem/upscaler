import pytest
import torch

from tests.helpers import TINY_CFG, make_tiny_checkpoint
from upscaler.models import build_model
from upscaler.utils import EMA, latest_checkpoint, load_generator, save_checkpoint


def test_ema_update_math():
    net = torch.nn.Linear(2, 2)
    ema = EMA(net, decay=0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    before = [p.clone() for p in ema.module.parameters()]
    ema.update(net)
    for b, e, p in zip(before, ema.module.parameters(), net.parameters()):
        assert torch.allclose(e, 0.5 * b + 0.5 * p)
        assert not e.requires_grad
    assert not ema.module.training


def test_save_load_roundtrip_with_optimizer(tmp_path):
    net = build_model(TINY_CFG)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net(torch.rand(1, 3, 8, 8)).mean().backward()
    opt.step()
    save_checkpoint(tmp_path / "ck" / "iter_5.pt", {"iter": 5, "net": net.state_dict(), "opt": opt.state_dict()})
    ck = torch.load(tmp_path / "ck" / "iter_5.pt", weights_only=True)
    net2 = build_model(TINY_CFG)
    net2.load_state_dict(ck["net"])
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    opt2.load_state_dict(ck["opt"])
    assert ck["iter"] == 5
    for a, b in zip(net.state_dict().values(), net2.state_dict().values()):
        assert torch.equal(a, b)
    assert opt2.state_dict()["state"][0]["exp_avg"].equal(opt.state_dict()["state"][0]["exp_avg"])
    assert not (tmp_path / "ck" / "iter_5.pt.tmp").exists()


def test_latest_checkpoint_numeric_order(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    for n in (9, 10, 2):
        (tmp_path / f"iter_{n}.pt").write_bytes(b"x")
    assert latest_checkpoint(tmp_path).name == "iter_10.pt"


def test_export_and_load_generator_same_output(tmp_path):
    path = make_tiny_checkpoint(tmp_path / "g.pt")
    torch.manual_seed(0)
    ref = build_model(TINY_CFG).eval()
    model = load_generator(path, expected_scale=4)
    x = torch.rand(1, 3, 6, 6)
    assert torch.allclose(model(x), ref(x))
    assert not model.training


def test_load_generator_accepts_training_checkpoint(tmp_path):
    net = build_model(TINY_CFG)
    save_checkpoint(tmp_path / "iter_1.pt", {"model_cfg": TINY_CFG, "ema": net.state_dict(), "iter": 1})
    assert load_generator(tmp_path / "iter_1.pt").scale == 4


def test_load_generator_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_generator(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="x2 model"):
        load_generator(make_tiny_checkpoint(tmp_path / "x2.pt", scale=2), expected_scale=4)
    bad = tmp_path / "bad.pt"
    save_checkpoint(bad, {"model_cfg": dict(TINY_CFG, num_feat=16), "params": build_model(TINY_CFG).state_dict()})
    with pytest.raises(ValueError, match="do not match"):
        load_generator(bad)
    junk = tmp_path / "junk.pt"
    torch.save({"hello": 1}, junk)
    with pytest.raises(ValueError, match="not an upscaler checkpoint"):
        load_generator(junk)
