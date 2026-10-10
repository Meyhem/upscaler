from pathlib import Path

import torch

from tests.helpers import make_val_dir, write_images
from upscaler.trainer import train
from upscaler.utils import load_config, load_generator

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, config="smoke.yaml", run="run", **train_over):
    cfg = load_config(ROOT / "configs" / config)
    tiles, val = tmp_path / "tiles", tmp_path / "val"
    if not tiles.exists():
        write_images(tiles, 8, 96, 96)
        make_val_dir(val, n=2, lr_hw=(16, 16))
    cfg["device"] = "cpu"
    cfg["run_dir"] = str(tmp_path / run)
    cfg["model"].update(num_feat=32, num_conv=8)
    cfg["data"].update(tiles_dir=str(tiles), crop=64, batch_size=4, num_workers=0, max_tiles=None)
    cfg["val"].update(dir=str(val), max_images=2, lpips=False, tile=0)
    cfg["train"].update(total_iters=20, log_every=5, val_every=10, ckpt_every=10, compile=False)
    cfg["train"].update(train_over)
    return cfg


def test_stage1_runs_writes_outputs_and_learns(tmp_path):
    cfg = _cfg(tmp_path, total_iters=60, lr=1e-3, val_every=60, ckpt_every=30)
    res = train(cfg)
    run = Path(cfg["run_dir"])
    assert res["iter"] == 60
    assert {"psnr", "ssim"} <= set(res["val"])
    assert (run / "config.yaml").exists() and any((run / "tb").iterdir())
    assert sorted(p.name for p in (run / "ckpt").iterdir()) == ["iter_30.pt", "iter_60.pt"]
    losses = [h["l_pix"] for h in res["history"]]
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3
    model = load_generator(run / "ema.pt", expected_scale=4)
    assert model(torch.rand(1, 3, 8, 8)).shape == (1, 3, 32, 32)


def test_resume_continues_from_latest_checkpoint(tmp_path):
    cfg = _cfg(tmp_path)
    train(cfg)
    ckpt_dir = Path(cfg["run_dir"]) / "ckpt"
    (ckpt_dir / "iter_20.pt").unlink()  # simulate a crash after iter 10's checkpoint
    res = train(cfg, resume=True)
    assert res["iter"] == 20
    assert [h["iter"] for h in res["history"]] == [15, 20]
    assert (ckpt_dir / "iter_20.pt").exists()


def test_stage2_gan_from_stage1(tmp_path):
    train(_cfg(tmp_path, total_iters=10))
    cfg = _cfg(tmp_path, config="stage2_gan.yaml", run="gan", total_iters=4, log_every=2,
               val_every=4, ckpt_every=4, init_from=str(tmp_path / "run" / "ema.pt"))
    cfg["gan"]["vgg_pretrained"] = False
    res = train(cfg)
    assert {"l_pix", "l_percep", "l_gan", "l_d"} <= set(res["history"][-1])
    ck = torch.load(Path(cfg["run_dir"]) / "ckpt" / "iter_4.pt", weights_only=True)
    assert "net_d" in ck and "opt_d" in ck
