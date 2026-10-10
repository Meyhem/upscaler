import json

from PIL import Image

import evaluate
from tests.helpers import make_tiny_checkpoint, make_val_dir, write_images


def test_run_writes_results_and_strips(tmp_path):
    ckpt = make_tiny_checkpoint(tmp_path / "stage1" / "ema.pt")
    val = make_val_dir(tmp_path / "val", n=2, lr_hw=(16, 16))
    real = tmp_path / "real"
    write_images(real, 1, 20, 24)
    out = tmp_path / "eval"
    res = evaluate.run([ckpt], out, val, real_dir=real, use_lpips=False, crop=8, tile=0)
    assert set(res) == {"bicubic", "stage1/ema"}
    assert json.loads((out / "results.json").read_text()) == res
    # val strip: nearest | bicubic | model | HR = 4 columns of 8*4 px
    assert Image.open(out / "val" / "v0.png").size == (4 * 32, 32)
    # real strip: nearest | bicubic | model
    assert Image.open(out / "real" / "img_000.png").size == (3 * 32, 32)


def test_cli_bad_checkpoint_returns_1(tmp_path, capsys):
    val = make_val_dir(tmp_path / "val")
    rc = evaluate.main(["--ckpt", str(tmp_path / "missing.pt"), "--out", str(tmp_path / "o"),
                        "--val-dir", str(val), "--no-lpips", "--cpu"])
    assert rc == 1
    assert "checkpoint not found" in capsys.readouterr().err
