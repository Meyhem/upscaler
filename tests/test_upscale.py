import pytest
from PIL import Image

import upscale
from tests.helpers import make_tiny_checkpoint, random_image


@pytest.fixture
def model(tmp_path):
    return str(make_tiny_checkpoint(tmp_path / "m.pt"))


def _run(*args):
    return upscale.main([*map(str, args), "--cpu"])


def test_single_file(tmp_path, model):
    src = tmp_path / "a.png"
    Image.fromarray(random_image(16, 20)).save(src)
    assert _run(src, "-o", tmp_path / "out.png", "--model", model) == 0
    out = Image.open(tmp_path / "out.png")
    assert out.size == (80, 64) and out.mode == "RGB"


def test_outscale_2_and_tiling(tmp_path, model):
    src = tmp_path / "a.png"
    Image.fromarray(random_image(16, 20)).save(src)
    assert _run(src, "-o", tmp_path / "o.png", "--model", model, "--outscale", 2, "--tile", 8) == 0
    assert Image.open(tmp_path / "o.png").size == (40, 32)


def test_rgba_keeps_alpha(tmp_path, model):
    src = tmp_path / "a.png"
    Image.new("RGBA", (10, 6), (200, 10, 10, 77)).save(src)
    assert _run(src, "-o", tmp_path / "o.png", "--model", model) == 0
    out = Image.open(tmp_path / "o.png")
    assert out.mode == "RGBA" and out.size == (40, 24)
    assert out.getchannel("A").getextrema() == (77, 77)


def test_grayscale_and_palette_inputs(tmp_path, model):
    Image.new("L", (7, 5), 120).save(tmp_path / "g.png")
    Image.new("RGB", (7, 5), (1, 2, 3)).convert("P").save(tmp_path / "p.png")
    for name in ("g", "p"):
        assert _run(tmp_path / f"{name}.png", "-o", tmp_path / f"{name}_o.png", "--model", model) == 0
        assert Image.open(tmp_path / f"{name}_o.png").size == (28, 20)


def test_one_pixel_image(tmp_path, model):
    Image.new("RGB", (1, 1), (9, 9, 9)).save(tmp_path / "px.png")
    assert _run(tmp_path / "px.png", "-o", tmp_path / "px_o.png", "--model", model) == 0
    assert Image.open(tmp_path / "px_o.png").size == (4, 4)


def test_exif_orientation_is_applied(tmp_path, model):
    img = Image.fromarray(random_image(10, 20))  # 20 wide x 10 tall as stored
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW when displayed
    img.save(tmp_path / "phone.jpg", exif=exif)
    assert _run(tmp_path / "phone.jpg", "-o", tmp_path / "o.png", "--model", model) == 0
    assert Image.open(tmp_path / "o.png").size == (40, 80)  # displayed as 10 wide x 20 tall


def test_directory_mode_skips_bad_files(tmp_path, model, capsys):
    d = tmp_path / "in"
    d.mkdir()
    Image.fromarray(random_image(8, 8)).save(d / "ok.jpg")
    (d / "broken.png").write_bytes(b"not an image")
    (d / "notes.txt").write_text("ignored")
    assert _run(d, "-o", tmp_path / "out", "--model", model) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["ok_x4.png"]
    assert "skipping" in capsys.readouterr().err


def test_errors(tmp_path, model, capsys):
    src = tmp_path / "a.png"
    Image.fromarray(random_image(8, 8)).save(src)
    assert _run(src, "-o", tmp_path / "o.png", "--model", tmp_path / "nope.pt") == 1
    x2 = make_tiny_checkpoint(tmp_path / "x2.pt", scale=2)
    assert _run(src, "-o", tmp_path / "o.png", "--model", x2) == 1
    (tmp_path / "bad.png").write_bytes(b"junk")
    assert _run(tmp_path / "bad.png", "-o", tmp_path / "o.png", "--model", model) == 1
    assert _run(tmp_path / "missing.png", "-o", tmp_path / "o.png", "--model", model) == 1
    (tmp_path / "empty").mkdir()
    assert _run(tmp_path / "empty", "-o", tmp_path / "o", "--model", model) == 1
    err = capsys.readouterr().err
    assert "checkpoint not found" in err and "x2 model" in err


def test_gpu_out_of_memory_is_reported_not_crashed(tmp_path, model, capsys, monkeypatch):
    import torch

    def boom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 8.58 GiB")

    monkeypatch.setattr(upscale, "tiled_forward", boom)
    src = tmp_path / "a.png"
    Image.fromarray(random_image(8, 8)).save(src)
    assert _run(src, "-o", tmp_path / "o.png", "--model", model) == 1
    d = tmp_path / "in"
    d.mkdir()
    Image.fromarray(random_image(8, 8)).save(d / "x.png")
    assert _run(d, "-o", tmp_path / "out", "--model", model) == 1  # skipped, not a traceback
    assert "--tile" in capsys.readouterr().err


def test_16bit_grayscale_is_scaled_not_clipped(tmp_path, model):
    import numpy as np

    grad = np.tile(np.linspace(0, 65535, 32).astype(np.uint16), (16, 1))
    Image.fromarray(grad).save(tmp_path / "g16.png")
    assert Image.open(tmp_path / "g16.png").mode.startswith("I")
    assert _run(tmp_path / "g16.png", "-o", tmp_path / "o.png", "--model", model, "--tile", 0) == 0
    mean = np.asarray(Image.open(tmp_path / "o.png")).mean()
    assert 90 < mean < 165  # ~127, not ~250


def test_icc_profile_is_preserved(tmp_path, model):
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(random_image(8, 8)).save(tmp_path / "p3.jpg", icc_profile=icc)
    assert _run(tmp_path / "p3.jpg", "-o", tmp_path / "o.png", "--model", model) == 0
    assert Image.open(tmp_path / "o.png").info.get("icc_profile") == icc
