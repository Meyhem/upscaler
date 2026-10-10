import numpy as np
from PIL import Image

from tests.helpers import random_image
from upscaler.data.prepare import (build_val_set, extract_tiles, find_hr_images, texture_score,
                                   tile_positions)


def test_tile_positions():
    assert tile_positions(1000, 400, 200) == [0, 200, 400, 600]
    assert tile_positions(450, 400, 200) == [0, 50]
    assert tile_positions(400, 400, 200) == [0]
    assert tile_positions(300, 400, 200) == []


def test_texture_score_flat_vs_textured():
    flat = np.full((64, 64, 3), 128, np.uint8)
    assert texture_score(flat) == 0.0
    assert texture_score(random_image(64, 64)) > 8.0


def test_find_hr_images_skips_lr(tmp_path):
    for rel in ["DIV2K/0001.png", "Flickr2K/Flickr2K_HR/000001.png", "Flickr2K/Flickr2K_LR_bicubic/X4/000001x4.png"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(p)
    names = [p.name for p in find_hr_images(tmp_path)]
    assert names == ["0001.png", "000001.png"]


def test_extract_tiles_drops_flat(tmp_path):
    tex = tmp_path / "tex.png"
    flat = tmp_path / "flat.png"
    Image.fromarray(random_image(120, 160)).save(tex)
    Image.fromarray(np.full((120, 160, 3), 90, np.uint8)).save(flat)
    kept, dropped = extract_tiles([tex, flat], tmp_path / "tiles", tile=64, stride=32)
    # positions: y in [0, 32, 56] x in [0, 32, 64, 96] -> 12 tiles per image
    assert (kept, dropped) == (12, 12)
    files = sorted(p.name for p in (tmp_path / "tiles").iterdir())
    assert files[0] == "tex_0000_0000.png" and len(files) == 12
    assert Image.open(tmp_path / "tiles" / files[0]).size == (64, 64)


def test_build_val_set_crops_and_is_deterministic(tmp_path):
    src = tmp_path / "0801.png"
    Image.fromarray(random_image(37, 50)).save(src)
    assert build_val_set([src], tmp_path / "a") == 1
    build_val_set([src], tmp_path / "b")
    assert Image.open(tmp_path / "a" / "hr" / "0801.png").size == (48, 36)
    assert Image.open(tmp_path / "a" / "lr" / "0801.png").size == (12, 9)
    assert (tmp_path / "a" / "lr" / "0801.png").read_bytes() == (tmp_path / "b" / "lr" / "0801.png").read_bytes()
