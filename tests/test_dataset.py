import pytest
import torch
from PIL import Image

from tests.helpers import make_val_dir, write_images
from upscaler.data.dataset import PairedImageDataset, TileDataset, list_images


def test_tile_dataset_crops_and_limits(tmp_path):
    write_images(tmp_path, 5, 80, 96)
    ds = TileDataset(tmp_path, crop=64, max_tiles=3)
    assert len(ds) == 3
    x = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert 0 <= x.min() and x.max() <= 1


def test_tile_dataset_seeded_augmentation(tmp_path):
    write_images(tmp_path, 1, 80, 96)
    ds = TileDataset(tmp_path, crop=64)
    torch.manual_seed(1)
    a = ds[0]
    torch.manual_seed(1)
    assert torch.equal(a, ds[0])


def test_tile_dataset_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        TileDataset(tmp_path, crop=64)
    write_images(tmp_path, 1, 32, 32)
    with pytest.raises(ValueError, match="smaller than crop"):
        TileDataset(tmp_path, crop=64)[0]


def test_list_images_filters_extensions(tmp_path):
    write_images(tmp_path, 2, 8, 8)
    (tmp_path / "notes.txt").write_text("x")
    Image.new("RGB", (4, 4)).save(tmp_path / "a.JPG")
    assert [p.name for p in list_images(tmp_path)] == ["a.JPG", "img_000.png", "img_001.png"]


def test_paired_dataset(tmp_path):
    make_val_dir(tmp_path, n=2, lr_hw=(8, 12))
    ds = PairedImageDataset(tmp_path)
    lr, hr, name = ds[1]
    assert len(ds) == 2 and name == "v1.png"
    assert lr.shape == (3, 8, 12) and hr.shape == (3, 32, 48)


def test_paired_dataset_missing_hr(tmp_path):
    make_val_dir(tmp_path, n=1)
    (tmp_path / "hr" / "v0.png").unlink()
    with pytest.raises(FileNotFoundError, match="v0.png"):
        PairedImageDataset(tmp_path)
