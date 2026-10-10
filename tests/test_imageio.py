import torch

from upscaler.imageio import read_rgb, tensor_to_pil, write_rgb, pil_to_tensor


def test_roundtrip_png(tmp_path):
    t = torch.randint(0, 256, (3, 7, 5)).float() / 255
    write_rgb(tmp_path / "sub" / "x.png", t)
    back = read_rgb(tmp_path / "sub" / "x.png")
    assert back.shape == (3, 7, 5)
    assert back.dtype == torch.float32
    assert torch.allclose(back, t, atol=1e-6)


def test_tensor_to_pil_clamps():
    t = torch.full((3, 2, 2), 2.0)
    t[0] = -1.0
    img = tensor_to_pil(t)
    assert img.mode == "RGB"
    back = pil_to_tensor(img)
    assert back[0].max() == 0.0 and back[1].min() == 1.0
