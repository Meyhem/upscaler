"""Upscale images x4 (or x2) with a trained upscaler.

python upscale.py photo.jpg -o photo_x4.png
python upscale.py ./photos -o ./out --fp16
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from upscaler.data.dataset import IMAGE_EXTS
from upscaler.imageio import pil_to_tensor, tensor_to_pil
from upscaler.tiling import tiled_forward
from upscaler.utils import load_generator

SCALE = 4
HIGH_BIT_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")
OOM_HINT = "out of GPU memory; try a smaller --tile (e.g. 256) or --cpu"


def _to_8bit(img: Image.Image) -> Image.Image:
    """16-bit / 32-bit grayscale -> 8-bit L (PIL's convert('RGB') would clip at 255)."""
    arr = np.asarray(img, dtype=np.float64)
    divisor = 1.0 if img.mode == "F" and arr.max(initial=0) <= 1.0 else 257.0
    arr = arr * 255.0 if divisor == 1.0 else arr / divisor
    return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8))


def load_image(path) -> tuple[torch.Tensor, Image.Image | None, bytes | None]:
    """RGB tensor (EXIF orientation applied), alpha channel if any, and the ICC profile if any."""
    with Image.open(path) as img:
        icc = img.info.get("icc_profile")
        img = ImageOps.exif_transpose(img)
        if img.mode in HIGH_BIT_MODES:
            img = _to_8bit(img)
        alpha = None
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            alpha = img.getchannel("A")
        return pil_to_tensor(img), alpha, icc


@torch.no_grad()
def _upscale(model, rgb: torch.Tensor, device, tile: int, tile_pad: int, fp16: bool) -> torch.Tensor:
    x = rgb[None].to(device)
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 and device.type == "cuda" else contextlib.nullcontext()
    with ctx:
        y = tiled_forward(model, x, SCALE, tile, tile_pad, out_device="cpu")
    return y[0].float().clamp_(0, 1)


def _process(src: Path, dst: Path, model, args, device) -> None:
    rgb, alpha, icc = load_image(src)
    out = tensor_to_pil(_upscale(model, rgb, device, args.tile, args.tile_pad, args.fp16))
    if args.outscale != SCALE:
        out = out.resize((rgb.shape[2] * args.outscale, rgb.shape[1] * args.outscale), Image.LANCZOS)
    if alpha is not None:
        out.putalpha(alpha.resize(out.size, Image.BICUBIC))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if icc:
        out.save(dst, icc_profile=icc)
    else:
        out.save(dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="image file or directory")
    ap.add_argument("-o", "--output", required=True, help="output file, or directory for directory input")
    ap.add_argument("--model", default="weights/upscaler_x4.pt")
    ap.add_argument("--outscale", type=int, choices=[2, 4], default=4)
    ap.add_argument("--tile", type=int, default=512, help="LR tile size; 0 disables tiling")
    ap.add_argument("--tile-pad", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    try:
        model = load_generator(args.model, device, expected_scale=SCALE)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    src = Path(args.input)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not files:
            print(f"error: no images found in {src}", file=sys.stderr)
            return 1
        ok = 0
        for p in tqdm(files, desc="upscaling"):
            try:
                _process(p, Path(args.output) / f"{p.stem}_x{args.outscale}.png", model, args, device)
                ok += 1
            except torch.OutOfMemoryError:
                print(f"warning: skipping {p}: {OOM_HINT}", file=sys.stderr)
            except (OSError, ValueError) as e:
                print(f"warning: skipping {p}: {e}", file=sys.stderr)
        return 0 if ok else 1

    if not src.is_file():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 1
    dst = Path(args.output)
    if dst.is_dir():
        dst = dst / f"{src.stem}_x{args.outscale}.png"
    try:
        _process(src, dst, model, args, device)
    except torch.OutOfMemoryError:
        print(f"error: cannot upscale {src}: {OOM_HINT}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"error: cannot upscale {src}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
