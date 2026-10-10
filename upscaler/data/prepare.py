"""One-time data preparation: HR training tiles + fixed degraded validation set.

python -m upscaler.data.prepare [--raw-dir datasets/raw] [--out-dir datasets]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from upscaler.data.dataset import IMAGE_EXTS
from upscaler.data.degrade import DegradeConfig, degrade
from upscaler.imageio import read_rgb, write_rgb

TRAIN_SOURCES = ("DIV2K_train_HR", "Flickr2K")
VAL_SOURCE = "DIV2K_valid_HR"


def find_hr_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and not any("LR" in part for part in p.relative_to(root).parts)
    )


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    if length < tile:
        return []
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def texture_score(tile_rgb_u8: np.ndarray) -> float:
    gray = cv2.cvtColor(tile_rgb_u8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).std())


def extract_tiles(images, out_dir, tile: int = 400, stride: int = 200, min_texture: float = 8.0) -> tuple[int, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    for path in tqdm(images, desc="tiles"):
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"))
        h, w = img.shape[:2]
        for y in tile_positions(h, tile, stride):
            for x in tile_positions(w, tile, stride):
                t = img[y:y + tile, x:x + tile]
                if texture_score(t) < min_texture:
                    dropped += 1
                    continue
                cv2.imwrite(str(out_dir / f"{Path(path).stem}_{y:04d}_{x:04d}.png"),
                            cv2.cvtColor(t, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
                kept += 1
    return kept, dropped


def build_val_set(hr_images, out_dir, scale: int = 4, seed: int = 1234,
                  degrade_cfg: DegradeConfig | None = None, device="cpu") -> int:
    out_dir = Path(out_dir)
    gen = torch.Generator(device=device).manual_seed(seed)
    cfg = degrade_cfg or DegradeConfig(scale=scale)
    count = 0
    for path in tqdm(hr_images, desc="val"):
        hr = read_rgb(path)
        _, h, w = hr.shape
        th, tw = h - h % scale, w - w % scale
        top, left = (h - th) // 2, (w - tw) // 2
        hr = hr[:, top:top + th, left:left + tw]
        lr = degrade(hr[None].to(device), gen, cfg)[0].cpu()
        write_rgb(out_dir / "hr" / f"{Path(path).stem}.png", hr)
        write_rgb(out_dir / "lr" / f"{Path(path).stem}.png", lr)
        count += 1
    return count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="datasets/raw")
    ap.add_argument("--out-dir", default="datasets")
    ap.add_argument("--tile", type=int, default=400)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--min-texture", type=float, default=8.0)
    ap.add_argument("--skip-tiles", action="store_true")
    ap.add_argument("--skip-val", action="store_true")
    args = ap.parse_args(argv)
    raw, out = Path(args.raw_dir), Path(args.out_dir)

    if not args.skip_tiles:
        images = [p for src in TRAIN_SOURCES if (raw / src).is_dir() for p in find_hr_images(raw / src)]
        if not images:
            print(f"no training images under {raw}/{{{','.join(TRAIN_SOURCES)}}}", file=sys.stderr)
            return 1
        kept, dropped = extract_tiles(images, out / "df2k_tiles", args.tile, args.stride, args.min_texture)
        total = kept + dropped
        print(f"tiles: kept {kept}, dropped {dropped} ({100 * dropped / max(total, 1):.1f}% low-texture) "
              f"from {len(images)} images")

    if not args.skip_val:
        val_images = find_hr_images(raw / VAL_SOURCE) if (raw / VAL_SOURCE).is_dir() else []
        if not val_images:
            print(f"no validation images under {raw / VAL_SOURCE}", file=sys.stderr)
            return 1
        print(f"val: wrote {build_val_set(val_images, out / 'val_x4')} pairs to {out / 'val_x4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
