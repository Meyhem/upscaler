"""Score checkpoints vs bicubic on the fixed validation set and save comparison strips.

python evaluate.py --ckpt runs/stage1/ema.pt --ckpt runs/stage2/ema.pt --out runs/eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from upscaler.data.dataset import PairedImageDataset, list_images
from upscaler.evaluation import bicubic_upscale, evaluate_pairs
from upscaler.imageio import read_rgb, write_rgb
from upscaler.metrics import LPIPSMetric
from upscaler.tiling import tiled_forward
from upscaler.utils import load_generator

SCALE = 4


def model_name(path) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def _center_crop(lr: torch.Tensor, crop: int):
    _, h, w = lr.shape
    ch, cw = min(crop, h), min(crop, w)
    t, l = (h - ch) // 2, (w - cw) // 2
    return lr[:, t:t + ch, l:l + cw], (t, l, ch, cw)


@torch.no_grad()
def _strip(lr_crop: torch.Tensor, models: dict, device, tile: int, hr_crop: torch.Tensor | None):
    x = lr_crop[None].to(device)
    cols = [F.interpolate(x, scale_factor=SCALE, mode="nearest"), bicubic_upscale(x, SCALE)]
    cols += [tiled_forward(m, x, SCALE, tile, 16).float().clamp(0, 1) for m in models.values()]
    if hr_crop is not None:
        cols.append(hr_crop[None].to(device))
    return torch.cat(cols, dim=-1)[0].cpu()


def run(ckpts, out_dir, val_dir, real_dir=None, device="cpu", max_images=None, use_lpips=True,
        crop: int = 128, tile: int = 512) -> dict:
    device = torch.device(device)
    out_dir = Path(out_dir)
    models = {model_name(c): load_generator(c, device, expected_scale=SCALE) for c in ckpts}
    lp = LPIPSMetric(device) if use_lpips else None

    results = {"bicubic": evaluate_pairs(bicubic_upscale, val_dir, device, max_images, lp)}
    for name, m in models.items():
        results[name] = evaluate_pairs(lambda x, m=m: tiled_forward(m, x, SCALE, tile, 16),
                                       val_dir, device, max_images, lp)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    keys = list(next(iter(results.values())))
    print(f"{'model':<28}" + "".join(f"{k.upper():>10}" for k in keys))
    for name, r in results.items():
        print(f"{name:<28}" + "".join(f"{r[k]:>10.4f}" for k in keys))

    ds = PairedImageDataset(val_dir)
    n = len(ds) if max_images is None else min(len(ds), max_images)
    for i in range(n):
        lr, hr, name = ds[i]
        lr_c, (t, l, ch, cw) = _center_crop(lr, crop)
        hr_c = hr[:, t * SCALE:(t + ch) * SCALE, l * SCALE:(l + cw) * SCALE]
        write_rgb(out_dir / "val" / f"{Path(name).stem}.png", _strip(lr_c, models, device, tile, hr_c))

    if real_dir is not None and Path(real_dir).is_dir():
        for p in list_images(real_dir):
            lr_c, _ = _center_crop(read_rgb(p), crop)
            write_rgb(out_dir / "real" / f"{p.stem}.png", _strip(lr_c, models, device, tile, None))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-dir", default="datasets/val_x4")
    ap.add_argument("--real-dir", default="datasets/real_test")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--crop", type=int, default=128, help="LR crop size for comparison strips")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    try:
        run(args.ckpt, args.out, args.val_dir, args.real_dir, device, args.max_images,
            not args.no_lpips, args.crop, args.tile)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
