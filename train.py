"""Train the upscaler.

python train.py --config configs/stage1_psnr.yaml [--resume] [--run-dir runs/x] [--device cuda]
"""
from __future__ import annotations

import argparse
import sys

from upscaler.trainer import train
from upscaler.utils import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in run_dir")
    ap.add_argument("--run-dir")
    ap.add_argument("--device", choices=["cpu", "cuda"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.run_dir:
        cfg["run_dir"] = args.run_dir
    if args.device:
        cfg["device"] = args.device
    res = train(cfg, resume=args.resume)
    print(f"done at iter {res['iter']}; final val: {res['val']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
