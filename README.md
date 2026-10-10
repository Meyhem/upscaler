# upscaler

A ×4 image super-resolution model for **real-world photos**, built and trained from scratch in PyTorch.

- **Model:** compact SRVGG-style CNN (~0.62M params, PixelShuffle ×4, global bilinear residual)
- **Data:** DF2K (DIV2K + Flickr2K) with Real-ESRGAN-style on-the-fly degradations (blur, resize, noise, JPEG — two passes)
- **Training:** stage 1 L1 (PSNR model) → stage 2 GAN fine-tune (L1 + VGG perceptual + U-Net discriminator)
- **Target hardware:** single RTX 5080 (16 GB), a few hours per stage

> **Status:** stage 1 trained and shipped as `weights/upscaler_x4.pt` (not in git). Stage 2 (GAN) not trained yet —
> the first run was lost to a GPU fault (Xid 79) at 5k/25k iters. Design: [spec](design/2026-10-10-image-upscaler-design.md),
> [plan](design/2026-10-10-image-upscaler-plan.md).

## Results (DIV2K validation, 100 images, degraded with seed 1234)

| model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| bicubic | 24.03 | 0.611 | 0.809 |
| stage 1 (L1, 200k iters, DIV2K tiles) | **24.28** | **0.664** | **0.574** |

Stage 1: 200k iters, batch 32, ~26 it/s on RTX 5080 (~2.1 h). Tiles: 400 px, stride 200, `--min-texture 5`
(DIV2K 45,929 tiles; Flickr2K adds 151,492 for stage 2).

## Usage

```bash
# setup
uv venv && uv pip install -e .

# data
python -m upscaler.data.download
python -m upscaler.data.prepare

# training
python train.py --config configs/smoke.yaml        # < 1 min sanity check
python train.py --config configs/stage1_psnr.yaml
python train.py --config configs/stage2_gan.yaml

# evaluation
python evaluate.py --ckpt runs/stage1/ema.pt --ckpt runs/stage2/ema.pt --out runs/eval

# upscale
python upscale.py photo.jpg -o photo_x4.png
python upscale.py ./photos -o ./out --fp16
```

## Layout

```
configs/      training configs (stage1, stage2, smoke)
upscaler/     library: data/, models/, losses, metrics, tiling, utils
train.py      training entry point
evaluate.py   PSNR / SSIM / LPIPS + comparison images
upscale.py    inference CLI with tiling
tests/        pytest suite (no dataset needed)
```

`datasets/`, `runs/` and `weights/` are gitignored.
