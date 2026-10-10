# upscaler

A ×4 image super-resolution model for **real-world photos**, built and trained from scratch in PyTorch.

- **Model:** compact SRVGG-style CNN (~0.62M params, PixelShuffle ×4, global bilinear residual)
- **Data:** DF2K (DIV2K + Flickr2K) with Real-ESRGAN-style on-the-fly degradations (blur, resize, noise, JPEG — two passes)
- **Training:** stage 1 L1 (PSNR model) → stage 2 GAN fine-tune (L1 + VGG perceptual + U-Net discriminator)
- **Target hardware:** single RTX 5080 (16 GB), a few hours per stage

> **Status:** design phase. See the [design spec](design/2026-10-10-image-upscaler-design.md).

## Planned usage

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
