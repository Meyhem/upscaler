# Image Upscaler — Design Spec

**Date:** 2026-10-10
**Status:** Approved in brainstorming, pending written-spec review

## 1. Goal

Build a ×4 image super-resolution model **from scratch** (random init, our own code and training) that upscales **real-world photos** — phone pictures, compressed/old web images — producing output that is visibly sharper and cleaner than bicubic interpolation.

### Constraints
- **From scratch:** the shipped generator is trained from random weights. The only pretrained component allowed is a **frozen ImageNet VGG19 used as a perceptual loss** during stage-2 training; it is never part of the shipped model. LPIPS (used only as an evaluation metric) also uses pretrained weights; this is acceptable for the same reason.
- **Scale:** ×4 only. ×2 is available at inference by downscaling the ×4 output.
- **Compute budget:** a few hours per training stage on a single RTX 5080 (16 GB, Blackwell).
- **Stack:** Python 3.12, PyTorch (CUDA ≥ 12.8 build), plain scripts + YAML configs, no training framework.

### Success criteria
1. Stage-1 model beats bicubic on PSNR, SSIM, and LPIPS on the validation set.
2. Stage-2 model beats stage-1 on LPIPS (PSNR may drop slightly — expected).
3. On the user's own real photos (`datasets/real_test/`), stage-2 output is visibly sharper with less JPEG blocking/noise than bicubic, judged by side-by-side comparison.
4. CLI upscales a 4K input on the 5080 without OOM and without visible tile seams.

## 2. Data pipeline

### 2.1 Datasets
- **Training:** DF2K = DIV2K train HR (800 images) + Flickr2K (2,650 images), ~25 GB raw.
- **Validation:** DIV2K validation HR (100 images).
- **Real test:** `datasets/real_test/` — user-supplied photos, no ground truth; used for visual comparison only.

### 2.2 Download — `upscaler/data/download.py`
Downloads DIV2K train/valid HR and Flickr2K archives from their official mirrors into `datasets/raw/<name>/`, each archive extracted into its own directory. Skips archives already present. Verifies archive size/checksum where the mirror publishes one.

### 2.3 Tile extraction — `upscaler/data/prepare.py` (run once)
- Crops each training HR image into **400×400 tiles, stride 200** → `datasets/df2k_tiles/*.png`.
- Discards low-texture tiles: tile is dropped if the standard deviation of its grayscale Laplacian is below a config threshold (default tuned so ≲10% of tiles are dropped).
- Validation images are not tiled.

### 2.4 Online degradation — `upscaler/data/degrade.py`
Runs **on GPU, per batch**, so each HR crop gets a fresh degradation every time it is seen. Simplified Real-ESRGAN second-order pipeline:

1. Blur: isotropic Gaussian / anisotropic Gaussian / generalized Gaussian, plus sinc (ringing) filter with low probability.
2. Random resize: up / down / keep, random choice of area / bilinear / bicubic.
3. Noise: Gaussian (color or gray) or Poisson.
4. JPEG compression, quality ∈ [30, 95].
5. Second pass of steps 1–4 with reduced probabilities and ranges.
6. Final resize to exactly HR/4 combined with an optional final sinc filter and JPEG pass (random order).

All probabilities and ranges are config values. The function takes an explicit `torch.Generator` so results are reproducible for a given seed. Output is clamped to [0, 1] and quantized to 8-bit levels to match real images.

### 2.5 Training pairs — `upscaler/data/dataset.py`
- Dataset returns random **256×256 HR crops** from tiles with random horizontal flip / vertical flip / 90° rotation.
- LR (64×64) is produced from the batch on GPU by `degrade.py`.
- Training target is the HR crop with **unsharp-mask sharpening** applied (Real-ESRGAN "USM ground truth"); the degradation input uses the unsharpened HR.

### 2.6 Validation set
`prepare.py` also builds a fixed validation set: each DIV2K valid HR image is center-cropped to a multiple of 4 and degraded once with seed `1234`. LR/HR pairs are saved to `datasets/val_x4/{lr,hr}/` so every run is evaluated on identical inputs.

## 3. Model

### 3.1 Generator — `upscaler/models/srvgg.py` (`SRVGGCompact`)
```
LR (3×H×W, [0,1])
 → Conv3×3(3→F) + PReLU
 → [Conv3×3(F→F) + PReLU] × N
 → Conv3×3(F→3·s²)
 → PixelShuffle(s)                     → 3×sH×sW
 + bilinear_upsample(LR, s)            (global residual)
 → output
```
- Defaults: `num_feat F=64`, `num_conv N=16`, `scale s=4` → ~1.2M parameters.
- Fully convolutional; arbitrary input size.
- Output clamped to [0, 1] only at inference (not during training, to keep gradients).
- Preset `tiny`: F=32, N=8 (smoke tests).

### 3.2 Discriminator — `upscaler/models/discriminator.py` (`UNetDiscriminatorSN`)
U-Net discriminator with spectral normalization (Real-ESRGAN design), `num_feat=64`, skip connections, per-pixel real/fake logits. ~4M params. Training only.

### 3.3 Registry — `upscaler/models/__init__.py`
`build_model(cfg)` maps `cfg.model.type` (e.g. `"srvgg_compact"`) to a class. Adding RRDBNet later = one new module + one registry entry.

## 4. Training — `train.py`

Single script, behavior selected by config (`configs/stage1_psnr.yaml`, `configs/stage2_gan.yaml`, `configs/smoke.yaml`).

### 4.1 Stage 1 — PSNR-oriented
| Item | Value |
|---|---|
| Loss | L1 |
| Optimizer | Adam(β=0.9, 0.99), lr 2e-4 |
| Schedule | cosine decay to 1e-6 |
| Batch | 32 (LR 64×64 / HR 256×256) |
| Iterations | 200k (target 2–4 h; measured on first run) |

### 4.2 Stage 2 — GAN fine-tune
- Generator initialized from stage-1 checkpoint (EMA weights).
- Generator loss: `1.0·L1 + 1.0·Perceptual + 0.1·GAN`.
  - Perceptual: L1 between frozen VGG19 features (layers conv1_2, conv2_2, conv3_4, conv4_4, conv5_4; weights 0.1, 0.1, 1, 1, 1), ImageNet-normalized inputs, before ReLU.
  - GAN: vanilla BCE-with-logits on per-pixel discriminator output.
- Discriminator loss: BCE real vs fake (fake detached).
- Optimizers: Adam(0.9, 0.99), lr 1e-4 for both, constant.
- Iterations: 100k (target 3–4 h).

### 4.3 Common
- **EMA** of generator weights, decay 0.999; EMA weights are used for validation and are the shipped model.
- Mixed precision **bf16** autocast; `torch.compile` on the generator behind a config flag (default on, auto-disabled if compilation fails).
- **Checkpoint** every 5k iters to `runs/<name>/ckpt/iter_<n>.pt` (generator, EMA, discriminator if any, optimizers, scheduler, iteration, RNG states). `--resume` loads the latest.
- **Validation** every 5k iters on `datasets/val_x4` (PSNR/SSIM/LPIPS, EMA weights).
- **Logging:** TensorBoard in `runs/<name>/tb/` — losses, lr, val metrics, fixed grid of 4 val crops (LR-bicubic | output | HR).
- **Reproducibility:** seed from config; resolved config copied to `runs/<name>/config.yaml`.
- **Smoke mode:** `configs/smoke.yaml` — tiny model, 200 iters, 50 tiles, batch 4; completes in < 1 min.

## 5. Evaluation — `evaluate.py`

`python evaluate.py --ckpt <path> [--ckpt <path> ...] --out runs/<name>/eval/`

- Metrics on `datasets/val_x4`:
  - **PSNR / SSIM** on Y channel (ITU-R BT.601), 4-px border crop.
  - **LPIPS** (AlexNet variant) on RGB.
- Rows: bicubic baseline + each checkpoint given. Printed as a table and written to `results.json`.
- Saves comparison strips (LR-nearest | bicubic | each model | HR) for val images and for `datasets/real_test/` (no HR column).

## 6. Inference CLI — `upscale.py`

```bash
python upscale.py INPUT -o OUTPUT [--model PATH] [--outscale {2,4}] [--tile 512] [--tile-pad 16] [--fp16] [--cpu]
```
- INPUT is a file or directory (jpg / jpeg / png / webp). Directory input writes `<stem>_x4.png` into OUTPUT dir.
- **Tiling** (`upscaler/tiling.py`): input split into `tile`-sized tiles with `tile-pad` overlap; each tile upscaled, padded borders discarded, tiles stitched. `--tile 0` disables tiling.
- `--outscale 2`: ×4 output resized down by 2 with Lanczos.
- Alpha channel: upscaled with bicubic and recombined.
- Default `--model`: `weights/upscaler_x4.pt` (EMA generator + model config stored inside the checkpoint so the CLI can rebuild the architecture).
- **Errors:** unreadable image → warning, skip (directory mode) or exit code 1 (single file). Missing checkpoint / architecture or scale mismatch → clear error, exit 1.

## 7. Project layout
```
upscaler/
├── pyproject.toml
├── configs/        stage1_psnr.yaml · stage2_gan.yaml · smoke.yaml
├── upscaler/
│   ├── data/       download.py · prepare.py · dataset.py · degrade.py
│   ├── models/     __init__.py · srvgg.py · discriminator.py
│   ├── losses.py   metrics.py   tiling.py   utils.py
├── train.py  evaluate.py  upscale.py
├── tests/
├── datasets/  runs/  weights/      (gitignored)
└── docs/superpowers/specs/
```
Environment: `uv` venv; PyTorch wheel for CUDA ≥ 12.8 (required for RTX 50-series / sm_120).

Dependencies: torch, torchvision, numpy, pillow, opencv-python-headless, pyyaml, tensorboard, lpips, scikit-image, tqdm, pytest.

## 8. Testing (pytest, no dataset required)
| Area | Checks |
|---|---|
| Model | output shape = 4× input for several sizes; param count ≈ 1.2M ±10% for defaults; gradients reach first conv |
| Discriminator | output shape = input spatial shape, 1 channel |
| Degradation | output = HR/4 exactly; values in [0,1]; same seed → identical output; each stage enabled alone changes the image |
| Tiling | tiled == untiled output within 1e-4 (fp32) on a 300×200 image with tile 64, tile_pad 24 (≥ generator receptive-field radius of 18 LR px; the CLI default of 16 is a speed/quality trade-off, not exact) |
| Metrics | PSNR(x, x) = capped max; PSNR/SSIM match scikit-image on a fixed pair |
| Losses | perceptual loss = 0 for identical inputs; GAN loss sign sanity |
| Checkpoint | save → load → identical weights, optimizer state, iteration |
| End-to-end | smoke training on synthetic images runs, loss decreases, checkpoint loads in `upscale.py` |

TDD: tests written alongside each unit.

## 9. Out of scope (for now)
- ×2-native, ×3, or arbitrary-scale models.
- RRDBNet / transformer architectures (registry leaves room).
- Larger datasets (LSDIR, OST), face-specific restoration, video.
- ONNX / TensorRT export, GUI.
