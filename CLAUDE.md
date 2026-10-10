# CLAUDE.md

×4 real-world photo super-resolution, trained from scratch in PyTorch. Design source of truth: `design/2026-10-10-image-upscaler-design.md` — read it before changing architecture, data pipeline, or training.

## Key decisions (don't change without asking)
- Generator is trained from random init. The only pretrained weights allowed are frozen VGG19 (perceptual loss, stage 2) and LPIPS (eval metric). Never initialize the generator from external weights.
- Scale is ×4 only. Generator: `SRVGGCompact` (F=64, N=16, ~1.2M params). New architectures go through the registry in `upscaler/models/__init__.py`.
- Degradation runs on GPU per batch (`upscaler/data/degrade.py`) and must stay seedable via an explicit `torch.Generator`.
- Shipped model = EMA generator weights; checkpoints embed the model config.
- Hardware: RTX 5080 16 GB (Blackwell, sm_120) → PyTorch must be a CUDA ≥ 12.8 build. Training budget: a few hours per stage.

## Commands
```bash
uv venv && uv pip install -e .          # setup
pytest -q                               # tests (no dataset needed)
python train.py --config configs/smoke.yaml   # end-to-end sanity, < 1 min
python train.py --config configs/stage1_psnr.yaml
python train.py --config configs/stage2_gan.yaml
python evaluate.py --ckpt <path> --out <dir>
python upscale.py <input> -o <output>
```

## Workflow
- All design docs and implementation plans live in `design/` (specs: `design/YYYY-MM-DD-<topic>-design.md`, plans: `design/YYYY-MM-DD-<topic>-plan.md`). Never write them to `docs/superpowers/...`.
- **Auto-push:** after every completed feature or fix (tests passing), commit and `git push` to `origin main` without asking.

## Conventions
- Plain PyTorch + YAML configs; no Lightning/Hydra/etc.
- TDD: add/adjust tests in `tests/` alongside each unit; tests must not require downloaded datasets.
- Image tensors are float RGB in [0, 1], shape (B, 3, H, W).
- PSNR/SSIM on Y channel with 4-px border crop.
- `datasets/`, `runs/`, `weights/` are gitignored — never commit data or checkpoints.
