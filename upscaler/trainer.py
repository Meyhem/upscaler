"""Training loop for stage 1 (L1) and stage 2 (GAN); behaviour selected by config."""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from upscaler.data.dataset import PairedImageDataset, TileDataset
from upscaler.data.degrade import DegradeConfig, degrade, usm_sharp
from upscaler.evaluation import evaluate_pairs
from upscaler.losses import PerceptualLoss, gan_loss
from upscaler.metrics import LPIPSMetric
from upscaler.models import build_model
from upscaler.models.discriminator import UNetDiscriminatorSN
from upscaler.tiling import tiled_forward
from upscaler.utils import EMA, export_generator, latest_checkpoint, load_generator, save_checkpoint, set_seed


def _infinite(loader):
    while True:
        yield from loader


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _maybe_compile(net, enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return net
    try:
        compiled = torch.compile(net)
        with torch.no_grad(), _autocast(device):
            compiled(torch.zeros(1, 3, 32, 32, device=device))
        return compiled
    except Exception as e:  # compilation problems must not kill a training run
        print(f"warning: torch.compile failed ({e}); continuing without it", flush=True)
        return net


@torch.no_grad()
def _sample_grid(model, val_dir, device, scale: int, n: int = 4, crop: int = 64) -> torch.Tensor:
    """Rows of [bicubic | model | HR] center crops for TensorBoard."""
    ds = PairedImageDataset(val_dir)
    rows = []
    for i in range(min(n, len(ds))):
        lr, hr, _ = ds[i]
        _, h, w = lr.shape
        c = min(crop, h, w)
        t, l = (h - c) // 2, (w - c) // 2
        lr_c = lr[None, :, t:t + c, l:l + c].to(device)
        hr_c = hr[None, :, t * scale:(t + c) * scale, l * scale:(l + c) * scale].to(device)
        bic = F.interpolate(lr_c, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)
        sr = model(lr_c).float().clamp(0, 1)
        rows.append(torch.cat([bic, sr, hr_c], dim=-1))
    width = max(r.shape[-1] for r in rows)
    rows = [F.pad(r, (0, width - r.shape[-1])) for r in rows]
    return torch.cat(rows, dim=-2)[0]


def train(cfg: dict, resume: bool = False) -> dict:
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg["seed"])
    tcfg, dcfg, vcfg, gcfg = cfg["train"], cfg["data"], cfg["val"], cfg.get("gan")
    scale = cfg["model"]["scale"]
    run_dir = Path(cfg["run_dir"])
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # --- models / optimizers ---------------------------------------------
    net = build_model(cfg["model"]).to(device)
    if tcfg.get("init_from") and not resume:
        net.load_state_dict(load_generator(tcfg["init_from"], device).state_dict())
    ema = EMA(net, tcfg["ema_decay"])
    total = tcfg["total_iters"]
    opt_g = torch.optim.Adam(net.parameters(), lr=tcfg["lr"], betas=(0.9, 0.99))
    sched = None
    if tcfg["scheduler"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=total, eta_min=tcfg["eta_min"])
    net_d = opt_d = percep = None
    if gcfg:
        net_d = UNetDiscriminatorSN(num_feat=gcfg["d_num_feat"]).to(device)
        opt_d = torch.optim.Adam(net_d.parameters(), lr=gcfg["d_lr"], betas=(0.9, 0.99))
        percep = PerceptualLoss(pretrained=gcfg["vgg_pretrained"]).to(device)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])

    start = 0
    if resume:
        path = latest_checkpoint(ckpt_dir)
        if path is None:
            raise FileNotFoundError(f"--resume given but no checkpoint in {ckpt_dir}")
        ck = torch.load(path, map_location=device, weights_only=True)
        net.load_state_dict(ck["net"])
        ema.load_state_dict(ck["ema"])
        opt_g.load_state_dict(ck["opt_g"])
        if sched is not None:
            sched.load_state_dict(ck["sched"])
        if net_d is not None:
            net_d.load_state_dict(ck["net_d"])
            opt_d.load_state_dict(ck["opt_d"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        gen.set_state(ck["degrade_rng"].cpu())
        start = ck["iter"]
        print(f"resumed from {path} at iter {start}", flush=True)

    # --- data -------------------------------------------------------------
    ds = TileDataset(dcfg["tiles_dir"], crop=dcfg["crop"], max_tiles=dcfg.get("max_tiles"))
    if len(ds) < dcfg["batch_size"]:
        raise ValueError(f"{len(ds)} tiles < batch_size {dcfg['batch_size']}")
    nw = dcfg["num_workers"]
    loader = DataLoader(ds, batch_size=dcfg["batch_size"], shuffle=True, num_workers=nw,
                        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=nw > 0)
    batches = _infinite(loader)
    degrade_cfg = DegradeConfig.from_dict(cfg.get("degradation"))
    if degrade_cfg.scale != scale:
        raise ValueError(f"degradation scale {degrade_cfg.scale} != model scale {scale}")

    fwd = _maybe_compile(net, tcfg.get("compile", False), device)
    lpips_metric = LPIPSMetric(device) if vcfg["lpips"] else None
    writer = SummaryWriter(str(run_dir / "tb"))
    history: list[dict] = []
    last_val: dict = {}
    w_pix = tcfg.get("pixel_weight", 1.0)

    # --- loop -------------------------------------------------------------
    net.train()
    t0 = time.time()
    for it in range(start + 1, total + 1):
        hr = next(batches).to(device, non_blocking=True)
        lr_img = degrade(hr, gen, degrade_cfg)
        gt = usm_sharp(hr) if tcfg["usm_gt"] else hr

        if net_d is not None:
            net_d.requires_grad_(False)
        with _autocast(device):
            sr = fwd(lr_img)
            losses = {"l_pix": F.l1_loss(sr.float(), gt) * w_pix}
            if net_d is not None:
                losses["l_percep"] = percep(sr, gt) * gcfg["perceptual_weight"]
                losses["l_gan"] = gan_loss(net_d(sr), True) * gcfg["gan_weight"]
            loss_g = sum(losses.values())
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        if sched is not None:
            sched.step()

        if net_d is not None:
            net_d.requires_grad_(True)
            with _autocast(device):
                l_d = gan_loss(net_d(hr), True) + gan_loss(net_d(sr.detach()), False)
            opt_d.zero_grad(set_to_none=True)
            l_d.backward()
            opt_d.step()
            losses["l_d"] = l_d
        ema.update(net)

        if it % tcfg["log_every"] == 0:
            vals = {k: float(v.detach()) for k, v in losses.items()}
            its = tcfg["log_every"] / max(time.time() - t0, 1e-9)
            t0 = time.time()
            for k, v in vals.items():
                writer.add_scalar(f"train/{k}", v, it)
            writer.add_scalar("train/lr", opt_g.param_groups[0]["lr"], it)
            writer.add_scalar("train/it_per_s", its, it)
            print(f"[{it}/{total}] " + " ".join(f"{k}={v:.4f}" for k, v in vals.items()) + f" | {its:.1f} it/s",
                  flush=True)
            history.append({"iter": it, **vals})

        # checkpoint before validation, so a crash during validation loses nothing
        if it % tcfg["ckpt_every"] == 0 or it == total:
            state = {"iter": it, "net": net.state_dict(), "ema": ema.state_dict(), "opt_g": opt_g.state_dict(),
                     "model_cfg": cfg["model"], "torch_rng": torch.get_rng_state(),
                     "degrade_rng": gen.get_state()}
            if sched is not None:
                state["sched"] = sched.state_dict()
            if net_d is not None:
                state["net_d"] = net_d.state_dict()
                state["opt_d"] = opt_d.state_dict()
            save_checkpoint(ckpt_dir / f"iter_{it}.pt", state)
            export_generator(run_dir / "ema.pt", cfg["model"], ema.state_dict())

        if it % tcfg["val_every"] == 0 or it == total:
            last_val = evaluate_pairs(lambda x: tiled_forward(ema.module, x, scale, vcfg["tile"], 16),
                                      vcfg["dir"], device, vcfg.get("max_images"), lpips_metric)
            for k, v in last_val.items():
                writer.add_scalar(f"val/{k}", v, it)
            writer.add_image("val/bicubic_sr_hr", _sample_grid(ema.module, vcfg["dir"], device, scale), it)
            print(f"[{it}/{total}] val " + " ".join(f"{k}={v:.4f}" for k, v in last_val.items()), flush=True)

    writer.close()
    return {"iter": max(start, total), "history": history, "val": last_val}
