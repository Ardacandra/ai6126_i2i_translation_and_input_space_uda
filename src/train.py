from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid, save_image

from src.config import DatasetPair, ExperimentConfig
from src.data import make_train_dataloader
from src.models import (
    PatchDiscriminator,
    ResnetGenerator,
    adapt_output_for_method,
    init_weights,
)


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    g_ab: nn.Module,
    g_ba: nn.Module,
    d_a: nn.Module,
    d_b: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d_a: torch.optim.Optimizer,
    opt_d_b: torch.optim.Optimizer,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "G_AB": g_ab.state_dict(),
            "G_BA": g_ba.state_dict(),
            "D_A": d_a.state_dict(),
            "D_B": d_b.state_dict(),
            "opt_G": opt_g.state_dict(),
            "opt_D_A": opt_d_a.state_dict(),
            "opt_D_B": opt_d_b.state_dict(),
        },
        checkpoint_path,
    )


def _save_sample_grid(
    sample_path: Path,
    real_a: torch.Tensor,
    fake_b: torch.Tensor,
    real_b: torch.Tensor,
) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    max_items = min(8, real_a.shape[0])
    grid = make_grid(
        torch.cat([real_a[:max_items], fake_b[:max_items], real_b[:max_items]], dim=0),
        nrow=max_items,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(grid, sample_path)


def train_one_experiment(
    config: ExperimentConfig,
    pair: DatasetPair,
    method: str,
) -> Tuple[Path, Path]:
    train_cfg = config.training
    exp_dir = config.output_root / pair.name / method
    checkpoints_dir = exp_dir / "checkpoints"
    samples_dir = exp_dir / "samples"
    log_path = exp_dir / "train_log.txt"

    _set_seed(config.seed)
    device = _select_device(config.device)
    loader, _, _ = make_train_dataloader(config.dataset_root, pair, train_cfg)
    loader_iter = iter(loader)

    g_ab = ResnetGenerator().to(device)
    g_ba = ResnetGenerator().to(device)
    d_a = PatchDiscriminator().to(device)
    d_b = PatchDiscriminator().to(device)

    g_ab.apply(init_weights)
    g_ba.apply(init_weights)
    d_a.apply(init_weights)
    d_b.apply(init_weights)

    opt_g = torch.optim.Adam(
        list(g_ab.parameters()) + list(g_ba.parameters()),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
    )
    opt_d_a = torch.optim.Adam(d_a.parameters(), lr=train_cfg.lr, betas=train_cfg.betas)
    opt_d_b = torch.optim.Adam(d_b.parameters(), lr=train_cfg.lr, betas=train_cfg.betas)

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    use_amp = train_cfg.use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    exp_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"pair={pair.name}, method={method}, device={device}\n")

    for epoch in range(1, train_cfg.epochs + 1):
        epoch_losses = {"g": 0.0, "d_a": 0.0, "d_b": 0.0}

        for _ in range(train_cfg.steps_per_epoch):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            real_a = batch["A"].to(device, non_blocking=True)
            real_b = batch["B"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                fake_b_raw = g_ab(real_a)
                fake_a_raw = g_ba(real_b)

                fake_b = adapt_output_for_method(
                    method=method,
                    generated=fake_b_raw,
                    source=real_a,
                    low_freq_ratio=train_cfg.spectral_low_freq_ratio,
                )
                fake_a = adapt_output_for_method(
                    method=method,
                    generated=fake_a_raw,
                    source=real_b,
                    low_freq_ratio=train_cfg.spectral_low_freq_ratio,
                )

                rec_a_raw = g_ba(fake_b)
                rec_b_raw = g_ab(fake_a)
                rec_a = adapt_output_for_method(
                    method=method,
                    generated=rec_a_raw,
                    source=fake_b,
                    low_freq_ratio=train_cfg.spectral_low_freq_ratio,
                )
                rec_b = adapt_output_for_method(
                    method=method,
                    generated=rec_b_raw,
                    source=fake_a,
                    low_freq_ratio=train_cfg.spectral_low_freq_ratio,
                )

                id_a_raw = g_ba(real_a)
                id_b_raw = g_ab(real_b)
                id_a = adapt_output_for_method(
                    method=method,
                    generated=id_a_raw,
                    source=real_a,
                    low_freq_ratio=train_cfg.spectral_low_freq_ratio,
                )
                id_b = adapt_output_for_method(
                    method=method,
                    generated=id_b_raw,
                    source=real_b,
                    low_freq_ratio=train_cfg.spectral_low_freq_ratio,
                )

                valid_b = torch.ones_like(d_b(real_b))
                valid_a = torch.ones_like(d_a(real_a))
                fake_target_b = torch.zeros_like(valid_b)
                fake_target_a = torch.zeros_like(valid_a)

                loss_gan_ab = criterion_gan(d_b(fake_b), valid_b)
                loss_gan_ba = criterion_gan(d_a(fake_a), valid_a)
                loss_cycle = criterion_cycle(rec_a, real_a) + criterion_cycle(rec_b, real_b)
                loss_identity = criterion_identity(id_a, real_a) + criterion_identity(id_b, real_b)
                loss_g = (
                    loss_gan_ab
                    + loss_gan_ba
                    + train_cfg.lambda_cycle * loss_cycle
                    + train_cfg.lambda_identity * loss_identity
                )

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)

            with autocast(enabled=use_amp):
                loss_d_a_real = criterion_gan(d_a(real_a), valid_a)
                loss_d_a_fake = criterion_gan(d_a(fake_a.detach()), fake_target_a)
                loss_d_a = 0.5 * (loss_d_a_real + loss_d_a_fake)

            opt_d_a.zero_grad(set_to_none=True)
            scaler.scale(loss_d_a).backward()
            scaler.step(opt_d_a)

            with autocast(enabled=use_amp):
                loss_d_b_real = criterion_gan(d_b(real_b), valid_b)
                loss_d_b_fake = criterion_gan(d_b(fake_b.detach()), fake_target_b)
                loss_d_b = 0.5 * (loss_d_b_real + loss_d_b_fake)

            opt_d_b.zero_grad(set_to_none=True)
            scaler.scale(loss_d_b).backward()
            scaler.step(opt_d_b)
            scaler.update()

            epoch_losses["g"] += float(loss_g.item())
            epoch_losses["d_a"] += float(loss_d_a.item())
            epoch_losses["d_b"] += float(loss_d_b.item())

        avg_g = epoch_losses["g"] / train_cfg.steps_per_epoch
        avg_d_a = epoch_losses["d_a"] / train_cfg.steps_per_epoch
        avg_d_b = epoch_losses["d_b"] / train_cfg.steps_per_epoch

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"epoch={epoch:03d}, loss_g={avg_g:.4f}, loss_d_a={avg_d_a:.4f}, loss_d_b={avg_d_b:.4f}\n"
            )

        if epoch % train_cfg.sample_every_epochs == 0 or epoch == 1:
            _save_sample_grid(
                samples_dir / f"epoch_{epoch:03d}.png",
                real_a.detach().cpu(),
                fake_b.detach().cpu(),
                real_b.detach().cpu(),
            )

        if epoch % train_cfg.save_every_epochs == 0 or epoch == train_cfg.epochs:
            _save_checkpoint(
                checkpoint_path=checkpoints_dir / f"epoch_{epoch:03d}.pt",
                epoch=epoch,
                g_ab=g_ab,
                g_ba=g_ba,
                d_a=d_a,
                d_b=d_b,
                opt_g=opt_g,
                opt_d_a=opt_d_a,
                opt_d_b=opt_d_b,
            )

    final_ckpt = checkpoints_dir / f"epoch_{train_cfg.epochs:03d}.pt"
    return final_ckpt, samples_dir
