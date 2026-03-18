#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchvision.utils import make_grid, save_image

from src.config import DatasetPair, load_config
from src.data import get_domain_images, make_eval_dataloader
from src.models import ResnetGenerator, adapt_output_for_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export side-by-side qualitative comparison for spatial vs spectral CycleGAN"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of source samples to export per pair",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/qualitative",
        help="Directory for qualitative comparison images",
    )
    return parser.parse_args()


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")
    return candidates[-1]


def _load_generator_ab(checkpoint_path: Path, device: torch.device) -> ResnetGenerator:
    model = ResnetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["G_AB"])
    model.eval()
    return model


def _denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def export_pair(
    config_path: Path,
    pair: DatasetPair,
    out_dir: Path,
    num_samples: int,
) -> Path:
    cfg = load_config(config_path)
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    src_images = get_domain_images(cfg.dataset_root, pair.source)
    tgt_images = get_domain_images(cfg.dataset_root, pair.target)

    loader_src = make_eval_dataloader(src_images, cfg.training.image_size, num_samples)
    loader_tgt = make_eval_dataloader(tgt_images, cfg.training.image_size, num_samples)
    src_batch = next(iter(loader_src))["image"].to(device)
    tgt_batch = next(iter(loader_tgt))["image"].to(device)

    ckpt_spatial = _find_latest_checkpoint(cfg.output_root / pair.name / "spatial" / "checkpoints")
    ckpt_spectral = _find_latest_checkpoint(cfg.output_root / pair.name / "spectral" / "checkpoints")

    g_spatial = _load_generator_ab(ckpt_spatial, device)
    g_spectral = _load_generator_ab(ckpt_spectral, device)

    with torch.no_grad():
        fake_spatial = g_spatial(src_batch)
        fake_spectral_raw = g_spectral(src_batch)
        fake_spectral = adapt_output_for_method(
            method="spectral",
            generated=fake_spectral_raw,
            source=src_batch,
            low_freq_ratio=cfg.training.spectral_low_freq_ratio,
        )

    rows = torch.cat(
        [
            _denorm(src_batch),
            _denorm(fake_spatial),
            _denorm(fake_spectral),
            _denorm(tgt_batch),
        ],
        dim=0,
    )

    nrow = src_batch.shape[0]
    grid = make_grid(rows, nrow=nrow)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair.name}_spatial_vs_spectral.png"
    save_image(grid, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    output_dir = Path(args.output_dir)

    print(f"Loaded config from: {config_path.resolve()}")
    print("Each output figure has rows: Source, Spatial, Spectral, Target reference")

    for pair in cfg.pairs:
        out_path = export_pair(config_path, pair, output_dir, args.num_samples)
        print(f"Saved qualitative comparison: {out_path}")


if __name__ == "__main__":
    main()
