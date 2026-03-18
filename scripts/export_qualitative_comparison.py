#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

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


def _draw_row_labels(
    grid: torch.Tensor,
    row_labels: list[str],
    row_height: int,
    padding: int,
    out_path: Path,
) -> None:
    grid_img = to_pil_image(grid)
    font = ImageFont.load_default()

    draw_probe = ImageDraw.Draw(grid_img)
    label_widths = []
    for label in row_labels:
        bbox = draw_probe.textbbox((0, 0), label, font=font)
        label_widths.append(bbox[2] - bbox[0])

    left_margin = max(label_widths) + 16
    canvas = Image.new("RGB", (grid_img.width + left_margin, grid_img.height), color=(255, 255, 255))
    canvas.paste(grid_img, (left_margin, 0))

    draw = ImageDraw.Draw(canvas)
    for row_idx, label in enumerate(row_labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        y_top = padding + row_idx * (row_height + padding)
        y_center = y_top + row_height // 2

        x = left_margin - text_w - 8
        y = y_center - text_h // 2
        draw.text((x, y), label, fill=(0, 0, 0), font=font)

    canvas.save(out_path)


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
    domain_set = {pair.source.strip().lower(), pair.target.strip().lower()}
    enforce_grayscale_channels = domain_set.issubset({"mnist", "usps"})

    # Shuffle image lists for diverse multi-category sampling while maintaining reproducibility
    import random
    rng = random.Random(42)
    rng.shuffle(src_images)
    rng.shuffle(tgt_images)

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
            enforce_grayscale_channels=enforce_grayscale_channels,
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
    grid_padding = 2
    grid = make_grid(rows, nrow=nrow, padding=grid_padding)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair.name}_spatial_vs_spectral.png"
    row_labels = ["Source", "Spatial", "Spectral", "Target"]
    _draw_row_labels(
        grid=grid,
        row_labels=row_labels,
        row_height=src_batch.shape[-2],
        padding=grid_padding,
        out_path=out_path,
    )
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
