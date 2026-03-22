#!/usr/bin/env python3
"""Export qualitative domain-translation comparisons for the Task II report.

For each dataset pair this script generates a side-by-side grid showing:
  Row 1 – Source images (original)
  Row 2 – CycleGAN (spatial) translated source
  Row 3 – Spectral CycleGAN translated source
  Row 4 – FDA translated source
  Row 5 – Target images (real)

The grids are saved as PNG files under ``--output-dir``.

Usage
-----
python scripts/export_uda_qualitative.py
python scripts/export_uda_qualitative.py --config config.yaml
python scripts/export_uda_qualitative.py --num-samples 8 --output-dir out/uda_qualitative
python scripts/export_uda_qualitative.py --pairs mnist_to_usps amazon_to_webcam
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from src.data import get_domain_images
from src.models import ResnetGenerator, adapt_output_for_method
from src.uda_data import (
    collect_image_paths,
    fda_transfer,
    get_domain_train_path,
    get_eval_transform as _eval_transform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export UDA qualitative comparison grids")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument(
        "--output-dir", type=str, default="out/uda_qualitative",
        help="Directory to write comparison PNG files",
    )
    p.add_argument("--pairs", nargs="*", default=None,
                   metavar="PAIR", help="Pair names to export (default: all)")
    return p.parse_args()


def _select_device(cfg: dict) -> torch.device:
    spec = cfg.get("device", "auto")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None


def _load_generator(ckpt_path: Path, device: torch.device) -> ResnetGenerator:
    g = ResnetGenerator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    g.load_state_dict(ckpt["G_AB"])
    g.eval()
    return g


def _load_images(
    image_paths: List[Path],
    n: int,
    image_size: int,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Load *n* images from *image_paths*, resize, normalize, return [N,3,H,W]."""
    rng = random.Random(seed)
    tfm = _eval_transform(image_size)
    chosen = rng.sample(list(image_paths), min(n, len(image_paths)))
    tensors = []
    for p in chosen:
        img = Image.open(p).convert("RGB")
        tensors.append(tfm(img))
    return torch.stack(tensors).to(device)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] → [0, 1] for grid display."""
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def _add_row_labels(
    grid_tensor: torch.Tensor,
    row_labels: List[str],
    row_height: int,
    padding: int,
    out_path: Path,
) -> None:
    """Attach row labels on the left side of the grid and save to *out_path*."""
    grid_img = to_pil_image(grid_tensor)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()

    draw_probe = ImageDraw.Draw(grid_img)
    max_lw = max(
        draw_probe.textbbox((0, 0), lbl, font=font)[2]
        for lbl in row_labels
    )
    left_margin = max_lw + 20

    canvas = Image.new("RGB", (grid_img.width + left_margin, grid_img.height),
                       color=(240, 240, 240))
    canvas.paste(grid_img, (left_margin, 0))
    draw = ImageDraw.Draw(canvas)

    for i, lbl in enumerate(row_labels):
        bb = draw.textbbox((0, 0), lbl, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        y_top = padding + i * (row_height + padding)
        yc = y_top + row_height // 2
        draw.text((left_margin - tw - 10, yc - th // 2), lbl, fill=(30, 30, 30), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-pair export
# ---------------------------------------------------------------------------

def export_pair(
    pair_cfg: dict,
    dataset_root: Path,
    cyclegan_root: Path,
    output_dir: Path,
    image_size: int,
    num_samples: int,
    device: torch.device,
) -> None:
    pair_name = pair_cfg["name"]
    src_domain = pair_cfg["source"]
    tgt_domain = pair_cfg["target"]

    print(f"\n{'─' * 50}")
    print(f"Pair: {pair_name}  ({src_domain} → {tgt_domain})")

    # Collect image paths
    src_paths = get_domain_images(dataset_root, src_domain)
    tgt_paths_raw = get_domain_images(dataset_root, tgt_domain)

    # Load source samples (fixed seed for reproducibility)
    src_tensors = _load_images(src_paths, num_samples, image_size, device, seed=42)
    tgt_tensors = _load_images(tgt_paths_raw, num_samples, image_size, device, seed=0)

    rows: List[torch.Tensor] = [_denorm(src_tensors)]
    row_labels: List[str] = ["Source"]

    fake_spatial: Optional[torch.Tensor] = None

    # ── CycleGAN spatial ──────────────────────────────────────────────────── #
    spatial_ckpt_dir = cyclegan_root / pair_name / "spatial" / "checkpoints"
    spatial_ckpt = _find_latest_checkpoint(spatial_ckpt_dir)
    if spatial_ckpt is not None:
        g_spatial = _load_generator(spatial_ckpt, device)
        with torch.no_grad():
            fake_spatial = adapt_output_for_method(
                method="spatial",
                generated=g_spatial(src_tensors),
                source=src_tensors,
                low_freq_ratio=0.2,
            )
        rows.append(_denorm(fake_spatial))
        row_labels.append("CycleGAN")
    else:
        print(f"  [SKIP] No spatial CycleGAN checkpoint → CycleGAN row omitted")

    # ── CyCADA (pixel stage) ──────────────────────────────────────────────── #
    if fake_spatial is not None:
        rows.append(_denorm(fake_spatial.clone()))
        row_labels.append("CyCADA")
    else:
        print("  [SKIP] No spatial checkpoint available → CyCADA row omitted")

    # ── Spectral CycleGAN ─────────────────────────────────────────────────── #
    spectral_ckpt_dir = cyclegan_root / pair_name / "spectral" / "checkpoints"
    spectral_ckpt = _find_latest_checkpoint(spectral_ckpt_dir)
    if spectral_ckpt is not None:
        domain_set = {src_domain.strip().lower(), tgt_domain.strip().lower()}
        enforce_gray = domain_set.issubset({"mnist", "usps"})
        g_spectral = _load_generator(spectral_ckpt, device)
        with torch.no_grad():
            fake_spectral = adapt_output_for_method(
                method="spectral",
                generated=g_spectral(src_tensors),
                source=src_tensors,
                low_freq_ratio=0.2,
                enforce_grayscale_channels=enforce_gray,
            )
        rows.append(_denorm(fake_spectral))
        row_labels.append("Spectral CycleGAN")
    else:
        print(f"  [SKIP] No spectral CycleGAN checkpoint → Spectral row omitted")

    # ── FDA ───────────────────────────────────────────────────────────────── #
    tgt_train_path = get_domain_train_path(dataset_root, tgt_domain)
    tgt_all_paths = collect_image_paths(tgt_train_path)
    if tgt_all_paths:
        tfm = _eval_transform(image_size)
        rng = random.Random(7)
        tgt_for_fda_list = [
            tfm(Image.open(rng.choice(tgt_all_paths)).convert("RGB"))
            for _ in range(src_tensors.shape[0])
        ]
        tgt_for_fda = torch.stack(tgt_for_fda_list).to(device)
        fda_imgs = fda_transfer(src_tensors, tgt_for_fda, beta=0.1)
        rows.append(_denorm(fda_imgs))
        row_labels.append("FDA")
    else:
        print(f"  [SKIP] No target images found for FDA row")

    # ── Target (real) ─────────────────────────────────────────────────────── #
    rows.append(_denorm(tgt_tensors))
    row_labels.append("Target")

    # ── Build and save grid ───────────────────────────────────────────────── #
    n = min(num_samples, src_tensors.shape[0])
    grid_rows = [row[:n].cpu() for row in rows]
    grid = make_grid(
        torch.cat(grid_rows, dim=0),
        nrow=n,
        padding=4,
        normalize=False,
    )

    row_h = image_size + 4  # image + padding
    out_path = output_dir / f"{pair_name}_comparison.png"
    _add_row_labels(grid, row_labels, row_height=row_h, padding=4, out_path=out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    cfg_file = Path(args.config)
    with cfg_file.open("r") as f:
        cfg = yaml.safe_load(f)

    cfg_root = cfg_file.parent
    dataset_root = (cfg_root / cfg["dataset"]["root"]).resolve()
    cyclegan_root = (cfg_root / cfg.get("cyclegan_output_root", "out/experiments")).resolve()
    output_dir = (cfg_root / args.output_dir).resolve()
    device = _select_device(cfg)

    tc = cfg.get("classifier_training", {})
    image_size = int(tc.get("image_size", 128))

    pair_cfgs = cfg["pairs"]
    if args.pairs:
        pair_cfgs = [p for p in pair_cfgs if p["name"] in args.pairs]

    print(f"Device     : {device}")
    print(f"Output dir : {output_dir}")
    print(f"Image size : {image_size}")
    print(f"Samples    : {args.num_samples}")

    for pair_cfg in pair_cfgs:
        export_pair(
            pair_cfg=pair_cfg,
            dataset_root=dataset_root,
            cyclegan_root=cyclegan_root,
            output_dir=output_dir,
            image_size=image_size,
            num_samples=args.num_samples,
            device=device,
        )

    print(f"\nAll grids saved to {output_dir}")


if __name__ == "__main__":
    main()
