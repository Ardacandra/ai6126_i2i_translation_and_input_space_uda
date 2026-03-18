#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.train import train_one_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CycleGAN and Spectral CycleGAN experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to experiment config yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    print(f"Loaded config from: {Path(args.config).resolve()}")
    print(f"Dataset root: {cfg.dataset_root}")
    print(f"Output root: {cfg.output_root}")

    for pair in cfg.pairs:
        for method in cfg.methods:
            print("=" * 80)
            print(f"Training: {pair.source} -> {pair.target} | method={method}")
            ckpt, sample_dir = train_one_experiment(cfg, pair, method)
            print(f"Saved final checkpoint: {ckpt}")
            print(f"Saved sample grids in: {sample_dir}")

    print("All configured experiments finished.")


if __name__ == "__main__":
    main()
