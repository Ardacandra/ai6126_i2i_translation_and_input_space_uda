#!/usr/bin/env python3
"""Benchmark five UDA methods across multiple dataset pairs.

Methods evaluated
-----------------
source_only       Classifier trained only on labeled source data.
cyclegan          Train on CycleGAN (spatial) translated source images.
spectral_cyclegan Train on spectral CycleGAN translated source images.
cycada            CyCADA: pixel adaptation (CycleGAN) + feature-level DANN.
fda               FDA: swap low-frequency amplitudes from random target images.

Usage
-----
# Run all methods and pairs defined in config.yaml
python scripts/run_uda_benchmark.py

# Custom config path
python scripts/run_uda_benchmark.py --config config.yaml

# Subset of methods / pairs
python scripts/run_uda_benchmark.py --methods source_only fda
python scripts/run_uda_benchmark.py --pairs mnist_to_usps svhn_to_mnist
python scripts/run_uda_benchmark.py --methods cyclegan --pairs amazon_to_webcam
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.uda_data import load_labeled_source, load_labeled_target_eval, make_loader
from src.uda_train import (
    evaluate,
    train_cycada,
    train_cyclegan_uda,
    train_fda_uda,
    train_source_only,
)

# ─── Constants ────────────────────────────────────────────────────────────────

ALL_METHODS = ["source_only", "cyclegan", "spectral_cyclegan", "cycada", "fda"]


# ─── Argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run UDA benchmark (Task II)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to merged config file (default: config.yaml)",
    )
    p.add_argument(
        "--methods", nargs="*", default=None,
        choices=ALL_METHODS,
        metavar="METHOD",
        help=f"Methods to run. Choices: {ALL_METHODS}. Default: all from config.",
    )
    p.add_argument(
        "--pairs", nargs="*", default=None,
        metavar="PAIR",
        help="Pair names to run (e.g. mnist_to_usps). Default: all from config.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Override device (e.g. cuda, cpu). Default: use config value.",
    )
    return p.parse_args()


# ─── Config helpers ───────────────────────────────────────────────────────────

def _select_device(cfg: dict, override: Optional[str]) -> torch.device:
    spec = override or cfg.get("device", "auto")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _find_cyclegan_checkpoint(
    cyclegan_root: Path,
    pair_name: str,
    task1_method: str,  # "spatial" or "spectral"
) -> Optional[Path]:
    """Return the latest epoch checkpoint for the given pair/method, or None."""
    ckpt_dir = cyclegan_root / pair_name / task1_method / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None


def _infer_num_classes(dataset_root: Path, source_domain: str, image_size: int) -> int:
    ds = load_labeled_source(dataset_root, source_domain, image_size, augment=False)
    return len(ds.classes)


# ─── Result I/O ───────────────────────────────────────────────────────────────

def _save_results(
    results: Dict[str, Dict[str, Optional[float]]],
    output_root: Path,
    pair_names: List[str],
    methods: List[str],
) -> None:
    """Save CSV + JSON and print a formatted accuracy table."""
    # CSV
    csv_path = output_root / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + pair_names)
        for method in methods:
            row = [method]
            for pn in pair_names:
                val = results[method].get(pn)
                row.append(f"{val * 100:.2f}" if val is not None else "N/A")
            writer.writerow(row)
    print(f"\nSaved results CSV : {csv_path}")

    # JSON (raw floats for programmatic access)
    json_path = output_root / "results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON: {json_path}")

    # Pretty-print table
    method_w = max(20, max(len(m) for m in methods) + 2)
    col_w = max(16, max(len(n) for n in pair_names) + 2)
    sep = "=" * (method_w + col_w * len(pair_names))
    print(f"\n{sep}")
    print(f"{'Method':<{method_w}}" + "".join(f"{n:>{col_w}}" for n in pair_names))
    print("-" * (method_w + col_w * len(pair_names)))
    for method in methods:
        row = f"{method:<{method_w}}"
        for pn in pair_names:
            val = results[method].get(pn)
            cell = f"{val * 100:.2f}%" if val is not None else "N/A"
            row += f"{cell:>{col_w}}"
        print(row)
    print(sep)


# ─── Main benchmark loop ──────────────────────────────────────────────────────

def run_benchmark(
    cfg_path: str,
    method_filter: Optional[List[str]],
    pair_filter: Optional[List[str]],
    device_override: Optional[str],
) -> None:
    cfg_file = Path(cfg_path)
    with cfg_file.open("r") as f:
        cfg = yaml.safe_load(f)

    cfg_root = cfg_file.parent
    dataset_root = (cfg_root / cfg["dataset"]["root"]).resolve()
    uda_output_cfg = cfg.get("uda_output", cfg.get("output", {}))
    output_root = (cfg_root / uda_output_cfg["root"]).resolve()
    cyclegan_root = (cfg_root / cfg.get("cyclegan_output_root", "out/experiments")).resolve()

    device = _select_device(cfg, device_override)
    print(f"Device : {device}")
    print(f"Dataset: {dataset_root}")
    print(f"Output : {output_root}")

    # Determine which methods & pairs to run
    methods = method_filter or cfg.get("uda_methods", ALL_METHODS)
    all_pair_cfgs = cfg["pairs"]
    pair_cfgs = [p for p in all_pair_cfgs
                 if pair_filter is None or p["name"] in pair_filter]

    # Classifier training hyperparameters
    tc = cfg.get("classifier_training", {})
    image_size = int(tc.get("image_size", 128))
    batch_size = int(tc.get("batch_size", 32))
    num_workers = int(tc.get("num_workers", 4))
    epochs = int(tc.get("epochs", 30))
    steps_per_epoch = int(tc.get("steps_per_epoch", 0))
    lr = float(tc.get("lr", 1e-3))
    weight_decay = float(tc.get("weight_decay", 1e-4))
    use_amp = bool(tc.get("use_amp", True))
    pretrained = bool(tc.get("pretrained_backbone", True))

    fda_beta = float(cfg.get("fda", {}).get("beta", 0.1))
    cycada_lambda_domain = float(cfg.get("cycada", {}).get("lambda_domain", 0.1))

    output_root.mkdir(parents=True, exist_ok=True)

    # results[method][pair_name] = accuracy (float) or None on failure
    results: Dict[str, Dict[str, Optional[float]]] = {m: {} for m in methods}

    for pair in pair_cfgs:
        pair_name = pair["name"]
        src = pair["source"]
        tgt = pair["target"]

        print(f"\n{'=' * 60}")
        print(f"Pair : {pair_name}  ({src} → {tgt})")

        try:
            num_classes = _infer_num_classes(dataset_root, src, image_size)
            print(f"Classes: {num_classes}")
        except Exception as exc:
            print(f"[ERROR] Cannot load source domain {src!r}: {exc}")
            for m in methods:
                results[m][pair_name] = None
            continue

        pair_out = output_root / pair_name

        # ── Common kwargs ────────────────────────────────────────────────── #
        common = dict(
            dataset_root=dataset_root,
            source_domain=src,
            target_domain=tgt,
            num_classes=num_classes,
            image_size=image_size,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            lr=lr,
            weight_decay=weight_decay,
            num_workers=num_workers,
            pretrained=pretrained,
            device=device,
            use_amp=use_amp,
        )

        for method in methods:
            method_out = pair_out / method
            print(f"\n  ▶ {method}")

            try:
                acc: Optional[float]

                # ── source_only ─────────────────────────────────────────── #
                if method == "source_only":
                    _, acc = train_source_only(
                        **common, output_dir=method_out
                    )

                # ── cyclegan ────────────────────────────────────────────── #
                elif method == "cyclegan":
                    ckpt = _find_cyclegan_checkpoint(cyclegan_root, pair_name, "spatial")
                    if ckpt is None:
                        print(f"    [SKIP] No spatial CycleGAN checkpoint for {pair_name}")
                        results[method][pair_name] = None
                        continue
                    print(f"    Checkpoint: {ckpt}")
                    _, acc = train_cyclegan_uda(
                        **common,
                        cyclegan_checkpoint=ckpt,
                        method="spatial",
                        output_dir=method_out,
                    )

                # ── spectral_cyclegan ────────────────────────────────────── #
                elif method == "spectral_cyclegan":
                    ckpt = _find_cyclegan_checkpoint(cyclegan_root, pair_name, "spectral")
                    if ckpt is None:
                        print(f"    [SKIP] No spectral CycleGAN checkpoint for {pair_name}")
                        results[method][pair_name] = None
                        continue
                    print(f"    Checkpoint: {ckpt}")
                    _, acc = train_cyclegan_uda(
                        **common,
                        cyclegan_checkpoint=ckpt,
                        method="spectral",
                        output_dir=method_out,
                    )

                # ── cycada ──────────────────────────────────────────────── #
                elif method == "cycada":
                    ckpt = _find_cyclegan_checkpoint(cyclegan_root, pair_name, "spatial")
                    if ckpt is None:
                        print(f"    [SKIP] No spatial CycleGAN checkpoint for CyCADA {pair_name}")
                        results[method][pair_name] = None
                        continue
                    print(f"    Checkpoint: {ckpt}")
                    _, acc = train_cycada(
                        **common,
                        cyclegan_checkpoint=ckpt,
                        lambda_domain=cycada_lambda_domain,
                        output_dir=method_out,
                    )

                # ── fda ─────────────────────────────────────────────────── #
                elif method == "fda":
                    _, acc = train_fda_uda(
                        **common,
                        fda_beta=fda_beta,
                        output_dir=method_out,
                    )

                else:
                    print(f"    [SKIP] Unknown method: {method!r}")
                    results[method][pair_name] = None
                    continue

                results[method][pair_name] = acc
                print(f"    Target accuracy: {acc * 100:.2f}%")

            except Exception as exc:
                print(f"    [ERROR] {method} on {pair_name}: {exc}")
                traceback.print_exc()
                results[method][pair_name] = None

    _save_results(
        results,
        output_root,
        pair_names=[p["name"] for p in pair_cfgs],
        methods=methods,
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        cfg_path=args.config,
        method_filter=args.methods,
        pair_filter=args.pairs,
        device_override=args.device,
    )
