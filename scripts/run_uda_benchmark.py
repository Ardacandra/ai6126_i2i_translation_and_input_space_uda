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
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.uda_data import fda_transfer, load_labeled_source, load_labeled_target_eval, make_loader
from src.uda_train import (
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
    checkpoint_epoch: Optional[int] = None,
) -> Optional[Path]:
    """Return requested epoch checkpoint (or latest) for the given pair/method."""
    ckpt_dir = cyclegan_root / pair_name / task1_method / "checkpoints"
    if not ckpt_dir.exists():
        return None
    if checkpoint_epoch is not None:
        ckpt_path = ckpt_dir / f"epoch_{int(checkpoint_epoch):03d}.pt"
        return ckpt_path if ckpt_path.exists() else None
    candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None


def _resolve_qualitative_checkpoint_epoch(
    cfg: dict,
    pair_cfg: dict,
    task1_method: str,  # "spatial" or "spectral"
) -> Optional[int]:
    """Resolve checkpoint epoch using qualitative config precedence.

    Precedence:
      pair.method -> pair.shared -> global.method -> global.shared -> latest
    """
    base_pair = pair_cfg.get("qualitative_checkpoint_epoch")
    qual_cfg = cfg.get("qualitative", {})
    if not isinstance(qual_cfg, dict):
        qual_cfg = {}
    base_global = qual_cfg.get("checkpoint_epoch")

    epoch = pair_cfg.get(f"qualitative_checkpoint_epoch_{task1_method}")
    if epoch is None:
        epoch = base_pair
    if epoch is None:
        epoch = qual_cfg.get(f"checkpoint_epoch_{task1_method}")
    if epoch is None:
        epoch = base_global

    return int(epoch) if epoch is not None else None


def _infer_num_classes(dataset_root: Path, source_domain: str, image_size: int) -> int:
    ds = load_labeled_source(dataset_root, source_domain, image_size, augment=False)
    return len(ds.classes)


@torch.no_grad()
def _evaluate_metrics(
    classifier,
    loader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Compute classification metrics on a labeled target loader."""
    classifier.eval()

    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = classifier(imgs).argmax(1)

        idx = labels * num_classes + preds
        batch_conf = torch.bincount(
            idx,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        conf += batch_conf.cpu()

    classifier.train()

    total = conf.sum().item()
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "balanced_accuracy": 0.0,
        }

    tp = conf.diag().to(torch.float64)
    support = conf.sum(dim=1).to(torch.float64)
    pred_count = conf.sum(dim=0).to(torch.float64)

    precision = torch.where(pred_count > 0, tp / pred_count, torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
    f1 = torch.where(
        (precision + recall) > 0,
        2.0 * precision * recall / (precision + recall),
        torch.zeros_like(tp),
    )

    valid = support > 0
    if valid.any():
        precision_macro = float(precision[valid].mean().item())
        recall_macro = float(recall[valid].mean().item())
        f1_macro = float(f1[valid].mean().item())
    else:
        precision_macro = recall_macro = f1_macro = 0.0

    accuracy = float(tp.sum().item() / total)
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "balanced_accuracy": recall_macro,
    }


@torch.no_grad()
def _collect_images(loader, max_samples: int, device: torch.device) -> torch.Tensor:
    """Collect up to *max_samples* normalized images from a dataloader."""
    batches = []
    remaining = max(max_samples, 1)
    for imgs, _ in loader:
        if remaining <= 0:
            break
        take = min(remaining, imgs.size(0))
        batches.append(imgs[:take].to(device, non_blocking=True))
        remaining -= take
    if not batches:
        raise RuntimeError("No images available to compute image quality metrics")
    return torch.cat(batches, dim=0)


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def _batch_gradient_magnitude_mean(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def _batch_laplacian_variance(x: torch.Tensor) -> torch.Tensor:
    gray = x.mean(dim=1, keepdim=True)
    up = torch.zeros_like(gray)
    down = torch.zeros_like(gray)
    left = torch.zeros_like(gray)
    right = torch.zeros_like(gray)
    up[..., 1:, :] = gray[..., :-1, :]
    down[..., :-1, :] = gray[..., 1:, :]
    left[..., :, 1:] = gray[..., :, :-1]
    right[..., :, :-1] = gray[..., :, 1:]
    lap = up + down + left + right - 4.0 * gray
    return lap.var()


@torch.no_grad()
def _translate_for_quality(
    method: str,
    src_imgs: torch.Tensor,
    tgt_imgs: torch.Tensor,
    device: torch.device,
    fda_beta: float,
    cyclegan_checkpoint: Optional[Path],
    spectral_enforce_grayscale: bool,
) -> torch.Tensor:
    """Return method-adapted source images for image-quality comparison."""
    if method == "source_only":
        return src_imgs

    if method in {"cyclegan", "spectral_cyclegan", "cycada"}:
        if cyclegan_checkpoint is None:
            raise RuntimeError(f"Missing checkpoint for image quality metric: {method}")
        from src.models import ResnetGenerator, adapt_output_for_method

        g_ab = ResnetGenerator().to(device)
        ckpt = torch.load(cyclegan_checkpoint, map_location=device, weights_only=False)
        g_ab.load_state_dict(ckpt["G_AB"])
        g_ab.eval()

        with torch.no_grad():
            generated = g_ab(src_imgs)
            mode = "spectral" if method == "spectral_cyclegan" else "spatial"
            enforce_gray = mode == "spectral" and spectral_enforce_grayscale
            translated = adapt_output_for_method(
                method=mode,
                generated=generated,
                source=src_imgs,
                low_freq_ratio=0.2,
                enforce_grayscale_channels=enforce_gray and src_imgs.shape[1] == 3,
            )
        return translated

    if method == "fda":
        b = src_imgs.shape[0]
        if tgt_imgs.shape[0] >= b:
            perm = torch.randperm(tgt_imgs.shape[0], device=tgt_imgs.device)[:b]
            tgt_sample = tgt_imgs[perm]
        else:
            idx = torch.randint(0, tgt_imgs.shape[0], (b,), device=tgt_imgs.device)
            tgt_sample = tgt_imgs[idx]
        return fda_transfer(src_imgs, tgt_sample, beta=fda_beta)

    raise RuntimeError(f"Unsupported method for image quality metrics: {method}")


@torch.no_grad()
def _evaluate_image_quality_metrics(
    method: str,
    src_imgs: torch.Tensor,
    tgt_imgs: torch.Tensor,
    device: torch.device,
    fda_beta: float,
    cyclegan_checkpoint: Optional[Path],
    spectral_enforce_grayscale: bool,
) -> Dict[str, float]:
    """Compute unpaired image-quality/domain-alignment metrics."""
    adapted = _translate_for_quality(
        method=method,
        src_imgs=src_imgs,
        tgt_imgs=tgt_imgs,
        device=device,
        fda_beta=fda_beta,
        cyclegan_checkpoint=cyclegan_checkpoint,
        spectral_enforce_grayscale=spectral_enforce_grayscale,
    )

    adapted01 = _to_01(adapted)
    target01 = _to_01(tgt_imgs)

    mean_src = adapted01.mean(dim=(0, 2, 3))
    mean_tgt = target01.mean(dim=(0, 2, 3))
    std_src = adapted01.std(dim=(0, 2, 3), unbiased=False)
    std_tgt = target01.std(dim=(0, 2, 3), unbiased=False)

    grad_src = _batch_gradient_magnitude_mean(adapted01)
    grad_tgt = _batch_gradient_magnitude_mean(target01)
    sharp_src = _batch_laplacian_variance(adapted01)
    sharp_tgt = _batch_laplacian_variance(target01)

    return {
        "img_mean_l1_to_target": float((mean_src - mean_tgt).abs().mean().item()),
        "img_std_l1_to_target": float((std_src - std_tgt).abs().mean().item()),
        "img_grad_l1_to_target": float((grad_src - grad_tgt).abs().item()),
        "img_sharpness": float(sharp_src.item()),
        "img_sharpness_gap_to_target": float((sharp_src - sharp_tgt).abs().item()),
    }


# ─── Result I/O ───────────────────────────────────────────────────────────────

def _save_results(
    results: Dict[str, Dict[str, Optional[Dict[str, float]]]],
    output_root: Path,
    pair_names: List[str],
    methods: List[str],
) -> None:
    """Save accuracy and detailed metric outputs, then print summary tables."""
    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "balanced_accuracy",
        "img_mean_l1_to_target",
        "img_std_l1_to_target",
        "img_grad_l1_to_target",
        "img_sharpness",
        "img_sharpness_gap_to_target",
    ]

    # Accuracy CSV (kept for compatibility)
    csv_path = output_root / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + pair_names)
        for method in methods:
            row = [method]
            for pn in pair_names:
                metrics = results[method].get(pn)
                val = metrics.get("accuracy") if metrics is not None else None
                row.append(f"{val * 100:.2f}" if val is not None else "N/A")
            writer.writerow(row)
    print(f"\nSaved results CSV : {csv_path}")

    # Long-format CSV with all metrics
    metrics_csv_path = output_root / "results_metrics.csv"
    with metrics_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "pair"] + metric_names)
        for method in methods:
            for pn in pair_names:
                metrics = results[method].get(pn)
                row = [method, pn]
                for m in metric_names:
                    val = metrics.get(m) if metrics is not None else None
                    row.append(f"{val:.6f}" if val is not None else "N/A")
                writer.writerow(row)
    print(f"Saved metric CSV  : {metrics_csv_path}")

    # Detailed JSON (raw floats for programmatic access)
    json_path = output_root / "results_detailed.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metric JSON : {json_path}")

    # Backward-compatible JSON containing only accuracy
    legacy_json_path = output_root / "results.json"
    legacy_results = {
        method: {
            pn: (
                results[method][pn]["accuracy"]
                if results[method].get(pn) is not None
                else None
            )
            for pn in pair_names
        }
        for method in methods
    }
    with legacy_json_path.open("w") as f:
        json.dump(legacy_results, f, indent=2)
    print(f"Saved results JSON: {legacy_json_path}")

    # Pretty-print accuracy table
    method_w = max(20, max(len(m) for m in methods) + 2)
    col_w = max(16, max(len(n) for n in pair_names) + 2)
    sep = "=" * (method_w + col_w * len(pair_names))
    print(f"\n{sep}")
    print("Accuracy")
    print(f"{'Method':<{method_w}}" + "".join(f"{n:>{col_w}}" for n in pair_names))
    print("-" * (method_w + col_w * len(pair_names)))
    for method in methods:
        row = f"{method:<{method_w}}"
        for pn in pair_names:
            metrics = results[method].get(pn)
            val = metrics.get("accuracy") if metrics is not None else None
            cell = f"{val * 100:.2f}%" if val is not None else "N/A"
            row += f"{cell:>{col_w}}"
        print(row)
    print(sep)

    # Pretty-print macro-F1 table for direct robustness comparison
    print(f"\n{sep}")
    print("Macro F1")
    print(f"{'Method':<{method_w}}" + "".join(f"{n:>{col_w}}" for n in pair_names))
    print("-" * (method_w + col_w * len(pair_names)))
    for method in methods:
        row = f"{method:<{method_w}}"
        for pn in pair_names:
            metrics = results[method].get(pn)
            val = metrics.get("f1_macro") if metrics is not None else None
            cell = f"{val * 100:.2f}%" if val is not None else "N/A"
            row += f"{cell:>{col_w}}"
        print(row)
    print(sep)

    # Pretty-print image quality alignment summary (lower is better)
    print(f"\n{sep}")
    print("Image Quality (mean/std/grad L1 to target; lower is better)")
    print(f"{'Method':<{method_w}}" + "".join(f"{n:>{col_w}}" for n in pair_names))
    print("-" * (method_w + col_w * len(pair_names)))
    for method in methods:
        row = f"{method:<{method_w}}"
        for pn in pair_names:
            metrics = results[method].get(pn)
            if metrics is None:
                cell = "N/A"
            else:
                composite = (
                    metrics["img_mean_l1_to_target"]
                    + metrics["img_std_l1_to_target"]
                    + metrics["img_grad_l1_to_target"]
                ) / 3.0
                cell = f"{composite:.4f}"
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

    # Determine which methods & pairs to run.
    # Task II requires benchmarking all five UDA approaches by default.
    if method_filter is not None:
        methods = method_filter
    else:
        configured_methods = cfg.get("uda_methods", ALL_METHODS)
        methods = [m for m in ALL_METHODS if m in set(configured_methods)]
        if len(methods) != len(ALL_METHODS):
            missing = [m for m in ALL_METHODS if m not in set(configured_methods)]
            print(
                "[WARN] config.uda_methods is missing required Task II methods: "
                f"{missing}. They will be included automatically."
            )
            methods = ALL_METHODS.copy()
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
    qm_cfg = cfg.get("quality_metrics", {})
    quality_samples = int(qm_cfg.get("max_samples", 256))
    seed = int(cfg.get("seed", 42))

    output_root.mkdir(parents=True, exist_ok=True)

    # results[method][pair_name] = metric dict or None on failure
    results: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {m: {} for m in methods}

    for pair in pair_cfgs:
        pair_name = pair["name"]
        src = pair["source"]
        tgt = pair["target"]
        domain_set = {src.strip().lower(), tgt.strip().lower()}
        spectral_enforce_grayscale = domain_set.issubset({"mnist", "usps"})
        spatial_epoch = _resolve_qualitative_checkpoint_epoch(cfg, pair, "spatial")
        spectral_epoch = _resolve_qualitative_checkpoint_epoch(cfg, pair, "spectral")

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
        tgt_eval_ds = load_labeled_target_eval(dataset_root, tgt, image_size)
        src_eval_ds = load_labeled_source(dataset_root, src, image_size, augment=False)
        src_eval_loader = make_loader(
            src_eval_ds,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        tgt_eval_loader = make_loader(
            tgt_eval_ds,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        src_quality_imgs = _collect_images(src_eval_loader, quality_samples, device)
        tgt_quality_imgs = _collect_images(tgt_eval_loader, quality_samples, device)

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
                classifier = None
                quality_ckpt: Optional[Path] = None

                # ── source_only ─────────────────────────────────────────── #
                if method == "source_only":
                    classifier, _ = train_source_only(
                        **common, output_dir=method_out
                    )

                # ── cyclegan ────────────────────────────────────────────── #
                elif method == "cyclegan":
                    ckpt = _find_cyclegan_checkpoint(
                        cyclegan_root,
                        pair_name,
                        "spatial",
                        checkpoint_epoch=spatial_epoch,
                    )
                    if ckpt is None:
                        if spatial_epoch is None:
                            print(f"    [SKIP] No spatial CycleGAN checkpoint for {pair_name}")
                        else:
                            print(
                                "    [SKIP] Requested spatial CycleGAN checkpoint "
                                f"epoch_{spatial_epoch:03d}.pt not found for {pair_name}"
                            )
                        results[method][pair_name] = None
                        continue
                    print(f"    Checkpoint: {ckpt}")
                    quality_ckpt = ckpt
                    classifier, _ = train_cyclegan_uda(
                        **common,
                        cyclegan_checkpoint=ckpt,
                        method="spatial",
                        output_dir=method_out,
                    )

                # ── spectral_cyclegan ────────────────────────────────────── #
                elif method == "spectral_cyclegan":
                    ckpt = _find_cyclegan_checkpoint(
                        cyclegan_root,
                        pair_name,
                        "spectral",
                        checkpoint_epoch=spectral_epoch,
                    )
                    if ckpt is None:
                        if spectral_epoch is None:
                            print(f"    [SKIP] No spectral CycleGAN checkpoint for {pair_name}")
                        else:
                            print(
                                "    [SKIP] Requested spectral CycleGAN checkpoint "
                                f"epoch_{spectral_epoch:03d}.pt not found for {pair_name}"
                            )
                        results[method][pair_name] = None
                        continue
                    print(f"    Checkpoint: {ckpt}")
                    quality_ckpt = ckpt
                    classifier, _ = train_cyclegan_uda(
                        **common,
                        cyclegan_checkpoint=ckpt,
                        method="spectral",
                        output_dir=method_out,
                    )

                # ── cycada ──────────────────────────────────────────────── #
                elif method == "cycada":
                    ckpt = _find_cyclegan_checkpoint(
                        cyclegan_root,
                        pair_name,
                        "spatial",
                        checkpoint_epoch=spatial_epoch,
                    )
                    if ckpt is None:
                        if spatial_epoch is None:
                            print(f"    [SKIP] No spatial CycleGAN checkpoint for CyCADA {pair_name}")
                        else:
                            print(
                                "    [SKIP] Requested spatial CycleGAN checkpoint "
                                f"epoch_{spatial_epoch:03d}.pt not found for CyCADA {pair_name}"
                            )
                        results[method][pair_name] = None
                        continue
                    print(f"    Checkpoint: {ckpt}")
                    quality_ckpt = ckpt
                    classifier, _ = train_cycada(
                        **common,
                        cyclegan_checkpoint=ckpt,
                        lambda_domain=cycada_lambda_domain,
                        output_dir=method_out,
                    )

                # ── fda ─────────────────────────────────────────────────── #
                elif method == "fda":
                    classifier, _ = train_fda_uda(
                        **common,
                        fda_beta=fda_beta,
                        output_dir=method_out,
                    )

                else:
                    print(f"    [SKIP] Unknown method: {method!r}")
                    results[method][pair_name] = None
                    continue

                metrics = _evaluate_metrics(
                    classifier=classifier,
                    loader=tgt_eval_loader,
                    device=device,
                    num_classes=num_classes,
                )
                torch.manual_seed(seed)
                random.seed(seed)
                quality_metrics = _evaluate_image_quality_metrics(
                    method=method,
                    src_imgs=src_quality_imgs,
                    tgt_imgs=tgt_quality_imgs,
                    device=device,
                    fda_beta=fda_beta,
                    cyclegan_checkpoint=quality_ckpt,
                    spectral_enforce_grayscale=spectral_enforce_grayscale,
                )
                metrics.update(quality_metrics)
                results[method][pair_name] = metrics
                print(
                    "    Metrics: "
                    f"acc={metrics['accuracy'] * 100:.2f}%  "
                    f"macro_f1={metrics['f1_macro'] * 100:.2f}%  "
                    f"macro_prec={metrics['precision_macro'] * 100:.2f}%  "
                    f"macro_rec={metrics['recall_macro'] * 100:.2f}%  "
                    f"img_mean_l1={metrics['img_mean_l1_to_target']:.4f}"
                )

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
