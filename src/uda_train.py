"""Training loops for the five Task II UDA methods.

Methods
-------
train_source_only
    Baseline: train on labeled source, evaluate on unlabeled target.

train_cyclegan_uda
    Train classifier on CycleGAN-translated source images (spatial *or*
    spectral variant).  The pre-trained G_AB from Task I is frozen; every
    source batch is translated on-the-fly before the classifier forward pass.

train_fda_uda
    Train classifier on FDA-translated source images.  Translation is applied
    inside the dataset's __getitem__ using random target images as amplitude
    donors.

train_cycada
    Simplified CyCADA: pixel-level adaptation (frozen CycleGAN G_AB) plus
    feature-level domain alignment via a Domain-Adversarial Neural Network
    with a Gradient Reversal Layer (GRL).

evaluate
    Compute top-1 accuracy of a classifier on a DataLoader.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.uda_data import (
    FDATransferDataset,
    collect_image_paths,
    get_domain_train_path,
    load_labeled_source,
    load_labeled_target_eval,
    load_unlabeled_target,
    make_loader,
)
from src.uda_models import DomainDiscriminator, ResNetClassifier, grad_reverse


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_optimizer(
    model: nn.Module,
    extra_modules: Optional[list] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Adam:
    params = list(model.parameters())
    if extra_modules:
        for m in extra_modules:
            params += list(m.parameters())
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _log_epoch(log_path: Path, epoch: int, **metrics: float) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        parts = [f"epoch={epoch:03d}"] + [f"{k}={v:.4f}" for k, v in metrics.items()]
        f.write(", ".join(parts) + "\n")


def _epoch_iter(loader: DataLoader, max_steps: int) -> Iterator:
    """Yield at most *max_steps* batches from *loader* per epoch."""
    it = iter(loader)
    for _ in range(max_steps):
        try:
            yield next(it)
        except StopIteration:
            return


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    classifier: ResNetClassifier,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return top-1 accuracy over *loader*."""
    classifier.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = classifier(imgs).argmax(1)
        correct += int((preds == labels).sum())
        total += labels.size(0)
    classifier.train()
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# 1. Source-only baseline
# ---------------------------------------------------------------------------

def train_source_only(
    dataset_root: Path,
    source_domain: str,
    target_domain: str,
    num_classes: int,
    *,
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 30,
    steps_per_epoch: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    pretrained: bool = True,
    device: torch.device,
    use_amp: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[ResNetClassifier, float]:
    """Train on labeled source data; return classifier and target accuracy."""
    src_ds = load_labeled_source(dataset_root, source_domain, image_size)
    tgt_ds = load_labeled_target_eval(dataset_root, target_domain, image_size)
    src_loader = make_loader(src_ds, batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)
    tgt_loader = make_loader(tgt_ds, batch_size, shuffle=False,
                             num_workers=num_workers)

    classifier = ResNetClassifier(num_classes, pretrained=pretrained).to(device)
    optimizer = _make_optimizer(classifier, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.txt"
        log_path.write_text("[source_only]\n")
    else:
        log_path = None

    max_steps = steps_per_epoch if steps_per_epoch > 0 else len(src_loader)

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = steps_done = 0
        for imgs, labels in _epoch_iter(src_loader, max_steps):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                loss = criterion(classifier(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss)
            steps_done += 1
        scheduler.step()
        if log_path is not None:
            _log_epoch(log_path, epoch, loss=total_loss / max(steps_done, 1))

    acc = evaluate(classifier, tgt_loader, device)
    if output_dir is not None:
        torch.save(classifier.state_dict(), output_dir / "classifier.pt")
    return classifier, acc


# ---------------------------------------------------------------------------
# 2. CycleGAN UDA  (spatial or spectral)
# ---------------------------------------------------------------------------

def train_cyclegan_uda(
    dataset_root: Path,
    source_domain: str,
    target_domain: str,
    num_classes: int,
    cyclegan_checkpoint: Path,
    *,
    method: str = "spatial",
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 30,
    steps_per_epoch: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    pretrained: bool = True,
    device: torch.device,
    use_amp: bool = True,
    low_freq_ratio: float = 0.2,
    output_dir: Optional[Path] = None,
) -> Tuple[ResNetClassifier, float]:
    """Train classifier on CycleGAN-translated source images.

    G_AB is loaded from *cyclegan_checkpoint*, frozen, and applied to every
    source batch before the classifier forward pass.  For the spectral method
    the low-frequency blend post-processing is also applied.
    """
    from src.models import ResnetGenerator, adapt_output_for_method

    # Load and freeze the pre-trained source→target generator
    g_ab = ResnetGenerator().to(device)
    ckpt = torch.load(cyclegan_checkpoint, map_location=device, weights_only=False)
    g_ab.load_state_dict(ckpt["G_AB"])
    g_ab.eval()
    for p in g_ab.parameters():
        p.requires_grad_(False)

    # Auto-detect grayscale mode for spectral method on digit datasets
    domain_set = {source_domain.strip().lower(), target_domain.strip().lower()}
    enforce_gray = method == "spectral" and domain_set.issubset({"mnist", "usps"})

    src_ds = load_labeled_source(dataset_root, source_domain, image_size)
    tgt_ds = load_labeled_target_eval(dataset_root, target_domain, image_size)
    src_loader = make_loader(src_ds, batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)
    tgt_loader = make_loader(tgt_ds, batch_size, shuffle=False,
                             num_workers=num_workers)

    classifier = ResNetClassifier(num_classes, pretrained=pretrained).to(device)
    optimizer = _make_optimizer(classifier, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.txt"
        log_path.write_text(f"[cyclegan_uda / {method}]\n")
    else:
        log_path = None

    max_steps = steps_per_epoch if steps_per_epoch > 0 else len(src_loader)

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = steps_done = 0
        for imgs, labels in _epoch_iter(src_loader, max_steps):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                fake_raw = g_ab(imgs)
                translated = adapt_output_for_method(
                    method=method,
                    generated=fake_raw,
                    source=imgs,
                    low_freq_ratio=low_freq_ratio,
                    enforce_grayscale_channels=enforce_gray,
                )

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                loss = criterion(classifier(translated), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss)
            steps_done += 1
        scheduler.step()
        if log_path is not None:
            _log_epoch(log_path, epoch, loss=total_loss / max(steps_done, 1))

    acc = evaluate(classifier, tgt_loader, device)
    if output_dir is not None:
        torch.save(classifier.state_dict(), output_dir / "classifier.pt")
    return classifier, acc


# ---------------------------------------------------------------------------
# 3. FDA UDA
# ---------------------------------------------------------------------------

def train_fda_uda(
    dataset_root: Path,
    source_domain: str,
    target_domain: str,
    num_classes: int,
    *,
    fda_beta: float = 0.1,
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 30,
    steps_per_epoch: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    pretrained: bool = True,
    device: torch.device,
    use_amp: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[ResNetClassifier, float]:
    """Train classifier on FDA-translated source images.

    For each source image a random target image is drawn from the target
    training split and its low-frequency amplitude is transplanted into the
    source spectrum (FDA, Yang & Soatto 2020).  No GAN training is required.
    """
    src_ds_raw = load_labeled_source(dataset_root, source_domain, image_size)
    tgt_path = get_domain_train_path(dataset_root, target_domain)
    tgt_paths = collect_image_paths(tgt_path)

    fda_ds = FDATransferDataset(
        source_dataset=src_ds_raw,
        target_image_paths=tgt_paths,
        image_size=image_size,
        beta=fda_beta,
    )
    # FDA uses random.randrange inside __getitem__; num_workers=0 avoids
    # potential cross-process random-state issues and PIL fork-safety concerns.
    fda_loader = make_loader(fda_ds, batch_size, shuffle=True,
                             num_workers=0, drop_last=True)

    tgt_ds = load_labeled_target_eval(dataset_root, target_domain, image_size)
    tgt_loader = make_loader(tgt_ds, batch_size, shuffle=False,
                             num_workers=num_workers)

    classifier = ResNetClassifier(num_classes, pretrained=pretrained).to(device)
    optimizer = _make_optimizer(classifier, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.txt"
        log_path.write_text(f"[fda_uda / beta={fda_beta}]\n")
    else:
        log_path = None

    max_steps = steps_per_epoch if steps_per_epoch > 0 else len(fda_loader)

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = steps_done = 0
        for imgs, labels in _epoch_iter(fda_loader, max_steps):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                loss = criterion(classifier(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss)
            steps_done += 1
        scheduler.step()
        if log_path is not None:
            _log_epoch(log_path, epoch, loss=total_loss / max(steps_done, 1))

    acc = evaluate(classifier, tgt_loader, device)
    if output_dir is not None:
        torch.save(classifier.state_dict(), output_dir / "classifier.pt")
    return classifier, acc


# ---------------------------------------------------------------------------
# 4. CyCADA
# ---------------------------------------------------------------------------

def train_cycada(
    dataset_root: Path,
    source_domain: str,
    target_domain: str,
    num_classes: int,
    cyclegan_checkpoint: Path,
    *,
    lambda_domain: float = 0.1,
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 30,
    steps_per_epoch: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    pretrained: bool = True,
    device: torch.device,
    use_amp: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[ResNetClassifier, float]:
    """CyCADA-style UDA: pixel-level (frozen CycleGAN) + feature-level (DANN).

    Training objective for the feature extractor / classifier:
      L = CE(f(G_AB(x_s)), y_s)
        + lambda_domain * L_domain(GRL(feats_src), GRL(feats_tgt))

    The Domain-Adversarial loss with Gradient Reversal encourages
    domain-invariant features at the penultimate layer.  ``alpha`` for the
    GRL is annealed using the schedule from Ganin et al. 2015.

    Reference:
        Hoffman et al., "CyCADA: Cycle-Consistent Adversarial Domain Adaptation",
        ICML 2018.
    """
    from src.models import ResnetGenerator, adapt_output_for_method

    # Frozen pixel-level adapter (spatial CycleGAN from Task I)
    g_ab = ResnetGenerator().to(device)
    ckpt = torch.load(cyclegan_checkpoint, map_location=device, weights_only=False)
    g_ab.load_state_dict(ckpt["G_AB"])
    g_ab.eval()
    for p in g_ab.parameters():
        p.requires_grad_(False)

    src_ds = load_labeled_source(dataset_root, source_domain, image_size)
    tgt_train_ds = load_unlabeled_target(dataset_root, target_domain, image_size)
    tgt_eval_ds = load_labeled_target_eval(dataset_root, target_domain, image_size)

    src_loader = make_loader(src_ds, batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)
    tgt_train_loader = make_loader(tgt_train_ds, batch_size, shuffle=True,
                                   num_workers=num_workers, drop_last=True)
    tgt_eval_loader = make_loader(tgt_eval_ds, batch_size, shuffle=False,
                                  num_workers=num_workers)

    classifier = ResNetClassifier(num_classes, pretrained=pretrained).to(device)
    domain_disc = DomainDiscriminator(feature_dim=classifier.feature_dim).to(device)

    # Joint optimizer for backbone + task head + domain discriminator
    optimizer = _make_optimizer(classifier, extra_modules=[domain_disc],
                                lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_task = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.txt"
        log_path.write_text(f"[cycada / lambda_domain={lambda_domain}]\n")
    else:
        log_path = None

    # Determine per-epoch step budget; must be finite for GRL alpha annealing.
    one_epoch_steps = min(len(src_loader), len(tgt_train_loader))
    max_steps = steps_per_epoch if steps_per_epoch > 0 else one_epoch_steps
    total_steps = max(epochs * max_steps, 1)
    global_step = 0

    for epoch in range(1, epochs + 1):
        classifier.train()
        domain_disc.train()
        tgt_iter = iter(tgt_train_loader)
        total_task = total_dom = steps_done = 0

        for src_imgs, src_labels in _epoch_iter(src_loader, max_steps):
            # Draw a matching target batch (cycle the target iterator as needed)
            try:
                tgt_imgs, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_train_loader)
                tgt_imgs, _ = next(tgt_iter)

            src_imgs = src_imgs.to(device, non_blocking=True)
            src_labels = src_labels.to(device, non_blocking=True)
            tgt_imgs = tgt_imgs.to(device, non_blocking=True)

            # Anneal GRL alpha: starts near 0, ends near 1 (Ganin et al. 2015)
            p = global_step / total_steps
            alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
            global_step += 1

            # Pixel-adapt source via frozen G_AB
            with torch.no_grad():
                adapted = adapt_output_for_method(
                    method="spatial",
                    generated=g_ab(src_imgs),
                    source=src_imgs,
                    low_freq_ratio=0.2,
                )

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                # Task loss on pixel-adapted source
                feats_src = classifier.get_features(adapted)
                loss_task = criterion_task(classifier.task_head(feats_src), src_labels)

                # Domain adversarial loss via GRL
                feats_tgt = classifier.get_features(tgt_imgs)
                dom_src = domain_disc(grad_reverse(feats_src, alpha))
                dom_tgt = domain_disc(grad_reverse(feats_tgt, alpha))
                loss_dom = (
                    criterion_domain(dom_src, torch.zeros_like(dom_src))
                    + criterion_domain(dom_tgt, torch.ones_like(dom_tgt))
                )
                loss = loss_task + lambda_domain * loss_dom

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_task += float(loss_task)
            total_dom += float(loss_dom)
            steps_done += 1

        scheduler.step()
        if log_path is not None:
            _log_epoch(
                log_path, epoch,
                loss_task=total_task / max(steps_done, 1),
                loss_domain=total_dom / max(steps_done, 1),
            )

    acc = evaluate(classifier, tgt_eval_loader, device)
    if output_dir is not None:
        torch.save(classifier.state_dict(), output_dir / "classifier.pt")
    return classifier, acc
