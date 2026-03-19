"""Data utilities for Task II UDA benchmark.

Provides:
  - RGBImageFolder         : ImageFolder that forces RGB mode for all images.
  - FDATransferDataset     : Wraps a source dataset with on-the-fly FDA spectral
                             amplitude transfer from a pool of target images.
  - fda_transfer           : Pure-tensor FDA transform (batch-safe, CPU/GPU).
  - load_labeled_source    : Load labeled source-domain training split.
  - load_labeled_target_eval: Load labeled target-domain evaluation split.
  - load_unlabeled_target  : Load unlabeled target-domain images for adversarial
                             alignment (labels present but ignored).
  - collect_image_paths    : Recursively collect all image paths under a directory.
  - make_loader            : Convenience DataLoader factory.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# Domain path tables
# ---------------------------------------------------------------------------

# Training split (labeled source) paths, relative to dataset_root
_DOMAIN_TRAIN_PATHS: Dict[str, str] = {
    "mnist": "MNIST/images/train",
    "usps": "USPS/images/train",
    "svhn": "SVHN/images/train",
    "amazon": "office31/amazon",
    "webcam": "office31/webcam",
    "dslr": "office31/dslr",
    "art": "OfficeHomeDataset/Art",
    "clipart": "OfficeHomeDataset/Clipart",
    "product": "OfficeHomeDataset/Product",
    "real-world": "OfficeHomeDataset/Real World",
    "photo": "pacs/photo",
    "sketch": "pacs/sketch",
    "cartoon": "pacs/cartoon",
    "art-painting": "pacs/art_painting",
}

# Evaluation split paths; test splits for digit datasets, full domain otherwise
_DOMAIN_EVAL_PATHS: Dict[str, str] = {
    "mnist": "MNIST/images/test",
    "usps": "USPS/images/test",
    "svhn": "SVHN/images/test",
    "amazon": "office31/amazon",
    "webcam": "office31/webcam",
    "dslr": "office31/dslr",
    "art": "OfficeHomeDataset/Art",
    "clipart": "OfficeHomeDataset/Clipart",
    "product": "OfficeHomeDataset/Product",
    "real-world": "OfficeHomeDataset/Real World",
    "photo": "pacs/photo",
    "sketch": "pacs/sketch",
    "cartoon": "pacs/cartoon",
    "art-painting": "pacs/art_painting",
}


def _normalize_domain(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _train_path(dataset_root: Path, domain: str) -> Path:
    key = _normalize_domain(domain)
    if key not in _DOMAIN_TRAIN_PATHS:
        raise KeyError(f"Unknown domain name: {domain!r}")
    p = dataset_root / _DOMAIN_TRAIN_PATHS[key]
    if not p.exists():
        raise FileNotFoundError(f"Training path not found for domain {domain!r}: {p}")
    return p


def _eval_path(dataset_root: Path, domain: str) -> Path:
    key = _normalize_domain(domain)
    if key not in _DOMAIN_EVAL_PATHS:
        raise KeyError(f"Unknown domain name: {domain!r}")
    p = dataset_root / _DOMAIN_EVAL_PATHS[key]
    if not p.exists():
        raise FileNotFoundError(f"Evaluation path not found for domain {domain!r}: {p}")
    return p


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


# Public alias for use in scripts outside this module
get_eval_transform = _eval_transform


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class RGBImageFolder(ImageFolder):
    """ImageFolder that coerces every image to RGB mode.

    Needed because digit datasets (MNIST, USPS) are stored as grayscale PNGs
    while the classifier backbone expects 3-channel input.
    """

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label  # type: ignore[return-value]


class FDATransferDataset(Dataset):
    """Source dataset with on-the-fly FDA-style amplitude transfer.

    For each source image the transform:
      1. Draws a random target image from *target_image_paths*.
      2. Replaces the low-frequency amplitude of the source spectrum with
         that of the target (FDA, Yang et al. 2020).
      3. Returns the FDA-translated image under the original source label.

    Args:
        source_dataset:    Labeled source dataset (e.g. RGBImageFolder).
        target_image_paths: Flat list of target-domain image file paths.
        image_size:        Spatial resolution used for the target images.
        beta:              Fraction of spatial frequencies treated as "low".
    """

    def __init__(
        self,
        source_dataset: RGBImageFolder,
        target_image_paths: List[str],
        image_size: int,
        beta: float = 0.1,
    ) -> None:
        self.source_dataset = source_dataset
        self.target_paths = target_image_paths
        self.beta = beta
        self._tgt_tfm = _eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        src_img, label = self.source_dataset[index]
        tgt_path = self.target_paths[random.randrange(len(self.target_paths))]
        tgt_img: torch.Tensor = self._tgt_tfm(Image.open(tgt_path).convert("RGB"))
        fda_img = fda_transfer(
            src_img.unsqueeze(0),
            tgt_img.unsqueeze(0),
            self.beta,
        ).squeeze(0)
        return fda_img, label


# ---------------------------------------------------------------------------
# FDA transform
# ---------------------------------------------------------------------------

def fda_transfer(
    source: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Swap low-frequency amplitudes from *target* into *source*.

    Implements the FDA frequency-domain amplitude swap described in:
      Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation",
      CVPR 2020.

    Args:
        source: Float tensor ``[B, C, H, W]`` in ``[-1, 1]``.
        target: Float tensor ``[B, C, H, W]`` in ``[-1, 1]``.
        beta:   Low-frequency ratio ``0 < beta < 0.5``.

    Returns:
        FDA-adapted source with the same shape and value range as *source*.
    """
    h, w = source.shape[-2], source.shape[-1]

    # Transform to frequency domain and centre DC component
    fft_src = torch.fft.fftshift(torch.fft.fft2(source, dim=(-2, -1)), dim=(-2, -1))
    fft_tgt = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1)), dim=(-2, -1))

    amp_src = torch.abs(fft_src)
    phase_src = fft_src / (amp_src + 1e-8)
    amp_tgt = torch.abs(fft_tgt)

    # Swap the rectangular low-frequency region
    cy, cx = h // 2, w // 2
    ry = max(1, int(h * beta / 2))
    rx = max(1, int(w * beta / 2))
    amp_mixed = amp_src.clone()
    amp_mixed[..., cy - ry : cy + ry, cx - rx : cx + rx] = (
        amp_tgt[..., cy - ry : cy + ry, cx - rx : cx + rx]
    )

    # Reconstruct and shift back
    fft_mixed = torch.fft.ifftshift(amp_mixed * phase_src, dim=(-2, -1))
    out = torch.fft.ifft2(fft_mixed, dim=(-2, -1)).real
    return out.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Loader factory functions
# ---------------------------------------------------------------------------

def load_labeled_source(
    dataset_root: Path,
    domain: str,
    image_size: int,
    augment: bool = True,
) -> RGBImageFolder:
    """Return a labeled, RGB-coerced ImageFolder for the source training split."""
    path = _train_path(dataset_root, domain)
    tfm = _train_transform(image_size) if augment else _eval_transform(image_size)
    return RGBImageFolder(str(path), transform=tfm)


def load_labeled_target_eval(
    dataset_root: Path,
    domain: str,
    image_size: int,
) -> RGBImageFolder:
    """Return a labeled ImageFolder for the target evaluation split."""
    path = _eval_path(dataset_root, domain)
    return RGBImageFolder(str(path), transform=_eval_transform(image_size))


def load_unlabeled_target(
    dataset_root: Path,
    domain: str,
    image_size: int,
) -> RGBImageFolder:
    """Return target-domain images for unsupervised alignment; labels are ignored."""
    path = _train_path(dataset_root, domain)
    return RGBImageFolder(str(path), transform=_train_transform(image_size))


def get_domain_train_path(dataset_root: Path, domain: str) -> Path:
    """Return the resolved training path for *domain*."""
    return _train_path(dataset_root, domain)


def collect_image_paths(root: Path) -> List[str]:
    """Recursively collect every image file path under *root*."""
    _allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _allowed
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
