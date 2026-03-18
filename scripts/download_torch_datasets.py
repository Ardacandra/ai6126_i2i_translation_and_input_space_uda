#!/usr/bin/env python3
"""Download MNIST, SVHN, and USPS using torchvision into dataset/.

Usage:
    python scripts/download_torch_datasets.py
    python scripts/download_torch_datasets.py --max-images-per-split 2000
    python scripts/download_torch_datasets.py --skip-image-export
"""

import argparse
from pathlib import Path
from typing import Optional

from torchvision import datasets


def _to_label_int(label: object) -> int:
    """Normalize dataset label types to int for folder naming."""
    return int(label)


def export_split_images(
    dataset_obj: object,
    image_root: Path,
    split_name: str,
    max_images: Optional[int] = None,
) -> int:
    """Export a torchvision split as images under split/label/ folders."""
    split_root = image_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for idx in range(len(dataset_obj)):
        if max_images is not None and saved_count >= max_images:
            break

        image, label = dataset_obj[idx]
        label_dir = split_root / str(_to_label_int(label))
        label_dir.mkdir(parents=True, exist_ok=True)

        image.save(label_dir / f"{idx:06d}.png")
        saved_count += 1

    return saved_count


def download_all(root: Path, export_images: bool, max_images_per_split: Optional[int]) -> None:
    """Download required torchvision datasets into dedicated dataset folders."""
    root.mkdir(parents=True, exist_ok=True)

    mnist_root = root / "MNIST"
    svhn_root = root / "SVHN"
    usps_root = root / "USPS"

    jobs = [
        (
            "MNIST",
            lambda: (
                datasets.MNIST(root=str(mnist_root), train=True, download=True),
                datasets.MNIST(root=str(mnist_root), train=False, download=True),
            ),
            mnist_root,
        ),
        (
            "SVHN",
            lambda: (
                datasets.SVHN(root=str(svhn_root), split="train", download=True),
                datasets.SVHN(root=str(svhn_root), split="test", download=True),
            ),
            svhn_root,
        ),
        (
            "USPS",
            lambda: (
                datasets.USPS(root=str(usps_root), train=True, download=True),
                datasets.USPS(root=str(usps_root), train=False, download=True),
            ),
            usps_root,
        ),
    ]

    for name, fn, dataset_dir in jobs:
        print(f"[START] Download {name}")
        train_split, test_split = fn()
        print(f"[DONE]  Download {name}")

        if export_images:
            image_root = dataset_dir / "images"
            print(f"[START] Export {name} split images -> {image_root}")
            train_saved = export_split_images(
                train_split,
                image_root,
                split_name="train",
                max_images=max_images_per_split,
            )
            test_saved = export_split_images(
                test_split,
                image_root,
                split_name="test",
                max_images=max_images_per_split,
            )
            print(
                f"[DONE]  Export {name}: train={train_saved} images, test={test_saved} images"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MNIST/SVHN/USPS and optionally export split images"
    )
    parser.add_argument(
        "--skip-image-export",
        action="store_true",
        help="Download datasets only, without exporting PNG images.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional cap for exported images in each split (train/test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = repo_root / "dataset"

    print(f"Downloading torchvision datasets into: {dataset_root}")
    download_all(
        dataset_root,
        export_images=not args.skip_image_export,
        max_images_per_split=args.max_images_per_split,
    )
    print("All downloads completed.")


if __name__ == "__main__":
    main()
