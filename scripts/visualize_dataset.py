#!/usr/bin/env python3
"""Verify and visualize all datasets are loaded correctly.

Usage:
    python scripts/visualize_dataset.py

Sample plots are saved to: out/viz/samples_*.png
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
from PIL import Image


def check_torchvision_datasets(dataset_root: Path) -> List[str]:
    """Check MNIST, SVHN, and USPS datasets."""
    lines = ["\n=== Torchvision Datasets ===\n"]

    torchvision_dsets = [
        (
            "MNIST",
            lambda: datasets.MNIST(
                root=str(dataset_root / "MNIST"), train=True, download=False
            ),
        ),
        (
            "SVHN",
            lambda: datasets.SVHN(
                root=str(dataset_root / "SVHN"), split="train", download=False
            ),
        ),
        (
            "USPS",
            lambda: datasets.USPS(
                root=str(dataset_root / "USPS"), train=True, download=False
            ),
        ),
    ]

    for name, loader_fn in torchvision_dsets:
        try:
            dset = loader_fn()
            lines.append(f"✓ {name:10} | Size: {len(dset):6}")
        except Exception as e:
            lines.append(f"✗ {name:10} | Error: {e}")

    return lines


def check_office31(dataset_root: Path) -> List[str]:
    """Check Office-31 dataset structure."""
    lines = ["\n=== Office-31 Dataset ===\n"]

    office31_root = dataset_root / "office31"
    domains = ["amazon", "dslr", "webcam"]

    for domain in domains:
        domain_path = office31_root / domain
        if domain_path.exists():
            image_count = len(list(domain_path.glob("**/*.jpg")))
            lines.append(f"✓ {domain:10} | Images: {image_count:6}")
        else:
            lines.append(f"✗ {domain:10} | Path not found: {domain_path}")

    return lines


def check_office_home(dataset_root: Path) -> List[str]:
    """Check Office-Home dataset structure."""
    lines = ["\n=== Office-Home Dataset ===\n"]

    office_home_root = dataset_root / "OfficeHomeDataset"
    domains = ["Art", "Clipart", "Product", "Real World"]

    for domain in domains:
        domain_path = office_home_root / domain
        if domain_path.exists():
            image_count = len(list(domain_path.glob("**/*.jpg")))
            lines.append(f"✓ {domain:15} | Images: {image_count:6}")
        else:
            lines.append(f"✗ {domain:15} | Path not found: {domain_path}")

    return lines


def check_pacs(dataset_root: Path) -> List[str]:
    """Check PACS dataset structure."""
    lines = ["\n=== PACS Dataset ===\n"]

    pacs_root = dataset_root / "pacs"
    domains = ["art_painting", "cartoon", "photo", "sketch"]

    for domain in domains:
        domain_path = pacs_root / domain
        if domain_path.exists():
            image_count = len(list(domain_path.glob("**/*.jpg")))
            lines.append(f"✓ {domain:15} | Images: {image_count:6}")
        else:
            lines.append(f"✗ {domain:15} | Path not found: {domain_path}")

    return lines


def plot_torchvision_samples(dataset_root: Path, output_dir: Path) -> None:
    """Plot 10 sample images from MNIST, SVHN, and USPS."""
    datasets_to_plot = [
        (
            "MNIST",
            lambda: datasets.MNIST(
                root=str(dataset_root / "MNIST"), train=True, download=False
            ),
        ),
        (
            "SVHN",
            lambda: datasets.SVHN(
                root=str(dataset_root / "SVHN"), split="train", download=False
            ),
        ),
        (
            "USPS",
            lambda: datasets.USPS(
                root=str(dataset_root / "USPS"), train=True, download=False
            ),
        ),
    ]

    for name, loader_fn in datasets_to_plot:
        try:
            dset = loader_fn()
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            axes = axes.flatten()

            # Prefer one sample per label first, then fill remaining slots.
            selected_indices: List[int] = []
            seen_labels = set()
            for idx in range(len(dset)):
                _, lbl = dset[idx]
                label_int = int(lbl)
                if label_int not in seen_labels:
                    seen_labels.add(label_int)
                    selected_indices.append(idx)
                if len(selected_indices) == 10:
                    break

            if len(selected_indices) < 10:
                for idx in range(len(dset)):
                    if idx in selected_indices:
                        continue
                    selected_indices.append(idx)
                    if len(selected_indices) == 10:
                        break

            for i, sample_idx in enumerate(selected_indices):
                img, label = dset[sample_idx]
                if isinstance(img, Image.Image):
                    axes[i].imshow(img, cmap="gray")
                else:
                    axes[i].imshow(np.asarray(img), cmap="gray")
                axes[i].set_title(f"Label: {label}", fontsize=8)
                axes[i].axis("off")

            for i in range(len(selected_indices), 10):
                axes[i].axis("off")

            fig.suptitle(f"{name} - 10 Sample Images", fontsize=14)
            plt.tight_layout()
            output_file = output_dir / f"samples_{name}.png"
            plt.savefig(output_file, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved {name} samples to {output_file.name}")
        except Exception as e:
            print(f"✗ Error plotting {name}: {e}")


def plot_image_folder_samples(
    dataset_root: Path, output_dir: Path, dataset_name: str, domains: List[str]
) -> None:
    """Plot 10 sample images from image folder datasets (Office-31, Office-Home, PACS)."""
    base_path = dataset_root / (
        "office31"
        if dataset_name == "Office-31"
        else "OfficeHomeDataset" if dataset_name == "Office-Home" else "pacs"
    )

    if not base_path.exists():
        print(f"✗ {dataset_name} path not found: {base_path}")
        return

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    img_count = 0

    # Build category -> image paths where category is domain/class_name.
    category_to_images: Dict[str, List[Path]] = {}
    for domain in domains:
        domain_path = base_path / domain
        if domain_path.exists():
            for class_dir in domain_path.iterdir():
                if not class_dir.is_dir():
                    continue
                images = sorted(
                    [
                        p
                        for p in class_dir.rglob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    ]
                )
                if images:
                    category_to_images[f"{domain}/{class_dir.name}"] = images

    selected_samples: List[Tuple[str, Path]] = []
    categories = sorted(category_to_images.keys())

    # First pass: one image from each category.
    for category in categories:
        selected_samples.append((category, category_to_images[category][0]))
        if len(selected_samples) == 10:
            break

    # Second pass: fill remaining slots by round-robin over categories.
    offset = 1
    while len(selected_samples) < 10 and categories:
        added = False
        for category in categories:
            images = category_to_images[category]
            if offset < len(images):
                selected_samples.append((category, images[offset]))
                added = True
                if len(selected_samples) == 10:
                    break
        if not added:
            break
        offset += 1

    for i, (category, image_path) in enumerate(selected_samples):
        try:
            img = Image.open(image_path).convert("RGB")
            axes[i].imshow(img)
            axes[i].set_title(category, fontsize=7)
            axes[i].axis("off")
            img_count += 1
        except Exception:
            axes[i].axis("off")

    for i in range(len(selected_samples), 10):
        axes[i].axis("off")

    fig.suptitle(f"{dataset_name} - 10 Sample Images", fontsize=14)
    plt.tight_layout()
    output_file = output_dir / f"samples_{dataset_name.replace('-', '_')}.png"
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close()
    if img_count > 0:
        print(f"✓ Saved {dataset_name} samples ({img_count} images) to {output_file.name}")
    else:
        print(f"✗ No images found in {dataset_name}")


def plot_all_samples(dataset_root: Path, output_dir: Path) -> None:
    """Generate sample image plots for all datasets."""
    print("\n=== Plotting Sample Images ===\n")

    # Plot torchvision datasets
    plot_torchvision_samples(dataset_root, output_dir)

    # Plot manual datasets
    plot_image_folder_samples(
        dataset_root, output_dir, "Office-31", ["amazon", "dslr", "webcam"]
    )
    plot_image_folder_samples(
        dataset_root, output_dir, "Office-Home", ["Art", "Clipart", "Product", "Real World"]
    )
    plot_image_folder_samples(
        dataset_root, output_dir, "PACS", ["art_painting", "cartoon", "photo", "sketch"]
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = repo_root / "dataset"
    output_dir = repo_root / "out" / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output verification checks to console
    print(f"Checking datasets at: {dataset_root}\n")
    print("=" * 50)
    print("\n".join(check_torchvision_datasets(dataset_root)))
    print("\n".join(check_office31(dataset_root)))
    print("\n".join(check_office_home(dataset_root)))
    print("\n".join(check_pacs(dataset_root)))
    print("\n" + "=" * 50)
    print("Dataset verification complete!")

    # Plot sample images
    plot_all_samples(dataset_root, output_dir)
    print(f"\nSample plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
