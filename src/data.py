from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import DatasetPair, TrainingConfig

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

_DOMAIN_PATHS: Dict[str, str] = {
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


def _normalize_domain_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _resolve_domain_path(dataset_root: Path, domain_name: str) -> Path:
    key = _normalize_domain_name(domain_name)
    if key not in _DOMAIN_PATHS:
        raise KeyError(f"Unsupported domain name: {domain_name}")
    path = dataset_root / _DOMAIN_PATHS[key]
    if not path.exists():
        raise FileNotFoundError(f"Domain path not found for {domain_name}: {path}")
    return path


def _collect_images(root: Path) -> List[Path]:
    images = sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in _ALLOWED_EXTENSIONS
        ]
    )
    if not images:
        raise RuntimeError(f"No images were found at {root}")
    return images


def get_domain_images(dataset_root: Path, domain_name: str) -> List[Path]:
    return _collect_images(_resolve_domain_path(dataset_root, domain_name))


class UnpairedImageDataset(Dataset):
    def __init__(self, images_a: List[Path], images_b: List[Path], image_size: int):
        self.images_a = images_a
        self.images_b = images_b
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return max(len(self.images_a), len(self.images_b))

    def _load_rgb(self, path: Path):
        # Some digit datasets are grayscale; forcing RGB keeps one model shape for all pairs.
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int):
        path_a = self.images_a[index % len(self.images_a)]
        path_b = self.images_b[random.randrange(len(self.images_b))]

        img_a = self.transform(self._load_rgb(path_a))
        img_b = self.transform(self._load_rgb(path_b))
        return {"A": img_a, "B": img_b, "path_A": str(path_a), "path_B": str(path_b)}


class SingleDomainImageDataset(Dataset):
    def __init__(self, images: List[Path], image_size: int):
        self.images = images
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        img = Image.open(path).convert("RGB")
        return {"image": self.transform(img), "path": str(path)}


def make_train_dataloader(
    dataset_root: Path,
    pair: DatasetPair,
    train_cfg: TrainingConfig,
) -> Tuple[DataLoader, List[Path], List[Path]]:
    src_root = _resolve_domain_path(dataset_root, pair.source)
    tgt_root = _resolve_domain_path(dataset_root, pair.target)

    images_a = _collect_images(src_root)
    images_b = _collect_images(tgt_root)

    dataset = UnpairedImageDataset(
        images_a=images_a,
        images_b=images_b,
        image_size=train_cfg.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, images_a, images_b


def make_eval_dataloader(images: List[Path], image_size: int, batch_size: int) -> DataLoader:
    dataset = SingleDomainImageDataset(images=images, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
