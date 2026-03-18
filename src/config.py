from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml

MethodType = Literal["spatial", "spectral"]
SpectralColorMode = Literal["auto", "grayscale", "rgb"]


@dataclass(frozen=True)
class DatasetPair:
    name: str
    source: str
    target: str


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int
    batch_size: int
    num_workers: int
    epochs: int
    steps_per_epoch: int
    lr: float
    betas: tuple[float, float]
    lambda_cycle: float
    lambda_identity: float
    spectral_low_freq_ratio: float
    spectral_color_mode: SpectralColorMode
    save_every_epochs: int
    sample_every_epochs: int
    sample_count: int
    use_amp: bool


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    device: str
    dataset_root: Path
    output_root: Path
    methods: List[MethodType]
    pairs: List[DatasetPair]
    training: TrainingConfig


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str | Path) -> ExperimentConfig:
    cfg_path = Path(config_path)
    raw = _read_yaml(cfg_path)

    pairs = [DatasetPair(**entry) for entry in raw["pairs"]]
    training = TrainingConfig(
        image_size=int(raw["training"]["image_size"]),
        batch_size=int(raw["training"]["batch_size"]),
        num_workers=int(raw["training"]["num_workers"]),
        epochs=int(raw["training"]["epochs"]),
        steps_per_epoch=int(raw["training"]["steps_per_epoch"]),
        lr=float(raw["training"]["lr"]),
        betas=(
            float(raw["training"]["betas"][0]),
            float(raw["training"]["betas"][1]),
        ),
        lambda_cycle=float(raw["training"]["lambda_cycle"]),
        lambda_identity=float(raw["training"]["lambda_identity"]),
        spectral_low_freq_ratio=float(raw["training"]["spectral_low_freq_ratio"]),
        spectral_color_mode=str(raw["training"].get("spectral_color_mode", "auto")).strip().lower(),
        save_every_epochs=int(raw["training"]["save_every_epochs"]),
        sample_every_epochs=int(raw["training"]["sample_every_epochs"]),
        sample_count=int(raw["training"]["sample_count"]),
        use_amp=bool(raw["training"]["use_amp"]),
    )

    if training.spectral_color_mode not in {"auto", "grayscale", "rgb"}:
        raise ValueError(
            "training.spectral_color_mode must be one of: auto, grayscale, rgb"
        )

    methods = [entry.strip().lower() for entry in raw["methods"]]
    for m in methods:
        if m not in {"spatial", "spectral"}:
            raise ValueError(f"Unsupported method in config: {m}")

    return ExperimentConfig(
        seed=int(raw["seed"]),
        device=str(raw.get("device", "auto")),
        dataset_root=(cfg_path.parent / raw["dataset"]["root"]).resolve(),
        output_root=(cfg_path.parent / raw["output"]["root"]).resolve(),
        methods=methods,
        pairs=pairs,
        training=training,
    )
