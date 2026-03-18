from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

MethodType = Literal["spatial", "spectral"]


class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc: int = 3, output_nc: int = 3, n_blocks: int = 6):
        super().__init__()

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            layers += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels = min(in_channels * 2, 512)

        for _ in range(n_blocks):
            layers.append(ResnetBlock(in_channels))

        out_channels = in_channels // 2
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels = max(in_channels // 2, 64)

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, output_nc, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3):
        super().__init__()

        def block(in_c: int, out_c: int, normalize: bool = True) -> list[nn.Module]:
            ops: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                ops.append(nn.InstanceNorm2d(out_c))
            ops.append(nn.LeakyReLU(0.2, inplace=True))
            return ops

        self.model = nn.Sequential(
            *block(input_nc, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def init_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in classname:
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


def low_frequency_blend(
    stylized: torch.Tensor,
    source: torch.Tensor,
    low_freq_ratio: float,
) -> torch.Tensor:
    """Blend low frequencies from stylized output with high frequencies from source."""
    h, w = stylized.shape[-2], stylized.shape[-1]
    fft_style = torch.fft.fftshift(torch.fft.fft2(stylized, dim=(-2, -1)), dim=(-2, -1))
    fft_source = torch.fft.fftshift(torch.fft.fft2(source, dim=(-2, -1)), dim=(-2, -1))

    cy, cx = h // 2, w // 2
    ry = max(1, int(h * low_freq_ratio / 2.0))
    rx = max(1, int(w * low_freq_ratio / 2.0))

    mask = torch.zeros((1, 1, h, w), dtype=torch.float32, device=stylized.device)
    mask[..., cy - ry : cy + ry, cx - rx : cx + rx] = 1.0

    mixed_fft = fft_style * mask + fft_source * (1.0 - mask)
    mixed = torch.fft.ifft2(torch.fft.ifftshift(mixed_fft, dim=(-2, -1)), dim=(-2, -1)).real
    return mixed.clamp(-1.0, 1.0)


def adapt_output_for_method(
    method: MethodType,
    generated: torch.Tensor,
    source: torch.Tensor,
    low_freq_ratio: float,
) -> torch.Tensor:
    if method == "spatial":
        return generated
    return low_frequency_blend(
        stylized=generated,
        source=source,
        low_freq_ratio=low_freq_ratio,
    )
