from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _gaussian_blur_mask(mask: torch.Tensor, sigma_y: float, sigma_x: float) -> torch.Tensor:
    radius_y = max(1, int(3.0 * sigma_y))
    radius_x = max(1, int(3.0 * sigma_x))

    y = torch.arange(-radius_y, radius_y + 1, device=mask.device, dtype=mask.dtype)
    x = torch.arange(-radius_x, radius_x + 1, device=mask.device, dtype=mask.dtype)
    kernel_y = torch.exp(-0.5 * (y / sigma_y) ** 2)
    kernel_x = torch.exp(-0.5 * (x / sigma_x) ** 2)
    kernel_2d = kernel_y[:, None] * kernel_x[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum().clamp_min(1e-12)
    kernel = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])

    blurred = F.conv2d(mask, kernel, padding=(radius_y, radius_x))
    return blurred / blurred.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def to_log_amplitude_map(image: torch.Tensor, output_channels: int = 3) -> torch.Tensor:
    """Map image to normalized log-amplitude spectrum for GAN training."""
    gray = image.mean(dim=1, keepdim=True)
    spectrum = torch.fft.fft2(gray, dim=(-2, -1))
    log_amp = torch.log1p(torch.abs(spectrum))

    mean = log_amp.mean(dim=(-2, -1), keepdim=True)
    std = log_amp.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    normalized = torch.tanh((log_amp - mean) / std)

    if output_channels == 1:
        return normalized
    return normalized.repeat(1, output_channels, 1, 1)


def low_frequency_blend(
    stylized: torch.Tensor,
    source: torch.Tensor,
    low_freq_ratio: float,
    enforce_grayscale_channels: bool = False,
) -> torch.Tensor:
    """Blend low frequencies from stylized output with high frequencies from source.

    Uses rfft2/irfft2 instead of fft2+fftshift so that irfft2 always returns a
    purely real tensor by construction.  The previous fft2 approach produced an
    off-by-one asymmetric mask ([cy-ry:cy+ry] is not centred on cy) which broke
    Hermitian symmetry in the mixed spectrum; the imaginary residual discarded by
    .real then appeared as per-channel phase-shift colour artifacts.
    """
    if enforce_grayscale_channels:
        stylized_proc = stylized.mean(dim=1, keepdim=True)
        source_proc = source.mean(dim=1, keepdim=True)
    else:
        stylized_proc = stylized
        source_proc = source

    h, w = stylized_proc.shape[-2], stylized_proc.shape[-1]
    fw = w // 2 + 1  # rfft2 retains only non-negative x-frequencies

    fft_style = torch.fft.rfft2(stylized_proc, dim=(-2, -1))
    fft_source = torch.fft.rfft2(source_proc, dim=(-2, -1))

    ry = max(1, int(h * low_freq_ratio / 2.0))
    rx = max(1, int(w * low_freq_ratio / 2.0))

    mask = torch.zeros((1, 1, h, fw), dtype=torch.float32, device=stylized_proc.device)
    # Low-freq rows: 0..ry-1 (positive freqs) and h-ry..h-1 (negative/conjugate freqs)
    mask[..., :ry, :rx] = 1.0
    mask[..., h - ry :, :rx] = 1.0

    sigma_y = max(1.0, float(ry) / 2.0)
    sigma_x = max(1.0, float(rx) / 2.0)
    mask = _gaussian_blur_mask(mask, sigma_y=sigma_y, sigma_x=sigma_x)

    mixed_fft = fft_style * mask + fft_source * (1.0 - mask)
    # irfft2 always returns a real tensor — no .real needed, no imaginary residual
    mixed = torch.fft.irfft2(mixed_fft, s=(h, w), dim=(-2, -1))

    if enforce_grayscale_channels and stylized.shape[1] > 1:
        mixed = mixed.repeat(1, stylized.shape[1], 1, 1)

    return mixed.clamp(-1.0, 1.0)


def adapt_output_for_method(
    method: MethodType,
    generated: torch.Tensor,
    source: torch.Tensor,
    low_freq_ratio: float,
    enforce_grayscale_channels: bool = False,
) -> torch.Tensor:
    if method == "spatial":
        return generated
    return low_frequency_blend(
        stylized=generated,
        source=source,
        low_freq_ratio=low_freq_ratio,
        enforce_grayscale_channels=enforce_grayscale_channels,
    )
