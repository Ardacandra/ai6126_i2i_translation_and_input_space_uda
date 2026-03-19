"""Neural network components for the Task II UDA benchmark.

Provides:
  - ResNetClassifier   : ResNet-18 backbone + linear task head.
                         The backbone is accessible for feature extraction,
                         which is needed by the CyCADA domain discriminator.
  - DomainDiscriminator: Binary MLP classifier used for feature-level domain
                         alignment (source = 0, target = 1).
  - grad_reverse       : Gradient Reversal Layer (GRL) from Ganin et al. 2015,
                         used inside the CyCADA training loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import models


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradReversalFn(Function):
    """Autograd function implementing the Gradient Reversal Layer."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # Negate and scale the gradient; return None for the non-tensor alpha arg
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply the Gradient Reversal Layer to *x* with scaling factor *alpha*.

    During the forward pass this is the identity.  During the backward pass
    the gradient is multiplied by ``-alpha``, encouraging the preceding
    network to produce features that are domain-invariant.

    Reference:
        Ganin et al., "Domain-Adversarial Training of Neural Networks",
        JMLR 2016.
    """
    return _GradReversalFn.apply(x, alpha)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ResNetClassifier(nn.Module):
    """ResNet-18 feature extractor with a linear classification head.

    Args:
        num_classes: Number of output classes.
        pretrained:  Whether to initialise the backbone with ImageNet weights.

    Attributes:
        feature_dim: Dimension of the penultimate feature vector (512 for
                     ResNet-18).
        backbone:    All layers of ResNet-18 except the final FC layer.
        task_head:   Linear mapping from *feature_dim* → *num_classes*.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        _backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.feature_dim: int = _backbone.fc.in_features  # 512 for ResNet-18
        # Drop the original classifier head; retain everything up to avg-pool
        self.backbone = nn.Sequential(*list(_backbone.children())[:-1])
        self.task_head = nn.Linear(self.feature_dim, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened 512-d feature vector (before classification)."""
        return self.backbone(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_head(self.get_features(x))


# ---------------------------------------------------------------------------
# Domain Discriminator
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """Binary MLP domain classifier for feature-level domain alignment.

    Predicts whether an input feature vector comes from the source (label 0)
    or target (label 1) domain.  When combined with :func:`grad_reverse` in
    the CyCADA training loop the feature extractor is pushed toward producing
    domain-invariant representations.

    Args:
        feature_dim: Dimension of the input feature vector.
        hidden_dim:  Width of the two hidden layers.
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
