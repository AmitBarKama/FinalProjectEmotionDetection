"""
MLP architecture for blendshape-based emotion recognition.

Architecture:
    Input(N) → [Linear → BatchNorm → ReLU → Dropout] × L → Linear(num_classes)

The model outputs raw logits; softmax should be applied externally for
inference, while training uses KL-divergence loss on log-softmax outputs.

Usage:
    python -m step4_training.model          # print model summary
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DROPOUT_RATES,
    HIDDEN_DIMS,
    NUM_BLENDSHAPES,
    NUM_EMOTIONS,
)

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class EmotionMLP(nn.Module):
    """Multi-layer perceptron for emotion classification from blendshapes.

    Parameters
    ----------
    input_dim : int
        Number of input blendshape features (default: 52).
    hidden_dims : list[int]
        Sizes of hidden layers (default: [128, 64, 32]).
    num_classes : int
        Number of emotion categories (default: 7).
    dropout_rates : list[float]
        Dropout probability after each hidden layer.  Must match
        ``len(hidden_dims)``.  Default: [0.3, 0.3, 0.2].
    """

    def __init__(
        self,
        input_dim: int = NUM_BLENDSHAPES,
        hidden_dims: Optional[list[int]] = None,
        num_classes: int = NUM_EMOTIONS,
        dropout_rates: Optional[list[float]] = None,
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims or list(HIDDEN_DIMS)
        dropout_rates = dropout_rates or list(DROPOUT_RATES)

        if len(hidden_dims) != len(dropout_rates):
            raise ValueError(
                f"hidden_dims ({len(hidden_dims)}) and dropout_rates "
                f"({len(dropout_rates)}) must have the same length."
            )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rates = dropout_rates

        # ── Build layers ─────────────────────────────────────────────
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim, drop_rate in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
            ])
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

        # Weight initialisation (Kaiming for ReLU layers)
        self._init_weights()

        # Print summary on creation
        self._print_summary()

    # ─────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """Apply Kaiming uniform init for linear layers, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, num_classes)."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities of shape (batch, num_classes)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _print_summary(self) -> None:
        """Log model architecture summary."""
        n_params = self.count_parameters()
        arch_str = f"Input({self.input_dim})"
        for h, d in zip(self.hidden_dims, self.dropout_rates):
            arch_str += f" → Linear({h})→BN→ReLU→Drop({d})"
        arch_str += f" → Linear({self.num_classes})"

        logger.info("=" * 65)
        logger.info("EmotionMLP Architecture")
        logger.info("-" * 65)
        logger.info("  %s", arch_str)
        logger.info("  Trainable parameters: %s", f"{n_params:,}")
        logger.info("=" * 65)
        logger.info("\n%s", self)


# ─────────────────────────────────────────────────────────────────────
# Main: print summary
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    """Instantiate the model and display summary information."""
    model = EmotionMLP()
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")

    # Quick forward pass sanity check
    dummy = torch.randn(4, model.input_dim)
    logits = model(dummy)
    probs = model.get_probabilities(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape:  {probs.shape}")
    print(f"Probs sum:    {probs.sum(dim=-1)}")
    print("\n✓ Model sanity check passed.")


if __name__ == "__main__":
    main()
