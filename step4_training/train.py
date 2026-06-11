"""
Training loop for the blendshape-based emotion recognition MLP.

Loss function — knowledge distillation from the teacher (HSEmotion):
    L = α · KL(log_softmax(logits/T), teacher/T) · T²
      + (1-α) · CrossEntropy(logits, hard_labels, label_smoothing=0.1)

Features:
  - Class-weighted cross-entropy via inverse frequency.
  - AdamW optimiser with cosine-annealing LR schedule.
  - Early stopping on validation macro-F1.
  - Best checkpoint + training history saved automatically.

Usage:
    python -m step4_training.train [--epochs 200] [--lr 1e-3] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EARLY_STOPPING_PATIENCE,
    EVALUATION_DIR,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LR_MIN,
    MAX_EPOCHS,
    NUM_EMOTIONS,
    WEIGHT_DECAY,
)
from step4_training.dataset_loader import create_dataloaders
from step4_training.model import EmotionMLP

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────
class DistillationLoss(nn.Module):
    """Combined knowledge-distillation loss.

    Parameters
    ----------
    alpha : float
        Weight for the soft-label (KL) term.  ``1 - alpha`` weights CE.
    temperature : float
        Softmax temperature for distillation.
    class_weights : torch.Tensor | None
        Per-class weights for the hard-label CE term.
    label_smoothing : float
        Label smoothing factor for CE.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 3.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = LABEL_SMOOTHING,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
        hard_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C) raw student logits
        soft_targets : (B, C) teacher probability vectors
        hard_targets : (B,) integer class indices
        """
        T = self.temperature

        # Soft loss: KL divergence on temperature-scaled distributions
        log_student = F.log_softmax(logits / T, dim=-1)
        teacher_soft = F.softmax(soft_targets / T, dim=-1)
        # KL expects log-probs as input and probs as target
        kl_loss = F.kl_div(log_student, teacher_soft, reduction="batchmean") * (T * T)

        # Hard loss: standard cross-entropy with label smoothing
        ce_loss = self.ce(logits, hard_targets)

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss


# ─────────────────────────────────────────────────────────────────────
# Compute class weights (inverse frequency)
# ─────────────────────────────────────────────────────────────────────
def compute_class_weights(train_loader, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training set."""
    counts = torch.zeros(NUM_EMOTIONS)
    for _, _, hard in train_loader:
        for label in hard:
            counts[label.item()] += 1
    # Inverse frequency
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * NUM_EMOTIONS  # normalise
    logger.info("Class weights: %s", weights.tolist())
    return weights.to(device)


# ─────────────────────────────────────────────────────────────────────
# Training & Validation Epoch
# ─────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: EmotionMLP,
    loader,
    criterion: DistillationLoss,
    optimizer: AdamW,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.  Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=100)
    for X, soft, hard in pbar:
        X, soft, hard = X.to(device), soft.to(device), hard.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, soft, hard)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == hard).sum().item()
        total += X.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(
    model: EmotionMLP,
    loader,
    criterion: DistillationLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run validation.  Returns (avg_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for X, soft, hard in loader:
        X, soft, hard = X.to(device), soft.to(device), hard.to(device)
        logits = model(X)
        loss = criterion(logits, soft, hard)

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(hard.cpu().tolist())

    n = max(len(all_preds), 1)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, macro_f1


# ─────────────────────────────────────────────────────────────────────
# Training Curves Plot
# ─────────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, save_dir: Path) -> None:
    """Generate loss + F1 training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # F1 + Accuracy
    ax2.plot(epochs, history["val_macro_f1"], label="Val Macro F1", linewidth=2, color="green")
    ax2.plot(epochs, history["train_acc"], label="Train Acc", linewidth=2, linestyle="--", alpha=0.7)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", linewidth=2, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Accuracy & Macro F1")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = save_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves → %s", path)


# ─────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────
def train(
    epochs: int = MAX_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    alpha: float = 0.7,
    temperature: float = 3.0,
    device_str: str = "auto",
) -> dict:
    """Full training procedure.

    Returns
    -------
    history : dict
        Training / validation metrics per epoch.
    """
    # ── Device ───────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # ── Data ─────────────────────────────────────────────────────────
    train_loader, val_loader, _, feature_cols = create_dataloaders(
        batch_size=batch_size, num_workers=4,
    )
    input_dim = len(feature_cols)
    logger.info("Input dimension: %d", input_dim)

    # ── Model ────────────────────────────────────────────────────────
    model = EmotionMLP(input_dim=input_dim).to(device)

    # ── Class weights + Loss ─────────────────────────────────────────
    class_weights = compute_class_weights(train_loader, device)
    criterion = DistillationLoss(
        alpha=alpha,
        temperature=temperature,
        class_weights=class_weights,
    ).to(device)

    # ── Optimiser & Scheduler ────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR_MIN)

    # ── History ──────────────────────────────────────────────────────
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_macro_f1": [], "lr": [],
    }

    best_f1 = 0.0
    patience_counter = 0

    logger.info("Starting training for up to %d epochs …", epochs)
    logger.info(
        "  alpha=%.2f | temperature=%.1f | lr=%.1e | batch=%d",
        alpha, temperature, lr, batch_size,
    )

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["lr"].append(current_lr)

        # Log
        logger.info(
            "Epoch %3d/%d | LR %.2e | Train L=%.4f A=%.4f | Val L=%.4f A=%.4f F1=%.4f",
            epoch, epochs, current_lr,
            train_loss, train_acc,
            val_loss, val_acc, val_f1,
        )

        # ── Checkpointing (best val macro-F1) ────────────────────────
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_macro_f1": best_f1,
                "input_dim": input_dim,
                "hidden_dims": model.hidden_dims,
                "num_classes": model.num_classes,
                "dropout_rates": model.dropout_rates,
                "feature_cols": feature_cols,
                "alpha": alpha,
                "temperature": temperature,
            }
            ckpt_path = CHECKPOINTS_DIR / "best_model.pt"
            torch.save(checkpoint, ckpt_path)
            logger.info("  ★ New best F1=%.4f → saved %s", best_f1, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(
                    "Early stopping triggered (patience=%d). Best F1=%.4f",
                    EARLY_STOPPING_PATIENCE, best_f1,
                )
                break

    elapsed = time.time() - start_time
    logger.info(
        "Training complete in %.1f min | Best val macro-F1: %.4f",
        elapsed / 60, best_f1,
    )

    # ── Save history ─────────────────────────────────────────────────
    history_path = CHECKPOINTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved history → %s", history_path)

    # ── Training curves ──────────────────────────────────────────────
    plot_training_curves(history, EVALUATION_DIR)

    return history


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the EmotionMLP model.")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Max epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--alpha", type=float, default=0.7, help="Soft-label weight (0-1)")
    parser.add_argument("--temperature", type=float, default=3.0, help="KD temperature")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto | cpu | cuda | mps",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        temperature=args.temperature,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
