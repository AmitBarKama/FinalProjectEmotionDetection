"""
Evaluate the trained EmotionMLP on the held-out test set.

Computes:
  1. Overall accuracy
  2. Macro / Weighted F1
  3. Per-class precision, recall, F1 (classification report)
  4. Normalised confusion matrix (heatmap)
  5. Teacher–student agreement rate
  6. Per-class KL divergence between student and teacher distributions

Saves metrics JSON + plots to config.EVALUATION_DIR.

Usage:
    python -m step4_training.evaluate [--device auto]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EMOTION_LABELS,
    EVALUATION_DIR,
    MIN_MACRO_F1,
    MIN_TEACHER_STUDENT_AGREEMENT,
    NUM_EMOTIONS,
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
# Load Model
# ─────────────────────────────────────────────────────────────────────
def load_model(device: torch.device) -> tuple[EmotionMLP, dict]:
    """Load the best checkpoint and return the model + checkpoint dict."""
    ckpt_path = CHECKPOINTS_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EmotionMLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        num_classes=ckpt["num_classes"],
        dropout_rates=ckpt["dropout_rates"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info(
        "Loaded model from epoch %d (best val F1=%.4f)",
        ckpt["epoch"], ckpt["best_val_macro_f1"],
    )
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────
# Gather Predictions
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def gather_predictions(model: EmotionMLP, loader, device: torch.device):
    """Run the model over a data loader and collect all results.

    Returns
    -------
    student_probs : np.ndarray  (N, C)  — student softmax probabilities
    teacher_probs : np.ndarray  (N, C)  — teacher probability vectors
    hard_labels   : np.ndarray  (N,)    — ground-truth integer labels
    """
    all_student_probs = []
    all_teacher_probs = []
    all_hard = []

    for X, soft, hard in tqdm(loader, desc="Evaluating", ncols=100):
        X = X.to(device)
        logits = model(X)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        all_student_probs.append(probs)
        all_teacher_probs.append(soft.numpy())
        all_hard.append(hard.numpy())

    return (
        np.concatenate(all_student_probs),
        np.concatenate(all_teacher_probs),
        np.concatenate(all_hard),
    )


# ─────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Path
) -> None:
    """Normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix → %s", save_path)


def plot_per_class_f1(report_dict: dict, save_path: Path) -> None:
    """Bar chart of per-class F1 scores."""
    labels = EMOTION_LABELS
    f1s = [report_dict[lbl]["f1-score"] for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, f1s, color="#4CAF50", alpha=0.85, edgecolor="black")
    ax.axhline(y=np.mean(f1s), color="red", linestyle="--", linewidth=1.5, label=f"Macro avg: {np.mean(f1s):.3f}")
    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Scores", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved per-class F1 chart → %s", save_path)


def plot_probability_calibration(
    student_probs: np.ndarray,
    teacher_probs: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    """Calibration-style plot comparing student and teacher confidence."""
    student_conf = student_probs.max(axis=1)
    teacher_conf = teacher_probs.max(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of max confidence
    axes[0].hist(teacher_conf, bins=n_bins, alpha=0.6, label="Teacher", color="#2196F3", edgecolor="black")
    axes[0].hist(student_conf, bins=n_bins, alpha=0.6, label="Student", color="#FF9800", edgecolor="black")
    axes[0].set_xlabel("Max Probability (Confidence)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Scatter: teacher vs student confidence
    axes[1].scatter(teacher_conf, student_conf, alpha=0.1, s=5, color="#4CAF50")
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect agreement")
    axes[1].set_xlabel("Teacher Confidence")
    axes[1].set_ylabel("Student Confidence")
    axes[1].set_title("Teacher vs Student Confidence")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved probability calibration → %s", save_path)


# ─────────────────────────────────────────────────────────────────────
# Per-class KL Divergence
# ─────────────────────────────────────────────────────────────────────
def compute_per_class_kl(
    student_probs: np.ndarray,
    teacher_probs: np.ndarray,
    hard_labels: np.ndarray,
) -> dict[str, float]:
    """Mean KL(teacher || student) per class."""
    kl_per_class: dict[str, float] = {}
    for idx, emo in enumerate(EMOTION_LABELS):
        mask = hard_labels == idx
        if mask.sum() == 0:
            kl_per_class[emo] = 0.0
            continue
        t = torch.from_numpy(teacher_probs[mask]).float().clamp(min=1e-8)
        s = torch.from_numpy(student_probs[mask]).float().clamp(min=1e-8)
        kl = F.kl_div(s.log(), t, reduction="batchmean").item()
        kl_per_class[emo] = round(kl, 6)
    return kl_per_class


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained EmotionMLP.")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto | cpu | cuda | mps",
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load model
    model, ckpt = load_model(device)

    # Load test data
    _, _, test_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE, num_workers=0,
    )

    # Gather predictions
    student_probs, teacher_probs, hard_labels = gather_predictions(
        model, test_loader, device,
    )
    student_preds = student_probs.argmax(axis=1)
    teacher_preds = teacher_probs.argmax(axis=1)

    # ── Metrics ──────────────────────────────────────────────────────
    accuracy = accuracy_score(hard_labels, student_preds)
    macro_f1 = f1_score(hard_labels, student_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(hard_labels, student_preds, average="weighted", zero_division=0)

    report = classification_report(
        hard_labels, student_preds,
        target_names=EMOTION_LABELS, output_dict=True, zero_division=0,
    )
    report_str = classification_report(
        hard_labels, student_preds,
        target_names=EMOTION_LABELS, zero_division=0,
    )

    # Teacher–student agreement
    agreement = (student_preds == teacher_preds).mean()

    # Per-class KL
    kl_per_class = compute_per_class_kl(student_probs, teacher_probs, hard_labels)

    # ── Print report ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"\nOverall Accuracy:          {accuracy:.4f}")
    print(f"Macro F1:                  {macro_f1:.4f}")
    print(f"Weighted F1:               {weighted_f1:.4f}")
    print(f"Teacher-Student Agreement: {agreement:.4f}")
    print(f"\n{report_str}")

    print("\nPer-Class KL Divergence (teacher ‖ student):")
    for emo, kl in kl_per_class.items():
        print(f"  {emo:<12s}: {kl:.6f}")

    # ── Quality gates ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("QUALITY GATES:")
    f1_pass = macro_f1 >= MIN_MACRO_F1
    agree_pass = agreement >= MIN_TEACHER_STUDENT_AGREEMENT
    print(f"  Macro F1 ≥ {MIN_MACRO_F1}:    {'✓ PASS' if f1_pass else '✗ FAIL'} ({macro_f1:.4f})")
    print(f"  Agreement ≥ {MIN_TEACHER_STUDENT_AGREEMENT}: {'✓ PASS' if agree_pass else '✗ FAIL'} ({agreement:.4f})")
    print("-" * 60)

    # ── Save metrics JSON ────────────────────────────────────────────
    metrics = {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "teacher_student_agreement": round(float(agreement), 6),
        "per_class_kl_divergence": kl_per_class,
        "per_class_metrics": {
            emo: {k: round(v, 6) for k, v in report[emo].items()}
            for emo in EMOTION_LABELS
        },
        "quality_gates": {
            "macro_f1_pass": bool(f1_pass),
            "agreement_pass": bool(agree_pass),
        },
        "checkpoint_epoch": ckpt["epoch"],
        "best_val_macro_f1": round(ckpt["best_val_macro_f1"], 6),
    }
    metrics_path = EVALUATION_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics → %s", metrics_path)

    # ── Plots ────────────────────────────────────────────────────────
    plot_confusion_matrix(
        hard_labels, student_preds, EVALUATION_DIR / "confusion_matrix.png",
    )
    plot_per_class_f1(report, EVALUATION_DIR / "per_class_f1.png")
    plot_probability_calibration(
        student_probs, teacher_probs,
        EVALUATION_DIR / "probability_calibration.png",
    )

    logger.info("Evaluation complete. All outputs saved to %s", EVALUATION_DIR)


if __name__ == "__main__":
    main()
