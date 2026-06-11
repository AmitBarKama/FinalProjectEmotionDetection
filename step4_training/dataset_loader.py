"""
PyTorch Dataset and DataLoader for blendshape-based emotion recognition.

Provides:
  - EmotionBlendshapeDataset: returns (blendshape_tensor, soft_label_tensor,
    hard_label_index) per sample.
  - Normalization: computes mean/std on the training split, saves to
    config.NORMALIZATION_PARAMS_FILE, and applies z-score standardisation.
  - create_dataloaders(): stratified train/val/test split with
    WeightedRandomSampler for class-imbalanced training.

Usage:
    python -m step4_training.dataset_loader          # quick smoke-test
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    BLENDSHAPE_NAMES,
    COMBINED_DATASET_FILE,
    EMOTION_LABELS,
    EMOTION_TO_IDX,
    NORMALIZATION_PARAMS_FILE,
    NUM_EMOTIONS,
    SELECTED_FEATURES_FILE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def load_selected_features() -> list[str]:
    """Load the selected feature list, falling back to all 52 blendshapes."""
    if SELECTED_FEATURES_FILE.exists():
        with open(SELECTED_FEATURES_FILE, "r") as f:
            features = json.load(f)
        logger.info("Loaded %d selected features from %s", len(features), SELECTED_FEATURES_FILE)
        return features
    logger.warning(
        "Selected features file not found (%s). Using all %d blendshapes.",
        SELECTED_FEATURES_FILE, len(BLENDSHAPE_NAMES),
    )
    return list(BLENDSHAPE_NAMES)


def _soft_label_columns() -> list[str]:
    """Return the column names for the 7 emotion probability scores."""
    # The combined dataset stores teacher probabilities in columns like
    # "prob_Anger", "prob_Disgust", …, "prob_Surprise".
    return [f"prob_{e}" for e in EMOTION_LABELS]


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────
class EmotionBlendshapeDataset(Dataset):
    """PyTorch dataset wrapping blendshape features + emotion labels.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain blendshape columns, ``predicted_emotion``, and
        probability columns ``prob_{emotion}`` for soft labels.
    feature_cols : list[str]
        Blendshape column names to use as input.
    mean : np.ndarray | None
        Per-feature mean for standardisation.
    std : np.ndarray | None
        Per-feature std for standardisation.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_cols: list[str],
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.feature_cols = feature_cols
        self.prob_cols = _soft_label_columns()

        # ── Blendshape features ──────────────────────────────────────
        X = dataframe[feature_cols].values.astype(np.float32)

        # Standardise if params provided
        if mean is not None and std is not None:
            std_safe = np.where(std == 0, 1.0, std)  # avoid div-by-zero
            X = (X - mean) / std_safe
        self.X = torch.from_numpy(X)

        # ── Soft labels (teacher probability vectors) ────────────────
        soft = dataframe[self.prob_cols].values.astype(np.float32)
        self.soft_labels = torch.from_numpy(soft)

        # ── Hard labels (integer index of predicted_emotion) ─────────
        hard = dataframe["predicted_emotion"].map(EMOTION_TO_IDX).values.astype(np.int64)
        self.hard_labels = torch.from_numpy(hard)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.soft_labels[idx], self.hard_labels[idx]


# ─────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────
def compute_normalization_params(
    df_train: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and save mean / std from the training set."""
    X_train = df_train[feature_cols].values.astype(np.float32)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    params = {
        "feature_cols": feature_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    NORMALIZATION_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NORMALIZATION_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Saved normalization params → %s", NORMALIZATION_PARAMS_FILE)

    return mean, std


def load_normalization_params() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load previously computed normalization parameters."""
    with open(NORMALIZATION_PARAMS_FILE, "r") as f:
        params = json.load(f)
    mean = np.array(params["mean"], dtype=np.float32)
    std = np.array(params["std"], dtype=np.float32)
    feature_cols = params["feature_cols"]
    return mean, std, feature_cols


# ─────────────────────────────────────────────────────────────────────
# Weighted Random Sampler
# ─────────────────────────────────────────────────────────────────────
def _make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler using inverse class frequency."""
    class_counts = np.bincount(labels, minlength=NUM_EMOTIONS).astype(np.float64)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)  # avoid div-by-zero
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────
def create_dataloaders(
    dataset_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build stratified train / val / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, feature_cols
    """
    path = dataset_path or COMBINED_DATASET_FILE
    logger.info("Loading dataset from %s", path)
    df = pd.read_parquet(path)
    logger.info("Total samples: %d", len(df))

    # ── Feature columns ──────────────────────────────────────────────
    feature_cols = load_selected_features()
    # Validate all columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # ── Stratified splits ────────────────────────────────────────────
    labels = df["predicted_emotion"].values
    val_test_frac = VAL_SPLIT + TEST_SPLIT
    test_frac_of_rest = TEST_SPLIT / val_test_frac

    idx_train, idx_temp = train_test_split(
        np.arange(len(df)),
        test_size=val_test_frac,
        stratify=labels,
        random_state=seed,
    )
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=test_frac_of_rest,
        stratify=labels[idx_temp],
        random_state=seed,
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    logger.info(
        "Split sizes → train: %d | val: %d | test: %d",
        len(df_train), len(df_val), len(df_test),
    )

    # ── Class distribution ───────────────────────────────────────────
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        dist = split_df["predicted_emotion"].value_counts().to_dict()
        logger.info("  %s class distribution: %s", name, dist)

    # ── Normalization (compute on train only) ────────────────────────
    mean, std = compute_normalization_params(df_train, feature_cols)

    # ── Build datasets ───────────────────────────────────────────────
    train_ds = EmotionBlendshapeDataset(df_train, feature_cols, mean, std)
    val_ds = EmotionBlendshapeDataset(df_val, feature_cols, mean, std)
    test_ds = EmotionBlendshapeDataset(df_test, feature_cols, mean, std)

    # ── Weighted sampler for training ────────────────────────────────
    train_labels = df_train["predicted_emotion"].map(EMOTION_TO_IDX).values
    sampler = _make_weighted_sampler(train_labels)

    # ── DataLoaders ──────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "DataLoaders ready → train batches: %d | val: %d | test: %d",
        len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader, feature_cols


# ─────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    """Quick smoke test: build loaders and iterate one batch."""
    train_loader, val_loader, test_loader, feature_cols = create_dataloaders(
        num_workers=0,
    )
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols[:5]} …")

    # Fetch one batch
    X, soft, hard = next(iter(train_loader))
    print(f"Batch shapes → X: {X.shape}, soft: {soft.shape}, hard: {hard.shape}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Soft label sum (should be ~1.0): {soft[0].sum():.4f}")
    print(f"Hard label sample: {hard[:8].tolist()}")
    print("\n✓ Dataset loader smoke test passed.")


if __name__ == "__main__":
    main()
