"""
Feature Selection Analysis for Blendshape-based Emotion Recognition.

Evaluates blendshape feature importance using three complementary methods:
  1. Mutual Information (information-theoretic)
  2. Random Forest feature importances (ensemble-based)
  3. L1-regularised Logistic Regression (sparsity-based)

Features appearing in the top-K of at least 2/3 methods are selected for
downstream training.  Results are saved as JSON reports and a comparative
bar chart.

Usage:
    python -m step4_training.feature_selection [--top-k 35]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BLENDSHAPE_NAMES,
    COMBINED_DATASET_FILE,
    EMOTION_LABELS,
    EVALUATION_DIR,
    FEATURE_SELECTION_REPORT,
    FEATURE_SELECTION_TOP_K,
    PROCESSED_DIR,
    SELECTED_FEATURES_FILE,
)

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress convergence warnings for L1 logistic regression
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────
# Helper: load dataset
# ─────────────────────────────────────────────────────────────────────
def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the combined dataset and extract blendshape features + labels.

    Returns
    -------
    df : pd.DataFrame
        Full dataset.
    X : np.ndarray, shape (n_samples, 52)
        Blendshape feature matrix.
    y : np.ndarray, shape (n_samples,)
        Integer-encoded emotion labels.
    """
    logger.info("Loading dataset from %s", path)
    df = pd.read_parquet(path)
    logger.info("Dataset shape: %s", df.shape)

    # Identify available blendshape columns
    bs_cols = [c for c in BLENDSHAPE_NAMES if c in df.columns]
    if not bs_cols:
        raise ValueError("No blendshape columns found in dataset.")
    logger.info("Found %d blendshape columns", len(bs_cols))

    X = df[bs_cols].values.astype(np.float32)

    # Encode target
    le = LabelEncoder()
    le.fit(EMOTION_LABELS)
    y = le.transform(df["predicted_emotion"].values)

    return df, X, y, bs_cols


# ─────────────────────────────────────────────────────────────────────
# Method 1: Mutual Information
# ─────────────────────────────────────────────────────────────────────
def compute_mutual_information(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """Compute mutual information between each feature and the target."""
    logger.info("Computing Mutual Information scores …")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    # Normalise to [0, 1]
    mi_max = mi_scores.max()
    if mi_max > 0:
        mi_scores = mi_scores / mi_max
    return {name: float(score) for name, score in zip(feature_names, mi_scores)}


# ─────────────────────────────────────────────────────────────────────
# Method 2: Random Forest
# ─────────────────────────────────────────────────────────────────────
def compute_random_forest_importance(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """Fit a Random Forest and extract normalised feature importances."""
    logger.info("Training Random Forest (n_estimators=200) …")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    imp_max = importances.max()
    if imp_max > 0:
        importances = importances / imp_max
    return {name: float(score) for name, score in zip(feature_names, importances)}


# ─────────────────────────────────────────────────────────────────────
# Method 3: L1 Regularisation (Logistic Regression)
# ─────────────────────────────────────────────────────────────────────
def compute_l1_importance(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """Fit L1-regularised Logistic Regression and use absolute coefficients."""
    logger.info("Training L1-regularised Logistic Regression …")
    lr = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        C=1.0,
        random_state=42,
        n_jobs=-1,
    )
    lr.fit(X, y)
    # Mean absolute coefficient across all classes
    coefs = np.abs(lr.coef_).mean(axis=0)
    coef_max = coefs.max()
    if coef_max > 0:
        coefs = coefs / coef_max
    return {name: float(score) for name, score in zip(feature_names, coefs)}


# ─────────────────────────────────────────────────────────────────────
# Ranking & Consensus
# ─────────────────────────────────────────────────────────────────────
def rank_features(scores: dict[str, float]) -> list[str]:
    """Return feature names sorted by descending importance."""
    return sorted(scores, key=scores.get, reverse=True)


def compute_consensus(
    rankings: dict[str, list[str]], top_k: int, min_methods: int = 2
) -> list[str]:
    """Return features that appear in the top-K of ≥ min_methods methods."""
    top_sets: dict[str, set[str]] = {
        method: set(ranked[:top_k]) for method, ranked in rankings.items()
    }
    all_features: set[str] = set()
    for s in top_sets.values():
        all_features |= s

    selected: list[str] = []
    for feat in all_features:
        count = sum(1 for s in top_sets.values() if feat in s)
        if count >= min_methods:
            selected.append(feat)

    # Sort selected features in the order they appear in BLENDSHAPE_NAMES
    bs_order = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}
    selected.sort(key=lambda f: bs_order.get(f, 999))
    return selected


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def plot_importance_comparison(
    all_scores: dict[str, dict[str, float]],
    selected_features: list[str],
    feature_names: list[str],
    save_path: Path,
) -> None:
    """Generate a grouped bar chart comparing importance across methods."""
    # Sort features by average importance
    avg_scores = {}
    for feat in feature_names:
        avg_scores[feat] = np.mean(
            [all_scores[m].get(feat, 0.0) for m in all_scores]
        )
    sorted_features = sorted(avg_scores, key=avg_scores.get, reverse=True)

    methods = list(all_scores.keys())
    n_features = len(sorted_features)
    n_methods = len(methods)
    x = np.arange(n_features)
    bar_width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(20, 8))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, method in enumerate(methods):
        scores = [all_scores[method].get(f, 0.0) for f in sorted_features]
        bars = ax.bar(
            x + i * bar_width, scores, bar_width,
            label=method.replace("_", " ").title(), color=colors[i], alpha=0.85,
        )

    # Highlight selected features
    for j, feat in enumerate(sorted_features):
        if feat in selected_features:
            ax.axvspan(
                j - 0.4, j + 0.4 + (n_methods - 1) * bar_width,
                alpha=0.08, color="green",
            )

    ax.set_xlabel("Blendshape Feature", fontsize=12)
    ax.set_ylabel("Normalised Importance", fontsize=12)
    ax.set_title("Feature Importance Comparison Across Methods", fontsize=14)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(sorted_features, rotation=90, fontsize=7)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved importance chart → %s", save_path)


# ─────────────────────────────────────────────────────────────────────
# Print Results Table
# ─────────────────────────────────────────────────────────────────────
def print_results_table(
    all_scores: dict[str, dict[str, float]],
    rankings: dict[str, list[str]],
    selected_features: list[str],
    top_k: int,
) -> None:
    """Print a formatted results table to stdout."""
    methods = list(all_scores.keys())
    header = f"{'Feature':<25}"
    for m in methods:
        header += f" | {m.replace('_', ' ').title():>20}"
    header += " | Selected"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Sort by average
    avg = {
        f: np.mean([all_scores[m].get(f, 0.0) for m in methods])
        for f in rankings[methods[0]]
    }
    for feat in sorted(avg, key=avg.get, reverse=True):
        row = f"{feat:<25}"
        for m in methods:
            score = all_scores[m].get(feat, 0.0)
            rank = rankings[m].index(feat) + 1 if feat in rankings[m] else -1
            row += f" | {score:8.4f} (#{rank:>2d})"
        sel = "  ✓" if feat in selected_features else ""
        row += f"  {sel}"
        print(row)

    print("=" * len(header))
    print(f"\nSelected {len(selected_features)} features (top-{top_k}, ≥2/3 methods)")
    print(f"Features: {selected_features}\n")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline feature selection analysis for blendshape features."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=FEATURE_SELECTION_TOP_K,
        help=f"Top-K features per method (default: {FEATURE_SELECTION_TOP_K})",
    )
    args = parser.parse_args()
    top_k: int = args.top_k

    # Load data
    df, X, y, feature_names = load_dataset(COMBINED_DATASET_FILE)
    logger.info(
        "Samples: %d | Features: %d | Classes: %d",
        X.shape[0], X.shape[1], len(np.unique(y)),
    )

    # ── Run all three methods ────────────────────────────────────────
    all_scores: dict[str, dict[str, float]] = {}
    rankings: dict[str, list[str]] = {}

    # 1. Mutual Information
    mi_scores = compute_mutual_information(X, y, feature_names)
    all_scores["mutual_information"] = mi_scores
    rankings["mutual_information"] = rank_features(mi_scores)

    # 2. Random Forest
    rf_scores = compute_random_forest_importance(X, y, feature_names)
    all_scores["random_forest"] = rf_scores
    rankings["random_forest"] = rank_features(rf_scores)

    # 3. L1 Regularisation
    l1_scores = compute_l1_importance(X, y, feature_names)
    all_scores["l1_regularization"] = l1_scores
    rankings["l1_regularization"] = rank_features(l1_scores)

    # ── Consensus ────────────────────────────────────────────────────
    selected = compute_consensus(rankings, top_k=top_k, min_methods=2)
    logger.info("Selected %d features via consensus (top-%d, ≥2/3 methods)", len(selected), top_k)

    # ── Print table ──────────────────────────────────────────────────
    print_results_table(all_scores, rankings, selected, top_k)

    # ── Save report JSON ─────────────────────────────────────────────
    report = {
        "top_k": top_k,
        "num_features_in": len(feature_names),
        "num_features_selected": len(selected),
        "selected_features": selected,
        "method_scores": {
            method: {
                "scores": scores,
                "ranking": rankings[method],
            }
            for method, scores in all_scores.items()
        },
    }
    FEATURE_SELECTION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_SELECTION_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved report → %s", FEATURE_SELECTION_REPORT)

    # ── Save selected features JSON ──────────────────────────────────
    with open(SELECTED_FEATURES_FILE, "w") as f:
        json.dump(selected, f, indent=2)
    logger.info("Saved selected features → %s", SELECTED_FEATURES_FILE)

    # ── Plot ─────────────────────────────────────────────────────────
    plot_path = EVALUATION_DIR / "feature_importance_comparison.png"
    plot_importance_comparison(all_scores, selected, feature_names, plot_path)

    logger.info("Feature selection complete.")


if __name__ == "__main__":
    main()
