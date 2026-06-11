"""
Analyse emotion label distribution and confidence statistics.

Loads the labelled dataset from ``config.EMOTION_LABELS_FILE`` and
generates four publication-quality plots saved to
``config.EMOTION_LABELS_DIR / 'analysis/'``:

1. **Class distribution** – bar chart of per-emotion counts.
2. **Confidence histogram** – overall and per-class.
3. **Probability correlation heatmap** – Pearson correlation of the
   7-class probability vectors.
4. **Confidence box plot** – per-emotion-class spread.

Usage
-----
    python -m step2_emotion_labeling.analyze_labels
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ── project imports ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    EMOTION_LABELS,
    EMOTION_LABELS_DIR,
    EMOTION_LABELS_FILE,
)

# ── logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── output directory ────────────────────────────────────────────────
ANALYSIS_DIR = EMOTION_LABELS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── style ───────────────────────────────────────────────────────────
PALETTE = sns.color_palette("Set2", n_colors=len(EMOTION_LABELS))
EMOTION_COLOR_MAP = dict(zip(EMOTION_LABELS, PALETTE))


def _apply_dark_theme() -> None:
    """Apply a clean dark Seaborn + Matplotlib theme."""
    sns.set_theme(style="darkgrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "#1e1e1e",
            "axes.facecolor": "#2a2a2a",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#cccccc",
            "text.color": "#cccccc",
            "xtick.color": "#aaaaaa",
            "ytick.color": "#aaaaaa",
            "grid.color": "#3a3a3a",
            "legend.facecolor": "#2a2a2a",
            "legend.edgecolor": "#444444",
            "savefig.facecolor": "#1e1e1e",
            "savefig.dpi": 200,
        }
    )


# ─────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> Path:
    """Bar chart showing the number of images per predicted emotion."""
    counts = df["predicted_emotion"].value_counts()
    # Ensure all emotions are present and in canonical order
    counts = counts.reindex(EMOTION_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[EMOTION_COLOR_MAP[e] for e in counts.index],
        edgecolor="#1e1e1e",
        linewidth=0.8,
    )
    # Annotate counts
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#cccccc",
        )

    ax.set_title("Emotion Class Distribution", fontsize=16, pad=12)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    out = ANALYSIS_DIR / "class_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_confidence_histograms(df: pd.DataFrame) -> Path:
    """Histogram of confidence scores: overall and per emotion."""
    n_emotions = len(EMOTION_LABELS)
    fig, axes = plt.subplots(
        2,
        (n_emotions + 1) // 2 + 1,
        figsize=(22, 10),
        gridspec_kw={"wspace": 0.35, "hspace": 0.45},
    )
    axes = axes.flatten()

    # Overall histogram
    ax0 = axes[0]
    ax0.hist(
        df["confidence"],
        bins=50,
        color="#66b3ff",
        edgecolor="#1e1e1e",
        alpha=0.85,
    )
    ax0.set_title("Overall Confidence", fontsize=13)
    ax0.set_xlabel("Confidence")
    ax0.set_ylabel("Count")

    # Per-emotion histograms
    for idx, emotion in enumerate(EMOTION_LABELS, start=1):
        ax = axes[idx]
        subset = df.loc[df["predicted_emotion"] == emotion, "confidence"]
        ax.hist(
            subset,
            bins=40,
            color=EMOTION_COLOR_MAP[emotion],
            edgecolor="#1e1e1e",
            alpha=0.85,
        )
        ax.set_title(emotion, fontsize=13)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")

    # Hide unused axes
    for j in range(len(EMOTION_LABELS) + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confidence Score Distributions", fontsize=17, y=1.01)
    plt.tight_layout()

    out = ANALYSIS_DIR / "confidence_histograms.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_probability_correlation(df: pd.DataFrame) -> Path:
    """Heatmap of Pearson correlations between the 7-class probabilities."""
    prob_cols = [f"prob_{e}" for e in EMOTION_LABELS]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        logger.warning("Missing probability columns: %s — skipping heatmap.", missing)
        return ANALYSIS_DIR  # nothing saved

    corr = df[prob_cols].corr()
    # Nicer labels
    corr.index = EMOTION_LABELS
    corr.columns = EMOTION_LABELS

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="#1e1e1e",
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Emotion Probability Correlations", fontsize=16, pad=12)
    plt.tight_layout()

    out = ANALYSIS_DIR / "probability_correlations.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_confidence_boxplot(df: pd.DataFrame) -> Path:
    """Box plot of confidence scores per emotion class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="predicted_emotion",
        y="confidence",
        order=EMOTION_LABELS,
        palette=EMOTION_COLOR_MAP,
        ax=ax,
        fliersize=2,
        linewidth=1.2,
    )
    ax.set_title("Confidence Scores per Emotion", fontsize=16, pad=12)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Confidence")
    plt.tight_layout()

    out = ANALYSIS_DIR / "confidence_boxplots.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a concise table of summary statistics."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  Emotion Label Analysis Summary")
    print(sep)
    print(f"  Total samples          : {len(df):,}")
    print(f"  Overall mean confidence: {df['confidence'].mean():.4f}")
    print(f"  Overall std  confidence: {df['confidence'].std():.4f}")
    print(f"  Min confidence         : {df['confidence'].min():.4f}")
    print(f"  Max confidence         : {df['confidence'].max():.4f}")

    print(f"\n  {'Emotion':<12s} {'Count':>7s} {'%':>6s} "
          f"{'Mean Conf':>10s} {'Std Conf':>10s}")
    print(f"  {'─'*12} {'─'*7} {'─'*6} {'─'*10} {'─'*10}")

    for emotion in EMOTION_LABELS:
        subset = df[df["predicted_emotion"] == emotion]
        count = len(subset)
        pct = 100 * count / len(df) if len(df) else 0
        mean_c = subset["confidence"].mean() if count else 0
        std_c = subset["confidence"].std() if count > 1 else 0
        print(
            f"  {emotion:<12s} {count:>7,d} {pct:>5.1f}% "
            f"{mean_c:>10.4f} {std_c:>10.4f}"
        )

    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not EMOTION_LABELS_FILE.exists():
        logger.error(
            "Emotion labels file not found: %s\n"
            "Run step2_emotion_labeling.label_emotions first.",
            EMOTION_LABELS_FILE,
        )
        sys.exit(1)

    logger.info("Loading %s …", EMOTION_LABELS_FILE)
    df = pd.read_parquet(EMOTION_LABELS_FILE)
    logger.info("Loaded %d records.", len(df))

    _apply_dark_theme()

    plot_class_distribution(df)
    plot_confidence_histograms(df)
    plot_probability_correlation(df)
    plot_confidence_boxplot(df)

    print_summary(df)
    logger.info("All plots saved to %s", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
