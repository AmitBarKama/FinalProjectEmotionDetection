"""
Validate and analyse extracted MediaPipe blendshapes.

Loads the blendshape parquet file produced by ``extract_blendshapes.py``
and produces:

1. Per-blendshape descriptive statistics (mean, std, min, max,
   percentage of near-zero values).
2. A list of near-zero-variance blendshapes (candidates for removal).
3. A list of highly correlated blendshape pairs (ρ > 0.95).
4. Three publication-quality plots saved to ``config.BLENDSHAPES_DIR``:
   - Box plot of all 52 blendshapes
   - Correlation heatmap
   - Histogram of non-zero activation rates

Usage
-----
    python -m step3_blendshape_extraction.validate_blendshapes
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend – must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BLENDSHAPE_NAMES,
    BLENDSHAPES_DIR,
    BLENDSHAPES_FILE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
NEAR_ZERO_THRESHOLD = 0.01   # value below which a coefficient counts as "near zero"
LOW_VARIANCE_THRESHOLD = 1e-4  # variance below which a blendshape is flagged
HIGH_CORR_THRESHOLD = 0.95   # absolute correlation above which a pair is flagged


# ═══════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═══════════════════════════════════════════════════════════════════════

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with per-blendshape descriptive statistics."""
    bs_cols = [c for c in BLENDSHAPE_NAMES if c in df.columns]
    data = df[bs_cols]

    stats = pd.DataFrame(
        {
            "mean": data.mean(),
            "std": data.std(),
            "min": data.min(),
            "max": data.max(),
            "variance": data.var(),
            "pct_near_zero": (data < NEAR_ZERO_THRESHOLD).mean() * 100,
        }
    )
    stats.index.name = "blendshape"
    return stats


def find_low_variance(stats: pd.DataFrame) -> list[str]:
    """Return blendshape names whose variance is below *LOW_VARIANCE_THRESHOLD*."""
    mask = stats["variance"] < LOW_VARIANCE_THRESHOLD
    return list(stats.index[mask])


def find_high_correlations(
    df: pd.DataFrame,
    threshold: float = HIGH_CORR_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Return pairs of blendshapes with |ρ| > *threshold*."""
    bs_cols = [c for c in BLENDSHAPE_NAMES if c in df.columns]
    corr = df[bs_cols].corr()

    pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()

    for i, c1 in enumerate(bs_cols):
        for j, c2 in enumerate(bs_cols):
            if j <= i:
                continue
            r = corr.loc[c1, c2]
            if abs(r) > threshold and (c1, c2) not in seen:
                pairs.append((c1, c2, round(r, 4)))
                seen.add((c1, c2))

    # Sort by absolute correlation descending
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def _style() -> None:
    """Apply a clean matplotlib / seaborn style."""
    sns.set_theme(style="whitegrid", font_scale=0.85)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def plot_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    """Box plot showing the distribution of each blendshape coefficient."""
    _style()
    bs_cols = [c for c in BLENDSHAPE_NAMES if c in df.columns]

    fig, ax = plt.subplots(figsize=(18, 6))
    data_melted = df[bs_cols].melt(var_name="Blendshape", value_name="Value")
    sns.boxplot(
        data=data_melted,
        x="Blendshape",
        y="Value",
        ax=ax,
        fliersize=1,
        linewidth=0.6,
        palette="viridis",
    )
    ax.set_title("Distribution of 52 Blendshape Coefficients", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Coefficient value")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)

    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Box plot saved to %s", save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """Correlation heatmap across all blendshapes."""
    _style()
    bs_cols = [c for c in BLENDSHAPE_NAMES if c in df.columns]
    corr = df[bs_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.1,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title("Blendshape Correlation Matrix", fontsize=13)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6, rotation=0)

    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Correlation heatmap saved to %s", save_path)


def plot_activation_histogram(df: pd.DataFrame, save_path: Path) -> None:
    """Histogram of non-zero activation rates (% of samples with value ≥ 0.01)."""
    _style()
    bs_cols = [c for c in BLENDSHAPE_NAMES if c in df.columns]
    activation_rates = (df[bs_cols] >= NEAR_ZERO_THRESHOLD).mean() * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    activation_rates_sorted = activation_rates.sort_values(ascending=False)
    colors = sns.color_palette("viridis", n_colors=len(activation_rates_sorted))
    ax.bar(
        range(len(activation_rates_sorted)),
        activation_rates_sorted.values,
        color=colors,
        edgecolor="none",
    )
    ax.set_xticks(range(len(activation_rates_sorted)))
    ax.set_xticklabels(activation_rates_sorted.index, rotation=90, fontsize=7)
    ax.set_ylabel("Non-zero activation rate (%)")
    ax.set_title("Blendshape Activation Rates (≥ 0.01)", fontsize=13)
    ax.axhline(10, color="red", linestyle="--", linewidth=0.8, label="10 % threshold")
    ax.legend(fontsize=8)

    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Activation-rate histogram saved to %s", save_path)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def validate_blendshapes() -> None:
    """Run the full validation pipeline."""
    # ── 1. Load data ──────────────────────────────────────────────────
    if not BLENDSHAPES_FILE.exists():
        logger.error(
            "Blendshapes file not found at %s. Run extract_blendshapes first.",
            BLENDSHAPES_FILE,
        )
        sys.exit(1)

    logger.info("Loading blendshapes from %s …", BLENDSHAPES_FILE)
    df = pd.read_parquet(BLENDSHAPES_FILE)
    logger.info("Loaded %d samples with %d columns.", len(df), len(df.columns))

    # ── 2. Descriptive statistics ─────────────────────────────────────
    stats = compute_statistics(df)
    logger.info("\n%s", stats.to_string())

    stats_path = BLENDSHAPES_DIR / "blendshape_statistics.csv"
    stats.to_csv(stats_path)
    logger.info("Statistics saved to %s", stats_path)

    # ── 3. Near-zero-variance features ────────────────────────────────
    low_var = find_low_variance(stats)
    if low_var:
        logger.warning(
            "Near-zero-variance blendshapes (candidates for removal): %s",
            ", ".join(low_var),
        )
    else:
        logger.info("✓  No near-zero-variance blendshapes found.")

    # ── 4. Highly correlated pairs ────────────────────────────────────
    high_corr = find_high_correlations(df)
    if high_corr:
        logger.info(
            "Highly correlated pairs (|r| > %.2f):", HIGH_CORR_THRESHOLD
        )
        for c1, c2, r in high_corr:
            logger.info("  %s  ↔  %s  :  r = %.4f", c1, c2, r)
    else:
        logger.info("✓  No highly correlated blendshape pairs found.")

    # ── 5. Generate plots ─────────────────────────────────────────────
    logger.info("Generating plots …")

    plot_boxplots(
        df,
        save_path=BLENDSHAPES_DIR / "blendshape_boxplots.png",
    )
    plot_correlation_heatmap(
        df,
        save_path=BLENDSHAPES_DIR / "blendshape_correlation_heatmap.png",
    )
    plot_activation_histogram(
        df,
        save_path=BLENDSHAPES_DIR / "blendshape_activation_rates.png",
    )

    # ── 6. Final summary ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total samples               : %d", len(df))
    logger.info("  Blendshape columns           : %d", len([c for c in BLENDSHAPE_NAMES if c in df.columns]))
    logger.info("  Near-zero-variance features  : %d  %s", len(low_var), low_var if low_var else "")
    logger.info("  Highly correlated pairs      : %d", len(high_corr))
    logger.info("  Plots saved to               : %s", BLENDSHAPES_DIR)
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate extracted blendshapes and generate diagnostic plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _ = parse_args()
    validate_blendshapes()
