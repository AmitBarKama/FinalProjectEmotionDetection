"""
Filter emotion labels by confidence and (optional) entropy thresholds.

Reads the full emotion label dataset from ``config.EMOTION_LABELS_FILE``,
applies the configured quality gates, and writes the filtered result to
``config.EMOTION_LABELS_FILTERED_FILE``.

Filtering criteria
------------------
1. **Confidence threshold** – keep only samples whose maximum class
   probability ≥ ``config.CONFIDENCE_THRESHOLD`` (default 0.5).
2. **Entropy threshold** (optional) – keep only samples whose
   Shannon entropy across the 7-class probability vector
   ≤ ``config.ENTROPY_THRESHOLD`` (default 1.5).
   Disable with ``--no-entropy``.

Usage
-----
    python -m step2_emotion_labeling.filter_labels
    python -m step2_emotion_labeling.filter_labels --confidence-threshold 0.6
    python -m step2_emotion_labeling.filter_labels --no-entropy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    EMOTION_LABELS,
    EMOTION_LABELS_FILE,
    EMOTION_LABELS_FILTERED_FILE,
    ENTROPY_THRESHOLD,
)

# ── logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Entropy helper
# ─────────────────────────────────────────────────────────────────────

def compute_entropy(df: pd.DataFrame) -> pd.Series:
    """Compute Shannon entropy for each row's 7-class probability vector.

    Returns a ``pd.Series`` of entropy values (one per row).
    Lower entropy → more certain prediction.
    """
    prob_cols = [f"prob_{e}" for e in EMOTION_LABELS]
    probs = df[prob_cols].values.astype(np.float64)
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log2(probs), axis=1)
    return pd.Series(entropy, index=df.index, name="entropy")


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def _per_class_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return per-emotion counts as a dict."""
    counts = df["predicted_emotion"].value_counts()
    return {e: int(counts.get(e, 0)) for e in EMOTION_LABELS}


def print_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    conf_thresh: float,
    ent_thresh: float | None,
) -> None:
    """Print a before/after comparison table."""
    sep = "─" * 65
    before_counts = _per_class_counts(df_before)
    after_counts = _per_class_counts(df_after)

    print(f"\n{sep}")
    print("  Emotion Label Filtering Report")
    print(sep)
    print(f"  Confidence threshold : ≥ {conf_thresh:.2f}")
    if ent_thresh is not None:
        print(f"  Entropy threshold    : ≤ {ent_thresh:.2f}")
    else:
        print(f"  Entropy threshold    : disabled")
    print()
    print(f"  Total BEFORE         : {len(df_before):>7,d}")
    print(f"  Total AFTER          : {len(df_after):>7,d}")
    removed = len(df_before) - len(df_after)
    pct_removed = 100 * removed / len(df_before) if len(df_before) else 0
    print(f"  Removed              : {removed:>7,d}  ({pct_removed:.1f}%)")

    print(f"\n  {'Emotion':<12s} {'Before':>8s} {'After':>8s} {'Removed':>8s} {'%Rem':>6s}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    for emotion in EMOTION_LABELS:
        b = before_counts[emotion]
        a = after_counts[emotion]
        r = b - a
        pct = 100 * r / b if b else 0
        print(f"  {emotion:<12s} {b:>8,d} {a:>8,d} {r:>8,d} {pct:>5.1f}%")

    # Confidence stats after filtering
    if len(df_after) > 0:
        print(f"\n  Post-filter confidence stats:")
        print(f"    Mean     : {df_after['confidence'].mean():.4f}")
        print(f"    Median   : {df_after['confidence'].median():.4f}")
        print(f"    Min      : {df_after['confidence'].min():.4f}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter emotion labels by confidence / entropy.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Minimum confidence to keep (default: {CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=ENTROPY_THRESHOLD,
        help=f"Maximum entropy to keep (default: {ENTROPY_THRESHOLD}).",
    )
    parser.add_argument(
        "--no-entropy",
        action="store_true",
        help="Disable entropy-based filtering.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    conf_thresh: float = args.confidence_threshold
    use_entropy: bool = not args.no_entropy
    ent_thresh: float | None = args.entropy_threshold if use_entropy else None

    # ── load ────────────────────────────────────────────────────────
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

    # ── confidence filter ───────────────────────────────────────────
    mask = df["confidence"] >= conf_thresh
    logger.info(
        "Confidence ≥ %.2f: %d / %d pass.",
        conf_thresh,
        mask.sum(),
        len(df),
    )

    # ── entropy filter ──────────────────────────────────────────────
    if use_entropy and ent_thresh is not None:
        entropy = compute_entropy(df)
        df["entropy"] = entropy
        entropy_mask = entropy <= ent_thresh
        logger.info(
            "Entropy ≤ %.2f: %d / %d pass.",
            ent_thresh,
            entropy_mask.sum(),
            len(df),
        )
        mask = mask & entropy_mask

    df_filtered = df.loc[mask].copy()

    # Drop helper column before saving
    if "entropy" in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=["entropy"])

    # ── report ──────────────────────────────────────────────────────
    print_report(df, df_filtered, conf_thresh, ent_thresh)

    # ── save ────────────────────────────────────────────────────────
    EMOTION_LABELS_FILTERED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(EMOTION_LABELS_FILTERED_FILE, index=False)
    logger.info(
        "Saved %d filtered records to %s",
        len(df_filtered),
        EMOTION_LABELS_FILTERED_FILE,
    )


if __name__ == "__main__":
    main()
