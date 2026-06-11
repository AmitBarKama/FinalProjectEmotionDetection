"""
Build the combined training dataset by merging emotion labels with blendshapes.

Performs an inner join of the filtered (or raw) emotion labels parquet with
the blendshapes parquet on ``image_id``.  The resulting dataset contains:

- 52 blendshape feature columns  (input)
- 7 emotion probability columns  (soft labels)
- ``predicted_emotion``          (hard label)
- ``confidence``                 (prediction confidence)

The merged dataset is saved to ``config.COMBINED_DATASET_FILE`` as parquet.

Usage
-----
    python -m step3_blendshape_extraction.build_dataset
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BLENDSHAPE_NAMES,
    BLENDSHAPES_FILE,
    COMBINED_DATASET_FILE,
    EMOTION_LABELS,
    EMOTION_LABELS_FILE,
    EMOTION_LABELS_FILTERED_FILE,
    MIN_DATASET_SIZE,
    HAPPINESS_CAP,
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
# Derived column names
# ---------------------------------------------------------------------------
PROB_COLUMNS = [f"prob_{e}" for e in EMOTION_LABELS]
# e.g. ["prob_Anger", "prob_Disgust", "prob_Fear", "prob_Happiness",
#        "prob_Neutral", "prob_Sadness", "prob_Surprise"]


# ═══════════════════════════════════════════════════════════════════════
# Core logic
# ═══════════════════════════════════════════════════════════════════════

def _load_emotion_labels() -> pd.DataFrame:
    """Load emotion labels, preferring the filtered version."""
    if EMOTION_LABELS_FILTERED_FILE.exists():
        path = EMOTION_LABELS_FILTERED_FILE
        logger.info("Using filtered emotion labels: %s", path)
    elif EMOTION_LABELS_FILE.exists():
        path = EMOTION_LABELS_FILE
        logger.warning(
            "Filtered labels not found – falling back to raw labels: %s", path
        )
    else:
        logger.error(
            "No emotion-label file found.  Expected one of:\n  %s\n  %s",
            EMOTION_LABELS_FILTERED_FILE,
            EMOTION_LABELS_FILE,
        )
        sys.exit(1)

    df = pd.read_parquet(path)
    logger.info("  Emotion labels loaded: %d samples, columns: %s", len(df), list(df.columns))
    return df


def _load_blendshapes() -> pd.DataFrame:
    """Load blendshape features."""
    if not BLENDSHAPES_FILE.exists():
        logger.error(
            "Blendshapes file not found at %s. Run extract_blendshapes first.",
            BLENDSHAPES_FILE,
        )
        sys.exit(1)

    df = pd.read_parquet(BLENDSHAPES_FILE)
    logger.info("  Blendshapes loaded: %d samples, %d columns", len(df), len(df.columns))
    return df


def build_dataset() -> pd.DataFrame:
    """Merge emotion labels with blendshapes and save the combined dataset.

    Returns
    -------
    pd.DataFrame
        The final merged dataset.
    """
    # ── 1. Load sources ──────────────────────────────────────────────
    logger.info("Loading source data …")
    emotions_df = _load_emotion_labels()
    blendshapes_df = _load_blendshapes()

    # Ensure image_id is string in both
    emotions_df["image_id"] = emotions_df["image_id"].astype(str)
    blendshapes_df["image_id"] = blendshapes_df["image_id"].astype(str)

    n_emotions = len(emotions_df)
    n_blendshapes = len(blendshapes_df)

    # ── 2. Inner join ─────────────────────────────────────────────────
    logger.info("Performing inner join on image_id …")
    merged = pd.merge(emotions_df, blendshapes_df, on="image_id", how="inner")
    
    # ── 2.5 Undersample Happiness if cap is configured ────────────────
    if "predicted_emotion" in merged.columns and HAPPINESS_CAP is not None:
        happy_mask = merged["predicted_emotion"] == "Happiness"
        happy_df = merged[happy_mask]
        other_df = merged[~happy_mask]
        
        n_happy = len(happy_df)
        if n_happy > HAPPINESS_CAP:
            logger.info("Undersampling Happiness class from %d to %d samples...", n_happy, HAPPINESS_CAP)
            # Use random_state for reproducible results
            happy_sampled = happy_df.sample(n=HAPPINESS_CAP, random_state=42)
            merged = pd.concat([happy_sampled, other_df], ignore_index=True)
        else:
            logger.info("Happiness samples (%d) is already below cap (%d). No undersampling performed.", n_happy, HAPPINESS_CAP)

    n_merged = len(merged)

    lost_emotions = n_emotions - n_merged
    lost_blendshapes = n_blendshapes - n_merged
    logger.info(
        "  Emotion samples without blendshapes : %d", lost_emotions,
    )
    logger.info(
        "  Blendshape samples without emotions : %d", lost_blendshapes,
    )
    logger.info("  Merged dataset size                : %d", n_merged)

    if n_merged == 0:
        logger.error("Merged dataset is empty – aborting.")
        sys.exit(1)

    # ── 3. Select & order columns ─────────────────────────────────────
    # Blendshape feature columns present in the data
    bs_cols = [c for c in BLENDSHAPE_NAMES if c in merged.columns]

    # Probability columns – try both forms (prob_anger / Anger)
    prob_cols_present: list[str] = []
    for pc in PROB_COLUMNS:
        if pc in merged.columns:
            prob_cols_present.append(pc)

    # If prob_ columns aren't there, look for raw emotion-name columns
    if not prob_cols_present:
        raw_prob_cols = [e for e in EMOTION_LABELS if e in merged.columns]
        if raw_prob_cols:
            # Rename to prob_<emotion> convention
            rename_map = {e: f"prob_{e}" for e in raw_prob_cols}
            merged.rename(columns=rename_map, inplace=True)
            prob_cols_present = list(rename_map.values())
            logger.info("  Renamed emotion columns to prob_* convention.")

    if not prob_cols_present:
        logger.error(
            "Could not find probability columns in emotion labels. "
            "Expected columns like prob_anger or Anger."
        )
        sys.exit(1)

    # Metadata columns
    meta_cols: list[str] = []
    if "predicted_emotion" in merged.columns:
        meta_cols.append("predicted_emotion")
    if "confidence" in merged.columns:
        meta_cols.append("confidence")

    final_cols = ["image_id"] + bs_cols + prob_cols_present + meta_cols
    merged = merged[final_cols]

    # ── 4. Class distribution ─────────────────────────────────────────
    if "predicted_emotion" in merged.columns:
        class_dist = merged["predicted_emotion"].value_counts().sort_index()
        class_pct = (
            merged["predicted_emotion"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        )
        logger.info("Class distribution:")
        for cls_name in class_dist.index:
            logger.info(
                "  %-12s  %6d  (%5.1f%%)",
                cls_name,
                class_dist[cls_name],
                class_pct[cls_name],
            )

    # ── 5. Quality gate ───────────────────────────────────────────────
    if n_merged < MIN_DATASET_SIZE:
        logger.warning(
            "⚠  Dataset size %d is below the quality gate (%d).",
            n_merged,
            MIN_DATASET_SIZE,
        )
    else:
        logger.info(
            "✓  Dataset size %d passes the quality gate (≥ %d).",
            n_merged,
            MIN_DATASET_SIZE,
        )

    # ── 6. Save ───────────────────────────────────────────────────────
    COMBINED_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(COMBINED_DATASET_FILE, index=False)
    logger.info("Combined dataset saved to %s", COMBINED_DATASET_FILE)

    # ── 7. Summary table ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DATASET BUILD SUMMARY")
    logger.info("=" * 60)
    logger.info("  Emotion label source     : %s",
                "filtered" if EMOTION_LABELS_FILTERED_FILE.exists() else "raw")
    logger.info("  Emotion label count      : %d", n_emotions)
    logger.info("  Blendshape count         : %d", n_blendshapes)
    logger.info("  Samples lost (emotions)  : %d", lost_emotions)
    logger.info("  Samples lost (blendshp.) : %d", lost_blendshapes)
    logger.info("  Final dataset size       : %d", n_merged)
    logger.info("  Feature columns          : %d blendshapes", len(bs_cols))
    logger.info("  Label columns            : %s", prob_cols_present)
    logger.info("  Output file              : %s", COMBINED_DATASET_FILE)
    logger.info("=" * 60)

    return merged


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the combined training dataset (blendshapes + emotion labels).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _ = parse_args()
    build_dataset()
