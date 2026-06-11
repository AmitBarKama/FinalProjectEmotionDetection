"""
Run emotion classification on all FFHQ images using HSEmotion.

For every image the script records:
- ``image_id``       – numeric id extracted from the filename
- ``filename``       – original filename
- ``predicted_emotion`` – argmax class label (7-class)
- ``confidence``     – max probability after renormalization
- ``prob_Anger`` … ``prob_Surprise`` – full 7-class probability vector

HSEmotion's ``enet_b0_8_best_afew`` model outputs **8 classes** (the 8th
being *Contempt*).  We drop Contempt and renormalise the remaining 7
probabilities so they sum to 1.

Checkpointing
-------------
Every ``--checkpoint-every`` images (default 1000) the partial results
DataFrame is flushed to disk so the run can be resumed with ``--resume``.

Usage
-----
    python -m step2_emotion_labeling.label_emotions
    python -m step2_emotion_labeling.label_emotions --device cpu --batch-size 16 --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── project imports ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    EMOTION_LABELS,
    EMOTION_LABELS_DIR,
    EMOTION_LABELS_FILE,
    EMOTION_MODEL_NAME,
    FFHQ_DIR,
    LABELING_BATCH_SIZE,
)

# ── logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── constants ───────────────────────────────────────────────────────
CHECKPOINT_FILE = EMOTION_LABELS_DIR / "_emotion_labels_checkpoint.parquet"

# HSEmotion 8-class order (the model's native class order)
# Mapping: 0-Anger, 1-Contempt, 2-Disgust, 3-Fear, 4-Happiness,
#          5-Neutral, 6-Sadness, 7-Surprise
HSEMOTION_8_CLASSES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

# Indices of the 7 classes we want (drop Contempt at index 1)
KEEP_INDICES = [i for i, c in enumerate(HSEMOTION_8_CLASSES) if c != "Contempt"]
# Verify mapping matches our config.EMOTION_LABELS order
_KEPT_NAMES = [HSEMOTION_8_CLASSES[i] for i in KEEP_INDICES]
assert _KEPT_NAMES == EMOTION_LABELS, (
    f"Class order mismatch: {_KEPT_NAMES} vs {EMOTION_LABELS}"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _gather_image_paths(directory: Path) -> list[Path]:
    """Return sorted image paths in *directory*."""
    paths = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda p: p.name)
    return paths


def _image_id_from_path(p: Path) -> int:
    """Extract numeric image ID from filename, e.g. '00042.png' → 42."""
    try:
        return int(p.stem)
    except ValueError:
        return hash(p.stem) & 0x7FFFFFFF  # fallback


def _map_8_to_7(probs_8: np.ndarray) -> np.ndarray:
    """Drop Contempt and renormalise the 7-class probabilities.

    Parameters
    ----------
    probs_8 : np.ndarray
        Shape ``(8,)`` — raw probabilities from HSEmotion.

    Returns
    -------
    np.ndarray
        Shape ``(7,)`` — renormalised probabilities for our 7 classes.
    """
    probs_7 = probs_8[KEEP_INDICES]
    total = probs_7.sum()
    if total > 0:
        probs_7 = probs_7 / total
    else:
        probs_7 = np.ones(7, dtype=np.float32) / 7.0
    return probs_7


# ─────────────────────────────────────────────────────────────────────
# Core labeling logic
# ─────────────────────────────────────────────────────────────────────
def label_images(
    image_paths: list[Path],
    device: str = "cpu",
    batch_size: int = 32,
    checkpoint_every: int = 1000,
    resume: bool = False,
) -> pd.DataFrame:
    """Run HSEmotionRecognizer on *image_paths* and return a DataFrame.

    Parameters
    ----------
    image_paths : list[Path]
        Paths to face images.
    device : str
        ``'cuda'`` or ``'cpu'``.
    batch_size : int
        Not used for per-image inference but kept for future batching.
    checkpoint_every : int
        Flush partial results every N images.
    resume : bool
        If ``True``, load the checkpoint and skip already-processed images.

    Returns
    -------
    pd.DataFrame
        Columns: image_id, filename, predicted_emotion, confidence,
        prob_Anger, prob_Disgust, prob_Fear, prob_Happiness,
        prob_Neutral, prob_Sadness, prob_Surprise.
    """
    # Monkey-patch torch.load to default weights_only=False for compatibility with PyTorch 2.6+
    import torch
    original_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load

    # Monkey-patch timm block classes to be compatible with older pickled models
    import timm.models._efficientnet_blocks
    for cls in (timm.models._efficientnet_blocks.DepthwiseSeparableConv, timm.models._efficientnet_blocks.InvertedResidual):
        original_getattr = cls.__getattr__
        def custom_getattr(self, name, orig_getattr=original_getattr):
            try:
                return orig_getattr(self, name)
            except AttributeError:
                if name in ("conv_s2d", "bn_s2d"):
                    return None
                if name == "aa":
                    import torch.nn as nn
                    return nn.Identity()
                raise
        cls.__getattr__ = custom_getattr

    from hsemotion.facial_emotions import HSEmotionRecognizer  # type: ignore[import-untyped]

    logger.info("Initialising HSEmotionRecognizer (model=%s, device=%s) …",
                EMOTION_MODEL_NAME, device)
    recognizer = HSEmotionRecognizer(
        model_name=EMOTION_MODEL_NAME,
        device=device,
    )

    # ── resume from checkpoint ──────────────────────────────────────
    records: list[dict] = []
    processed_filenames: set[str] = set()

    if resume and CHECKPOINT_FILE.exists():
        logger.info("Resuming from checkpoint %s …", CHECKPOINT_FILE)
        df_ckpt = pd.read_parquet(CHECKPOINT_FILE)
        records = df_ckpt.to_dict("records")
        processed_filenames = set(df_ckpt["filename"])
        logger.info("Checkpoint has %d records.", len(records))

    # Filter out already-processed images
    remaining = [p for p in image_paths if p.name not in processed_filenames]
    logger.info(
        "Images to process: %d  (already done: %d)",
        len(remaining),
        len(processed_filenames),
    )

    t0 = time.perf_counter()
    prob_cols = [f"prob_{e}" for e in EMOTION_LABELS]

    for i, img_path in enumerate(
        tqdm(remaining, desc="Labeling emotions", unit="img")
    ):
        # Load image as BGR numpy array (what OpenCV / HSEmotion expects)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("Could not read %s — skipping.", img_path.name)
            continue

        # HSEmotionRecognizer.predict_emotions expects a face crop as
        # a numpy BGR image.  FFHQ images are already aligned face crops.
        try:
            emotion, scores = recognizer.predict_emotions(
                img_bgr, logits=False
            )
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", img_path.name, exc)
            continue

        scores = np.asarray(scores, dtype=np.float32)

        # Map 8-class → 7-class
        if scores.shape[0] == 8:
            probs_7 = _map_8_to_7(scores)
        elif scores.shape[0] == 7:
            probs_7 = scores / (scores.sum() + 1e-9)
        else:
            logger.warning(
                "Unexpected score length %d for %s — skipping.",
                scores.shape[0],
                img_path.name,
            )
            continue

        pred_idx = int(np.argmax(probs_7))
        pred_emotion = EMOTION_LABELS[pred_idx]
        confidence = float(probs_7[pred_idx])

        record = {
            "image_id": _image_id_from_path(img_path),
            "filename": img_path.name,
            "predicted_emotion": pred_emotion,
            "confidence": confidence,
        }
        for j, col in enumerate(prob_cols):
            record[col] = float(probs_7[j])
        records.append(record)

        # ── periodic checkpoint ─────────────────────────────────────
        if (i + 1) % checkpoint_every == 0:
            _save_checkpoint(records)
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            logger.info(
                "Checkpoint at %d images (%.1f img/s).", i + 1, rate
            )

    # Final checkpoint
    _save_checkpoint(records)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Finished labeling %d images in %.1f s (%.1f img/s).",
        len(remaining),
        elapsed,
        len(remaining) / elapsed if elapsed > 0 else 0,
    )

    df = pd.DataFrame(records)
    return df


def _save_checkpoint(records: list[dict]) -> None:
    """Flush *records* to the checkpoint parquet file."""
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_parquet(CHECKPOINT_FILE, index=False)


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label FFHQ images with emotions via HSEmotion.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference: 'cuda' or 'cpu' (default: cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=LABELING_BATCH_SIZE,
        help=f"Batch size (default: {LABELING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Save checkpoint every N images (default: 1000).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not FFHQ_DIR.exists():
        logger.error("FFHQ directory not found: %s", FFHQ_DIR)
        logger.error("Run step1_dataset.download_ffhq first.")
        sys.exit(1)

    image_paths = _gather_image_paths(FFHQ_DIR)
    if not image_paths:
        logger.error("No images found in %s", FFHQ_DIR)
        sys.exit(1)

    logger.info("Found %d images in %s", len(image_paths), FFHQ_DIR)
    logger.info("Model         : %s", EMOTION_MODEL_NAME)
    logger.info("Device        : %s", args.device)
    logger.info("Checkpoint    : every %d images", args.checkpoint_every)
    logger.info("Output file   : %s", EMOTION_LABELS_FILE)

    df = label_images(
        image_paths=image_paths,
        device=args.device,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    # Save final results
    EMOTION_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EMOTION_LABELS_FILE, index=False)
    logger.info("Saved %d records to %s", len(df), EMOTION_LABELS_FILE)

    # Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Removed checkpoint file.")

    # Quick summary
    print(f"\n{'─' * 50}")
    print("  Emotion Labeling Summary")
    print(f"{'─' * 50}")
    print(f"  Total images labelled : {len(df)}")
    print(f"  Mean confidence       : {df['confidence'].mean():.3f}")
    print(f"  Median confidence     : {df['confidence'].median():.3f}")
    print(f"\n  Per-class counts:")
    for emotion in EMOTION_LABELS:
        count = (df["predicted_emotion"] == emotion).sum()
        pct = 100 * count / len(df) if len(df) else 0
        print(f"    {emotion:<12s}: {count:>6d}  ({pct:5.1f}%)")
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()
