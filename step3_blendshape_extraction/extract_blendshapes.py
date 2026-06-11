"""
Extract MediaPipe Face Blendshapes from FFHQ thumbnail images.

This script processes every image in FFHQ_DIR through the MediaPipe
FaceLandmarker (with blendshapes enabled) and records 52 ARKit-compatible
blendshape coefficients per detected face.  Images where no face is
detected are silently skipped and counted.

Outputs
-------
- config.BLENDSHAPES_FILE  (parquet)  – image_id + 52 blendshape columns
- Checkpoint files in config.BLENDSHAPES_DIR for safe resume

Usage
-----
    python -m step3_blendshape_extraction.extract_blendshapes
    python -m step3_blendshape_extraction.extract_blendshapes --resume
    python -m step3_blendshape_extraction.extract_blendshapes --max-images 5000
"""

import argparse
import logging
import sys
import time
import urllib.request
from pathlib import Path

import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BLENDSHAPE_NAMES,
    BLENDSHAPES_DIR,
    BLENDSHAPES_FILE,
    FFHQ_DIR,
    MEDIAPIPE_MODEL_PATH,
    MEDIAPIPE_MODEL_URL,
    MIN_BLENDSHAPE_EXTRACTION_RATE,
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
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_INTERVAL = 1000
CHECKPOINT_FILE = BLENDSHAPES_DIR / "blendshapes_checkpoint.parquet"
MISSED_LOG_FILE = BLENDSHAPES_DIR / "missed_images.txt"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _download_model() -> None:
    """Download the MediaPipe face-landmarker model if it is not present."""
    if MEDIAPIPE_MODEL_PATH.exists():
        logger.info("MediaPipe model already present at %s", MEDIAPIPE_MODEL_PATH)
        return

    MEDIAPIPE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe model from %s …", MEDIAPIPE_MODEL_URL)

    try:
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, str(MEDIAPIPE_MODEL_PATH))
        size_mb = MEDIAPIPE_MODEL_PATH.stat().st_size / (1024 * 1024)
        logger.info("Download complete (%.1f MB).", size_mb)
    except Exception as exc:
        logger.error("Failed to download model: %s", exc)
        if MEDIAPIPE_MODEL_PATH.exists():
            MEDIAPIPE_MODEL_PATH.unlink()
        raise


def _collect_image_paths(max_images: int | None = None) -> list[Path]:
    """Recursively collect image files from FFHQ_DIR, sorted for determinism."""
    paths = sorted(
        p
        for p in FFHQ_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if max_images is not None:
        paths = paths[:max_images]
    logger.info("Found %d images in %s", len(paths), FFHQ_DIR)
    return paths


def _image_id_from_path(path: Path) -> str:
    """Derive a stable image_id from the filename (stem without extension)."""
    return path.stem


def _load_checkpoint() -> set[str]:
    """Load already-processed image_ids from the checkpoint file."""
    if not CHECKPOINT_FILE.exists():
        return set()
    df = pd.read_parquet(CHECKPOINT_FILE)
    ids = set(df["image_id"].tolist())
    logger.info("Loaded checkpoint with %d processed images.", len(ids))
    return ids


def _save_checkpoint(records: list[dict]) -> None:
    """Persist current records to the checkpoint parquet file."""
    df = pd.DataFrame(records)
    df.to_parquet(CHECKPOINT_FILE, index=False)
    logger.info("Checkpoint saved (%d records).", len(records))


# ═══════════════════════════════════════════════════════════════════════
# Main extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_blendshapes(
    resume: bool = False,
    max_images: int | None = None,
) -> pd.DataFrame:
    """Run blendshape extraction on all FFHQ images.

    Parameters
    ----------
    resume : bool
        If True, skip images already present in the checkpoint file.
    max_images : int or None
        Cap the number of images to process (useful for testing).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``image_id`` + 52 blendshape names.
    """
    # ── 1. Ensure the model is available ──────────────────────────────
    _download_model()

    # ── 2. Build the FaceLandmarker ───────────────────────────────────
    base_options = mp.tasks.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # ── 3. Gather image paths & handle resume ─────────────────────────
    all_paths = _collect_image_paths(max_images)

    if resume:
        done_ids = _load_checkpoint()
        # Also load previous records so we can extend them
        if CHECKPOINT_FILE.exists():
            records: list[dict] = pd.read_parquet(CHECKPOINT_FILE).to_dict("records")
        else:
            records = []
    else:
        done_ids: set[str] = set()
        records = []

    paths_to_process = [
        p for p in all_paths if _image_id_from_path(p) not in done_ids
    ]
    logger.info(
        "%d images to process (%d already done).",
        len(paths_to_process),
        len(done_ids),
    )

    # ── 4. Process images ─────────────────────────────────────────────
    missed_images: list[str] = []
    processed_since_ckpt = 0
    t_start = time.perf_counter()

    for img_path in tqdm(paths_to_process, desc="Extracting blendshapes", unit="img"):
        image_id = _image_id_from_path(img_path)

        try:
            mp_image = mp.Image.create_from_file(str(img_path))
            result = landmarker.detect(mp_image)
        except Exception as exc:
            logger.debug("Error processing %s: %s", img_path.name, exc)
            missed_images.append(image_id)
            continue

        # Check for face detection
        if not result.face_blendshapes or len(result.face_blendshapes) == 0:
            missed_images.append(image_id)
            continue

        # Extract the 52 blendshape scores
        face_bs = result.face_blendshapes[0]  # first (and only) face
        bs_dict: dict[str, float] = {
            cat.category_name: cat.score for cat in face_bs
        }

        row = {"image_id": image_id}
        for name in BLENDSHAPE_NAMES:
            row[name] = bs_dict.get(name, 0.0)

        records.append(row)
        processed_since_ckpt += 1

        # Periodic checkpoint
        if processed_since_ckpt >= CHECKPOINT_INTERVAL:
            _save_checkpoint(records)
            processed_since_ckpt = 0

    landmarker.close()

    # ── 5. Final save ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    df = pd.DataFrame(records)

    if len(df) == 0:
        logger.warning("No blendshapes were extracted. Output file will not be created.")
        return df

    df.to_parquet(BLENDSHAPES_FILE, index=False)
    logger.info("Saved blendshapes to %s", BLENDSHAPES_FILE)

    # Clean up checkpoint after successful full save
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint file removed after successful save.")

    # Save missed images log
    if missed_images:
        with open(MISSED_LOG_FILE, "w") as f:
            f.write("\n".join(missed_images) + "\n")
        logger.info("Missed-image log saved to %s", MISSED_LOG_FILE)

    # ── 6. Summary report ─────────────────────────────────────────────
    total_attempted = len(paths_to_process)
    faces_detected = len(records) - len(done_ids)
    faces_missed = len(missed_images)
    detection_rate = faces_detected / total_attempted if total_attempted else 0.0

    logger.info("=" * 60)
    logger.info("BLENDSHAPE EXTRACTION REPORT")
    logger.info("=" * 60)
    logger.info("  Total images attempted : %d", total_attempted)
    logger.info("  Faces detected (new)   : %d", faces_detected)
    logger.info("  Faces missed (new)     : %d", faces_missed)
    logger.info("  Detection rate         : %.2f%%", detection_rate * 100)
    logger.info("  Total records in file  : %d", len(df))
    logger.info("  Elapsed time           : %.1f s", elapsed)
    logger.info("  Speed                  : %.1f img/s", total_attempted / elapsed if elapsed > 0 else 0)
    logger.info("=" * 60)

    # Quality gate
    if detection_rate < MIN_BLENDSHAPE_EXTRACTION_RATE:
        logger.warning(
            "⚠  Detection rate %.2f%% is below the quality gate (%.0f%%).",
            detection_rate * 100,
            MIN_BLENDSHAPE_EXTRACTION_RATE * 100,
        )
    else:
        logger.info("✓  Detection rate passes the quality gate.")

    return df


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe blendshapes from FFHQ images.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint (skip already-processed images).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_blendshapes(resume=args.resume, max_images=args.max_images)
