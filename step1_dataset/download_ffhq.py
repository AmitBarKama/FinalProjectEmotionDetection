"""
Download FFHQ 128×128 thumbnail images.

Primary source : Hugging Face (datasets library or direct HTTP)
Fallback source: Kaggle mirror

Features
--------
- Resumable: skips images that already exist on disk.
- Progress bar via tqdm.
- Configurable via CLI: ``--num-images``, ``--source``.

Usage
-----
    python -m step1_dataset.download_ffhq
    python -m step1_dataset.download_ffhq --num-images 5000 --source huggingface
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from tqdm import tqdm

# ── project imports ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FFHQ_DIR, DATA_DIR  # noqa: E402

# ── logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── constants ───────────────────────────────────────────────────────
TOTAL_FFHQ_IMAGES = 70_000

# Hugging Face dataset — FFHQ 128x128 thumbnails
HF_DATASET_NAME = "merkol/ffhq-256"
# Direct download URLs for the zip archives on HuggingFace
# The thumbnails128x128 dataset is typically hosted as zip shards
HF_THUMBNAIL_URL_TEMPLATE = (
    "https://huggingface.co/datasets/merkol/ffhq-256/resolve/main/"
    "data/thumbnails128x128/thumbnail128x128_batch_{batch_idx:02d}.zip"
)
# Alternative: single-archive approach
HF_SINGLE_ARCHIVE_URLS = [
    "https://huggingface.co/datasets/nhimwei76/ffhq-128/resolve/main/ffhq_128.zip",
]

# Kaggle mirror
KAGGLE_DATASET = "greatgamedota/ffhq-face-data-set"


def _count_existing_images(directory: Path) -> set[str]:
    """Return set of existing PNG filenames in *directory*."""
    if not directory.exists():
        return set()
    return {f.name for f in directory.glob("*.png")}


def _expected_filename(index: int) -> str:
    """Return the expected PNG filename for a given image index (0-based)."""
    return f"{index:05d}.png"


# ─────────────────────────────────────────────────────────────────────
# Hugging Face Download — via `datasets` library
# ─────────────────────────────────────────────────────────────────────
def _download_huggingface_datasets_lib(
    num_images: int,
    dest_dir: Path,
    existing: set[str],
) -> int:
    """Download using the HuggingFace ``datasets`` library (streaming).

    Returns the number of newly saved images.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "The `datasets` library is not installed. "
            "Install with: pip install datasets"
        )
        raise

    logger.info("Loading FFHQ dataset from Hugging Face via `datasets` library …")
    # Stream to avoid downloading everything at once
    ds = load_dataset(HF_DATASET_NAME, split="train", streaming=True)

    saved = 0
    pbar = tqdm(total=num_images, desc="Downloading (HF datasets)", unit="img")

    # Update progress for already-existing images we will skip
    skipped = 0

    for idx, sample in enumerate(ds):
        if saved + skipped >= num_images:
            break

        fname = _expected_filename(idx)
        dest_path = dest_dir / fname

        if fname in existing:
            skipped += 1
            pbar.update(1)
            continue

        # The sample should contain an image (PIL Image)
        img = sample.get("image") or sample.get("img")
        if img is None:
            logger.warning("Sample %d has no 'image' key; skipping.", idx)
            continue

        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img))

        # Resize to 128x128 if needed
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.LANCZOS)

        img.save(dest_path, format="PNG")
        saved += 1
        pbar.update(1)

    pbar.close()
    logger.info(
        "HF datasets: saved %d new images (%d skipped as existing).",
        saved,
        skipped,
    )
    return saved


# ─────────────────────────────────────────────────────────────────────
# Hugging Face Download — direct HTTP (zip archive)
# ─────────────────────────────────────────────────────────────────────
def _download_huggingface_http(
    num_images: int,
    dest_dir: Path,
    existing: set[str],
) -> int:
    """Download FFHQ 128×128 thumbnails via direct HTTP zip archive.

    Returns the number of newly saved images.
    """
    saved = 0
    zip_tmp = DATA_DIR / "_ffhq128_tmp.zip"

    for url in HF_SINGLE_ARCHIVE_URLS:
        logger.info("Trying direct HTTP download: %s", url)
        try:
            resp = requests.head(url, timeout=15, allow_redirects=True)
            if resp.status_code != 200:
                logger.warning("HEAD returned %d, skipping URL.", resp.status_code)
                continue
        except requests.RequestException as exc:
            logger.warning("HEAD request failed (%s), skipping URL.", exc)
            continue

        content_length = int(resp.headers.get("Content-Length", 0))
        logger.info(
            "Downloading archive (%.1f MB) …",
            content_length / 1024 / 1024 if content_length else 0,
        )

        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(zip_tmp, "wb") as f:
                    pbar = tqdm(
                        total=content_length or None,
                        desc="Downloading ZIP",
                        unit="B",
                        unit_scale=True,
                    )
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        pbar.update(len(chunk))
                    pbar.close()
        except requests.RequestException as exc:
            logger.error("Download failed: %s", exc)
            if zip_tmp.exists():
                zip_tmp.unlink()
            continue

        # Extract images from ZIP
        logger.info("Extracting images from archive …")
        try:
            with zipfile.ZipFile(zip_tmp, "r") as zf:
                image_members = [
                    m
                    for m in zf.namelist()
                    if m.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                image_members.sort()

                for member in tqdm(
                    image_members[:num_images],
                    desc="Extracting",
                    unit="img",
                ):
                    basename = Path(member).name
                    # Normalise filename to 5-digit PNG
                    stem = Path(basename).stem
                    try:
                        idx = int(stem)
                        fname = _expected_filename(idx)
                    except ValueError:
                        fname = basename

                    if fname in existing:
                        continue

                    dest_path = dest_dir / fname
                    with zf.open(member) as src:
                        img = Image.open(src)
                        if img.size != (128, 128):
                            img = img.resize((128, 128), Image.LANCZOS)
                        img.save(dest_path, format="PNG")
                        saved += 1

        except zipfile.BadZipFile:
            logger.error("Corrupted zip archive.")
        finally:
            if zip_tmp.exists():
                zip_tmp.unlink()

        if saved > 0:
            break  # success — don't try other URLs

    logger.info("HTTP download: saved %d new images.", saved)
    return saved


def download_huggingface(num_images: int, dest_dir: Path) -> int:
    """Try HuggingFace ``datasets`` library first, fall back to HTTP."""
    existing = _count_existing_images(dest_dir)
    remaining = num_images - len(existing)

    if remaining <= 0:
        logger.info(
            "Already have %d / %d images — nothing to download.",
            len(existing),
            num_images,
        )
        return 0

    logger.info(
        "%d images already present; need %d more.",
        len(existing),
        remaining,
    )

    # Attempt 1: datasets library (streaming, most reliable)
    try:
        saved = _download_huggingface_datasets_lib(num_images, dest_dir, existing)
        if saved > 0:
            return saved
    except Exception as exc:
        logger.warning("datasets-library download failed: %s", exc)

    # Attempt 2: direct HTTP archive
    try:
        saved = _download_huggingface_http(num_images, dest_dir, existing)
        if saved > 0:
            return saved
    except Exception as exc:
        logger.warning("HTTP archive download failed: %s", exc)

    logger.error("All Hugging Face download methods failed.")
    return 0


# ─────────────────────────────────────────────────────────────────────
# Kaggle Fallback
# ─────────────────────────────────────────────────────────────────────
def download_kaggle(num_images: int, dest_dir: Path) -> int:
    """Download FFHQ thumbnails from Kaggle.

    Requires ``kaggle`` CLI to be installed and configured
    (``~/.kaggle/kaggle.json``).

    Returns the number of newly saved images.
    """
    try:
        import kaggle  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "The `kaggle` package is not installed. "
            "Install with: pip install kaggle"
        )
        return 0

    existing = _count_existing_images(dest_dir)
    remaining = num_images - len(existing)
    if remaining <= 0:
        logger.info("Already have %d images — nothing to download.", len(existing))
        return 0

    kaggle_tmp = DATA_DIR / "_kaggle_tmp"
    kaggle_tmp.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading FFHQ from Kaggle dataset: %s", KAGGLE_DATASET)
    try:
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(kaggle_tmp),
            unzip=True,
            quiet=False,
        )
    except Exception as exc:
        logger.error("Kaggle download failed: %s", exc)
        return 0

    # Move images into dest_dir
    saved = 0
    all_images = sorted(kaggle_tmp.rglob("*.png"))
    for img_path in tqdm(all_images[:num_images], desc="Moving images", unit="img"):
        stem = img_path.stem
        try:
            idx = int(stem)
            fname = _expected_filename(idx)
        except ValueError:
            fname = img_path.name

        if fname in existing:
            continue

        dest_path = dest_dir / fname
        shutil.move(str(img_path), str(dest_path))
        saved += 1

    # Cleanup
    shutil.rmtree(kaggle_tmp, ignore_errors=True)
    logger.info("Kaggle: saved %d new images.", saved)
    return saved


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FFHQ 128×128 thumbnail images.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=TOTAL_FFHQ_IMAGES,
        help=f"Number of images to download (default: {TOTAL_FFHQ_IMAGES}).",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "kaggle"],
        default="huggingface",
        help="Download source (default: huggingface).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_images: int = args.num_images
    source: str = args.source

    FFHQ_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Target directory : %s", FFHQ_DIR)
    logger.info("Requested images : %d", num_images)
    logger.info("Primary source   : %s", source)

    if source == "huggingface":
        saved = download_huggingface(num_images, FFHQ_DIR)
        if saved == 0 and len(_count_existing_images(FFHQ_DIR)) < num_images:
            logger.info("Falling back to Kaggle …")
            saved = download_kaggle(num_images, FFHQ_DIR)
    else:
        saved = download_kaggle(num_images, FFHQ_DIR)
        if saved == 0 and len(_count_existing_images(FFHQ_DIR)) < num_images:
            logger.info("Falling back to Hugging Face …")
            saved = download_huggingface(num_images, FFHQ_DIR)

    total = len(_count_existing_images(FFHQ_DIR))
    logger.info("Done. Total images in %s: %d", FFHQ_DIR, total)


if __name__ == "__main__":
    main()
