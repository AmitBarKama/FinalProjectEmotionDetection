"""
Verify the downloaded FFHQ thumbnail dataset.

Checks
------
- Total image count in ``config.FFHQ_DIR``.
- Each image can be opened / decoded by PIL without errors.
- File size statistics (min / max / avg).
- Image format and dimensions.
- Summarises corrupted images (if any).

Usage
-----
    python -m step1_dataset.verify_dataset
"""

from __future__ import annotations

import logging
import statistics
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ── project imports ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FFHQ_DIR  # noqa: E402

# ── logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def gather_image_paths(directory: Path) -> list[Path]:
    """Return sorted list of image file paths in *directory*."""
    paths = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda p: p.name)
    return paths


def verify_images(
    image_paths: list[Path],
) -> tuple[list[Path], list[dict]]:
    """Open every image; return (corrupted_paths, stat_dicts).

    Each *stat_dict* has keys: ``path``, ``size_bytes``, ``format``,
    ``width``, ``height``.
    """
    corrupted: list[Path] = []
    stats: list[dict] = []

    for img_path in tqdm(image_paths, desc="Verifying images", unit="img"):
        size_bytes = img_path.stat().st_size
        try:
            with Image.open(img_path) as img:
                img.verify()  # quick structural check
            # Re-open to get pixel data (verify() can miss decoding errors)
            with Image.open(img_path) as img:
                img.load()
                stats.append(
                    {
                        "path": img_path,
                        "size_bytes": size_bytes,
                        "format": img.format or "UNKNOWN",
                        "width": img.width,
                        "height": img.height,
                    }
                )
        except Exception as exc:
            logger.warning("Corrupted: %s — %s", img_path.name, exc)
            corrupted.append(img_path)

    return corrupted, stats


def print_report(
    total: int,
    corrupted: list[Path],
    stats: list[dict],
) -> bool:
    """Print a human-readable verification report.

    Returns ``True`` if the dataset passes, ``False`` otherwise.
    """
    sep = "─" * 60
    print(f"\n{sep}")
    print("  FFHQ Dataset Verification Report")
    print(sep)

    print(f"  Directory      : {FFHQ_DIR}")
    print(f"  Total files    : {total}")
    print(f"  Valid images   : {len(stats)}")
    print(f"  Corrupted      : {len(corrupted)}")

    if stats:
        sizes = [s["size_bytes"] for s in stats]
        formats = {s["format"] for s in stats}
        dimensions = {(s["width"], s["height"]) for s in stats}

        print(f"\n  File sizes:")
        print(f"    Min          : {min(sizes):>10,} bytes")
        print(f"    Max          : {max(sizes):>10,} bytes")
        print(f"    Mean         : {statistics.mean(sizes):>10,.0f} bytes")
        if len(sizes) > 1:
            print(f"    Std-dev      : {statistics.stdev(sizes):>10,.0f} bytes")
        print(f"    Total        : {sum(sizes):>10,} bytes  "
              f"({sum(sizes) / 1024 / 1024:.1f} MB)")

        print(f"\n  Formats        : {', '.join(sorted(formats))}")
        print(f"  Dimensions     : {', '.join(f'{w}×{h}' for w, h in sorted(dimensions))}")
    else:
        print("\n  ⚠  No valid images found.")

    if corrupted:
        print(f"\n  Corrupted files ({len(corrupted)}):")
        for p in corrupted[:20]:
            print(f"    • {p.name}")
        if len(corrupted) > 20:
            print(f"    … and {len(corrupted) - 20} more")

    # Pass / fail
    passed = len(corrupted) == 0 and len(stats) > 0
    print(f"\n  {'✅  PASS' if passed else '❌  FAIL'}")
    print(sep + "\n")
    return passed


def main() -> None:
    logger.info("Scanning %s …", FFHQ_DIR)

    if not FFHQ_DIR.exists():
        logger.error("FFHQ directory does not exist: %s", FFHQ_DIR)
        sys.exit(1)

    image_paths = gather_image_paths(FFHQ_DIR)
    if not image_paths:
        logger.error("No image files found in %s", FFHQ_DIR)
        print_report(0, [], [])
        sys.exit(1)

    logger.info("Found %d image files. Starting verification …", len(image_paths))
    corrupted, stats = verify_images(image_paths)

    passed = print_report(
        total=len(image_paths),
        corrupted=corrupted,
        stats=stats,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
