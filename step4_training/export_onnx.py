"""
Export the trained EmotionMLP to ONNX format with validation and benchmarks.

Steps:
  1. Load best checkpoint and normalization parameters.
  2. Export via torch.onnx.export (opset 15, dynamic batch axis).
  3. Validate: compare PyTorch vs ONNX Runtime outputs on test samples.
  4. Benchmark: measure average inference latency over 1000 runs.
  5. Quality gates: model size < MAX_ONNX_SIZE_KB, latency < MAX_ONNX_INFERENCE_MS.
  6. Save metadata JSON alongside the ONNX file.

Usage:
    python -m step4_training.export_onnx [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CHECKPOINTS_DIR,
    EMOTION_LABELS,
    EXPORTED_DIR,
    MAX_ONNX_INFERENCE_MS,
    MAX_ONNX_SIZE_KB,
    NORMALIZATION_PARAMS_FILE,
    ONNX_MODEL_PATH,
    ONNX_OPSET_VERSION,
    SELECTED_FEATURES_FILE,
)
from step4_training.model import EmotionMLP

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────
def load_checkpoint(device: torch.device) -> tuple[EmotionMLP, dict]:
    """Load best model checkpoint."""
    ckpt_path = CHECKPOINTS_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EmotionMLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        num_classes=ckpt["num_classes"],
        dropout_rates=ckpt["dropout_rates"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info("Loaded model from epoch %d", ckpt["epoch"])
    return model, ckpt


def load_normalization_params() -> dict:
    """Load normalization parameters from JSON."""
    if not NORMALIZATION_PARAMS_FILE.exists():
        raise FileNotFoundError(f"Normalization params not found: {NORMALIZATION_PARAMS_FILE}")
    with open(NORMALIZATION_PARAMS_FILE, "r") as f:
        return json.load(f)


def load_selected_features() -> list[str]:
    """Load selected features list from JSON."""
    if not SELECTED_FEATURES_FILE.exists():
        logger.warning("Selected features file not found, using checkpoint info.")
        return []
    with open(SELECTED_FEATURES_FILE, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────────────────
def export_to_onnx(model: EmotionMLP, input_dim: int, save_path: Path) -> None:
    """Export the PyTorch model to ONNX format."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, input_dim)
    logger.info("Exporting to ONNX (opset %d) → %s", ONNX_OPSET_VERSION, save_path)

    torch.onnx.export(
        model.cpu(),
        dummy_input,
        str(save_path),
        opset_version=ONNX_OPSET_VERSION,
        input_names=["blendshapes"],
        output_names=["emotion_probabilities"],
        dynamic_axes={
            "blendshapes": {0: "batch_size"},
            "emotion_probabilities": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    logger.info("ONNX export complete.")


# ─────────────────────────────────────────────────────────────────────
# Validation: PyTorch vs ONNX Runtime
# ─────────────────────────────────────────────────────────────────────
def validate_onnx(model: EmotionMLP, onnx_path: Path, input_dim: int, n_samples: int = 50) -> float:
    """Compare PyTorch and ONNX Runtime outputs.

    Returns
    -------
    max_diff : float
        Maximum absolute difference across all test samples.
    """
    import onnxruntime as ort

    logger.info("Validating ONNX model against PyTorch (%d samples) …", n_samples)

    session = ort.InferenceSession(str(onnx_path))
    model.cpu().eval()

    max_diff = 0.0
    for _ in range(n_samples):
        x_np = np.random.randn(1, input_dim).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # PyTorch
        with torch.no_grad():
            pt_out = model(x_torch).numpy()

        # ONNX Runtime
        ort_out = session.run(None, {"blendshapes": x_np})[0]

        diff = np.abs(pt_out - ort_out).max()
        max_diff = max(max_diff, diff)

    logger.info("Max absolute difference: %.2e", max_diff)
    return max_diff


# ─────────────────────────────────────────────────────────────────────
# Benchmark Latency
# ─────────────────────────────────────────────────────────────────────
def benchmark_latency(onnx_path: Path, input_dim: int, n_runs: int = 1000) -> float:
    """Measure average inference latency in milliseconds."""
    import onnxruntime as ort

    logger.info("Benchmarking latency (%d runs) …", n_runs)
    session = ort.InferenceSession(str(onnx_path))
    x_np = np.random.randn(1, input_dim).astype(np.float32)

    # Warm-up
    for _ in range(50):
        session.run(None, {"blendshapes": x_np})

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"blendshapes": x_np})
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_runs) * 1000
    logger.info("Average latency: %.3f ms", avg_ms)
    return avg_ms


# ─────────────────────────────────────────────────────────────────────
# Save Metadata
# ─────────────────────────────────────────────────────────────────────
def save_metadata(
    onnx_path: Path,
    ckpt: dict,
    norm_params: dict,
    selected_features: list[str],
    model_size_kb: float,
    latency_ms: float,
) -> None:
    """Save a sidecar JSON with everything needed at inference time."""
    metadata = {
        "model_file": onnx_path.name,
        "input_features": selected_features or norm_params.get("feature_cols", []),
        "input_dim": ckpt["input_dim"],
        "normalization_params": {
            "mean": norm_params["mean"],
            "std": norm_params["std"],
        },
        "emotion_labels": EMOTION_LABELS,
        "model_architecture": {
            "hidden_dims": ckpt["hidden_dims"],
            "dropout_rates": ckpt["dropout_rates"],
            "num_classes": ckpt["num_classes"],
        },
        "training_info": {
            "epoch": ckpt["epoch"],
            "best_val_macro_f1": round(ckpt["best_val_macro_f1"], 6),
            "alpha": ckpt.get("alpha"),
            "temperature": ckpt.get("temperature"),
        },
        "onnx_opset_version": ONNX_OPSET_VERSION,
        "model_size_kb": round(model_size_kb, 2),
        "avg_latency_ms": round(latency_ms, 4),
    }

    meta_path = onnx_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata → %s", meta_path)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Export EmotionMLP to ONNX.")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for loading model (cpu recommended for export)",
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=1000,
        help="Number of ONNX Runtime runs for latency benchmark",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load assets ──────────────────────────────────────────────────
    model, ckpt = load_checkpoint(device)
    norm_params = load_normalization_params()
    selected_features = load_selected_features()
    input_dim = ckpt["input_dim"]

    logger.info("Model: input_dim=%d, params=%d", input_dim, model.count_parameters())

    # ── Export ────────────────────────────────────────────────────────
    export_to_onnx(model, input_dim, ONNX_MODEL_PATH)

    # ── Model size ───────────────────────────────────────────────────
    model_size_kb = os.path.getsize(ONNX_MODEL_PATH) / 1024
    logger.info("ONNX model size: %.1f KB", model_size_kb)

    # ── Validate ─────────────────────────────────────────────────────
    max_diff = validate_onnx(model, ONNX_MODEL_PATH, input_dim)
    if max_diff < 1e-5:
        logger.info("✓ Validation PASSED (max diff %.2e < 1e-5)", max_diff)
    else:
        logger.warning("✗ Validation FAILED (max diff %.2e ≥ 1e-5)", max_diff)

    # ── Benchmark ────────────────────────────────────────────────────
    avg_latency = benchmark_latency(ONNX_MODEL_PATH, input_dim, n_runs=args.benchmark_runs)

    # ── Quality gates ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("ONNX EXPORT REPORT")
    print("=" * 55)
    print(f"  Model path:      {ONNX_MODEL_PATH}")
    print(f"  Parameter count: {model.count_parameters():,}")
    print(f"  Model size:      {model_size_kb:.1f} KB")
    print(f"  Avg latency:     {avg_latency:.3f} ms")
    print(f"  Max PT↔ORT diff: {max_diff:.2e}")

    size_pass = model_size_kb <= MAX_ONNX_SIZE_KB
    latency_pass = avg_latency <= MAX_ONNX_INFERENCE_MS
    diff_pass = max_diff < 1e-5

    print("\n  QUALITY GATES:")
    print(f"    Size ≤ {MAX_ONNX_SIZE_KB} KB:        {'✓ PASS' if size_pass else '✗ FAIL'} ({model_size_kb:.1f} KB)")
    print(f"    Latency ≤ {MAX_ONNX_INFERENCE_MS} ms:    {'✓ PASS' if latency_pass else '✗ FAIL'} ({avg_latency:.3f} ms)")
    print(f"    Numerical match:     {'✓ PASS' if diff_pass else '✗ FAIL'} ({max_diff:.2e})")
    print("=" * 55)

    if not all([size_pass, latency_pass, diff_pass]):
        logger.warning("Some quality gates FAILED — review the report above.")

    # ── Metadata ─────────────────────────────────────────────────────
    save_metadata(
        ONNX_MODEL_PATH, ckpt, norm_params, selected_features,
        model_size_kb, avg_latency,
    )

    logger.info("ONNX export pipeline complete.")


if __name__ == "__main__":
    main()
