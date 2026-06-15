"""
Live webcam "test" for the trained emotion model.

Opens the MacBook camera and, in a single window, shows:
  • the camera feed with the MediaPipe face mesh drawn live on your face,
  • a side panel listing every blendshape the model consumes (value 0–1) with
    a fill bar showing how "full" each one is,
  • the model's emotion output (per-class probabilities + dominant emotion).

The demo is model-agnostic: it reads the active model's input feature list,
normalization (mean/std), and emotion labels from the ONNX sidecar metadata
(models/exported/emotion_model.json), so it adapts automatically whether the
model uses 34 or all 52 blendshapes.

Pipeline per frame:
    webcam → MediaPipe FaceLandmarker (52 blendshapes) → pick model features
    → z-score normalize → ONNX (logits) → softmax → emotions.

Note: the exported ONNX outputs raw *logits* (despite the "emotion_probabilities"
output name), so softmax is applied here.

Usage:
    python -m step5_live_demo.live_demo [--camera 0] [--no-smooth]

Keys:
    q / ESC : quit
    s       : toggle temporal (EMA) smoothing of the emotion output
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    MEDIAPIPE_MODEL_PATH,
    ONNX_MODEL_PATH,
)

import mediapipe as mp  # noqa: E402

# NOTE: this mediapipe build is tasks-only (no `mediapipe.solutions` /
# `framework.formats`), so the face mesh is drawn directly with OpenCV from the
# Tasks-API landmark coordinates rather than the legacy drawing utilities.

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── layout / style constants ─────────────────────────────────────────
DISPLAY_H = 720          # camera frame is scaled to this height
PANEL_W = 480            # width of the side panel
BG = (24, 24, 28)        # panel background (BGR)
BAR_BG = (60, 60, 66)
WHITE = (240, 240, 240)
GREY = (170, 170, 175)
DIM = (120, 120, 128)
AMBER = (0, 200, 255)    # dominant-emotion highlight (BGR)
GREEN = (0, 200, 80)
BLUE = (220, 160, 60)
RED = (60, 60, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MESH_COLOR = (0, 220, 120)   # face-mesh line color (BGR)


# ═════════════════════════════════════════════════════════════════════
# Small helpers
# ═════════════════════════════════════════════════════════════════════
def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def put_text(img, text, x, y, scale=0.45, color=WHITE, thick=1) -> None:
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_value_bar(img, x, y, w, h, value, fill_color) -> None:
    """Draw a 0–1 fill bar with a background track."""
    value = max(0.0, min(1.0, float(value)))
    cv2.rectangle(img, (x, y), (x + w, y + h), BAR_BG, -1)
    fw = int(round(w * value))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), fill_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (95, 95, 100), 1)


def draw_face_mesh(img, landmarks) -> None:
    """Draw a live face mesh by Delaunay-triangulating the landmark points.

    Uses only OpenCV + the normalized landmark coordinates, since this
    mediapipe build ships no drawing utilities. The mesh deforms with the
    face as expressions change.
    """
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for px, py in pts:
        if 0 <= px < w and 0 <= py < h:
            subdiv.insert((float(px), float(py)))

    for t in subdiv.getTriangleList():
        p1, p2, p3 = (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))
        # skip triangles whose vertices fall outside the frame (hull artifacts)
        if all(0 <= x < w and 0 <= y < h for x, y in (p1, p2, p3)):
            cv2.line(img, p1, p2, MESH_COLOR, 1)
            cv2.line(img, p2, p3, MESH_COLOR, 1)
            cv2.line(img, p3, p1, MESH_COLOR, 1)

    for px, py in pts:
        cv2.circle(img, (px, py), 1, WHITE, -1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════
# Asset loading
# ═════════════════════════════════════════════════════════════════════
def load_model_assets():
    """Load the ONNX session + metadata (features, mean/std, emotion labels)."""
    import onnxruntime as ort

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")
    meta_path = ONNX_MODEL_PATH.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    input_name = session.get_inputs()[0].name

    features = meta["input_features"]
    mean = np.asarray(meta["normalization_params"]["mean"], dtype=np.float32)
    std = np.asarray(meta["normalization_params"]["std"], dtype=np.float32)
    std = np.where(std < 1e-7, 1e-7, std)  # guard div-by-zero
    labels = meta["emotion_labels"]

    logger.info(
        "Model loaded: %d input features, %d emotions (input tensor '%s').",
        len(features), len(labels), input_name,
    )
    return session, input_name, features, mean, std, labels


def build_landmarker():
    """Create a MediaPipe FaceLandmarker in VIDEO mode with blendshapes."""
    if not MEDIAPIPE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MEDIAPIPE_MODEL_PATH}\n"
            "Run the blendshape-extraction step once to download it."
        )
    base_options = mp.tasks.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


# ═════════════════════════════════════════════════════════════════════
# Panel rendering
# ═════════════════════════════════════════════════════════════════════
def render_panel(height, features, raw_values, probs, labels, smoothing, fps,
                 infer_every=1):
    """Build the side panel image (BGR)."""
    panel = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)

    # ── header ───────────────────────────────────────────────────────
    smooth_txt = "smoothed" if smoothing else "raw"
    put_text(panel, f"{len(features)} inputs - {smooth_txt} - 1/{infer_every} fr",
             14, 22, scale=0.46, color=GREY)
    put_text(panel, f"{fps:4.1f} FPS", PANEL_W - 88, 22, scale=0.46, color=DIM)

    # ── emotion section ──────────────────────────────────────────────
    y = 40
    put_text(panel, "EMOTION", 14, y + 18, scale=0.7, color=WHITE, thick=2)
    y += 34

    if probs is None:
        put_text(panel, "no face detected", 14, y + 14, scale=0.5, color=RED)
        y += 30
    else:
        dom = int(np.argmax(probs))
        for i, label in enumerate(labels):
            p = float(probs[i])
            is_dom = i == dom
            ry = y + i * 30
            color = AMBER if is_dom else GREY
            put_text(panel, label, 14, ry + 17, scale=0.5, color=color,
                     thick=2 if is_dom else 1)
            draw_value_bar(panel, 132, ry + 4, 232, 17, p,
                           GREEN if is_dom else DIM)
            put_text(panel, f"{p * 100:4.0f}%", 372, ry + 17, scale=0.46,
                     color=color, thick=2 if is_dom else 1)
        y += len(labels) * 30

    # ── divider ──────────────────────────────────────────────────────
    y += 10
    cv2.line(panel, (12, y), (PANEL_W - 12, y), (70, 70, 76), 1)
    y += 22
    put_text(panel, f"BLENDSHAPES ({len(features)})", 14, y, scale=0.6,
             color=WHITE, thick=2)
    y += 14

    # ── blendshape grid (1 or 2 columns to fit) ──────────────────────
    n = len(features)
    area_h = height - y - 12
    ncols = 1 if n <= 24 else 2
    rows_per_col = math.ceil(n / ncols)
    row_h = max(10, min(24, area_h // max(rows_per_col, 1)))
    col_w = (PANEL_W - 16) // ncols

    for i, name in enumerate(features):
        col = i // rows_per_col
        row = i % rows_per_col
        x = 8 + col * col_w
        ry = y + row * row_h
        val = float(raw_values[i]) if raw_values is not None else 0.0
        put_text(panel, name[:13], x, ry + row_h - 4, scale=0.34, color=GREY)
        bar_x = x + 96
        bar_w = col_w - 132
        draw_value_bar(panel, bar_x, ry + 2, bar_w, max(6, row_h - 6), val, BLUE)
        put_text(panel, f"{val:.2f}", x + col_w - 34, ry + row_h - 4,
                 scale=0.34, color=GREY)

    return panel


# ═════════════════════════════════════════════════════════════════════
# Main loop
# ═════════════════════════════════════════════════════════════════════
def run(camera_index: int = 0, smoothing: bool = True, ema_alpha: float = 0.35,
        infer_every: int = 5) -> None:
    session, input_name, features, mean, std, labels = load_model_assets()
    landmarker = build_landmarker()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}. On macOS, grant the "
            "terminal/IDE Camera access in System Settings → Privacy & Security "
            "→ Camera, then rerun."
        )

    infer_every = max(1, int(infer_every))
    logger.info(
        "Camera opened. Inferring every %d frame(s). "
        "Keys: q/ESC quit, s smoothing, [ / ] change rate.",
        infer_every,
    )

    ema_probs: np.ndarray | None = None
    last_probs: np.ndarray | None = None   # held between inference frames
    last_raw: np.ndarray | None = None     # held between inference frames
    frame_idx = 0
    t_prev = cv2.getTickCount()
    fps = 0.0
    win = "Emotion Live Test"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Failed to read frame; stopping.")
                break

            # scale to display height
            h0, w0 = frame.shape[:2]
            scale = DISPLAY_H / h0
            frame = cv2.resize(frame, (int(round(w0 * scale)), DISPLAY_H))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            timestamp_ms = int(frame_idx * 33)  # strictly increasing
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            has_face = bool(result.face_landmarks)
            has_bs = bool(result.face_blendshapes)

            if has_face:
                # draw the live face mesh every frame so tracking stays smooth
                draw_face_mesh(frame, result.face_landmarks[0])
            else:
                # face lost: drop held values so the panel shows "no face"
                last_raw = None
                last_probs = None
                ema_probs = None

            # Throttle the model: only re-evaluate every `infer_every` frames.
            # Between updates the panel reuses the last (EMA-smoothed) result,
            # which is what removes the emotion jitter.
            if has_bs and frame_idx % infer_every == 0:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                last_raw = np.array(
                    [bs.get(name, 0.0) for name in features], dtype=np.float32
                )
                z = ((last_raw - mean) / std).reshape(1, -1).astype(np.float32)
                logits = session.run(None, {input_name: z})[0][0]
                probs = softmax(logits)

                if smoothing:
                    ema_probs = (
                        probs if ema_probs is None
                        else ema_alpha * probs + (1 - ema_alpha) * ema_probs
                    )
                    last_probs = ema_probs
                else:
                    ema_probs = None
                    last_probs = probs

            frame_idx += 1

            # mirror for a natural selfie view (mesh flips with the face)
            frame = cv2.flip(frame, 1)

            # FPS (smoothed)
            t_now = cv2.getTickCount()
            dt = (t_now - t_prev) / cv2.getTickFrequency()
            t_prev = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            panel = render_panel(
                frame.shape[0], features, last_raw, last_probs, labels,
                smoothing, fps, infer_every,
            )
            canvas = np.hstack([frame, panel])
            cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("s"):
                smoothing = not smoothing
                ema_probs = None
                logger.info("Smoothing %s", "ON" if smoothing else "OFF")
            if key == ord("]"):  # slower / calmer
                infer_every = min(30, infer_every + 1)
                logger.info("infer_every = %d", infer_every)
            if key == ord("["):  # faster / more responsive
                infer_every = max(1, infer_every - 1)
                logger.info("infer_every = %d", infer_every)
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        logger.info("Demo closed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam emotion-model test.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable EMA smoothing of the emotion output")
    parser.add_argument("--infer-every", type=int, default=5,
                        help="Run the model every N frames; higher = calmer / less "
                             "jitter (default 5). Tune live with [ and ].")
    parser.add_argument("--ema-alpha", type=float, default=0.35,
                        help="EMA smoothing factor 0-1; lower = smoother (default 0.35)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        camera_index=args.camera,
        smoothing=not args.no_smooth,
        ema_alpha=args.ema_alpha,
        infer_every=args.infer_every,
    )
