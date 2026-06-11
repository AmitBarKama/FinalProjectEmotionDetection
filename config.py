"""
Global configuration for the Emotion Recognition Pipeline.

All paths, constants, hyperparameters, and definitions used across
the pipeline steps are centralized here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Project Root
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────
# Data Directories
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
FFHQ_DIR = DATA_DIR / "ffhq_thumbnails"
EMOTION_LABELS_DIR = DATA_DIR / "emotion_labels"
BLENDSHAPES_DIR = DATA_DIR / "blendshapes"
COMBINED_DATASET_DIR = DATA_DIR / "combined_dataset"
PROCESSED_DIR = DATA_DIR / "processed"

# ──────────────────────────────────────────────────────────────────────
# Model Directories
# ──────────────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_DIR = MODELS_DIR / "exported"
EVALUATION_DIR = MODELS_DIR / "evaluation"

# ──────────────────────────────────────────────────────────────────────
# MediaPipe Model
# ──────────────────────────────────────────────────────────────────────
MEDIAPIPE_MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
MEDIAPIPE_MODEL_PATH = MEDIAPIPE_MODEL_DIR / "face_landmarker_v2_with_blendshapes.task"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ──────────────────────────────────────────────────────────────────────
# Create all directories
# ──────────────────────────────────────────────────────────────────────
for d in [
    FFHQ_DIR, EMOTION_LABELS_DIR, BLENDSHAPES_DIR,
    COMBINED_DATASET_DIR, PROCESSED_DIR,
    CHECKPOINTS_DIR, EXPORTED_DIR, EVALUATION_DIR,
    MEDIAPIPE_MODEL_DIR,
    EMOTION_LABELS_DIR / "analysis",
]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Emotion Definitions
# ──────────────────────────────────────────────────────────────────────
EMOTION_LABELS = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]
NUM_EMOTIONS = len(EMOTION_LABELS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTION_LABELS)}

# ──────────────────────────────────────────────────────────────────────
# MediaPipe Blendshape Names (52 ARKit-compatible)
# ──────────────────────────────────────────────────────────────────────
BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]
NUM_BLENDSHAPES = len(BLENDSHAPE_NAMES)

# ──────────────────────────────────────────────────────────────────────
# Dataset Generation Settings
# ──────────────────────────────────────────────────────────────────────
# EmotiEffLib / HSEmotion settings
EMOTION_MODEL_NAME = "enet_b0_8_best_afew"  # Best 7-class model
EMOTION_DEVICE = "cuda"  # Set to "cpu" if no GPU available

# Confidence filtering thresholds
CONFIDENCE_THRESHOLD = 0.5       # Minimum confidence to keep a sample
ENTROPY_THRESHOLD = 1.5          # Maximum entropy (lower = more certain)
HAPPINESS_CAP = 5000             # Cap the maximum number of Happiness samples to balance the dataset


# Batch processing
LABELING_BATCH_SIZE = 32         # Images processed per batch in emotion labeling
BLENDSHAPE_BATCH_SIZE = 1        # MediaPipe processes one image at a time

# ──────────────────────────────────────────────────────────────────────
# Feature Selection Settings
# ──────────────────────────────────────────────────────────────────────
FEATURE_SELECTION_TOP_K = 35     # Top K features to keep from each method
FEATURE_SELECTION_METHODS = [
    "mutual_information",
    "random_forest",
    "l1_regularization",
]

# ──────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 200
WEIGHT_DECAY = 1e-4

# MLP architecture
HIDDEN_DIMS = [128, 64, 32]
DROPOUT_RATES = [0.3, 0.3, 0.2]

# Label smoothing (used as fallback, primary approach is KL on soft labels)
LABEL_SMOOTHING = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# Learning rate scheduler
LR_SCHEDULER = "cosine"  # "cosine" | "step" | "plateau"
LR_MIN = 1e-6

# ──────────────────────────────────────────────────────────────────────
# ONNX Export Settings
# ──────────────────────────────────────────────────────────────────────
ONNX_OPSET_VERSION = 15
ONNX_MODEL_PATH = EXPORTED_DIR / "emotion_model.onnx"

# ──────────────────────────────────────────────────────────────────────
# File Paths for Intermediate Outputs
# ──────────────────────────────────────────────────────────────────────
EMOTION_LABELS_FILE = EMOTION_LABELS_DIR / "emotion_labels.parquet"
EMOTION_LABELS_FILTERED_FILE = EMOTION_LABELS_DIR / "emotion_labels_filtered.parquet"
BLENDSHAPES_FILE = BLENDSHAPES_DIR / "blendshapes.parquet"
COMBINED_DATASET_FILE = COMBINED_DATASET_DIR / "dataset.parquet"
FEATURE_SELECTION_REPORT = PROCESSED_DIR / "feature_selection_report.json"
SELECTED_FEATURES_FILE = PROCESSED_DIR / "selected_features.json"
NORMALIZATION_PARAMS_FILE = PROCESSED_DIR / "normalization_params.json"

# ──────────────────────────────────────────────────────────────────────
# Quality Gates
# ──────────────────────────────────────────────────────────────────────
MIN_DATASET_SIZE = 40_000
MIN_BLENDSHAPE_EXTRACTION_RATE = 0.90
MIN_TEACHER_STUDENT_AGREEMENT = 0.80
MIN_MACRO_F1 = 0.55
MAX_ONNX_SIZE_KB = 200
MAX_ONNX_INFERENCE_MS = 1.0
