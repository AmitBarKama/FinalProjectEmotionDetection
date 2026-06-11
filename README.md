# 🎮 Real-Time Emotion Recognition from MediaPipe Blendshapes

A complete pipeline for training a lightweight neural network that predicts player emotions
from webcam-captured facial blendshapes, designed for adaptive horror game experiences in Unity.

## Architecture

```
┌─────────────────── OFFLINE PIPELINE (Python) ───────────────────┐
│                                                                  │
│  FFHQ Images ──→ EmotiEffLib ──→ Emotion Labels                 │
│       │                              │                           │
│       └──→ MediaPipe ──→ Blendshapes │                           │
│                              │       │                           │
│                              └───┬───┘                           │
│                                  ▼                               │
│                          Combined Dataset                        │
│                                  │                               │
│                          Feature Selection                       │
│                                  │                               │
│                          MLP Training (PyTorch)                  │
│                                  │                               │
│                          ONNX Export                              │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────── RUNTIME PIPELINE (Unity) ────────────────────┐
│                                                                  │
│  Webcam → MediaPipe → Blendshapes → ONNX Model → Emotions      │
│                                                     │            │
│                                          Temporal Smoothing      │
│                                                     │            │
│                                          Game Adaptation         │
└──────────────────────────────────────────────────────────────────┘
```

> **Note:** The emotion classifier (EmotiEffLib) is used **only during dataset generation**
> and is NOT included in the final Unity runtime system.

## Project Structure

```
├── config.py                          # Global configuration
├── requirements.txt                   # Python dependencies
│
├── step1_dataset/                     # FFHQ download & verification
├── step2_emotion_labeling/            # Automatic emotion classification
├── step3_blendshape_extraction/       # MediaPipe blendshape extraction
├── step4_training/                    # Model training & export
│
├── data/                              # Generated data (gitignored)
├── models/                            # Saved models & exports
├── unity_integration/                 # C# scripts for Unity
└── notebooks/                         # Jupyter notebooks
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download FFHQ Dataset

```bash
python -m step1_dataset.download_ffhq
python -m step1_dataset.verify_dataset
```

### 3. Generate Emotion Labels

```bash
python -m step2_emotion_labeling.label_emotions
python -m step2_emotion_labeling.analyze_labels
python -m step2_emotion_labeling.filter_labels
```

### 4. Extract Blendshapes & Build Dataset

```bash
python -m step3_blendshape_extraction.extract_blendshapes
python -m step3_blendshape_extraction.validate_blendshapes
python -m step3_blendshape_extraction.build_dataset
```

### 5. Train & Export Model

```bash
python -m step4_training.feature_selection
python -m step4_training.train
python -m step4_training.evaluate
python -m step4_training.export_onnx
```

### 6. Deploy to Unity

Copy `models/exported/emotion_model.onnx` to your Unity project and use the
scripts in `unity_integration/` as reference implementations.

## Target Emotions

| Emotion   | Description                    |
|-----------|--------------------------------|
| Anger     | Frustration, irritation        |
| Disgust   | Revulsion, distaste            |
| Fear      | Anxiety, dread                 |
| Happiness | Joy, amusement                 |
| Neutral   | Baseline, no strong emotion    |
| Sadness   | Sorrow, disappointment         |
| Surprise  | Startle, astonishment          |

## Runtime Requirements

- Very low inference latency (<1ms on CPU)
- No PCA or expensive runtime transformations
- Emotion probabilities interpretable as intensities
- Temporal smoothing via configurable moving window

## License

This project uses:
- FFHQ Dataset: Creative Commons BY-NC-SA 4.0
- EmotiEffLib: Apache-2.0 (offline use only)
- MediaPipe: Apache-2.0
