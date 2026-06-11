# Unity Integration — Emotion Recognition from Blendshapes

Real-time emotion recognition in Unity using an ONNX model powered by
**Unity Sentis**. The pipeline takes 52 ARKit-compatible face blendshape
values (from MediaPipe or any ARKit source) and outputs probabilities
for 7 emotions:

| Index | Emotion   |
|-------|-----------|
| 0     | Anger     |
| 1     | Disgust   |
| 2     | Fear      |
| 3     | Happiness |
| 4     | Neutral   |
| 5     | Sadness   |
| 6     | Surprise  |

---

## Prerequisites

| Requirement         | Version       | Notes                                  |
|---------------------|---------------|----------------------------------------|
| Unity Editor        | **2023.2+**   | LTS recommended                        |
| Unity Sentis        | **2.x**       | Install via Package Manager            |
| ONNX Model          | Opset 15      | Exported by `step4_training/export.py`  |
| Normalization Params| JSON          | `data/processed/normalization_params.json` |

---

## Step-by-Step Setup

### 1. Install Unity Sentis

1. Open your Unity project.
2. Go to **Window → Package Manager**.
3. Click the **+** button → **Add package by name…**
4. Enter: `com.unity.sentis`
5. Click **Add**. Wait for installation to complete.

> **Tip:** If you're on Unity 6+, Sentis may already be pre-installed.
> Check the Package Manager's "In Project" tab.

### 2. Import the ONNX Model

1. Locate the exported model at:
   ```
   models/exported/emotion_model.onnx
   ```
2. Drag the `.onnx` file into your Unity project's `Assets/Models/`
   folder (create the folder if it doesn't exist).
3. Unity will automatically import it as a `ModelAsset`. Click on it in
   the Project window to verify the import settings.

### 3. Import the C# Scripts

Copy these scripts into your Unity project (e.g. `Assets/Scripts/Emotion/`):

- `EmotionInference.cs` — Core inference component
- `TemporalSmoother.cs` — Frame-to-frame smoothing

### 4. Add EmotionInference Component

1. Create an empty GameObject (or use your existing face-tracking object):
   **GameObject → Create Empty** → name it `EmotionSystem`.
2. In the Inspector, click **Add Component** → search for
   `EmotionInference`.
3. Assign the **Model Asset** field by dragging your imported ONNX model
   onto it.

### 5. Configure Normalization Parameters

The model expects z-score normalized inputs. The mean and standard
deviation arrays are saved during training at:

```
data/processed/normalization_params.json
```

This JSON file contains:

```json
{
  "mean": [0.123, 0.456, ...],
  "std":  [0.789, 0.012, ...]
}
```

**To populate the Inspector arrays:**

- **Option A (Manual):** Expand `Normalization Mean` and
  `Normalization Std` in the Inspector. Set the array size to match
  your feature count (52 or fewer if feature selection was applied),
  then enter each value.

- **Option B (Editor Script):** Use the helper below to auto-load from
  JSON at edit time:

```csharp
// NormalizationLoader.cs — place in an Editor/ folder
using UnityEngine;
using UnityEditor;
using System.IO;

public class NormalizationLoader : EditorWindow
{
    [MenuItem("Tools/Load Normalization Params")]
    static void LoadParams()
    {
        string path = EditorUtility.OpenFilePanel(
            "Select normalization_params.json", "", "json");
        if (string.IsNullOrEmpty(path)) return;

        string json = File.ReadAllText(path);
        var data = JsonUtility.FromJson<NormData>(json);

        var inference = FindObjectOfType<EmotionInference>();
        if (inference == null)
        {
            Debug.LogError("No EmotionInference component found in scene.");
            return;
        }

        // Use SerializedObject to write to private serialized fields
        var so = new SerializedObject(inference);
        SetArray(so, "normalizationMean", data.mean);
        SetArray(so, "normalizationStd", data.std);
        so.ApplyModifiedProperties();

        Debug.Log($"Loaded {data.mean.Length} normalization parameters.");
    }

    static void SetArray(SerializedObject so, string propName, float[] values)
    {
        var prop = so.FindProperty(propName);
        prop.arraySize = values.Length;
        for (int i = 0; i < values.Length; i++)
            prop.GetArrayElementAtIndex(i).floatValue = values[i];
    }

    [System.Serializable]
    class NormData { public float[] mean; public float[] std; }
}
```

### 6. Connect MediaPipe Blendshape Input

Feed blendshape values from your face-tracking solution into
`EmotionInference.ClassifyEmotion()`. The exact integration depends on
your tracking package:

**ARKit (via AR Foundation):**
```csharp
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

// In your face-tracking script:
ARFace face = ...; // from ARFaceManager
float[] blendshapes = new float[52];
// Map ARKit blend shape coefficients to the 52-element array
// in the order defined by EmotionInference.BlendshapeNames
```

**MediaPipe (via custom plugin or UDP stream):**
```csharp
// Parse the 52 blendshape values from your MediaPipe bridge
float[] blendshapes = mediaPipeBridge.GetBlendshapes();
```

### 7. Add Temporal Smoothing

1. On the same (or a nearby) GameObject, click **Add Component** →
   `TemporalSmoother`.
2. Configure in the Inspector:
   - **Window Size:** 5 (default) — increase for smoother output
   - **Smoothing Mode:** SMA or EMA
   - **EMA Alpha:** 0.2 (lower = smoother)

---

## Example Usage

```csharp
using UnityEngine;

public class EmotionDisplay : MonoBehaviour
{
    [SerializeField] private EmotionInference emotionInference;
    [SerializeField] private TemporalSmoother temporalSmoother;
    [SerializeField] private TMPro.TextMeshProUGUI emotionLabel;

    private void Update()
    {
        // 1. Get blendshapes from your face tracking source
        float[] blendshapes = GetBlendshapesFromTracker();
        if (blendshapes == null) return;

        // 2. Run inference
        float[] rawProbabilities = emotionInference.ClassifyEmotion(blendshapes);
        if (rawProbabilities == null) return;

        // 3. Apply temporal smoothing
        float[] smoothed = temporalSmoother.Smooth(rawProbabilities);

        // 4. Get the dominant emotion
        var dominant = emotionInference.GetDominantEmotion();
        float confidence = emotionInference.GetEmotionIntensity(dominant);

        // 5. Update UI
        emotionLabel.text = $"{dominant}: {confidence:P0}";

        // Or use the smoothed values for something else:
        // e.g., drive avatar blend shapes, trigger animations, etc.
        Debug.Log($"Dominant: {dominant} ({confidence:P1})");
    }

    private float[] GetBlendshapesFromTracker()
    {
        // TODO: Replace with your actual blendshape source
        return new float[52];
    }
}
```

---

## Performance Tips

### Model Size & Speed
- The exported MLP is **< 200 KB** — well within mobile budgets.
- Inference takes **< 1 ms** on CPU, making `BackendType.CPU` the
  optimal choice (avoids GPU synchronization overhead).

### Reducing Overhead
1. **Don't run every frame.** Emotions don't change at 60+ fps.
   Run inference every 3–5 frames and rely on the
   `TemporalSmoother` to interpolate:
   ```csharp
   if (Time.frameCount % 3 == 0)
   {
       float[] probs = emotionInference.ClassifyEmotion(blendshapes);
       smoothed = temporalSmoother.Smooth(probs);
   }
   ```
2. **Feature Selection:** If the training pipeline selected a subset of
   blendshapes (e.g. 35 out of 52), set `Input Feature Count` in the
   Inspector accordingly and only pass the selected features.
3. **Warm-up:** The first inference is automatically run in `Start()`
   to avoid a hitch on the first real frame.

### Memory
- The `TemporalSmoother` allocates a small circular buffer on first
  use. With `windowSize=5` and 7 emotions, that's only 140 bytes.
- Sentis `Worker` handles its own memory pool — just call `Dispose()`
  in `OnDestroy()` (already handled).

---

## Troubleshooting

### "No ModelAsset assigned"
**Cause:** The `Model Asset` field in the EmotionInference Inspector is
empty.
**Fix:** Drag the imported `.onnx` file from `Assets/Models/` onto the
field.

### "normalizationMean is not set"
**Cause:** The normalization arrays are empty.
**Fix:** Load values from `normalization_params.json` using the editor
script (see Step 5 above) or enter them manually.

### "Expected 52 blendshape values, got N"
**Cause:** Mismatch between your blendshape source and the model input
size.
**Fix:** Check `Input Feature Count` in the Inspector. If feature
selection reduced the count (e.g. to 35), update the field and ensure
you're passing only the selected features in the correct order.

### Model outputs look wrong / always Neutral
- Verify the normalization mean and std values match the ones from
  training. Incorrect normalization is the #1 cause of bad predictions.
- Ensure blendshape values are in **the same order** as
  `EmotionInference.BlendshapeNames`.
- Check that your ONNX model was exported with opset 15 (as configured
  in `config.py`).

### Sentis package not found
**Cause:** Unity version too old or Sentis not installed.
**Fix:** Update to Unity 2023.2+ and install `com.unity.sentis` from
the Package Manager.

### Performance spikes on first frame
This is handled by the warm-up inference in `Start()`. If you still see
a hitch, ensure `Start()` runs before your first `ClassifyEmotion()`
call (adjust Script Execution Order if needed).

---

## File Reference

| File                        | Description                                  |
|-----------------------------|----------------------------------------------|
| `EmotionInference.cs`       | Core inference MonoBehaviour (Sentis + ONNX)  |
| `TemporalSmoother.cs`       | Frame-to-frame probability smoothing         |
| `emotion_model.onnx`        | Trained MLP model (exported from PyTorch)     |
| `normalization_params.json` | Z-score mean/std from training data           |
| `selected_features.json`    | Feature subset (if feature selection applied) |

---

## Architecture Overview

```
┌──────────────────────┐
│  Face Tracking       │
│  (MediaPipe / ARKit) │
└──────────┬───────────┘
           │ 52 blendshape floats
           ▼
┌──────────────────────┐
│  EmotionInference    │
│  ┌────────────────┐  │
│  │ Z-Score Norm   │  │
│  │ (mean / std)   │  │
│  └───────┬────────┘  │
│          ▼           │
│  ┌────────────────┐  │
│  │ MLP (Sentis)   │  │
│  │ ONNX on CPU    │  │
│  └───────┬────────┘  │
│          ▼           │
│  ┌────────────────┐  │
│  │ Softmax        │  │
│  └───────┬────────┘  │
└──────────┼───────────┘
           │ 7 emotion probabilities
           ▼
┌──────────────────────┐
│  TemporalSmoother    │
│  (SMA / EMA)         │
└──────────┬───────────┘
           │ smoothed probabilities
           ▼
┌──────────────────────┐
│  Your Application    │
│  (UI, avatar, etc.)  │
└──────────────────────┘
```
