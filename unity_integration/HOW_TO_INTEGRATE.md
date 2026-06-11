# Unity Sentis Integration Guide for Agent/Developer

This document details how to integrate and use the trained **`emotion_model.onnx`** model correctly in Unity using the **Unity Sentis** framework. 

---

## 1. Model Specifications

*   **Model Format:** ONNX (Opset 15/18)
*   **Location:** `models/exported/emotion_model.onnx`
*   **Model Architecture:** MLP (Multi-Layer Perceptron)
*   **Input Tensor Name:** `input`
*   **Input Tensor Shape:** `(1, 34)` — Batch size of 1, 34 selected facial blendshape features.
*   **Output Tensor Name:** `output` (or the default output node)
*   **Output Tensor Shape:** `(1, 7)` — Probabilities for 7 emotions:
    1.  `Anger`
    2.  `Disgust`
    3.  `Fear`
    4.  `Happiness`
    5.  `Neutral`
    6.  `Sadness`
    7.  `Surprise`

---

## 2. Feature Mapping & Sequence (CRITICAL)

The ONNX model was trained using a subset of **34 select features** extracted via feature selection (rather than all 52 ARKit blendshapes). The model expects input values in the **exact sequence** below.

### Selected 34 Features (in order):
```json
[
  "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
  "cheekPuff", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookUpLeft", "eyeSquintLeft",
  "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawOpen",
  "jawRight", "mouthDimpleRight", "mouthFrownRight", "mouthFunnel", "mouthLowerDownLeft",
  "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
  "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
  "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
  "mouthUpperUpLeft", "mouthUpperUpRight"
]
```

---

## 3. Z-Score Normalization Constants

The raw blendshape values ($x$) must be normalized using z-score normalization before passing them to the model:

$$z = \frac{x - \mu}{\sigma}$$

Because the mean ($\mu$) and standard deviation ($\sigma$) parameters are computed dynamically based on the final training split, **do not manually hardcode them**. 

Always load them directly from the exported [`data/processed/normalization_params.json`](file:///Users/amit/Final%20project/data/processed/normalization_params.json) file using the automated **Normalization Loader** Editor script (provided in `unity_integration/README.md`). This guarantees that your Unity runtime values match the training statistics perfectly.

---

## 4. Reference Integration Code

Create a wrapper script (e.g. `EmotionBridge.cs`) to map 52 standard ARKit blendshapes from your face tracker to the 34 features needed by `EmotionInference.cs`.

### `EmotionBridge.cs`
```csharp
using System.Collections.Generic;
using UnityEngine;

public class EmotionBridge : MonoBehaviour
{
    [Header("Components")]
    [SerializeField] private EmotionInference emotionInference;
    [SerializeField] private TemporalSmoother temporalSmoother;

    // Ordered list of the 34 features expected by the model
    private static readonly string[] SelectedFeatureNames = new string[]
    {
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookUpLeft", "eyeSquintLeft",
        "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawOpen",
        "jawRight", "mouthDimpleRight", "mouthFrownRight", "mouthFunnel", "mouthLowerDownLeft",
        "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
        "mouthUpperUpLeft", "mouthUpperUpRight"
    };

    /// <summary>
    /// Processes standard 52 blendshape values, extracts the 34 selected features,
    /// runs inference, and returns smoothed emotion probabilities.
    /// </summary>
    /// <param name="arkitBlendshapes">Dictionary of ARKit blendshape name -> value (0.0 to 1.0)</param>
    /// <returns>7 emotion probabilities (smoothed)</returns>
    public float[] ProcessFrame(Dictionary<string, float> arkitBlendshapes)
    {
        if (emotionInference == null || !emotionInference.IsReady)
        {
            return null;
        }

        // 1. Map 52 blendshapes to the 34 selected model inputs
        float[] modelInput = new float[34];
        for (int i = 0; i < SelectedFeatureNames.Length; i++)
        {
            string featureName = SelectedFeatureNames[i];
            if (arkitBlendshapes.TryGetValue(featureName, out float val))
            {
                modelInput[i] = val;
            }
            else
            {
                modelInput[i] = 0.0f; // Default if missing
                Debug.LogWarning($"[EmotionBridge] Missing expected blendshape: {featureName}");
            }
        }

        // 2. Classify raw emotion probabilities
        float[] rawProbabilities = emotionInference.ClassifyEmotion(modelInput);
        if (rawProbabilities == null) return null;

        // 3. Smooth temporal outputs (SMA/EMA) to avoid jittering
        if (temporalSmoother != null)
        {
            return temporalSmoother.Smooth(rawProbabilities);
        }

        return rawProbabilities;
    }
}
```

---

## 5. Setup Checklist for Agent/Developer

1.  **Unity Sentis Package**: Install `com.unity.sentis` from Package Manager.
2.  **Import Assets**:
    *   Drag `emotion_model.onnx` into `Assets/Models/`.
    *   Copy `EmotionInference.cs`, `TemporalSmoother.cs`, and `EmotionBridge.cs` to your scripts folder.
3.  **Scene Configuration**:
    *   Add `EmotionInference` and `TemporalSmoother` components to a GameObject.
    *   Drag the imported `emotion_model.onnx` into the `Model Asset` field of `EmotionInference`.
    *   Set **`Input Feature Count`** to **`34`** in the `EmotionInference` inspector.
4.  **Inject Normalization Parameters**:
    *   Use the **Normalization Loader** Editor script (detailed in `unity_integration/README.md`) to automatically parse `normalization_params.json` and fill `normalizationMean` and `normalizationStd` in the inspector.
5.  **Performance Tuning**:
    *   For performance, run inference every 3-5 frames (e.g. 15-20 times per second) instead of every single frame, as human emotions change slowly.
