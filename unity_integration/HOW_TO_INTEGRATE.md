# Unity Sentis Integration Guide for Agent/Developer

This document details how to integrate and use the trained **`emotion_model.onnx`** model correctly in Unity using the **Unity Sentis** framework. 

---

## 1. Model Specifications

*   **Model Format:** ONNX (Opset 15)
*   **Location:** `models/exported/emotion_model.onnx`
*   **Model Architecture:** MLP (Multi-Layer Perceptron)
*   **Input Tensor Name:** `input`
*   **Input Tensor Shape:** `(1, 32)` — Batch size of 1, 32 selected facial blendshape features.
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

The ONNX model was trained using a subset of **32 select features** extracted via feature selection (rather than all 52 ARKit blendshapes). The model expects input values in the **exact sequence** below.

### Selected 32 Features (in order):
```json
[
  "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
  "cheekPuff", "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
  "eyeWideLeft", "eyeWideRight", "jawForward", "jawOpen", "mouthDimpleLeft",
  "mouthDimpleRight", "mouthFrownRight", "mouthFunnel", "mouthLowerDownLeft",
  "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
  "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthSmileLeft",
  "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
  "mouthUpperUpRight"
]
```

---

## 3. Z-Score Normalization Constants

The raw blendshape values ($x$) must be normalized using z-score normalization before passing them to the model:

$$z = \frac{x - \mu}{\sigma}$$

Here are the pre-calculated **Mean ($\mu$)** and **Std Dev ($\sigma$)** values for the 32 features in the exact corresponding order:

| Index | Feature Name | Mean ($\mu$) | Std ($\sigma$) |
| :--- | :--- | :--- | :--- |
| **0** | `browDownLeft` | `0.1584337` | `0.2091646` |
| **1** | `browDownRight` | `0.1598119` | `0.2067910` |
| **2** | `browInnerUp` | `0.0986554` | `0.1739993` |
| **3** | `browOuterUpLeft` | `0.1287214` | `0.1839892` |
| **4** | `browOuterUpRight` | `0.0973631` | `0.1520101` |
| **5** | `cheekPuff` | `0.0005417` | `0.0005100` |
| **6** | `eyeBlinkLeft` | `0.1651758` | `0.1183827` |
| **7** | `eyeBlinkRight` | `0.1341262` | `0.1116499` |
| **8** | `eyeSquintLeft` | `0.4710794` | `0.1847358` |
| **9** | `eyeSquintRight` | `0.4081563` | `0.1909593` |
| **10** | `eyeWideLeft` | `0.0139011` | `0.0262353` |
| **11** | `eyeWideRight` | `0.0178294` | `0.0390001` |
| **12** | `jawForward` | `0.0018469` | `0.0021216` |
| **13** | `jawOpen` | `0.0558740` | `0.0882164` |
| **14** | `mouthDimpleLeft` | `0.0189840` | `0.0394242` |
| **15** | `mouthDimpleRight` | `0.0235995` | `0.0398540` |
| **16** | `mouthFrownRight` | `0.0024705` | `0.0142826` |
| **17** | `mouthFunnel` | `0.0124883` | `0.0266607` |
| **18** | `mouthLowerDownLeft` | `0.1116235` | `0.1517227` |
| **19** | `mouthLowerDownRight` | `0.1679708` | `0.1980454` |
| **20** | `mouthPressLeft` | `0.0940272` | `0.0852945` |
| **21** | `mouthPressRight` | `0.1233680` | `0.1113784` |
| **22** | `mouthPucker` | `0.0249447` | `0.1022173` |
| **23** | `mouthRollLower` | `0.0148792` | `0.0441618` |
| **24** | `mouthRollUpper` | `0.0443501` | `0.0703641` |
| **25** | `mouthShrugLower` | `0.0281448` | `0.0878666` |
| **26** | `mouthSmileLeft` | `0.6929000` | `0.3009902` |
| **27** | `mouthSmileRight` | `0.6869228` | `0.3048645` |
| **28** | `mouthStretchLeft` | `0.0638041` | `0.1019856` |
| **29** | `mouthStretchRight` | `0.0787980` | `0.1185814` |
| **30** | `mouthUpperUpLeft` | `0.3130709` | `0.3008308` |
| **31** | `mouthUpperUpRight` | `0.3336166` | `0.3106966` |

---

## 4. Reference Integration Code

Create a wrapper script (e.g. `EmotionBridge.cs`) to map 52 ARKit blendshapes from your face tracker to the 32 features needed by `EmotionInference.cs`.

### `EmotionBridge.cs`
```csharp
using System.Collections.Generic;
using UnityEngine;

public class EmotionBridge : MonoBehaviour
{
    [Header("Components")]
    [SerializeField] private EmotionInference emotionInference;
    [SerializeField] private TemporalSmoother temporalSmoother;

    // Ordered list of the 32 features expected by the model
    private static readonly string[] SelectedFeatureNames = new string[]
    {
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight", "jawForward", "jawOpen", "mouthDimpleLeft",
        "mouthDimpleRight", "mouthFrownRight", "mouthFunnel", "mouthLowerDownLeft",
        "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthSmileLeft",
        "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
        "mouthUpperUpRight"
    };

    /// <summary>
    /// Processes standard 52 blendshape values, extracts the 32 selected features,
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

        // 1. Map 52 blendshapes to the 32 selected model inputs
        float[] modelInput = new float[32];
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
    *   Set **`Input Feature Count`** to `32` in the `EmotionInference` inspector.
4.  **Inject Normalization Parameters**:
    *   Use the **Normalization Loader** Editor script (detailed in `unity_integration/README.md`) to automatically parse `normalization_params.json` and fill `normalizationMean` and `normalizationStd` in the inspector.
5.  **Performance Tuning**:
    *   For performance, run inference every 3-5 frames (e.g. 15-20 times per second) instead of every single frame, as human emotions change slowly.
