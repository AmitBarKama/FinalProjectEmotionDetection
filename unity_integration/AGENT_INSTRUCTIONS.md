# AI Developer Guide: Unity Sentis Emotion Model Integration

This document details how to load and run the lightweight MLP emotion recognition model (`emotion_model.onnx`) inside your Unity project.

---

## 1. Files to Import
Transfer these **5 files** from the Python workspace into your Unity Assets folder:

1.  **ONNX Model:** `models/exported/emotion_model.onnx` -> Save to `Assets/Models/`
2.  **Mean/Std Normalization Params:** `data/processed/normalization_params.json` -> Save to `Assets/` (or your Editor folder)
3.  **Inference Script:** `unity_integration/EmotionInference.cs` -> Save to `Assets/Scripts/`
4.  **Smoothing Script:** `unity_integration/TemporalSmoother.cs` -> Save to `Assets/Scripts/`
5.  **Integration Helper:** `unity_integration/HOW_TO_INTEGRATE.md` -> (For your reference)

---

## 2. Model Specifications & Mechanics

*   **Format:** ONNX (Opset 15/18 compatible with Unity Sentis 2.0+).
*   **Backend:** Running on `BackendType.CPU` is recommended for this lightweight MLP (~15K parameters) to avoid GPU synchronization latency.
*   **Input Name:** `input`
*   **Input Shape:** `(1, 34)` — 34 selected facial blendshapes (rather than all 52).
*   **Output Name:** `output` (or default output node)
*   **Output Shape:** `(1, 7)` — Float logits corresponding to 7 emotions in this exact index order:
    `0: Anger`, `1: Disgust`, `2: Fear`, `3: Happiness`, `4: Neutral`, `5: Sadness`, `6: Surprise`

---

## 3. Input Sequence (34 Selected Blendshapes)
The model expects raw blendshape inputs in this **exact order**. If you change this order, inputs will be scrambled, and predictions will fail.

```csharp
private static readonly string[] SelectedFeatureNames = new string[]
{
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookUpLeft", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawOpen",
    "jawRight", "mouthDimpleRight", "mouthFrownRight", "mouthFunnel", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight"
};
```

---

## 4. Normalization and Pre-processing
Raw blendshape inputs ($x$) must be normalized using z-score normalization before creating the input tensor:
$$z = \frac{x - \mu}{\sigma}$$
*   **Mean ($\mu$)** and **Std Dev ($\sigma$)** arrays are stored in `normalization_params.json`.
*   Use the **Normalization Loader** Editor script (found in `unity_integration/README.md`) to automatically inject these 34 mean/std values into your `EmotionInference` inspector arrays. **Do not leave these arrays at zero.**

---

## 5. C# Bridge Script Template (`EmotionBridge.cs`)
Attach this script to your face tracking object to extract the 34 features from your tracker's 52-blendshape stream, feed them to the inference engine, and smooth the output.

```csharp
using System.Collections.Generic;
using UnityEngine;

public class EmotionBridge : MonoBehaviour
{
    [SerializeField] private EmotionInference emotionInference;
    [SerializeField] private TemporalSmoother temporalSmoother;

    // The exact 34 features expected by the model
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
    /// Call this inside your tracking Update loop.
    /// </summary>
    /// <param name="arkitBlendshapes">Dictionary mapping ARKit name to value [0.0 - 1.0]</param>
    public float[] ProcessFrame(Dictionary<string, float> arkitBlendshapes)
    {
        if (emotionInference == null || !emotionInference.IsReady) return null;

        // 1. Map tracking values to 34 ordered inputs
        float[] modelInput = new float[34];
        for (int i = 0; i < SelectedFeatureNames.Length; i++)
        {
            string featureName = SelectedFeatureNames[i];
            if (arkitBlendshapes.TryGetValue(featureName, out float val))
                modelInput[i] = val;
            else
                modelInput[i] = 0.0f; // Default if tracking loses features
        }

        // 2. Classify raw probabilities
        float[] rawProbabilities = emotionInference.ClassifyEmotion(modelInput);
        if (rawProbabilities == null) return null;

        // 3. Apply moving window smoothing
        if (temporalSmoother != null)
            return temporalSmoother.Smooth(rawProbabilities);

        return rawProbabilities;
    }
}
```

---

## 6. Sentis 2.0+ Tensor Output Handling
When reading model outputs, **do not index the GPU tensor directly** (`cpuOutput[i]` is deprecated for 2D shapes). Instead, retrieve the data using `DownloadToArray()` to download the data to a CPU array safely:

```csharp
// Inside EmotionInference.cs:
var cpuOutput = outputTensor.ReadbackAndClone();
float[] outputArray = cpuOutput.DownloadToArray(); // Safe flat CPU array copy

for (int i = 0; i < NumEmotions; i++)
{
    _outputBuffer[i] = outputArray[i]; // Store in local buffer
}
cpuOutput.Dispose();
```

---

## 7. Step-by-Step Integration Checklist
1.  Verify the `com.unity.sentis` package is installed in your project.
2.  Import `emotion_model.onnx` into your project and drag it into the **`Model Asset`** field of the `EmotionInference` inspector.
3.  Set **`Input Feature Count`** to **`34`** in the `EmotionInference` inspector.
4.  Run the normalisation loader script to import `normalization_params.json` into the inspector.
5.  Call `emotionInference.ClassifyEmotion()` every 3 to 5 frames (emotions do not need 60Hz updates; temporal smoothing handles frame interpolation).
