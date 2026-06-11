// ═══════════════════════════════════════════════════════════════════════
// EmotionInference.cs
// Unity Sentis integration for real-time emotion recognition from
// MediaPipe / ARKit face blendshapes.
//
// Pipeline: Blendshapes (52 floats) → Z-Score Normalization →
//           MLP (ONNX via Sentis) → 7 Emotion Probabilities
//
// Emotions (index order): Anger, Disgust, Fear, Happiness,
//                         Neutral, Sadness, Surprise
// ═══════════════════════════════════════════════════════════════════════

using System;
using UnityEngine;
using Unity.Sentis;

/// <summary>
/// Performs real-time emotion classification from ARKit-compatible face
/// blendshape values using an ONNX model loaded through Unity Sentis.
/// <para>
/// Attach this component to any GameObject, assign the ONNX
/// <see cref="modelAsset"/>, and populate the normalization arrays
/// (mean / std) exported during training. Call
/// <see cref="ClassifyEmotion"/> each frame with fresh blendshape data.
/// </para>
/// </summary>
public class EmotionInference : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────
    // Emotion Definitions
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// The seven emotion categories predicted by the model, matching
    /// the label order used during training.
    /// </summary>
    public enum EmotionType
    {
        Anger    = 0,
        Disgust  = 1,
        Fear     = 2,
        Happiness = 3,
        Neutral  = 4,
        Sadness  = 5,
        Surprise = 6
    }

    /// <summary>Human-readable names parallel to <see cref="EmotionType"/>.</summary>
    public static readonly string[] EmotionNames =
    {
        "Anger", "Disgust", "Fear", "Happiness",
        "Neutral", "Sadness", "Surprise"
    };

    /// <summary>Total number of emotion classes.</summary>
    public const int NumEmotions = 7;

    // ─────────────────────────────────────────────────────────────────
    // ARKit / MediaPipe Blendshape Names (52 values)
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// The 52 blendshape names produced by MediaPipe FaceLandmarker,
    /// in the exact order expected by the model input tensor.
    /// The first entry (<c>_neutral</c>) is the base pose weight.
    /// </summary>
    public static readonly string[] BlendshapeNames =
    {
        "_neutral",
        "browDownLeft",      "browDownRight",     "browInnerUp",
        "browOuterUpLeft",   "browOuterUpRight",
        "cheekPuff",         "cheekSquintLeft",   "cheekSquintRight",
        "eyeBlinkLeft",      "eyeBlinkRight",
        "eyeLookDownLeft",   "eyeLookDownRight",
        "eyeLookInLeft",     "eyeLookInRight",
        "eyeLookOutLeft",    "eyeLookOutRight",
        "eyeLookUpLeft",     "eyeLookUpRight",
        "eyeSquintLeft",     "eyeSquintRight",
        "eyeWideLeft",       "eyeWideRight",
        "jawForward",        "jawLeft",           "jawOpen",
        "jawRight",
        "mouthClose",
        "mouthDimpleLeft",   "mouthDimpleRight",
        "mouthFrownLeft",    "mouthFrownRight",
        "mouthFunnel",       "mouthLeft",
        "mouthLowerDownLeft","mouthLowerDownRight",
        "mouthPressLeft",    "mouthPressRight",
        "mouthPucker",       "mouthRight",
        "mouthRollLower",    "mouthRollUpper",
        "mouthShrugLower",   "mouthShrugUpper",
        "mouthSmileLeft",    "mouthSmileRight",
        "mouthStretchLeft",  "mouthStretchRight",
        "mouthUpperUpLeft",  "mouthUpperUpRight",
        "noseSneerLeft",     "noseSneerRight"
    };

    /// <summary>Full blendshape vector length (all 52 ARKit shapes).</summary>
    public const int NumBlendshapes = 52;

    // ─────────────────────────────────────────────────────────────────
    // Serialized Fields (Inspector)
    // ─────────────────────────────────────────────────────────────────

    [Header("Model")]
    [Tooltip("The ONNX model asset exported from the training pipeline.")]
    [SerializeField] private ModelAsset modelAsset;

    [Header("Normalization Parameters")]
    [Tooltip("Per-feature mean values from training (length must match model input).")]
    [SerializeField] private float[] normalizationMean;

    [Tooltip("Per-feature standard deviation values from training (length must match model input).")]
    [SerializeField] private float[] normalizationStd;

    [Header("Options")]
    [Tooltip("Number of input features the model expects. Set to 52 for all " +
             "blendshapes, or fewer if feature selection was applied.")]
    [SerializeField] private int inputFeatureCount = NumBlendshapes;

    // ─────────────────────────────────────────────────────────────────
    // Runtime State
    // ─────────────────────────────────────────────────────────────────

    private Worker _worker;
    private float[] _outputBuffer;
    private bool _isReady;

    /// <summary>
    /// <c>true</c> once the model has been loaded and the warm-up
    /// inference has completed successfully.
    /// </summary>
    public bool IsReady => _isReady;

    // ─────────────────────────────────────────────────────────────────
    // Unity Lifecycle
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Loads the ONNX model, creates a Sentis <see cref="Worker"/>
    /// on the CPU backend, and performs a warm-up inference to prime
    /// the pipeline.
    /// </summary>
    private void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("[EmotionInference] No ModelAsset assigned. " +
                           "Drag your ONNX file into the inspector.");
            return;
        }

        // Validate normalization arrays
        if (!ValidateNormalization())
            return;

        try
        {
            // Load the ONNX model into a Sentis runtime model
            Model runtimeModel = ModelLoader.Load(modelAsset);

            // CPU backend is ideal for this tiny MLP — avoids GPU
            // sync overhead and keeps the main thread responsive.
            _worker = new Worker(runtimeModel, BackendType.CPU);
            _outputBuffer = new float[NumEmotions];

            // Warm-up: run a dummy inference so the first real call
            // doesn't pay JIT / allocation costs.
            WarmUp();

            _isReady = true;
            Debug.Log("[EmotionInference] Model loaded and warm-up complete.");
        }
        catch (Exception ex)
        {
            Debug.LogError($"[EmotionInference] Failed to load model: {ex.Message}");
        }
    }

    /// <summary>
    /// Releases Sentis resources to avoid memory leaks.
    /// </summary>
    private void OnDestroy()
    {
        _worker?.Dispose();
        _worker = null;
        _isReady = false;
    }

    // ─────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs the emotion classification model on raw blendshape values.
    /// <para>
    /// The input is z-score normalized using the stored training
    /// statistics before being fed to the model.
    /// </para>
    /// </summary>
    /// <param name="blendshapes">
    /// Raw blendshape values from MediaPipe / ARKit.  Length must
    /// equal <see cref="inputFeatureCount"/> (default 52).
    /// </param>
    /// <returns>
    /// Array of 7 floats representing the probability of each
    /// <see cref="EmotionType"/>. Values sum to ~1.0.
    /// Returns <c>null</c> if the model is not ready or the input
    /// length is invalid.
    /// </returns>
    public float[] ClassifyEmotion(float[] blendshapes)
    {
        if (!_isReady)
        {
            Debug.LogWarning("[EmotionInference] Model not ready.");
            return null;
        }

        if (blendshapes == null || blendshapes.Length != inputFeatureCount)
        {
            Debug.LogWarning($"[EmotionInference] Expected {inputFeatureCount} " +
                             $"blendshape values, got {blendshapes?.Length ?? 0}.");
            return null;
        }

        // ── 1. Normalize: z = (x - mean) / std ──────────────────────
        float[] normalized = new float[inputFeatureCount];
        for (int i = 0; i < inputFeatureCount; i++)
        {
            float std = normalizationStd[i];
            // Guard against division by zero (constant features)
            if (std < 1e-7f)
                std = 1e-7f;
            normalized[i] = (blendshapes[i] - normalizationMean[i]) / std;
        }

        // ── 2. Create input tensor (batch=1, features=N) ────────────
        using var inputTensor = new Tensor<float>(
            new TensorShape(1, inputFeatureCount), normalized);

        // ── 3. Run inference ─────────────────────────────────────────
        _worker.Schedule(inputTensor);

        // ── 4. Read output (batch=1, classes=7) ──────────────────────
        using var outputTensor = _worker.PeekOutput() as Tensor<float>;

        // Make the data readable on the CPU
        var cpuOutput = outputTensor.ReadbackAndClone();

        for (int i = 0; i < NumEmotions; i++)
        {
            _outputBuffer[i] = cpuOutput[i];
        }

        cpuOutput.Dispose();

        // ── 5. Apply softmax (if model doesn't include it) ──────────
        Softmax(_outputBuffer);

        return (float[])_outputBuffer.Clone();
    }

    /// <summary>
    /// Returns the <see cref="EmotionType"/> with the highest
    /// probability from the last classification.
    /// </summary>
    /// <returns>
    /// The dominant emotion, or <see cref="EmotionType.Neutral"/>
    /// if no inference has been performed yet.
    /// </returns>
    public EmotionType GetDominantEmotion()
    {
        if (_outputBuffer == null)
            return EmotionType.Neutral;

        int maxIdx = 0;
        float maxVal = _outputBuffer[0];
        for (int i = 1; i < NumEmotions; i++)
        {
            if (_outputBuffer[i] > maxVal)
            {
                maxVal = _outputBuffer[i];
                maxIdx = i;
            }
        }
        return (EmotionType)maxIdx;
    }

    /// <summary>
    /// Returns the predicted probability (0.0 – 1.0) for a specific
    /// emotion from the last classification.
    /// </summary>
    /// <param name="emotion">The emotion to query.</param>
    /// <returns>
    /// Probability value, or 0.0 if no inference has been performed.
    /// </returns>
    public float GetEmotionIntensity(EmotionType emotion)
    {
        if (_outputBuffer == null)
            return 0f;

        int idx = (int)emotion;
        if (idx < 0 || idx >= NumEmotions)
            return 0f;

        return _outputBuffer[idx];
    }

    /// <summary>
    /// Returns a formatted string showing all emotion probabilities.
    /// Useful for debugging overlays.
    /// </summary>
    public string GetEmotionSummary()
    {
        if (_outputBuffer == null)
            return "No inference performed yet.";

        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < NumEmotions; i++)
        {
            sb.AppendLine($"{EmotionNames[i]}: {_outputBuffer[i]:P1}");
        }
        return sb.ToString();
    }

    // ─────────────────────────────────────────────────────────────────
    // Private Helpers
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Validates that the normalization arrays are present and have
    /// the correct length.
    /// </summary>
    private bool ValidateNormalization()
    {
        if (normalizationMean == null || normalizationMean.Length == 0)
        {
            Debug.LogError("[EmotionInference] normalizationMean is not set. " +
                           "Populate it with values from normalization_params.json.");
            return false;
        }

        if (normalizationStd == null || normalizationStd.Length == 0)
        {
            Debug.LogError("[EmotionInference] normalizationStd is not set. " +
                           "Populate it with values from normalization_params.json.");
            return false;
        }

        if (normalizationMean.Length != inputFeatureCount)
        {
            Debug.LogError($"[EmotionInference] normalizationMean length " +
                           $"({normalizationMean.Length}) does not match " +
                           $"inputFeatureCount ({inputFeatureCount}).");
            return false;
        }

        if (normalizationStd.Length != inputFeatureCount)
        {
            Debug.LogError($"[EmotionInference] normalizationStd length " +
                           $"({normalizationStd.Length}) does not match " +
                           $"inputFeatureCount ({inputFeatureCount}).");
            return false;
        }

        return true;
    }

    /// <summary>
    /// Performs a single inference with zero-valued input to warm up
    /// the Sentis pipeline (allocations, JIT, etc.).
    /// </summary>
    private void WarmUp()
    {
        float[] dummy = new float[inputFeatureCount];
        // All zeros → after normalization this becomes -mean/std,
        // which is fine for a throwaway warm-up pass.

        using var inputTensor = new Tensor<float>(
            new TensorShape(1, inputFeatureCount), dummy);

        _worker.Schedule(inputTensor);

        // Read and discard the output to ensure the full pipeline
        // executes synchronously.
        using var outputTensor = _worker.PeekOutput() as Tensor<float>;
        var cpuOutput = outputTensor.ReadbackAndClone();
        cpuOutput.Dispose();

        Debug.Log("[EmotionInference] Warm-up inference completed.");
    }

    /// <summary>
    /// In-place softmax over the output buffer. Converts raw logits
    /// to probabilities that sum to 1.0.
    /// <para>
    /// If your ONNX model already includes a softmax output layer,
    /// this becomes a no-op numerically (softmax of probabilities
    /// sharpens slightly but remains valid). For best accuracy,
    /// ensure your export omits the final softmax so this method
    /// handles it.
    /// </para>
    /// </summary>
    private static void Softmax(float[] values)
    {
        // Find max for numerical stability
        float max = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > max) max = values[i];
        }

        // Exponentiate and sum
        float sum = 0f;
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Mathf.Exp(values[i] - max);
            sum += values[i];
        }

        // Normalize
        if (sum > 0f)
        {
            for (int i = 0; i < values.Length; i++)
                values[i] /= sum;
        }
    }
}
