// ═══════════════════════════════════════════════════════════════════════
// TemporalSmoother.cs
// Smooths per-frame emotion probability outputs to prevent jarring
// transitions in real-time applications.
//
// Supports two modes:
//   1. Simple Moving Average (SMA) – equal weight over a window
//   2. Exponential Moving Average (EMA) – recent frames weighted more
//
// Thread-safe: all public methods are protected by a lock, so the
// smoother can be fed from a background thread if needed.
// ═══════════════════════════════════════════════════════════════════════

using System;
using UnityEngine;

/// <summary>
/// Temporal smoother for float-array signals (e.g. emotion
/// probabilities). Maintains a sliding window of recent values and
/// returns a smoothed result on each call to <see cref="Smooth"/>.
/// <para>
/// Attach to the same GameObject as <see cref="EmotionInference"/>
/// or use as a standalone utility.
/// </para>
/// </summary>
public class TemporalSmoother : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────
    // Smoothing Mode
    // ─────────────────────────────────────────────────────────────────

    /// <summary>Selects the smoothing algorithm.</summary>
    public enum SmoothingMode
    {
        /// <summary>
        /// Simple Moving Average – unweighted mean of the last
        /// <see cref="windowSize"/> frames.
        /// </summary>
        SMA,

        /// <summary>
        /// Exponential Moving Average – blends the current sample
        /// with the previous smoothed value using
        /// <see cref="emaAlpha"/>.
        /// </summary>
        EMA
    }

    // ─────────────────────────────────────────────────────────────────
    // Serialized Fields (Inspector)
    // ─────────────────────────────────────────────────────────────────

    [Header("Smoothing Configuration")]
    [Tooltip("Number of past frames to consider for SMA. " +
             "Higher = smoother but more latent.")]
    [Range(1, 30)]
    [SerializeField] private int windowSize = 5;

    [Tooltip("Smoothing algorithm to use.")]
    [SerializeField] private SmoothingMode smoothingMode = SmoothingMode.SMA;

    [Tooltip("EMA blending factor (0–1). Lower = smoother. " +
             "Typical values: 0.1–0.3 for gentle smoothing.")]
    [Range(0.01f, 1.0f)]
    [SerializeField] private float emaAlpha = 0.2f;

    // ─────────────────────────────────────────────────────────────────
    // Runtime State
    // ─────────────────────────────────────────────────────────────────

    /// <summary>Circular buffer of recent raw probability snapshots.</summary>
    private float[][] _buffer;

    /// <summary>Write position in the circular buffer.</summary>
    private int _head;

    /// <summary>Number of frames stored so far (up to windowSize).</summary>
    private int _count;

    /// <summary>Length of each probability vector (set on first call).</summary>
    private int _vectorLength;

    /// <summary>EMA accumulator (only used in EMA mode).</summary>
    private float[] _emaState;

    /// <summary>Thread-safety lock.</summary>
    private readonly object _lock = new object();

    // ─────────────────────────────────────────────────────────────────
    // Public Properties
    // ─────────────────────────────────────────────────────────────────

    /// <summary>Current window size.</summary>
    public int WindowSize
    {
        get => windowSize;
        set
        {
            if (value < 1)
                throw new ArgumentOutOfRangeException(nameof(value),
                    "Window size must be at least 1.");
            windowSize = value;
            Reset();
        }
    }

    /// <summary>Current smoothing mode.</summary>
    public SmoothingMode Mode
    {
        get => smoothingMode;
        set
        {
            smoothingMode = value;
            Reset();
        }
    }

    /// <summary>EMA alpha (blending factor).</summary>
    public float EmaAlpha
    {
        get => emaAlpha;
        set => emaAlpha = Mathf.Clamp(value, 0.01f, 1.0f);
    }

    // ─────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Feeds a new raw probability vector into the smoother and
    /// returns the temporally smoothed result.
    /// </summary>
    /// <param name="rawProbabilities">
    /// Raw per-class probabilities from the current frame.
    /// Length must be consistent across calls.
    /// </param>
    /// <returns>
    /// Smoothed probability array of the same length. The caller
    /// receives a new array (safe to cache or modify).
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown if <paramref name="rawProbabilities"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown if the vector length changes between calls without a
    /// <see cref="Reset"/>.
    /// </exception>
    public float[] Smooth(float[] rawProbabilities)
    {
        if (rawProbabilities == null)
            throw new ArgumentNullException(nameof(rawProbabilities));

        lock (_lock)
        {
            // First call – initialize buffers
            if (_buffer == null || _vectorLength == 0)
            {
                InitializeBuffers(rawProbabilities.Length);
            }

            // Safety check: vector length must not change mid-stream
            if (rawProbabilities.Length != _vectorLength)
            {
                throw new ArgumentException(
                    $"Expected vector of length {_vectorLength}, " +
                    $"got {rawProbabilities.Length}. Call Reset() first " +
                    "if the input dimensionality changed.");
            }

            switch (smoothingMode)
            {
                case SmoothingMode.SMA:
                    return SmoothSMA(rawProbabilities);
                case SmoothingMode.EMA:
                    return SmoothEMA(rawProbabilities);
                default:
                    return (float[])rawProbabilities.Clone();
            }
        }
    }

    /// <summary>
    /// Clears all internal history. Call this when switching scenes,
    /// resetting tracking, or changing configuration.
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _buffer = null;
            _emaState = null;
            _head = 0;
            _count = 0;
            _vectorLength = 0;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Private Helpers
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Allocates the circular buffer and EMA state for a given
    /// vector length.
    /// </summary>
    private void InitializeBuffers(int vectorLength)
    {
        _vectorLength = vectorLength;
        _buffer = new float[windowSize][];
        for (int i = 0; i < windowSize; i++)
            _buffer[i] = new float[vectorLength];

        _emaState = new float[vectorLength];
        _head = 0;
        _count = 0;
    }

    /// <summary>
    /// Simple Moving Average: stores the raw input in a circular
    /// buffer and returns the element-wise mean of all stored frames.
    /// </summary>
    private float[] SmoothSMA(float[] raw)
    {
        // Copy raw values into the current buffer slot
        Array.Copy(raw, _buffer[_head], _vectorLength);

        // Advance the circular write head
        _head = (_head + 1) % windowSize;
        _count = Math.Min(_count + 1, windowSize);

        // Compute element-wise average over filled slots
        float[] result = new float[_vectorLength];
        for (int slot = 0; slot < _count; slot++)
        {
            for (int j = 0; j < _vectorLength; j++)
            {
                result[j] += _buffer[slot][j];
            }
        }

        float invCount = 1f / _count;
        for (int j = 0; j < _vectorLength; j++)
        {
            result[j] *= invCount;
        }

        return result;
    }

    /// <summary>
    /// Exponential Moving Average:
    /// <c>state = alpha * raw + (1 - alpha) * state</c>.
    /// On the first frame the state is initialized to the raw input.
    /// </summary>
    private float[] SmoothEMA(float[] raw)
    {
        // Also store in the circular buffer (useful for debugging /
        // potential future introspection)
        Array.Copy(raw, _buffer[_head], _vectorLength);
        _head = (_head + 1) % windowSize;
        _count = Math.Min(_count + 1, windowSize);

        if (_count == 1)
        {
            // First frame: initialize EMA state directly
            Array.Copy(raw, _emaState, _vectorLength);
        }
        else
        {
            float oneMinusAlpha = 1f - emaAlpha;
            for (int j = 0; j < _vectorLength; j++)
            {
                _emaState[j] = emaAlpha * raw[j] + oneMinusAlpha * _emaState[j];
            }
        }

        return (float[])_emaState.Clone();
    }
}
