using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.InferenceEngine;
using UnityEngine.Android;

/// <summary>
/// Runs the syringe keypoint model (YOLO-pose style, 4 keypoints) on the passthrough feed.
///
/// Quest 3 frame-rate strategy (same as <see cref="ARArmDetection.CustomArmDetector"/>):
///  - inference is layer-sliced via Worker.ScheduleIterable, so each rendered frame only
///    dispatches _layersPerFrame layers of GPU work instead of the whole 640x640 network
///    in one spike;
///  - the output readback is asynchronous (ReadbackRequest polled across frames), so the
///    CPU never blocks waiting on the GPU. The earlier "async never completed" attempt
///    failed because it re-Scheduled and re-Peeked the output every Update, invalidating
///    the tensor whose readback it was polling - the fix is to not start a new inference
///    while one is scheduled or awaiting readback, exactly as CustomArmDetector does;
///  - a duty-cycle gap between inferences keeps the GPU (and headset thermals) from
///    running the model back-to-back forever;
///  - between completed inferences the last detection state is served from the cached
///    public properties.
/// </summary>
public class CustomSyringeDetector : MonoBehaviour
{
    [Header("Model Configuration")]
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private BackendType _backend = BackendType.GPUCompute;
    [Tooltip("Only used when the model has a dynamic input shape; static ONNX exports declare " +
             "their own size and that is used instead.")]
    [SerializeField] private int _inputSize = 640;
    [Tooltip("Quantize model weights to FP16 at load. Halves weight memory and is faster on the " +
             "Quest GPU with no practical accuracy loss.")]
    [SerializeField] private bool _quantizeToFp16 = true;
    [Tooltip("Set low (0.05) so the spheres reliably APPEAR at all for pipeline verification, " +
             "even out-of-context (e.g. no practice arm in view) where confidence runs lower than " +
             "usual. This trades away some false-positive rejection - raise it back up (0.15-0.25) " +
             "once real detection is confirmed working end-to-end and you're tuning for accuracy " +
             "instead of just visibility.")]
    [SerializeField, Range(0f, 1f)] private float _confidenceThreshold = 0.05f;

    [Header("Performance")]
    [Tooltip("How many model layers to dispatch per rendered frame. Spreads the GPU cost of one " +
             "inference over several frames so the app holds native frame rate. Lower = smoother " +
             "frames but fewer detections per second. 0 = dispatch the whole model in one frame.")]
    [SerializeField, Range(0, 64)] private int _layersPerFrame = 14;
    [Tooltip("Minimum seconds between the END of one inference and the START of the next. " +
             "Caps the detector's GPU duty cycle so it doesn't run back-to-back and overheat " +
             "the headset. 0.2 s + the layer-sliced dispatch time ≈ 3-4 detections/s.")]
    [SerializeField] private float _inferenceInterval = 0.2f;
    [Tooltip("Dispose and rebuild the inference Worker this often (seconds). The InferenceEngine " +
             "GPU backend pools scratch memory it never hands back during a long session (observed " +
             "multi-GB growth on the arm model); recycling the worker returns it. The loaded model " +
             "is cached so the rebuild is cheap. 0 = never.")]
    [SerializeField] private float _workerRecycleSeconds = 25f;

    [Header("Debug")]
    [Tooltip("Logs HighestConfidence + IsSyringeDetected to Logcat every ~1 second, tagged " +
             "'[DetectorDebug]'. Use this to tell apart 'model genuinely sees nothing' (confidence " +
             "stays near 0.00-0.05) from 'model sees something but it's just under threshold' " +
             "(confidence bounces around below _confidenceThreshold) - especially useful when " +
             "testing without the practice arm, where the model may be seeing an out-of-distribution " +
             "scene it was never trained on.")]
    [SerializeField] private bool logDebugInfo = true;
    private float _lastDebugLogTime;

    [Header("Texture Input Source")]
    public ARArmDetection.PassthroughCameraSource cameraSourceComponent;

    private Model _model;   // loaded + quantized once, reused across worker recycles
    private Worker _worker;
    private Tensor<float> _inputTensor;
    private IEnumerator _scheduleSteps;
    private Tensor<float> _pendingOutput;
    private bool _readbackPending;
    private float _nextInferenceAllowedTime;
    private float _nextReadbackWarningTime;
    private float _workerCreatedTime;
    private int _tensorWidth;
    private int _tensorHeight;

    // GPU-side letterbox target: the raw camera texture (e.g. 1280x960, 4:3) is aspect-preserved
    // cropped/scaled into this square RT before conversion to a tensor, instead of being
    // squash-stretched directly into 640x640 (which distorts the syringe's proportions relative
    // to what the model was trained on).
    private RenderTexture _processedRT;

    // TextureConverter.ToTensor(Texture, Tensor) allocates a NEW CommandBuffer on every call and
    // never disposes it (InferenceEngine 2.6.1) - record into one reusable buffer instead.
    private CommandBuffer _toTensorCommandBuffer;

    // Cached so detection frames don't scan the scene. Re-looked-up on a slow retry timer while
    // null because facilitator mode can be bootstrapped after this component starts.
    private ARArmDetection.Facilitator.FacilitatorModeController _facilitator;
    private float _nextFacilitatorSearchTime;

    // Kill switch: stop running a model that throws on every inference instead of spinning
    // forever (log spam + GPU leak from aborted inferences). Same as CustomArmDetector.
    private const int MaxConsecutiveFailures = 8;
    private int _consecutiveFailures;
    private bool _permanentlyFailed;

    // Public properties that the visualizer script can read
    public bool IsSyringeDetected { get; private set; }
    public float HighestConfidence { get; private set; }
    public Vector2[] NormalizedKeypoints { get; private set; } = new Vector2[4];
    /// <summary>Per-keypoint confidence (the 3rd value of each x,y,conf triple), same order as
    /// NormalizedKeypoints. Was previously discarded entirely - exposed so callers (e.g.
    /// SyringeDebugHUD) can tell "unreliable point" apart from "correct point in an odd spot".</summary>
    public float[] KeypointConfidences { get; private set; } = new float[4];

    void Start()
    {
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
        }

        // Auto-discover the Camera Passthrough source if it wasn't dropped in the inspector
        if (cameraSourceComponent == null)
        {
            cameraSourceComponent = UnityEngine.Object.FindAnyObjectByType<ARArmDetection.PassthroughCameraSource>();
        }

        try
        {
            if (modelAsset != null)
            {
                LoadModel();
                Debug.Log($"CustomSyringeDetector: Neural Engine initialized successfully ({_tensorWidth}x{_tensorHeight}).");
            }
            else
            {
                Debug.LogError("CustomSyringeDetector: Missing Model Asset! Please attach your .onnx model.");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"CustomSyringeDetector: Neural Engine crashed during initialization! Error details: {e.Message}");
        }
    }

    private void LoadModel()
    {
        var model = ModelLoader.Load(modelAsset);

        // The ONNX's declared input size is authoritative; the Inspector value is only a
        // fallback for dynamic-shape models.
        _tensorWidth = _inputSize;
        _tensorHeight = _inputSize;
        if (model.inputs.Count > 0 && model.inputs[0].shape.IsStatic())
        {
            var declared = model.inputs[0].shape.ToTensorShape();
            if (declared.rank == 4 && declared[2] > 0 && declared[3] > 0)
            {
                _tensorHeight = declared[2];
                _tensorWidth = declared[3];
            }
        }

        if (_quantizeToFp16)
            ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);

        _model = model;
        CreateWorker();

        _processedRT = new RenderTexture(_tensorWidth, _tensorHeight, 0, RenderTextureFormat.ARGB32);
        _processedRT.Create();
    }

    private void CreateWorker()
    {
        _worker = new Worker(_model, _backend);
        _inputTensor = new Tensor<float>(new TensorShape(1, 3, _tensorHeight, _tensorWidth));
        _workerCreatedTime = Time.unscaledTime;
    }

    void Update()
    {
        if (_permanentlyFailed || _worker == null || _inputTensor == null || cameraSourceComponent == null) return;

        try
        {
            // 1. An async readback is in flight: poll it, never block. Cached detection
            //    state keeps serving consumers meanwhile.
            if (_readbackPending)
            {
                TryConsumeReadback();
                return;
            }

            // 2. An inference is mid-dispatch: push a few more layers this frame.
            if (_scheduleSteps != null)
            {
                if (!AdvanceSchedule())
                    BeginReadback();
                return;
            }

            // 3. Idle: respect the duty-cycle gap before starting the next inference.
            if (Time.unscaledTime < _nextInferenceAllowedTime) return;

            // Safe point to recycle the worker (nothing scheduled or awaiting readback).
            RecycleWorkerIfDue();

            Texture liveTex = cameraSourceComponent.CurrentTexture;
            if (liveTex == null) return;

            WriteCroppedInput(liveTex);

            if (_layersPerFrame > 0)
            {
                _scheduleSteps = _worker.ScheduleIterable(_inputTensor);
                if (AdvanceSchedule()) return;   // more layers next frame
            }
            else
            {
                _worker.Schedule(_inputTensor);
            }

            BeginReadback();
        }
        catch (Exception ex)
        {
            _scheduleSteps = null;
            _readbackPending = false;
            _pendingOutput = null;

            if (++_consecutiveFailures >= MaxConsecutiveFailures)
            {
                _permanentlyFailed = true;
                Debug.LogError($"[CustomSyringeDetector] Permanently disabled after {_consecutiveFailures} " +
                               $"consecutive failures. Last error: {ex}");
                OnDisableCleanup();
            }
            else
            {
                Debug.LogError($"[CustomSyringeDetector] Inference failed ({_consecutiveFailures}/{MaxConsecutiveFailures}): {ex}");
            }
        }
    }

    /// <summary>
    /// Aspect-preserving center-crop of the camera frame into the square model input
    /// (no stretching), then tensor conversion via a reusable CommandBuffer.
    /// </summary>
    private void WriteCroppedInput(Texture liveTex)
    {
        float textureAspect = (float)liveTex.width / liveTex.height;

        Vector2 scale = Vector2.one;
        Vector2 offset = Vector2.zero;
        if (textureAspect > 1f)
        {
            scale.x = 1f / textureAspect;
            offset.x = (1f - scale.x) / 2f;
        }
        else if (textureAspect < 1f)
        {
            scale.y = textureAspect;
            offset.y = (1f - scale.y) / 2f;
        }

        Graphics.Blit(liveTex, _processedRT, scale, offset);

        if (SystemInfo.supportsComputeShaders)
        {
            _toTensorCommandBuffer ??= new CommandBuffer { name = "CustomSyringeDetector.ToTensor" };
            _toTensorCommandBuffer.Clear();
            _toTensorCommandBuffer.ToTensor(_processedRT, _inputTensor);
            Graphics.ExecuteCommandBuffer(_toTensorCommandBuffer);
        }
        else
        {
            TextureConverter.ToTensor(_processedRT, _inputTensor, new TextureTransform());
        }
    }

    /// <summary>Dispatches up to _layersPerFrame layers. Returns true while layers remain.</summary>
    private bool AdvanceSchedule()
    {
        int budget = Mathf.Max(1, _layersPerFrame);
        for (int i = 0; i < budget; i++)
        {
            if (!_scheduleSteps.MoveNext())
            {
                _scheduleSteps = null;
                return false;
            }
        }
        return true;
    }

    private void BeginReadback()
    {
        _pendingOutput = _worker.PeekOutput() as Tensor<float>;
        if (_pendingOutput == null) return;

        _pendingOutput.ReadbackRequest();
        _readbackPending = true;
    }

    private void TryConsumeReadback()
    {
        if (_pendingOutput == null)
        {
            _readbackPending = false;
            return;
        }

        if (!_pendingOutput.IsReadbackRequestDone()) return;

        // IsReadbackRequestDone() only reports .done, not .hasError - a readback that finished
        // WITH an error still counts as done and ReadbackAndClone() throws (common on Quest).
        // A second ReadbackAndClone() issues a fresh blocking readback of the same still-valid
        // buffer, recovering the frame; if that also fails we drop it and ride the cached state.
        Tensor<float> cpuOutput = null;
        try
        {
            cpuOutput = _pendingOutput.ReadbackAndClone();
        }
        catch (Exception ex)
        {
            if (Time.unscaledTime >= _nextReadbackWarningTime)
            {
                _nextReadbackWarningTime = Time.unscaledTime + 5f;
                Debug.LogWarning($"[CustomSyringeDetector] Async readback failed ({ex.Message}); retrying synchronously.");
            }
            try { cpuOutput = _pendingOutput.ReadbackAndClone(); }
            catch { /* drop this frame; next inference starts after the duty-cycle gap */ }
        }

        if (cpuOutput != null)
        {
            using (cpuOutput)
                ParseSyringeOutput(cpuOutput);
            _consecutiveFailures = 0;
        }

        _pendingOutput = null;
        _readbackPending = false;
        _nextInferenceAllowedTime = Time.unscaledTime + Mathf.Max(0f, _inferenceInterval);
    }

    /// <summary>
    /// Periodically disposes and rebuilds the Worker to release GPU scratch memory the
    /// InferenceEngine backend pools but never returns during a long continuous run.
    /// Only called when the pipeline is idle so no in-flight work is lost.
    /// </summary>
    private void RecycleWorkerIfDue()
    {
        if (_workerRecycleSeconds <= 0f || _model == null) return;
        if (Time.unscaledTime - _workerCreatedTime < _workerRecycleSeconds) return;

        _worker?.Dispose();
        _inputTensor?.Dispose();
        CreateWorker();
    }

    private void ParseSyringeOutput(Tensor<float> output)
    {
        // The tensor is already a CPU clone (ReadbackAndClone), so read it in place -
        // DownloadToArray would copy the whole output into a fresh managed array every
        // inference and feed the GC for nothing.
        ReadOnlySpan<float> predictionData = output.AsReadOnlySpan();

        int totalAnchors = output.shape[2]; // 8400

        int bestAnchorIndex = 0;
        HighestConfidence = -1f;

        for (int i = 0; i < totalAnchors; i++)
        {
            float confidence = predictionData[4 * totalAnchors + i]; // Feature index 4 is object confidence
            if (confidence > HighestConfidence)
            {
                HighestConfidence = confidence;
                bestAnchorIndex = i;
            }
        }

        if (HighestConfidence > _confidenceThreshold) // Threshold for detection
        {
            IsSyringeDetected = true;
            int kptOffset = 5; // Keypoints start at feature index 5

            for (int j = 0; j < 4; j++)
            {
                // Extract using features-first stride math
                float pixelX = predictionData[(kptOffset + j * 3) * totalAnchors + bestAnchorIndex];
                float pixelY = predictionData[(kptOffset + 1 + j * 3) * totalAnchors + bestAnchorIndex];
                float kptConf = predictionData[(kptOffset + 2 + j * 3) * totalAnchors + bestAnchorIndex];

                // Save clean normalized 0.0 to 1.0 coordinates
                NormalizedKeypoints[j] = new Vector2(pixelX / _tensorWidth, pixelY / _tensorHeight);
                KeypointConfidences[j] = kptConf;
            }

            NotifyFacilitator();
        }
        else
        {
            IsSyringeDetected = false;
        }

        if (logDebugInfo && Time.time - _lastDebugLogTime >= 1f)
        {
            _lastDebugLogTime = Time.time;
            Debug.Log($"[DetectorDebug] HighestConfidence={HighestConfidence:F4} threshold={_confidenceThreshold:F2} " +
                       $"IsSyringeDetected={IsSyringeDetected}");
        }
    }

    // if a step requires an external syringe placement verification hook.
    private void NotifyFacilitator()
    {
        if (_facilitator == null)
        {
            // Scene scans are too expensive to run on every detection; retry on a slow
            // timer because facilitator mode can be bootstrapped after this scene loads.
            if (Time.unscaledTime < _nextFacilitatorSearchTime) return;
            _nextFacilitatorSearchTime = Time.unscaledTime + 2f;
            _facilitator = UnityEngine.Object.FindAnyObjectByType<ARArmDetection.Facilitator.FacilitatorModeController>();
            if (_facilitator == null) return;
        }

        _facilitator.SignalCompletion("SyringeDetected");
    }

    private void OnDisable() => OnDisableCleanup();

    private void OnDisableCleanup()
    {
        _scheduleSteps = null;
        _readbackPending = false;
        _pendingOutput = null;
        _worker?.Dispose();
        _worker = null;
        _inputTensor?.Dispose();
        _inputTensor = null;
        _toTensorCommandBuffer?.Dispose();
        _toTensorCommandBuffer = null;

        if (_processedRT != null)
        {
            _processedRT.Release();
            Destroy(_processedRT);
            _processedRT = null;
        }
    }
}
