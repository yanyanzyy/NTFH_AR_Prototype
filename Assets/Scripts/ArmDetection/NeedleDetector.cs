using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace ARArmDetection
{
    /// <summary>
    /// Runs the NEEDLE pose model (YOLO11n-pose) exported from Ultralytics, alongside
    /// the arm-only model in <see cref="CustomArmDetector"/>. Only the needle class is
    /// read from it — arms keep coming from the (better) dedicated arm model.
    ///
    /// Default layout matches Assets/Models/arm-needle-pose-320.onnx (== lucas.onnx),
    /// features-first [1, 18, N]:
    ///   0..3   box cx, cy, w, h   (input-pixel scale, letterboxed)
    ///   4..5   class scores       {0: arm, 1: needle}
    ///   6..17  4 kpts (kx,ky,v)   needle: kpt0 = tip (contact point), kpt3 = hub/plunger
    /// The tip is always kpt0 and the hub the LAST keypoint, so a future needle-only
    /// export (e.g. [1,11,N]: 1 class, 2 kpts) works by setting Num Classes = 1 and
    /// Needle Class Id = 0 — no code change.
    ///
    /// Preprocessing and scheduling mirror CustomArmDetector exactly: the input size is
    /// read from the ONNX itself, frames are letterboxed (aspect-fit + Ultralytics gray
    /// pad), inference is layer-sliced via ScheduleIterable, and the output readback is
    /// asynchronous. Between completed inferences the last needle is served from cache.
    /// </summary>
    public class NeedleDetector : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] private ModelAsset _modelAsset;
        [SerializeField] private BackendType _backend = BackendType.GPUCompute;
        [Tooltip("Only used when the model has a dynamic input shape. Static ONNX inputs " +
                 "(all Ultralytics exports) override this with the model's own size.")]
        [SerializeField] private int _inputSize = 320;
        [Tooltip("Quantize model weights to FP16 at load. Halves weight memory and is faster on the " +
                 "Quest GPU with no practical accuracy loss for this model.")]
        [SerializeField] private bool _quantizeToFp16 = true;

        [Header("Output layout")]
        [Tooltip("Number of classes in the assigned model's output. arm-needle-pose-320.onnx has 2 " +
                 "(arm, needle); a needle-only export would have 1. Keypoint count is derived from " +
                 "this: K = (features - 4 - numClasses) / 3.")]
        [SerializeField, Range(1, 4)] private int _numClasses = 2;
        [Tooltip("Which class index is the needle in the assigned model. arm-needle-pose-320.onnx: " +
                 "{0: arm, 1: needle} -> 1. A needle-only export -> 0.")]
        [SerializeField] private int _needleClassId = 1;

        [Header("Scheduling (frame-rate vs detection latency)")]
        [Tooltip("How many model layers to dispatch per rendered frame. Spreads the GPU cost of one " +
                 "inference over several frames so the app holds native frame rate. Lower = smoother " +
                 "frames but fewer detections per second. 0 = dispatch the whole model in one frame.")]
        [SerializeField, Range(0, 64)] private int _layersPerFrame = 14;
        [Tooltip("Minimum seconds between the END of one inference and the START of the next. " +
                 "Caps the GPU duty cycle so the needle model doesn't run back-to-back forever " +
                 "while an arm is locked (heat -> throttling -> lag).")]
        [SerializeField] private float _minSecondsBetweenInferences = 0.25f;
        [Tooltip("Dispose and rebuild the inference Worker this often (seconds) while running, to " +
                 "release GPU scratch memory the InferenceEngine backend pools but never returns " +
                 "during a long run. The model is cached so the rebuild is cheap. 0 = never.")]
        [SerializeField] private float _workerRecycleSeconds = 25f;

        [Header("Filtering")]
        [SerializeField, Range(0f, 1f)] private float _confidenceThreshold = 0.4f;

        private Model _model;
        private Worker _worker;
        private Tensor<float> _inputTensor;
        private IEnumerator _scheduleSteps;
        private Tensor<float> _pendingOutput;
        private bool _readbackPending;
        private int _pendingImageWidth;
        private int _pendingImageHeight;
        private string _lastOutputShape = "-";
        private float _nextInferenceAllowedTime;
        private float _nextReadbackWarningTime;
        private float _workerCreatedTime;

        private int _tensorWidth;
        private int _tensorHeight;

        private RenderTexture _letterboxSquare;
        private RenderTexture _letterboxContent;
        private bool _letterboxUnsupportedWarned;
        // See CustomArmDetector: TextureConverter.ToTensor(Texture, Tensor) leaks a
        // CommandBuffer per call; record into one reusable buffer instead.
        private CommandBuffer _toTensorCommandBuffer;

        private int _pendingPadX;
        private int _pendingPadY;
        private float _pendingInvFitX = 1f;
        private float _pendingInvFitY = 1f;

        private NeedleDetection _latest;
        private bool _hasLatest;

        // Kill switch: if the assigned model throws on every inference (e.g. an ONNX with an
        // operator InferenceEngine can't run — the arm-needle export throws KeyNotFoundException
        // '423' per frame), stop running it. Left unchecked that spins forever, spamming the log
        // and leaking GPU memory from each aborted inference. Fail fast instead.
        private const int MaxConsecutiveFailures = 8;
        private int _consecutiveFailures;
        private bool _permanentlyFailed;

        public bool IsReady => isActiveAndEnabled && !_permanentlyFailed && _worker != null && _inputTensor != null;
        public bool LastRunConsumedNewResult { get; private set; }
        public float LastNeedleMaxScore { get; private set; }
        public float ConfidenceThreshold => _confidenceThreshold;
        public string Status { get; private set; } = "Not started";

        /// <summary>Best needle from the most recent COMPLETED inference. Returns false when
        /// that inference found no needle above the confidence threshold.</summary>
        public bool TryGetLatest(out NeedleDetection needle)
        {
            needle = _latest;
            return _hasLatest;
        }

        private void OnEnable() => LoadModel();

        private void OnDisable()
        {
            DisposeWorker();
            _model = null;
            _toTensorCommandBuffer?.Dispose();
            _toTensorCommandBuffer = null;
            ReleaseLetterboxTextures();
        }

        private void DisposeWorker()
        {
            _scheduleSteps = null;
            _readbackPending = false;
            _pendingOutput = null;
            _worker?.Dispose();
            _worker = null;
            _inputTensor?.Dispose();
            _inputTensor = null;
        }

        private void CreateWorker()
        {
            _worker = new Worker(_model, _backend);
            _inputTensor = new Tensor<float>(new TensorShape(1, 3, _tensorHeight, _tensorWidth));
            _workerCreatedTime = Time.unscaledTime;
        }

        /// <summary>Periodically rebuilds the Worker to release pooled GPU memory. Only fires
        /// when idle (nothing scheduled or awaiting readback). See CustomArmDetector.</summary>
        private void RecycleWorkerIfDue()
        {
            if (_workerRecycleSeconds <= 0f || _model == null) return;
            if (_readbackPending || _scheduleSteps != null) return;
            if (Time.unscaledTime - _workerCreatedTime < _workerRecycleSeconds) return;

            _worker?.Dispose();
            _worker = null;
            _inputTensor?.Dispose();
            _inputTensor = null;
            CreateWorker();
        }

        /// <summary>Leak-free replacement for TextureConverter.ToTensor(Texture, Tensor):
        /// records the conversion into a single reusable CommandBuffer.</summary>
        private void WriteTextureToInputTensor(Texture src)
        {
            if (SystemInfo.supportsComputeShaders)
            {
                _toTensorCommandBuffer ??= new CommandBuffer { name = "NeedleDetector.ToTensor" };
                _toTensorCommandBuffer.Clear();
                _toTensorCommandBuffer.ToTensor(src, _inputTensor);
                Graphics.ExecuteCommandBuffer(_toTensorCommandBuffer);
            }
            else
            {
                TextureConverter.ToTensor(src, _inputTensor);
            }
        }

        private void ReleaseLetterboxTextures()
        {
            if (_letterboxSquare != null) { _letterboxSquare.Release(); _letterboxSquare = null; }
            if (_letterboxContent != null) { _letterboxContent.Release(); _letterboxContent = null; }
        }

        public void Run(Texture source)
        {
            LastRunConsumedNewResult = false;

            if (!IsReady)
            {
                Status = _modelAsset == null ? "No needle model assigned" : "Model not ready";
                return;
            }

            if (source == null || source.width <= 0 || source.height <= 0)
            {
                Status = "Waiting: no camera texture";
                return;
            }

            try
            {
                if (_readbackPending)
                {
                    TryConsumeReadback();
                    return;
                }

                if (_scheduleSteps != null)
                {
                    if (AdvanceSchedule())
                    {
                        Status = "dispatching layers";
                        return;
                    }
                    BeginReadback();
                    return;
                }

                // Duty-cycle gate, mirroring CustomArmDetector.
                if (Time.unscaledTime < _nextInferenceAllowedTime)
                {
                    Status = "cooling down";
                    return;
                }

                RecycleWorkerIfDue();

                WriteLetterboxedInput(source);
                _pendingImageWidth = source.width;
                _pendingImageHeight = source.height;

                if (_layersPerFrame > 0)
                {
                    _scheduleSteps = _worker.ScheduleIterable(_inputTensor);
                    if (AdvanceSchedule())
                    {
                        Status = "inference started";
                        return;
                    }
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
                Status = $"{ex.GetType().Name}: {ex.Message}";

                if (++_consecutiveFailures >= MaxConsecutiveFailures)
                {
                    _permanentlyFailed = true;
                    DisposeWorker();   // release GPU memory; IsReady is now false so Run early-outs
                    Status = $"DISABLED after {_consecutiveFailures} failures ({ex.GetType().Name}). " +
                             "Model is incompatible with InferenceEngine; re-export it. " +
                             "Injection falls back to hand tracking.";
                    Debug.LogError($"[NeedleDetector] Permanently disabled after {_consecutiveFailures} " +
                                   $"consecutive failures. Last error: {ex}");
                }
                else
                {
                    Debug.LogError($"[NeedleDetector] Run failed ({_consecutiveFailures}/{MaxConsecutiveFailures}): {ex}");
                }
            }
        }

        private void WriteLetterboxedInput(Texture source)
        {
            if ((SystemInfo.copyTextureSupport & CopyTextureSupport.Basic) == 0)
            {
                if (!_letterboxUnsupportedWarned)
                {
                    _letterboxUnsupportedWarned = true;
                    Debug.LogWarning("[NeedleDetector] CopyTexture unsupported on this GPU; " +
                                     "using stretched (non-letterboxed) model input.");
                }
                _pendingPadX = 0;
                _pendingPadY = 0;
                _pendingInvFitX = source.width / (float)_tensorWidth;
                _pendingInvFitY = source.height / (float)_tensorHeight;
                WriteTextureToInputTensor(source);
                return;
            }

            float fit = Mathf.Min(_tensorWidth / (float)source.width,
                                  _tensorHeight / (float)source.height);
            int contentW = Mathf.Clamp(Mathf.RoundToInt(source.width * fit), 1, _tensorWidth);
            int contentH = Mathf.Clamp(Mathf.RoundToInt(source.height * fit), 1, _tensorHeight);
            int padX = (_tensorWidth - contentW) / 2;
            int padY = (_tensorHeight - contentH) / 2;

            if (_letterboxSquare == null || !_letterboxSquare.IsCreated() ||
                _letterboxSquare.width != _tensorWidth || _letterboxSquare.height != _tensorHeight)
            {
                if (_letterboxSquare != null) _letterboxSquare.Release();
                _letterboxSquare = new RenderTexture(_tensorWidth, _tensorHeight, 0, RenderTextureFormat.ARGB32);
                _letterboxSquare.Create();
                ClearToUltralyticsGray(_letterboxSquare);
            }
            if (_letterboxContent == null || !_letterboxContent.IsCreated() ||
                _letterboxContent.width != contentW || _letterboxContent.height != contentH)
            {
                if (_letterboxContent != null) _letterboxContent.Release();
                _letterboxContent = new RenderTexture(contentW, contentH, 0, RenderTextureFormat.ARGB32);
                _letterboxContent.Create();
                ClearToUltralyticsGray(_letterboxSquare);
            }

            Graphics.Blit(source, _letterboxContent);
            Graphics.CopyTexture(_letterboxContent, 0, 0, 0, 0, contentW, contentH,
                                 _letterboxSquare, 0, 0, padX, padY);
            WriteTextureToInputTensor(_letterboxSquare);

            _pendingPadX = padX;
            _pendingPadY = padY;
            _pendingInvFitX = source.width / (float)contentW;
            _pendingInvFitY = source.height / (float)contentH;
        }

        private static void ClearToUltralyticsGray(RenderTexture rt)
        {
            var previous = RenderTexture.active;
            RenderTexture.active = rt;
            GL.Clear(false, true, new Color(114f / 255f, 114f / 255f, 114f / 255f, 1f));
            RenderTexture.active = previous;
        }

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
            if (_pendingOutput == null)
            {
                Status = "Model output is not float tensor";
                return;
            }

            _pendingOutput.ReadbackRequest();
            _readbackPending = true;
            Status = "awaiting readback";
        }

        private void TryConsumeReadback()
        {
            if (_pendingOutput == null)
            {
                _readbackPending = false;
                Status = "Pending output was lost";
                return;
            }

            if (!_pendingOutput.IsReadbackRequestDone())
            {
                Status = "readback pending";
                return;
            }

            // Same recovery as CustomArmDetector: an async readback that finished WITH an
            // error still reports done, so retry once with a blocking readback before
            // dropping the frame.
            Tensor<float> cpuOutput = null;
            try
            {
                cpuOutput = _pendingOutput.ReadbackAndClone();
            }
            catch (Exception ex)
            {
                // Throttled to avoid per-inference logcat spam (log I/O costs frame time).
                if (Time.unscaledTime >= _nextReadbackWarningTime)
                {
                    _nextReadbackWarningTime = Time.unscaledTime + 5f;
                    Debug.LogWarning($"[NeedleDetector] Async readback failed ({ex.Message}); retrying synchronously.");
                }
                try
                {
                    cpuOutput = _pendingOutput.ReadbackAndClone();
                }
                catch (Exception ex2)
                {
                    Status = $"readback dropped ({ex2.GetType().Name})";
                }
            }

            if (cpuOutput != null)
            {
                using (cpuOutput)
                    ParsePoseOutput(cpuOutput, _pendingImageWidth, _pendingImageHeight);
                _consecutiveFailures = 0;   // a clean inference clears the failure streak
                LastRunConsumedNewResult = true;
                Status = $"needle={(_hasLatest ? _latest.Confidence.ToString("F2") : "none")} " +
                         $"max={LastNeedleMaxScore:F3} conf>={_confidenceThreshold:F2} shape={_lastOutputShape}";
            }

            _pendingOutput = null;
            _readbackPending = false;
            _nextInferenceAllowedTime = Time.unscaledTime + Mathf.Max(0f, _minSecondsBetweenInferences);
        }

        private void LoadModel()
        {
            DisposeWorker();
            _model = null;
            _hasLatest = false;

            if (_modelAsset == null)
            {
                Status = "No needle model assigned";
                return;
            }

            var model = ModelLoader.Load(_modelAsset);

            _tensorWidth = _inputSize;
            _tensorHeight = _inputSize;
            if (model.inputs.Count > 0 && model.inputs[0].shape.IsStatic())
            {
                var declared = model.inputs[0].shape.ToTensorShape();
                if (declared.rank == 4 && declared[2] > 0 && declared[3] > 0)
                {
                    _tensorHeight = declared[2];
                    _tensorWidth = declared[3];
                    if (_tensorWidth != _inputSize || _tensorHeight != _inputSize)
                        Debug.LogWarning($"[NeedleDetector] _inputSize is {_inputSize} but the model " +
                                         $"declares {_tensorWidth}x{_tensorHeight}; using the model's size.");
                }
            }

            if (_quantizeToFp16)
                ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
            _model = model;
            CreateWorker();
            Status = $"Model loaded ({_tensorWidth}x{_tensorHeight})";
        }

        private void ParsePoseOutput(Tensor<float> output, int imageWidth, int imageHeight)
        {
            LastNeedleMaxScore = 0f;
            _hasLatest = false;

            var shape = output.shape;
            _lastOutputShape = shape.ToString();
            // The tensor is already a CPU clone (ReadbackAndClone), so read it in place —
            // DownloadToArray would copy the whole output into a fresh managed array every
            // inference and feed the GC for nothing.
            ReadOnlySpan<float> data = output.AsReadOnlySpan();

            if (!TryGetYoloLayout(shape, out int rows, out int features, out bool featuresFirst))
            {
                Status = $"Unsupported pose output shape {shape}";
                return;
            }

            // features = 4 box + numClasses scores + 3 * K keypoints
            int kptFloats = features - 4 - _numClasses;
            if (kptFloats < 3 || kptFloats % 3 != 0)
            {
                Status = $"Output {shape} doesn't match Num Classes = {_numClasses} " +
                         $"(expected 4 + {_numClasses} + 3*K features)";
                return;
            }
            int keypointCount = kptFloats / 3;
            int scoreChannel = 4 + Mathf.Clamp(_needleClassId, 0, _numClasses - 1);
            int kptOffset = 4 + _numClasses;

            float bestScore = 0f;
            int bestRow = -1;
            for (int i = 0; i < rows; i++)
            {
                float score = ReadFeature(data, shape, i, scoreChannel, featuresFirst);
                if (score > LastNeedleMaxScore) LastNeedleMaxScore = score;
                if (score < _confidenceThreshold || score <= bestScore) continue;
                bestScore = score;
                bestRow = i;
            }

            if (bestRow < 0) return;

            float cx = ReadFeature(data, shape, bestRow, 0, featuresFirst);
            float cy = ReadFeature(data, shape, bestRow, 1, featuresFirst);
            float w = ReadFeature(data, shape, bestRow, 2, featuresFirst);
            float h = ReadFeature(data, shape, bestRow, 3, featuresFirst);
            Rect bounds = XywhToImageBounds(cx, cy, w, h, imageWidth, imageHeight);
            if (bounds.width < 2f || bounds.height < 2f) return;

            _latest = new NeedleDetection
            {
                ImageBounds = bounds,
                Confidence = bestScore,
                // Labeling convention: kpt0 = tip (the contact point), the LAST kpt = hub.
                // Holds for both the 4-kpt combined model and a 2-kpt needle-only export.
                TipImage = ReadKeypoint(data, shape, bestRow, kptOffset, 0, featuresFirst, imageWidth, imageHeight),
                HubImage = ReadKeypoint(data, shape, bestRow, kptOffset, keypointCount - 1, featuresFirst, imageWidth, imageHeight),
            };
            _hasLatest = true;
        }

        private bool TryGetYoloLayout(TensorShape shape, out int rows, out int features, out bool featuresFirst)
        {
            rows = 0;
            features = 0;
            featuresFirst = false;

            if (shape.rank == 3)
            {
                int d1 = shape[1];
                int d2 = shape[2];
                if (d1 <= 32 && d2 > d1)
                {
                    features = d1; rows = d2; featuresFirst = true;
                    return features >= 8;
                }
                features = d2; rows = d1; featuresFirst = false;
                return features >= 8;
            }

            if (shape.rank == 2)
            {
                int d0 = shape[0];
                int d1 = shape[1];
                if (d0 <= 32 && d1 > d0)
                {
                    features = d0; rows = d1; featuresFirst = true;
                    return features >= 8;
                }
                features = d1; rows = d0; featuresFirst = false;
                return features >= 8;
            }

            return false;
        }

        private float ReadFeature(ReadOnlySpan<float> data, TensorShape shape, int row, int feature, bool featuresFirst)
        {
            if (shape.rank == 3)
            {
                int rows = featuresFirst ? shape[2] : shape[1];
                int features = featuresFirst ? shape[1] : shape[2];
                return featuresFirst ? data[feature * rows + row] : data[row * features + feature];
            }

            int rows2 = featuresFirst ? shape[1] : shape[0];
            int features2 = featuresFirst ? shape[0] : shape[1];
            return featuresFirst ? data[feature * rows2 + row] : data[row * features2 + feature];
        }

        private Vector2 ReadKeypoint(ReadOnlySpan<float> data, TensorShape shape, int row, int kptOffset, int k,
                                     bool featuresFirst, int imageWidth, int imageHeight)
        {
            float kx = ReadFeature(data, shape, row, kptOffset + k * 3, featuresFirst);
            float ky = ReadFeature(data, shape, row, kptOffset + 1 + k * 3, featuresFirst);
            bool normalized = Mathf.Max(Mathf.Abs(kx), Mathf.Abs(ky)) <= 2f;
            if (normalized) { kx *= _tensorWidth; ky *= _tensorHeight; }
            return new Vector2(
                Mathf.Clamp((kx - _pendingPadX) * _pendingInvFitX, 0f, imageWidth),
                Mathf.Clamp((ky - _pendingPadY) * _pendingInvFitY, 0f, imageHeight));
        }

        private Rect XywhToImageBounds(float cx, float cy, float w, float h, int imageWidth, int imageHeight)
        {
            bool normalized = Mathf.Max(Mathf.Abs(cx), Mathf.Abs(cy), Mathf.Abs(w), Mathf.Abs(h)) <= 2f;
            if (normalized)
            {
                cx *= _tensorWidth; w *= _tensorWidth;
                cy *= _tensorHeight; h *= _tensorHeight;
            }

            float width = Mathf.Abs(w) * _pendingInvFitX;
            float height = Mathf.Abs(h) * _pendingInvFitY;
            float centerX = (cx - _pendingPadX) * _pendingInvFitX;
            float centerY = (cy - _pendingPadY) * _pendingInvFitY;

            float xMin = Mathf.Clamp(centerX - width * 0.5f, 0f, imageWidth);
            float yMin = Mathf.Clamp(centerY - height * 0.5f, 0f, imageHeight);
            float xMax = Mathf.Clamp(centerX + width * 0.5f, 0f, imageWidth);
            float yMax = Mathf.Clamp(centerY + height * 0.5f, 0f, imageHeight);

            return new Rect(xMin, yMin, Mathf.Max(0f, xMax - xMin), Mathf.Max(0f, yMax - yMin));
        }
    }
}
