using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Runs the ARM-ONLY pose model (YOLO11n-pose) exported from Ultralytics.
    /// Single class, two keypoints:
    ///   kpt0 = proximal (near elbow / insertion zone), kpt1 = distal (wrist)
    ///
    /// Arms are returned as <see cref="PersonDetection"/> (proximal mapped to the
    /// shoulder keypoint slot, distal to the wrist slot) so the existing
    /// ArmDetectionManager / overlay pipeline consumes them unchanged.
    ///
    /// Expected ONNX output (features-first [1, 11, N]):
    ///   0..3  box cx,cy,w,h   4 arm score   5..10 kpts (kx0,ky0,v0, kx1,ky1,v1)
    /// Legacy 2-class exports ([1, 12, N] / [1, 18, N]) are still parsed - the arm
    /// score is read from channel 4 + _armClassId - so an older combined ONNX keeps
    /// working until the arm-only model is assigned.
    ///
    /// Quest 3 frame-rate strategy:
    ///  - inference is layer-sliced via Worker.ScheduleIterable, so each rendered
    ///    frame only dispatches _layersPerFrame layers of GPU work instead of the
    ///    whole network in one spike;
    ///  - the output readback is asynchronous (ReadbackRequest), so the CPU never
    ///    blocks waiting on the GPU;
    ///  - between completed inferences the last detections are served from cache
    ///    (the manager's smoothing/lock rides across those frames).
    /// </summary>
    public class CustomArmDetector : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] private ModelAsset _modelAsset;
        [SerializeField] private BackendType _backend = BackendType.GPUCompute;
        [SerializeField] private int _inputSize = 320;
        [Tooltip("Quantize model weights to FP16 at load. Halves weight memory and is faster on the " +
                 "Quest GPU with no practical accuracy loss for this model.")]
        [SerializeField] private bool _quantizeToFp16 = true;

        [Header("Scheduling (frame-rate vs detection latency)")]
        [Tooltip("How many model layers to dispatch per rendered frame. Spreads the GPU cost of one " +
                 "inference over several frames so the app holds native frame rate. Lower = smoother " +
                 "frames but fewer detections per second. 0 = dispatch the whole model in one frame.")]
        [SerializeField, Range(0, 64)] private int _layersPerFrame = 14;

        [Header("Filtering")]
        [SerializeField, Range(0f, 1f)] private float _confidenceThreshold = 0.25f;
        [SerializeField, Range(0f, 1f)] private float _nmsIoUThreshold = 0.45f;
        [SerializeField, Range(1, 50)] private int _maxDetections = 5;

        [Header("Legacy 2-class models only")]
        [Tooltip("Only used when an old 2-class (arm+needle) ONNX is assigned: which class index is " +
                 "the arm. Ignored by the arm-only [1,11,N] export.")]
        [SerializeField] private int _armClassId = 1;

        private Worker _worker;
        private Tensor<float> _inputTensor;
        private IEnumerator _scheduleSteps;
        private Tensor<float> _pendingOutput;
        private bool _readbackPending;
        private int _pendingImageWidth;
        private int _pendingImageHeight;
        private readonly List<PoseCandidate> _candidates = new();
        private readonly List<PersonDetection> _detections = new();
        private string _lastOutputShape = "-";

        public bool IsReady => isActiveAndEnabled && _worker != null && _inputTensor != null;
        public bool LastRunConsumedNewResult { get; private set; }
        public bool LastRunScheduledInference { get; private set; }
        public float LastArmOnlyMaxScore { get; private set; }
        public float ConfidenceThreshold => _confidenceThreshold;
        public string Status { get; private set; } = "Not started";

        private struct PoseCandidate
        {
            public Rect Bounds;
            public float Score;
            public Vector2 K0;   // proximal (near elbow)
            public Vector2 K1;   // distal (wrist)
        }

        private void OnEnable() => LoadModel();

        private void OnDisable()
        {
            _scheduleSteps = null;
            _readbackPending = false;
            _pendingOutput = null;
            _worker?.Dispose();
            _worker = null;
            _inputTensor?.Dispose();
            _inputTensor = null;
        }

        public List<PersonDetection> Run(Texture source)
        {
            LastRunConsumedNewResult = false;
            LastRunScheduledInference = false;

            if (!IsReady)
            {
                Status = _modelAsset == null ? "No pose model assigned" : "Model not ready";
                return _detections;
            }

            if (source == null || source.width <= 0 || source.height <= 0)
            {
                Status = "Waiting: no camera texture";
                return _detections;
            }

            try
            {
                if (_readbackPending)
                {
                    TryConsumeReadback();
                    return _detections;
                }

                if (_scheduleSteps != null)
                {
                    if (AdvanceSchedule())
                    {
                        Status = $"dispatching layers; using {_detections.Count} cached arm(s)";
                        return _detections;
                    }
                    BeginReadback();
                    return _detections;
                }

                // Start a new inference from the freshest camera frame.
                TextureConverter.ToTensor(source, _inputTensor);
                _pendingImageWidth = source.width;
                _pendingImageHeight = source.height;
                LastRunScheduledInference = true;

                if (_layersPerFrame > 0)
                {
                    _scheduleSteps = _worker.ScheduleIterable(_inputTensor);
                    if (AdvanceSchedule())
                    {
                        Status = $"inference started; using {_detections.Count} cached arm(s)";
                        return _detections;
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
                Debug.LogError($"[CustomArmDetector] Run failed: {ex}");
            }

            return _detections;
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
            if (_pendingOutput == null)
            {
                Status = "Model output is not float tensor";
                return;
            }

            _pendingOutput.ReadbackRequest();
            _readbackPending = true;
            Status = $"awaiting readback; using {_detections.Count} cached arm(s)";
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
                Status = $"readback pending; using {_detections.Count} cached arm(s)";
                return;
            }

            // IsReadbackRequestDone() only reports .done, not .hasError - an async GPU
            // readback that finished WITH an error is still "done", so ReadbackAndClone()
            // throws "Cannot access the data as it is not available" (common on Quest).
            // The failed async request has already been consumed, so a second
            // ReadbackAndClone() issues a fresh *blocking* readback of the same still-valid
            // buffer - recovering this frame instead of dropping it. If that also fails we
            // drop the frame and re-run next frame, riding on the cached detections.
            Tensor<float> cpuOutput = null;
            try
            {
                cpuOutput = _pendingOutput.ReadbackAndClone();
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[CustomArmDetector] Async readback failed ({ex.Message}); retrying synchronously.");
                try
                {
                    cpuOutput = _pendingOutput.ReadbackAndClone();
                }
                catch (Exception ex2)
                {
                    Status = $"readback dropped ({ex2.GetType().Name}); using {_detections.Count} cached arm(s)";
                    Debug.LogWarning($"[CustomArmDetector] Readback dropped: {ex2.Message}");
                }
            }

            if (cpuOutput != null)
            {
                using (cpuOutput)
                    ParsePoseOutput(cpuOutput, _pendingImageWidth, _pendingImageHeight);
                LastRunConsumedNewResult = true;
                Status = $"arms={_detections.Count} max={LastArmOnlyMaxScore:F3} " +
                         $"conf>={_confidenceThreshold:F2} shape={_lastOutputShape}";
            }

            _pendingOutput = null;
            _readbackPending = false;
        }

        private void LoadModel()
        {
            _scheduleSteps = null;
            _readbackPending = false;
            _pendingOutput = null;
            _worker?.Dispose();
            _worker = null;
            _inputTensor?.Dispose();
            _inputTensor = null;

            if (_modelAsset == null)
            {
                Status = "No pose model assigned";
                return;
            }

            var model = ModelLoader.Load(_modelAsset);
            if (_quantizeToFp16)
                ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
            _worker = new Worker(model, _backend);
            _inputTensor = new Tensor<float>(new TensorShape(1, 3, _inputSize, _inputSize));
            Status = "Model loaded";
        }

        private void ParsePoseOutput(Tensor<float> output, int imageWidth, int imageHeight)
        {
            LastArmOnlyMaxScore = 0f;
            _candidates.Clear();
            _detections.Clear();

            var shape = output.shape;
            _lastOutputShape = shape.ToString();
            var data = output.DownloadToArray();

            if (!TryGetYoloLayout(shape, out int rows, out int features, out bool featuresFirst))
            {
                Status = $"Unsupported pose output shape {shape}";
                return;
            }

            int scoreChannel;
            int kptOffset;
            if (features == 11)
            {
                // Arm-only export: 4 box + 1 score + 2 kpts.
                scoreChannel = 4;
                kptOffset = 5;
            }
            else if (features == 12 || features == 18)
            {
                // Legacy 2-class exports (2 or 4 kpts): 4 box + 2 scores + kpts.
                scoreChannel = 4 + Mathf.Clamp(_armClassId, 0, 1);
                kptOffset = 6;
            }
            else
            {
                Status = $"Unsupported pose output shape {shape} (expected 11, 12 or 18 features)";
                return;
            }

            for (int i = 0; i < rows; i++)
            {
                float score = ReadFeature(data, shape, i, scoreChannel, featuresFirst);
                if (score > LastArmOnlyMaxScore) LastArmOnlyMaxScore = score;
                if (score < _confidenceThreshold) continue;

                float cx = ReadFeature(data, shape, i, 0, featuresFirst);
                float cy = ReadFeature(data, shape, i, 1, featuresFirst);
                float w = ReadFeature(data, shape, i, 2, featuresFirst);
                float h = ReadFeature(data, shape, i, 3, featuresFirst);

                Rect bounds = XywhToImageBounds(cx, cy, w, h, imageWidth, imageHeight);
                if (bounds.width < 2f || bounds.height < 2f) continue;

                _candidates.Add(new PoseCandidate
                {
                    Bounds = bounds,
                    Score = score,
                    K0 = ReadKeypoint(data, shape, i, kptOffset, 0, featuresFirst, imageWidth, imageHeight),
                    K1 = ReadKeypoint(data, shape, i, kptOffset, 1, featuresFirst, imageWidth, imageHeight),
                });
            }

            _candidates.Sort((a, b) => b.Score.CompareTo(a.Score));

            var keptBounds = new List<Rect>();
            for (int i = 0; i < _candidates.Count && _detections.Count < _maxDetections; i++)
            {
                var c = _candidates[i];
                bool suppressed = false;
                for (int j = 0; j < keptBounds.Count; j++)
                {
                    if (IoU(c.Bounds, keptBounds[j]) > _nmsIoUThreshold)
                    {
                        suppressed = true;
                        break;
                    }
                }
                if (suppressed) continue;
                keptBounds.Add(c.Bounds);
                _detections.Add(ToPersonDetection(c));
            }
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
                    return features >= 11;
                }
                features = d2; rows = d1; featuresFirst = false;
                return features >= 11;
            }

            if (shape.rank == 2)
            {
                int d0 = shape[0];
                int d1 = shape[1];
                if (d0 <= 32 && d1 > d0)
                {
                    features = d0; rows = d1; featuresFirst = true;
                    return features >= 11;
                }
                features = d1; rows = d0; featuresFirst = false;
                return features >= 11;
            }

            return false;
        }

        private float ReadFeature(float[] data, TensorShape shape, int row, int feature, bool featuresFirst)
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

        private Vector2 ReadKeypoint(float[] data, TensorShape shape, int row, int kptOffset, int k,
                                     bool featuresFirst, int imageWidth, int imageHeight)
        {
            float kx = ReadFeature(data, shape, row, kptOffset + k * 3, featuresFirst);
            float ky = ReadFeature(data, shape, row, kptOffset + 1 + k * 3, featuresFirst);
            bool normalized = Mathf.Max(Mathf.Abs(kx), Mathf.Abs(ky)) <= 2f;
            float scaleX = normalized ? imageWidth : imageWidth / (float)_inputSize;
            float scaleY = normalized ? imageHeight : imageHeight / (float)_inputSize;
            return new Vector2(
                Mathf.Clamp(kx * scaleX, 0f, imageWidth),
                Mathf.Clamp(ky * scaleY, 0f, imageHeight));
        }

        private Rect XywhToImageBounds(float cx, float cy, float w, float h, int imageWidth, int imageHeight)
        {
            bool normalized = Mathf.Max(Mathf.Abs(cx), Mathf.Abs(cy), Mathf.Abs(w), Mathf.Abs(h)) <= 2f;
            float scaleX = normalized ? imageWidth : imageWidth / (float)_inputSize;
            float scaleY = normalized ? imageHeight : imageHeight / (float)_inputSize;

            float width = Mathf.Abs(w) * scaleX;
            float height = Mathf.Abs(h) * scaleY;
            float centerX = cx * scaleX;
            float centerY = cy * scaleY;

            float xMin = Mathf.Clamp(centerX - width * 0.5f, 0f, imageWidth);
            float yMin = Mathf.Clamp(centerY - height * 0.5f, 0f, imageHeight);
            float xMax = Mathf.Clamp(centerX + width * 0.5f, 0f, imageWidth);
            float yMax = Mathf.Clamp(centerY + height * 0.5f, 0f, imageHeight);

            return new Rect(xMin, yMin, Mathf.Max(0f, xMax - xMin), Mathf.Max(0f, yMax - yMin));
        }

        /// <summary>
        /// Maps an arm pose candidate to the COCO-slot PersonDetection the manager
        /// expects: proximal -&gt; shoulder, distal -&gt; wrist, midpoint -&gt; elbow.
        /// Both Left and Right slots are filled so the manager's side scan finds it.
        /// </summary>
        private PersonDetection ToPersonDetection(PoseCandidate c)
        {
            var keypoints = new Keypoint[17];
            Vector2 proximal = c.K0;
            Vector2 distal = c.K1;
            Vector2 elbow = (proximal + distal) * 0.5f;

            FillArmKeypoints(keypoints, Side.Left, proximal, elbow, distal, c.Score);
            FillArmKeypoints(keypoints, Side.Right, proximal, elbow, distal, c.Score);

            return new PersonDetection
            {
                ImageBounds = c.Bounds,
                Confidence = c.Score,
                Keypoints = keypoints,
            };
        }

        private static void FillArmKeypoints(Keypoint[] keypoints, Side side,
                                             Vector2 shoulder, Vector2 elbow, Vector2 wrist,
                                             float confidence)
        {
            int shoulderIdx = side == Side.Left ? (int)CocoKeypoint.LeftShoulder : (int)CocoKeypoint.RightShoulder;
            int elbowIdx = side == Side.Left ? (int)CocoKeypoint.LeftElbow : (int)CocoKeypoint.RightElbow;
            int wristIdx = side == Side.Left ? (int)CocoKeypoint.LeftWrist : (int)CocoKeypoint.RightWrist;

            keypoints[shoulderIdx] = new Keypoint { ImagePos = shoulder, Confidence = confidence };
            keypoints[elbowIdx] = new Keypoint { ImagePos = elbow, Confidence = confidence };
            keypoints[wristIdx] = new Keypoint { ImagePos = wrist, Confidence = confidence };
        }

        private static float IoU(Rect a, Rect b)
        {
            float xMin = Mathf.Max(a.xMin, b.xMin);
            float yMin = Mathf.Max(a.yMin, b.yMin);
            float xMax = Mathf.Min(a.xMax, b.xMax);
            float yMax = Mathf.Min(a.yMax, b.yMax);

            float intersection = Mathf.Max(0f, xMax - xMin) * Mathf.Max(0f, yMax - yMin);
            float union = a.width * a.height + b.width * b.height - intersection;
            return union <= 0f ? 0f : intersection / union;
        }
    }
}
