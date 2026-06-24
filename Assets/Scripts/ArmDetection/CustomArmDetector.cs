using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Runs the arm + needle POSE model (YOLO11n-pose) exported from Ultralytics.
    /// Two classes, two keypoints each:
    ///   class 0 = arm    -> kpt0 = proximal (near elbow), kpt1 = distal (wrist)
    ///   class 1 = needle -> kpt0 = tip (contact point),   kpt1 = hub (back of needle)
    ///
    /// Arms are returned as <see cref="PersonDetection"/> (proximal mapped to the
    /// shoulder keypoint slot, distal to the wrist slot) so the existing
    /// ArmDetectionManager / overlay pipeline consumes them unchanged. Needles are
    /// exposed separately via <see cref="LastNeedles"/> for contact detection.
    ///
    /// Expected ONNX output (features-first [1, 12, N]):
    ///   0..3  box cx,cy,w,h   4..5 class scores   6..11 kpts (kx0,ky0,v0, kx1,ky1,v1)
    /// </summary>
    public class CustomArmDetector : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] private ModelAsset _modelAsset;
        [SerializeField] private BackendType _backend = BackendType.GPUCompute;
        [SerializeField] private int _inputSize = 320;

        [Header("Filtering")]
        [SerializeField, Range(0f, 1f)] private float _confidenceThreshold = 0.25f;
        [SerializeField, Range(0f, 1f)] private float _nmsIoUThreshold = 0.45f;
        [SerializeField, Range(1, 50)] private int _maxDetections = 5;

        [Header("Class mapping (must match the model's training names dict)")]
        [Tooltip("Class index the pose model uses for the arm. Check the ONNX metadata's 'names' field after export.")]
        [SerializeField] private int _armClassId = 1;
        [Tooltip("Class index the pose model uses for the needle/syringe. Check the ONNX metadata's 'names' field after export.")]
        [SerializeField] private int _needleClassId = 0;

        private Worker _worker;
        private Tensor<float> _inputTensor;
        private readonly List<PersonDetection> _detections = new();
        private readonly List<NeedleDetection> _needles = new();
        private readonly List<PoseCandidate> _candidates = new();
        private string _lastOutputShape = "-";

        public bool IsReady => isActiveAndEnabled && _worker != null && _inputTensor != null;
        public bool LastRunConsumedNewResult { get; private set; }
        public bool LastRunScheduledInference { get; private set; }
        public float LastArmOnlyMaxScore { get; private set; }
        public float ConfidenceThreshold => _confidenceThreshold;
        public string Status { get; private set; } = "Not started";

        /// <summary>Needles detected on the most recent run (tip + hub in image space).</summary>
        public IReadOnlyList<NeedleDetection> LastNeedles => _needles;

        private struct PoseCandidate
        {
            public int Cls;
            public Rect Bounds;
            public float Score;
            public Vector2 K0;   // arm: proximal | needle: tip
            public Vector2 K1;   // arm: distal   | needle: hub
        }

        private void OnEnable() => LoadModel();

        private void OnDisable()
        {
            _worker?.Dispose();
            _worker = null;
            _inputTensor?.Dispose();
            _inputTensor = null;
        }

        public List<PersonDetection> Run(Texture source)
        {
            LastRunConsumedNewResult = false;
            LastRunScheduledInference = false;
            LastArmOnlyMaxScore = 0f;
            _detections.Clear();
            _needles.Clear();

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
                TextureConverter.ToTensor(source, _inputTensor);
                _worker.Schedule(_inputTensor);
                LastRunScheduledInference = true;

                var output = _worker.PeekOutput() as Tensor<float>;
                if (output == null)
                {
                    Status = "Model output is not float tensor";
                    return _detections;
                }

                using var cpuOutput = output.ReadbackAndClone();
                ParsePoseOutput(cpuOutput, source.width, source.height);

                LastRunConsumedNewResult = true;
                Status = $"arms={_detections.Count} needles={_needles.Count} " +
                         $"max={LastArmOnlyMaxScore:F3} conf>={_confidenceThreshold:F2} shape={_lastOutputShape}";
            }
            catch (Exception ex)
            {
                Status = $"{ex.GetType().Name}: {ex.Message}";
                Debug.LogError($"[CustomArmDetector] Run failed: {ex}");
            }

            return _detections;
        }

        private void LoadModel()
        {
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
            _worker = new Worker(model, _backend);
            _inputTensor = new Tensor<float>(new TensorShape(1, 3, _inputSize, _inputSize));
            Status = "Model loaded";
        }

        private void ParsePoseOutput(Tensor<float> output, int imageWidth, int imageHeight)
        {
            _candidates.Clear();
            _detections.Clear();
            _needles.Clear();

            var shape = output.shape;
            _lastOutputShape = shape.ToString();
            var data = output.DownloadToArray();

            if (!TryGetYoloLayout(shape, out int rows, out int features, out bool featuresFirst) || features < 12)
            {
                Status = $"Unsupported pose output shape {shape} (need >=12 features)";
                return;
            }

            for (int i = 0; i < rows; i++)
            {
                float armScore = ReadFeature(data, shape, i, 4 + _armClassId, featuresFirst);
                float needleScore = ReadFeature(data, shape, i, 4 + _needleClassId, featuresFirst);
                int cls = needleScore > armScore ? _needleClassId : _armClassId;
                float score = Mathf.Max(armScore, needleScore);

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
                    Cls = cls,
                    Bounds = bounds,
                    Score = score,
                    K0 = ReadKeypoint(data, shape, i, 0, featuresFirst, imageWidth, imageHeight),
                    K1 = ReadKeypoint(data, shape, i, 1, featuresFirst, imageWidth, imageHeight),
                });
            }

            _candidates.Sort((a, b) => b.Score.CompareTo(a.Score));

            // Class-aware NMS, then route arms vs needles to their consumers.
            var kept = new List<PoseCandidate>();
            for (int i = 0; i < _candidates.Count && kept.Count < _maxDetections; i++)
            {
                var c = _candidates[i];
                bool suppressed = false;
                for (int j = 0; j < kept.Count; j++)
                {
                    if (kept[j].Cls == c.Cls && IoU(c.Bounds, kept[j].Bounds) > _nmsIoUThreshold)
                    {
                        suppressed = true;
                        break;
                    }
                }
                if (suppressed) continue;
                kept.Add(c);

                if (c.Cls == _armClassId)
                    _detections.Add(ToPersonDetection(c));
                else
                    _needles.Add(new NeedleDetection { TipImage = c.K0, HubImage = c.K1, Confidence = c.Score });
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
                    return features >= 12;
                }
                features = d2; rows = d1; featuresFirst = false;
                return features >= 12;
            }

            if (shape.rank == 2)
            {
                int d0 = shape[0];
                int d1 = shape[1];
                if (d0 <= 32 && d1 > d0)
                {
                    features = d0; rows = d1; featuresFirst = true;
                    return features >= 12;
                }
                features = d1; rows = d0; featuresFirst = false;
                return features >= 12;
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

        private Vector2 ReadKeypoint(float[] data, TensorShape shape, int row, int k,
                                     bool featuresFirst, int imageWidth, int imageHeight)
        {
            float kx = ReadFeature(data, shape, row, 6 + k * 3, featuresFirst);
            float ky = ReadFeature(data, shape, row, 7 + k * 3, featuresFirst);
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
