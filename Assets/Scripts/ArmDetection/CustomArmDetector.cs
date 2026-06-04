using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Runs a one-class custom arm detector exported from Ultralytics/Roboflow.
    /// Expected model output is YOLO-style x,y,w,h,confidence for class "arm".
    /// </summary>
    public class CustomArmDetector : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] private ModelAsset _modelAsset;
        [SerializeField] private BackendType _backend = BackendType.GPUCompute;
        [SerializeField] private int _inputSize = 640;

        [Header("Filtering")]
        [SerializeField, Range(0f, 1f)] private float _confidenceThreshold = 0.12f;
        [SerializeField, Range(0f, 1f)] private float _nmsIoUThreshold = 0.45f;
        [SerializeField, Range(1, 50)] private int _maxDetections = 8;
        [Tooltip("Enable if the ONNX output is x1,y1,x2,y2 instead of cx,cy,w,h. Auto usually works.")]
        [SerializeField] private bool _forceXyxyOutput = false;

        private Worker _worker;
        private Tensor<float> _inputTensor;
        private readonly List<PersonDetection> _detections = new();
        private readonly List<Candidate> _candidates = new();
        private string _lastOutputShape = "-";

        public bool IsReady => isActiveAndEnabled && _worker != null && _inputTensor != null;
        public bool LastRunConsumedNewResult { get; private set; }
        public bool LastRunScheduledInference { get; private set; }
        public bool LastRunWasArmOnlyFallback => _detections.Count > 0;
        public float LastArmOnlyMaxScore { get; private set; }
        public float ConfidenceThreshold => _confidenceThreshold;
        public string Status { get; private set; } = "Not started";

        private struct Candidate
        {
            public Rect Bounds;
            public float Score;
        }

        private void OnEnable()
        {
            LoadModel();
        }

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

            if (!IsReady)
            {
                Status = _modelAsset == null ? "No custom arm model assigned" : "Model not ready";
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
                ParseYoloOutput(cpuOutput, source.width, source.height);

                LastRunConsumedNewResult = true;
                Status = _detections.Count > 0
                    ? $"OK max={LastArmOnlyMaxScore:F3} shape={_lastOutputShape}"
                    : $"0 arms max={LastArmOnlyMaxScore:F3} conf<{_confidenceThreshold:F2} shape={_lastOutputShape}";
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
                Status = "No custom arm model assigned";
                return;
            }

            var model = ModelLoader.Load(_modelAsset);
            _worker = new Worker(model, _backend);
            _inputTensor = new Tensor<float>(new TensorShape(1, 3, _inputSize, _inputSize));
            Status = "Model loaded";
        }

        private void ParseYoloOutput(Tensor<float> output, int imageWidth, int imageHeight)
        {
            _candidates.Clear();
            _detections.Clear();

            var shape = output.shape;
            _lastOutputShape = shape.ToString();
            var data = output.DownloadToArray();

            int rows;
            int features;
            bool featuresFirst;

            if (!TryGetYoloLayout(shape, out rows, out features, out featuresFirst))
            {
                Status = $"Unsupported output shape {shape}";
                return;
            }

            for (int i = 0; i < rows; i++)
            {
                float score = ReadFeature(data, shape, i, 4, featuresFirst);
                if (score > LastArmOnlyMaxScore) LastArmOnlyMaxScore = score;
                if (score < _confidenceThreshold) continue;

                float cx = ReadFeature(data, shape, i, 0, featuresFirst);
                float cy = ReadFeature(data, shape, i, 1, featuresFirst);
                float w = ReadFeature(data, shape, i, 2, featuresFirst);
                float h = ReadFeature(data, shape, i, 3, featuresFirst);

                Rect bounds = LooksLikeXyxy(cx, cy, w, h)
                    ? XyxyToImageBounds(cx, cy, w, h, imageWidth, imageHeight)
                    : XywhToImageBounds(cx, cy, w, h, imageWidth, imageHeight);
                if (bounds.width < 2f || bounds.height < 2f) continue;

                _candidates.Add(new Candidate { Bounds = bounds, Score = score });
            }

            _candidates.Sort((a, b) => b.Score.CompareTo(a.Score));

            for (int i = 0; i < _candidates.Count && _detections.Count < _maxDetections; i++)
            {
                var candidate = _candidates[i];
                bool suppressed = false;

                for (int j = 0; j < _detections.Count; j++)
                {
                    if (IoU(candidate.Bounds, _detections[j].ImageBounds) > _nmsIoUThreshold)
                    {
                        suppressed = true;
                        break;
                    }
                }

                if (!suppressed)
                    _detections.Add(ToPersonDetection(candidate));
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

                if (d1 <= 16 && d2 > d1)
                {
                    features = d1;
                    rows = d2;
                    featuresFirst = true;
                    return features >= 5;
                }

                features = d2;
                rows = d1;
                featuresFirst = false;
                return features >= 5;
            }

            if (shape.rank == 2)
            {
                int d0 = shape[0];
                int d1 = shape[1];

                if (d0 <= 16 && d1 > d0)
                {
                    features = d0;
                    rows = d1;
                    featuresFirst = true;
                    return features >= 5;
                }

                features = d1;
                rows = d0;
                featuresFirst = false;
                return features >= 5;
            }

            return false;
        }

        private float ReadFeature(float[] data, TensorShape shape, int row, int feature, bool featuresFirst)
        {
            if (shape.rank == 3)
            {
                int rows = featuresFirst ? shape[2] : shape[1];
                int features = featuresFirst ? shape[1] : shape[2];
                return featuresFirst
                    ? data[feature * rows + row]
                    : data[row * features + feature];
            }

            int rows2 = featuresFirst ? shape[1] : shape[0];
            int features2 = featuresFirst ? shape[0] : shape[1];
            return featuresFirst
                ? data[feature * rows2 + row]
                : data[row * features2 + feature];
        }

        private bool LooksLikeXyxy(float x1, float y1, float x2, float y2)
        {
            if (_forceXyxyOutput) return true;
            return x2 > x1 && y2 > y1;
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

        private Rect XyxyToImageBounds(float x1, float y1, float x2, float y2, int imageWidth, int imageHeight)
        {
            bool normalized = Mathf.Max(Mathf.Abs(x1), Mathf.Abs(y1), Mathf.Abs(x2), Mathf.Abs(y2)) <= 2f;
            float scaleX = normalized ? imageWidth : imageWidth / (float)_inputSize;
            float scaleY = normalized ? imageHeight : imageHeight / (float)_inputSize;

            float xMin = Mathf.Clamp(Mathf.Min(x1, x2) * scaleX, 0f, imageWidth);
            float yMin = Mathf.Clamp(Mathf.Min(y1, y2) * scaleY, 0f, imageHeight);
            float xMax = Mathf.Clamp(Mathf.Max(x1, x2) * scaleX, 0f, imageWidth);
            float yMax = Mathf.Clamp(Mathf.Max(y1, y2) * scaleY, 0f, imageHeight);

            return new Rect(xMin, yMin, Mathf.Max(0f, xMax - xMin), Mathf.Max(0f, yMax - yMin));
        }

        private PersonDetection ToPersonDetection(Candidate candidate)
        {
            var keypoints = new Keypoint[17];
            Rect b = candidate.Bounds;

            Vector2 shoulder;
            Vector2 elbow;
            Vector2 wrist;

            if (b.width >= b.height)
            {
                shoulder = new Vector2(b.xMin, b.center.y);
                elbow = b.center;
                wrist = new Vector2(b.xMax, b.center.y);
            }
            else
            {
                shoulder = new Vector2(b.center.x, b.yMin);
                elbow = b.center;
                wrist = new Vector2(b.center.x, b.yMax);
            }

            FillArmKeypoints(keypoints, Side.Left, shoulder, elbow, wrist, candidate.Score);
            FillArmKeypoints(keypoints, Side.Right, shoulder, elbow, wrist, candidate.Score);

            return new PersonDetection
            {
                ImageBounds = b,
                Confidence = candidate.Score,
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
