using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Converts MediaPipe Hands landmarks into the existing arm-detection shape.
    /// MediaPipe gives a reliable wrist/hand; this component infers elbow/shoulder
    /// image points so ArmDetectionManager can keep rendering a whole-arm overlay.
    /// </summary>
    public class MediaPipeHandArmDetector : MonoBehaviour
    {
        [SerializeField] private PassthroughCameraSource _cameraSource;
        [Tooltip("How far from wrist toward forearm to synthesize the shoulder point, as a fraction of image height.")]
        [SerializeField] private float _estimatedArmLengthImageFraction = 0.32f;
        [Tooltip("Fixed arm box length as fraction of image height.")]
        [SerializeField] private float _fixedBoxLengthImageFraction = 0.58f;
        [Tooltip("Fixed arm box thickness as fraction of image height.")]
        [SerializeField] private float _fixedBoxThicknessImageFraction = 0.16f;
        [SerializeField, Range(0f, 1f)] private float _directionSmoothing = 0.85f;
        [Tooltip("Keep the latest MediaPipe result alive for this many Unity frames.")]
        [SerializeField, Range(1, 120)] private int _staleAfterFrames = 45;
        [SerializeField, Range(0f, 1f)] private float _landmarkSmoothing = 0.75f;
        [SerializeField, Range(0f, 1f)] private float _defaultConfidence = 0.95f;
        [Tooltip("Extra smoothing for the final generated arm box/keypoints.")]
        [SerializeField, Range(0f, 1f)] private float _boxSmoothing = 0.65f;

        private bool _hasStableDirection;
        private Vector2 _stableForearmDir;
        private bool _hasStableArm;
        private Vector2 _stableShoulder;
        private Vector2 _stableElbow;
        private Vector2 _stableWrist;
        private bool _hasStableBounds;
        private Rect _stableBounds;
        private readonly List<PersonDetection> _detections = new(1);
        private readonly Vector2[] _latestLandmarks01 = new Vector2[21];
        private bool _hasSmoothedLandmarks;
        private bool _hasLandmarks;
        private bool _hasUnreadLandmarks;
        private int _lastLandmarkFrame = -1000;
        private Side _side = Side.Right;
        private float _confidence;

        public bool IsReady => isActiveAndEnabled;
        public bool LastRunConsumedNewResult { get; private set; }
        public bool LastRunScheduledInference => false;
        public bool LastRunWasArmOnlyFallback => _detections.Count > 0;
        public float LastArmOnlyMaxScore => _hasLandmarks ? _confidence : 0f;
        public string Status { get; private set; } = "Waiting: no MediaPipe landmarks";

        private void Reset()
        {
            _cameraSource = GetComponentInParent<PassthroughCameraSource>()
                         ?? FindAnyObjectByType<PassthroughCameraSource>();
        }

        public void SetNormalizedLandmarks(IReadOnlyList<Vector2> landmarks01,
                                           Side side = Side.Right,
                                           float confidence = -1f)
        {
            if (landmarks01 == null || landmarks01.Count < 21) return;

            float t = 1f - _landmarkSmoothing;

            for (int i = 0; i < 21; i++)
            {
                _latestLandmarks01[i] = _hasSmoothedLandmarks
                    ? Vector2.Lerp(_latestLandmarks01[i], landmarks01[i], t)
                    : landmarks01[i];
            }

            _hasSmoothedLandmarks = true;

            _side = side;
            _confidence = confidence >= 0f ? confidence : _defaultConfidence;
            _hasLandmarks = true;
            _hasUnreadLandmarks = true;
            _lastLandmarkFrame = Time.frameCount;
            Status = "OK";
        }

        public void ClearLandmarks()
        {
            _hasLandmarks = false;
            _hasUnreadLandmarks = false;
            _hasStableBounds = false;
            _hasStableDirection = false;
            _hasSmoothedLandmarks = false;
            _hasStableArm = false;
            _detections.Clear();
            Status = "Waiting: no MediaPipe hand";
        }

        public List<PersonDetection> Run(Texture _)
        {
            LastRunConsumedNewResult = false;
            _detections.Clear();

            if (!_hasLandmarks)
            {
                Status = "Waiting: no MediaPipe hand";
                return _detections;
            }

            if (Time.frameCount - _lastLandmarkFrame > _staleAfterFrames)
            {
                ClearLandmarks();
                Status = "Waiting: MediaPipe hand stale";
                return _detections;
            }

            if (_cameraSource == null)
                _cameraSource = GetComponentInParent<PassthroughCameraSource>()
                             ?? FindAnyObjectByType<PassthroughCameraSource>();

            int width = _cameraSource != null && _cameraSource.Width > 0 ? _cameraSource.Width : 1280;
            int height = _cameraSource != null && _cameraSource.Height > 0 ? _cameraSource.Height : 720;

            var keypoints = new Keypoint[17];
            int shoulderIdx = _side == Side.Left ? (int)CocoKeypoint.LeftShoulder : (int)CocoKeypoint.RightShoulder;
            int elbowIdx    = _side == Side.Left ? (int)CocoKeypoint.LeftElbow    : (int)CocoKeypoint.RightElbow;
            int wristIdx    = _side == Side.Left ? (int)CocoKeypoint.LeftWrist    : (int)CocoKeypoint.RightWrist;

            Vector2 wrist = ToImage(_latestLandmarks01[0], width, height);

            Vector2 indexMcp = ToImage(_latestLandmarks01[5], width, height);
            Vector2 middleMcp = ToImage(_latestLandmarks01[9], width, height);
            Vector2 pinkyMcp = ToImage(_latestLandmarks01[17], width, height);
            Vector2 middleTip = ToImage(_latestLandmarks01[12], width, height);

            Vector2 palm = (indexMcp + middleMcp + pinkyMcp) / 3f;

            // Fingers point palm -> middle tip, so arm points opposite.
            Vector2 rawDir = palm - middleTip;

            if (rawDir.sqrMagnitude < 16f)
                rawDir = wrist - palm;

            if (rawDir.sqrMagnitude < 16f)
                rawDir = new Vector2(_side == Side.Left ? -1f : 1f, 0f);

            rawDir.Normalize();

            if (!_hasStableDirection)
            {
                _stableForearmDir = rawDir;
                _hasStableDirection = true;
            }
            else
            {
                float t = 1f - _directionSmoothing;
                _stableForearmDir = Vector2.Lerp(_stableForearmDir, rawDir, t).normalized;
            }

            Vector2 forearmDir = _stableForearmDir;

            float armLength = Mathf.Clamp(height * _fixedBoxLengthImageFraction, 260f, 760f);
            Vector2 shoulder = ClampImage(wrist + forearmDir * armLength, width, height);
            Vector2 elbow = Vector2.Lerp(shoulder, wrist, 0.52f);

            StabilizeArmPoints(ref shoulder, ref elbow, ref wrist);

            keypoints[shoulderIdx] = new Keypoint { ImagePos = shoulder, Confidence = _confidence };
            keypoints[elbowIdx]    = new Keypoint { ImagePos = elbow,    Confidence = _confidence };
            keypoints[wristIdx]    = new Keypoint { ImagePos = wrist,    Confidence = _confidence };

            Rect bounds = BuildBounds(shoulder, elbow, wrist, width, height);
            bounds = StabilizeBounds(bounds);
            _detections.Add(new PersonDetection
            {
                Confidence = _confidence,
                ImageBounds = bounds,
                Keypoints = keypoints,
            });

            LastRunConsumedNewResult = _hasUnreadLandmarks;
            _hasUnreadLandmarks = false;
            Status = "OK";
            return _detections;
        }

        private void StabilizeArmPoints(ref Vector2 shoulder, ref Vector2 elbow, ref Vector2 wrist)
        {
            float t = 1f - _boxSmoothing;

            if (!_hasStableArm)
            {
                _stableShoulder = shoulder;
                _stableElbow = elbow;
                _stableWrist = wrist;
                _hasStableArm = true;
            }
            else
            {
                _stableShoulder = Vector2.Lerp(_stableShoulder, shoulder, t);
                _stableElbow = Vector2.Lerp(_stableElbow, elbow, t);
                _stableWrist = Vector2.Lerp(_stableWrist, wrist, t);
            }

            shoulder = _stableShoulder;
            elbow = _stableElbow;
            wrist = _stableWrist;
        }

        private Rect StabilizeBounds(Rect bounds)
        {
            if (!_hasStableBounds)
            {
                _stableBounds = bounds;
                _hasStableBounds = true;
                return _stableBounds;
            }

            float t = 1f - _boxSmoothing;

            Vector2 center = Vector2.Lerp(_stableBounds.center, bounds.center, t);

            // Keep size mostly constant. Only slowly follow size changes.
            float boxWidth = Mathf.Lerp(_stableBounds.width, bounds.width, t * 0.15f);
            float boxHeight = Mathf.Lerp(_stableBounds.height, bounds.height, t * 0.15f);

            _stableBounds = new Rect(
                center.x - boxWidth * 0.5f,
                center.y - boxHeight * 0.5f,
                boxWidth,
                boxHeight
            );

            return _stableBounds;
        }

        private static Vector2 ToImage(Vector2 normalized, int width, int height)
            => new(normalized.x * width, normalized.y * height);

        private static Vector2 ClampImage(Vector2 p, int width, int height)
            => new(Mathf.Clamp(p.x, 0f, width), Mathf.Clamp(p.y, 0f, height));

        private Rect BuildBounds(Vector2 shoulder, Vector2 elbow, Vector2 wrist, int width, int height)
        {
            Vector2 axis = shoulder - wrist;
            if (axis.sqrMagnitude < 1f)
                axis = Vector2.right;

            axis.Normalize();

            float boxLength = Mathf.Clamp(height * 0.48f, 300f, 620f);
            float boxHeight = Mathf.Clamp(height * 0.18f, 110f, 210f);

            // Anchor the box from wrist toward inferred shoulder.
            Vector2 center = wrist + axis * (boxLength * 0.5f);

            float minX = center.x - boxLength * 0.5f;
            float minY = center.y - boxHeight * 0.5f;
            float maxX = center.x + boxLength * 0.5f;
            float maxY = center.y + boxHeight * 0.5f;

            minX = Mathf.Clamp(minX, 0f, width);
            minY = Mathf.Clamp(minY, 0f, height);
            maxX = Mathf.Clamp(maxX, 0f, width);
            maxY = Mathf.Clamp(maxY, 0f, height);

            return new Rect(minX, minY, Mathf.Max(1f, maxX - minX), Mathf.Max(1f, maxY - minY));
        }
    }
}
