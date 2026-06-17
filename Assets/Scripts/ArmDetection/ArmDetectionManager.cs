using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

namespace ARArmDetection
{
    [DefaultExecutionOrder(100)]
    public class ArmDetectionManager : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private PassthroughCameraSource _cameraSource;
        [SerializeField] private CustomArmDetector _customArmDetector;
        [SerializeField] private MediaPipeHandArmDetector _mediaPipeDetector;
        [SerializeField] private WearerHandFilter _wearerFilter;
        [SerializeField] private ArmOverlay _overlay;

        [Header("Detector selection")]
        [Tooltip("Use the trained custom arm model as the whole-arm detector.")]
        [SerializeField] private bool _useCustomArmDetector = true;
        [Tooltip("If the custom detector finds no arm, use MediaPipe hand landmarks as a fallback.")]
        [SerializeField] private bool _fallbackToMediaPipe = true;

        [Header("Depth estimation")]
        [SerializeField] private EnvironmentRaycastManager _depthRaycaster;
        [SerializeField] private float _assumedPersonHeightMeters = 1.7f;
        [SerializeField] private float _minPersonBboxPixels = 10f;
        [SerializeField] private float _minDepthMeters = 0.3f;
        [SerializeField] private float _maxDepthMeters = 8f;

        [Header("Arm filtering")]
        [SerializeField, Range(0f, 1f)] private float _keypointConfidence = 0.01f;
        [SerializeField] private bool _skipWearerFilterInArmOnlyMode = true;
        [SerializeField] private float _armOnlyStabilityRadiusPixels = 420f;
        [SerializeField, Range(0f, 1f)] private float _armOnlyImageSmoothing = 0.65f;
        [SerializeField, Range(0f, 1f)] private float _armOnlyWorldSmoothing = 0.55f;
        [SerializeField, Range(0, 90)] private int _armOnlyLockLostFrames = 30;
        [SerializeField, Range(1, 12)] private int _customDetectorEveryNFrames = 1;

        [Header("Debug")]
        [SerializeField] private ArmDetectionDebugHUD _debugHUD;
        [SerializeField] private bool _bypassWearerFilter = true;

        public int InferenceCount { get; private set; }
        public int LastPersonCount { get; private set; }
        public bool LastFoundArm { get; private set; }
        public string ManagerStatus { get; private set; } = "Not started";
        public string LastArmStatus { get; private set; } = "-";
        public bool IsArmOnlyMode => true;
        public List<PersonDetection> LastDetections { get; } = new();
        public int SelectedDetectionIndex { get; private set; } = -1;
        public Side SelectedDetectionSide { get; private set; } = Side.Left;
        public float LastMaxArmScore => Mathf.Max(
            _customArmDetector != null ? _customArmDetector.LastArmOnlyMaxScore : 0f,
            _mediaPipeDetector != null ? _mediaPipeDetector.LastArmOnlyMaxScore : 0f);

        public void SetArmOnlyMode(bool armOnly)
        {
            Debug.Log("[ArmManager] Arm detection mode is fixed to ARM-ONLY.");
        }

        /// <summary>True while the tracker has a valid smoothed arm in world space.</summary>
        public bool IsLocked => _hasSmoothedArmWorld && _stableArmLostFrames < _armOnlyLockLostFrames;

        /// <summary>
        /// Returns the current smoothed shoulder and wrist world positions.
        /// Returns false when no arm is locked (no valid detection yet).
        /// </summary>
        public bool TryGetArmEndpoints(out Vector3 shoulder, out Vector3 wrist)
        {
            shoulder = _smoothedShoulderWorld;
            wrist    = _smoothedWristWorld;
            return IsLocked;
        }

        private struct ArmCandidate
        {
            public Vector3 Shoulder;
            public Vector3 Wrist;
            public float Depth;
            public float Score;
            public Vector2 ShoulderImage;
            public Vector2 ElbowImage;
            public Vector2 WristImage;
            public Side Side;
        }

        private bool _hasStableArmImage;
        private Side _stableArmSide;
        private Vector2 _stableArmMidpointImage;
        private bool _hasSmoothedArmWorld;
        private Vector3 _smoothedShoulderWorld;
        private Vector3 _smoothedWristWorld;
        private PersonDetection _stableArmDetection;
        private int _stableArmLostFrames;
        private bool _currentDetectionsAreMediaPipe;
        private bool _currentDetectionsAreCustom;
        private int _customDetectorFrameCounter;
        private readonly List<PersonDetection> _cachedCustomDetections = new();

        private void Reset()
        {
            _cameraSource = GetComponentInChildren<PassthroughCameraSource>();
            _customArmDetector = GetComponentInChildren<CustomArmDetector>();
            _mediaPipeDetector = GetComponentInChildren<MediaPipeHandArmDetector>();
            _wearerFilter = GetComponentInChildren<WearerHandFilter>();
            _overlay = GetComponentInChildren<ArmOverlay>();
        }

        private void Start()
        {
            if (_customArmDetector == null) _customArmDetector = GetComponentInChildren<CustomArmDetector>();
            if (_mediaPipeDetector == null) _mediaPipeDetector = GetComponentInChildren<MediaPipeHandArmDetector>();
            if (_cameraSource == null) Debug.LogError("[ArmManager] _cameraSource is NULL.");
            if (_customArmDetector == null && _mediaPipeDetector == null) Debug.LogError("[ArmManager] No detector assigned.");
            if (_overlay == null) Debug.LogError("[ArmManager] _overlay is NULL.");
        }

        private void Update()
        {
            if (Time.frameCount % 120 == 0)
            {
                Debug.Log($"[ArmManager] cam={_cameraSource != null} custom={_customArmDetector != null} " +
                          $"mp={_mediaPipeDetector != null} " +
                          $"overlay={_overlay != null} hasFrame={(_cameraSource != null && _cameraSource.HasFrame)} " +
                          $"modelReady={HasReadyDetector()} inferenceCount={InferenceCount} armStatus={LastArmStatus}");
            }

            if (_cameraSource == null) { ManagerStatus = "ERR: CameraSource not assigned"; return; }
            if (_customArmDetector == null && _mediaPipeDetector == null) { ManagerStatus = "ERR: No detector assigned"; return; }
            if (_overlay == null) { ManagerStatus = "ERR: Overlay not assigned"; return; }
            if (!_cameraSource.HasFrame) { ManagerStatus = "Waiting: no camera frame"; return; }
            if (!HasReadyDetector()) { ManagerStatus = "Waiting: detector not ready"; return; }

            _currentDetectionsAreCustom = false;
            _currentDetectionsAreMediaPipe = false;
            List<PersonDetection> persons = RunPrimaryDetector();
            LastPersonCount = persons.Count;

            LastDetections.Clear();
            LastDetections.AddRange(persons);

            bool found = false;
            ArmCandidate best = default;
            int selectedIdx = -1;
            Side selectedSide = Side.Left;
            int bboxPassCount = 0;
            int keypointPassCount = 0;
            int filterPassCount = 0;
            int depthRaycastHits = 0;

            for (int i = 0; i < persons.Count; i++)
            {
                var p = persons[i];
                float bboxSize = Mathf.Max(p.ImageBounds.width, p.ImageBounds.height);
                if (bboxSize < _minPersonBboxPixels) continue;
                bboxPassCount++;

                float depth = EstimateDepth(p);
                if (TryArm(p, Side.Left, depth, ref found, ref best, ref keypointPassCount, ref filterPassCount, ref depthRaycastHits))
                {
                    selectedIdx = i;
                    selectedSide = Side.Left;
                }
                if (TryArm(p, Side.Right, depth, ref found, ref best, ref keypointPassCount, ref filterPassCount, ref depthRaycastHits))
                {
                    selectedIdx = i;
                    selectedSide = Side.Right;
                }
            }

            if (found)
            {
                StabilizeArmOnlyCandidate(ref best, selectedSide, selectedIdx);
            }
            else
            {
                TryUseLockedArmOnlyCandidate(ref found, ref best, ref selectedIdx, ref selectedSide);
            }

            SelectedDetectionIndex = found ? selectedIdx : -1;
            SelectedDetectionSide = selectedSide;

            if (found)
            {
                string depthSrc = depthRaycastHits > 0 ? "depth-API" : "heuristic";
                string source = _currentDetectionsAreMediaPipe ? "MediaPipe fallback" : "custom arm";
                LastArmStatus = $"OK - {best.Depth:F1} m away ({depthSrc}) [{source}]";
            }
            else if (persons.Count == 0)
            {
                LastArmStatus = _currentDetectionsAreMediaPipe
                    ? $"0 hands ({_mediaPipeDetector.Status})"
                    : $"0 arms ({(_customArmDetector != null ? _customArmDetector.Status : "custom detector missing")})";
            }
            else if (bboxPassCount == 0)
            {
                LastArmStatus = $"{persons.Count} detection(s) bbox too small (<{_minPersonBboxPixels}px)";
            }
            else if (keypointPassCount == 0)
            {
                LastArmStatus = $"Keypoints too low (conf<{_keypointConfidence:F2})";
            }
            else if (filterPassCount == 0)
            {
                LastArmStatus = "All arms blocked by WearerFilter";
            }
            else
            {
                LastArmStatus = "No arm selected";
            }

            var camTransform = _cameraSource.CameraTransform;
            _overlay.Render(found ? (best.Shoulder, best.Wrist) : null, camTransform);

            LastFoundArm = found;
            _debugHUD?.ReportDetections(persons.Count, found ? 1 : 0);
        }

        public float GetEstimatedDepth(PersonDetection p) => EstimateDepth(p);

        private bool HasReadyDetector()
        {
            bool customReady = _useCustomArmDetector && _customArmDetector != null && _customArmDetector.IsReady;
            bool mediaPipeReady = _mediaPipeDetector != null && _mediaPipeDetector.IsReady;
            return customReady || mediaPipeReady;
        }

        private List<PersonDetection> RunPrimaryDetector()
        {
            bool canRunCustom = _useCustomArmDetector && _customArmDetector != null && _customArmDetector.IsReady;
            if (canRunCustom)
            {
                _customDetectorFrameCounter++;
                bool shouldRun = _cachedCustomDetections.Count == 0
                              || _customDetectorFrameCounter % Mathf.Max(1, _customDetectorEveryNFrames) == 0;

                if (shouldRun)
                {
                    var detections = _customArmDetector.Run(_cameraSource.CurrentTexture);
                    if (_customArmDetector.LastRunScheduledInference) InferenceCount++;

                    _cachedCustomDetections.Clear();
                    _cachedCustomDetections.AddRange(detections);
                }

                if (_cachedCustomDetections.Count > 0 || !_fallbackToMediaPipe || _mediaPipeDetector == null || !_mediaPipeDetector.IsReady)
                {
                    _currentDetectionsAreCustom = true;
                    ManagerStatus = "Running [Custom arm detector]";
                    return _cachedCustomDetections;
                }
            }

            if (_mediaPipeDetector != null && _mediaPipeDetector.IsReady)
            {
                var mediaPipePersons = _mediaPipeDetector.Run(null);
                if (_mediaPipeDetector.LastRunConsumedNewResult) InferenceCount++;
                _currentDetectionsAreMediaPipe = true;
                ManagerStatus = "Running [MediaPipe fallback]";
                return mediaPipePersons;
            }

            _currentDetectionsAreCustom = true;
            return _cachedCustomDetections;
        }

        private float EstimateDepth(PersonDetection p)
        {
            float imageH = _cameraSource.Height;
            if (imageH <= 0f) return _minDepthMeters;
            float focalPx = (imageH * 0.5f) / Mathf.Tan(_cameraSource.VerticalFovRadians * 0.5f);

            float armDepth = EstimateDepthFromArmLength(p, focalPx);
            if (armDepth > 0f) return armDepth;

            float bboxH = Mathf.Max(1f, p.ImageBounds.height);
            float depth = _assumedPersonHeightMeters * focalPx / bboxH;
            return Mathf.Clamp(depth, _minDepthMeters, _maxDepthMeters);
        }

        private float EstimateDepthFromArmLength(PersonDetection p, float focalPx)
        {
            const float AssumedArmLengthMeters = 0.65f;
            float bestConf = 0f;
            Vector2 bestShoulder = default;
            Vector2 bestWrist = default;

            foreach (var pair in new[]
            {
                ((int)CocoKeypoint.LeftShoulder, (int)CocoKeypoint.LeftWrist),
                ((int)CocoKeypoint.RightShoulder, (int)CocoKeypoint.RightWrist),
            })
            {
                float conf = Mathf.Min(p.Keypoints[pair.Item1].Confidence, p.Keypoints[pair.Item2].Confidence);
                if (conf > bestConf)
                {
                    bestConf = conf;
                    bestShoulder = p.Keypoints[pair.Item1].ImagePos;
                    bestWrist = p.Keypoints[pair.Item2].ImagePos;
                }
            }

            if (bestConf < 0.05f) return 0f;
            float armPx = Vector2.Distance(bestShoulder, bestWrist);
            if (armPx < 5f) return 0f;
            return Mathf.Clamp(AssumedArmLengthMeters * focalPx / armPx, _minDepthMeters, _maxDepthMeters);
        }

        private bool TryArm(PersonDetection p, Side side, float depthHeuristic,
                            ref bool found, ref ArmCandidate best,
                            ref int keypointPassCount, ref int filterPassCount,
                            ref int depthRaycastHits)
        {
            int shoulderIdx = side == Side.Left ? (int)CocoKeypoint.LeftShoulder : (int)CocoKeypoint.RightShoulder;
            int elbowIdx = side == Side.Left ? (int)CocoKeypoint.LeftElbow : (int)CocoKeypoint.RightElbow;
            int wristIdx = side == Side.Left ? (int)CocoKeypoint.LeftWrist : (int)CocoKeypoint.RightWrist;

            var shoulder = p.Keypoints[shoulderIdx];
            var elbow = p.Keypoints[elbowIdx];
            var wrist = p.Keypoints[wristIdx];

            if (shoulder.Confidence < _keypointConfidence || wrist.Confidence < _keypointConfidence) return false;
            keypointPassCount++;

            var arm = new ArmDetection
            {
                Side = side,
                ShoulderImage = shoulder.ImagePos,
                ElbowImage = elbow.ImagePos,
                WristImage = wrist.ImagePos,
                Confidence = Mathf.Min(shoulder.Confidence, wrist.Confidence),
            };

            bool skipWearerFilter = _bypassWearerFilter || (IsArmOnlyMode && _skipWearerFilterInArmOnlyMode);
            if (!skipWearerFilter && _wearerFilter != null && _wearerFilter.IsWearerArm(arm, _cameraSource)) return false;
            filterPassCount++;

            Vector3 shoulderWorld = ProjectImagePoint(arm.ShoulderImage, depthHeuristic, out bool _);
            Vector3 wristWorld = ProjectImagePoint(arm.WristImage, depthHeuristic, out bool wristHit);
            if (wristHit) depthRaycastHits++;

            float effectiveDepth = Vector3.Distance(_cameraSource.CameraPose.position, wristWorld);
            float score = arm.Confidence - effectiveDepth * 0.015f;

            if (_hasStableArmImage && _stableArmSide == side)
            {
                Vector2 midpoint = (arm.ShoulderImage + arm.WristImage) * 0.5f;
                float jump = Vector2.Distance(midpoint, _stableArmMidpointImage);
                float continuity = Mathf.Clamp01(1f - jump / Mathf.Max(1f, _armOnlyStabilityRadiusPixels));
                score += continuity * 0.5f;
            }

            if (!found || score > best.Score)
            {
                best = new ArmCandidate
                {
                    Shoulder = shoulderWorld,
                    Wrist = wristWorld,
                    Depth = effectiveDepth,
                    Score = score,
                    ShoulderImage = arm.ShoulderImage,
                    ElbowImage = arm.ElbowImage,
                    WristImage = arm.WristImage,
                    Side = side,
                };
                found = true;
                return true;
            }
            return false;
        }

        private void StabilizeArmOnlyCandidate(ref ArmCandidate best, Side selectedSide, int selectedIdx)
        {
            Vector2 midpoint = (best.ShoulderImage + best.WristImage) * 0.5f;
            float imageT = 1f - _armOnlyImageSmoothing;

            if (_hasStableArmImage)
            {
                best.ShoulderImage = Vector2.Lerp(_stableArmDetection.Keypoints[(int)(selectedSide == Side.Left ? CocoKeypoint.LeftShoulder : CocoKeypoint.RightShoulder)].ImagePos, best.ShoulderImage, imageT);
                best.ElbowImage = Vector2.Lerp(_stableArmDetection.Keypoints[(int)(selectedSide == Side.Left ? CocoKeypoint.LeftElbow : CocoKeypoint.RightElbow)].ImagePos, best.ElbowImage, imageT);
                best.WristImage = Vector2.Lerp(_stableArmDetection.Keypoints[(int)(selectedSide == Side.Left ? CocoKeypoint.LeftWrist : CocoKeypoint.RightWrist)].ImagePos, best.WristImage, imageT);
                midpoint = (best.ShoulderImage + best.WristImage) * 0.5f;
                _stableArmMidpointImage = Vector2.Lerp(_stableArmMidpointImage, midpoint, imageT);
            }
            else
            {
                _stableArmMidpointImage = midpoint;
            }

            _stableArmSide = selectedSide;
            _hasStableArmImage = true;
            _stableArmLostFrames = 0;

            if (!_hasSmoothedArmWorld)
            {
                _smoothedShoulderWorld = best.Shoulder;
                _smoothedWristWorld = best.Wrist;
                _hasSmoothedArmWorld = true;
            }
            else
            {
                float t = 1f - _armOnlyWorldSmoothing;
                _smoothedShoulderWorld = Vector3.Lerp(_smoothedShoulderWorld, best.Shoulder, t);
                _smoothedWristWorld = Vector3.Lerp(_smoothedWristWorld, best.Wrist, t);
            }

            best.Shoulder = _smoothedShoulderWorld;
            best.Wrist = _smoothedWristWorld;

            if (selectedIdx >= 0 && selectedIdx < LastDetections.Count)
            {
                _stableArmDetection = BuildSmoothedDetection(LastDetections[selectedIdx], best, selectedSide);
                LastDetections[selectedIdx] = _stableArmDetection;
            }
        }

        private bool TryUseLockedArmOnlyCandidate(ref bool found, ref ArmCandidate best, ref int selectedIdx, ref Side selectedSide)
        {
            if (!_hasStableArmImage || _stableArmLostFrames >= _armOnlyLockLostFrames)
            {
                ResetArmStability();
                return false;
            }

            _stableArmLostFrames++;
            LastDetections.Clear();
            LastDetections.Add(_stableArmDetection);
            selectedIdx = 0;
            selectedSide = _stableArmSide;
            best = new ArmCandidate
            {
                Shoulder = _smoothedShoulderWorld,
                Wrist = _smoothedWristWorld,
                Depth = Vector3.Distance(_cameraSource.CameraPose.position, _smoothedWristWorld),
                Score = _stableArmDetection.Confidence,
                ShoulderImage = _stableArmDetection.Keypoints[(int)(selectedSide == Side.Left ? CocoKeypoint.LeftShoulder : CocoKeypoint.RightShoulder)].ImagePos,
                ElbowImage = _stableArmDetection.Keypoints[(int)(selectedSide == Side.Left ? CocoKeypoint.LeftElbow : CocoKeypoint.RightElbow)].ImagePos,
                WristImage = _stableArmDetection.Keypoints[(int)(selectedSide == Side.Left ? CocoKeypoint.LeftWrist : CocoKeypoint.RightWrist)].ImagePos,
                Side = selectedSide,
            };
            found = true;
            return true;
        }

        private static PersonDetection BuildSmoothedDetection(PersonDetection source, ArmCandidate arm, Side side)
        {
            var keypoints = (Keypoint[])source.Keypoints.Clone();
            int shoulderIdx = side == Side.Left ? (int)CocoKeypoint.LeftShoulder : (int)CocoKeypoint.RightShoulder;
            int elbowIdx = side == Side.Left ? (int)CocoKeypoint.LeftElbow : (int)CocoKeypoint.RightElbow;
            int wristIdx = side == Side.Left ? (int)CocoKeypoint.LeftWrist : (int)CocoKeypoint.RightWrist;

            keypoints[shoulderIdx] = new Keypoint { ImagePos = arm.ShoulderImage, Confidence = keypoints[shoulderIdx].Confidence };
            keypoints[elbowIdx] = new Keypoint { ImagePos = arm.ElbowImage, Confidence = keypoints[elbowIdx].Confidence };
            keypoints[wristIdx] = new Keypoint { ImagePos = arm.WristImage, Confidence = keypoints[wristIdx].Confidence };

            return new PersonDetection
            {
                Confidence = source.Confidence,
                ImageBounds = BuildArmImageBounds(arm.ShoulderImage, arm.ElbowImage, arm.WristImage),
                Keypoints = keypoints,
            };
        }

        private static Rect BuildArmImageBounds(Vector2 shoulder, Vector2 elbow, Vector2 wrist)
        {
            const float Padding = 28f;
            float minX = Mathf.Min(Mathf.Min(shoulder.x, elbow.x), wrist.x) - Padding;
            float minY = Mathf.Min(Mathf.Min(shoulder.y, elbow.y), wrist.y) - Padding;
            float maxX = Mathf.Max(Mathf.Max(shoulder.x, elbow.x), wrist.x) + Padding;
            float maxY = Mathf.Max(Mathf.Max(shoulder.y, elbow.y), wrist.y) + Padding;
            return new Rect(minX, minY, maxX - minX, maxY - minY);
        }

        private void ResetArmStability()
        {
            _hasStableArmImage = false;
            _hasSmoothedArmWorld = false;
            _stableArmLostFrames = 0;
        }

        private Vector3 ProjectImagePoint(Vector2 imagePoint, float fallbackDepth, out bool usedRaycast)
        {
            usedRaycast = false;

            if (_depthRaycaster != null && EnvironmentRaycastManager.IsSupported)
            {
                var ray = _cameraSource.ImagePointToRay(imagePoint);
                if (_depthRaycaster.Raycast(ray, out var hit, _maxDepthMeters))
                {
                    usedRaycast = true;
                    return hit.point;
                }
            }

            return _cameraSource.ImagePointToWorld(imagePoint, fallbackDepth);
        }
    }
}
