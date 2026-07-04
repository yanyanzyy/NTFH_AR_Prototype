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

        [Header("Depth-based arm axis (bypasses keypoint regression)")]
        [Tooltip("When a depth raycaster is available, estimate the arm's 3D axis by sampling a grid " +
                 "of points across the detected box and raycasting them against real-world depth, then " +
                 "running PCA on the resulting point cloud. Use this when the model's keypoint head has " +
                 "no reliable training signal for the Arm class (box/class only) - it derives orientation " +
                 "from sensed geometry instead of the (currently untrained) keypoint regression.")]
        [SerializeField] private bool _useDepthAxisEstimation = true;
        [Tooltip("Sample grid resolution across the box (grid x grid points raycasted).")]
        [SerializeField, Range(3, 12)] private int _depthSampleGrid = 6;
        [Tooltip("Flip which depth-derived axis endpoint is treated as shoulder vs wrist. Toggle once " +
                 "after visually checking overlay orientation against your fixed mannequin setup.")]
        [SerializeField] private bool _swapDepthAxisEndpoints = false;

        [Header("Fixed-axis fallback (no depth sensing or trained keypoints needed)")]
        [Tooltip("When depth-axis estimation can't get hits (e.g. raycasting against a handheld prop " +
                 "that was never scanned into the room mesh), fall back to a FIXED real-world direction " +
                 "for the arm axis instead of the untrained keypoints. Works because the mannequin is " +
                 "stationary between sessions - calibrate once by rotating _fixedArmYawDegrees in the " +
                 "Inspector at Play time until the overlay points the right way, then leave it.")]
        [SerializeField] private bool _useFixedWorldAxis = true;
        [Tooltip("Rotation around the world Y axis (degrees) that the arm points along, assuming it " +
                 "lies roughly horizontal. 0 = world +Z, 90 = world +X, etc. Calibrate this in Play mode.")]
        [SerializeField] private float _fixedArmYawDegrees = 0f;
        [Tooltip("Real length of the mannequin forearm in metres (placed centred on the detected box).")]
        [SerializeField] private float _fixedArmLengthMeters = 0.55f;
        [Tooltip("Keep the fixed-axis overlay at a stable depth so small box-size jitter doesn't make it zoom.")]
        [SerializeField] private bool _stabilizeFixedAxisDepth = true;
        [Tooltip("How strongly to smooth fixed-axis depth (0 = raw depth, 1 = frozen).")]
        [SerializeField, Range(0f, 1f)] private float _fixedAxisDepthSmoothing = 0.97f;
        [Tooltip("Flip which fixed-axis endpoint is treated as shoulder vs wrist.")]
        [SerializeField] private bool _swapFixedAxisEndpoints = false;

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

        /// <summary>Explains the last result of TryEstimateArmAxisFromDepth — why it succeeded
        /// or failed this frame. Surfaced in the debug HUD to diagnose depth-API issues.</summary>
        public string DepthAxisStatus { get; private set; } = "Not attempted yet";

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
        private int _customDetectorFrameCounter;
        private readonly List<PersonDetection> _cachedCustomDetections = new();

        private bool _hasStableFixedAxisDepth;
        private float _stableFixedAxisDepth;

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

            Vector3 shoulderWorld, wristWorld;
            if (TryEstimateArmAxisFromDepth(p.ImageBounds, out var axisEndA, out var axisEndB))
            {
                shoulderWorld = axisEndA;
                wristWorld = axisEndB;
                depthRaycastHits++;
            }
            else if (_useFixedWorldAxis)
            {
                // For a stationary mannequin, a stable heuristic depth is less jittery than
                // per-frame Depth API raycasts, which can make the overlay appear to zoom.
                float stableDepth = GetStableFixedAxisDepth(depthHeuristic);
                Vector3 centerWorld = _cameraSource.ImagePointToWorld(p.ImageBounds.center, stableDepth);
                GetFixedAxisEndpoints(centerWorld, out shoulderWorld, out wristWorld);
                DepthAxisStatus = $"Fixed-axis fallback - yaw={_fixedArmYawDegrees:F0} deg, " +
                                  $"length={_fixedArmLengthMeters:F2} m, depth={stableDepth:F2} m";
            }
            else
            {
                shoulderWorld = ProjectImagePoint(arm.ShoulderImage, depthHeuristic, out bool _);
                wristWorld = ProjectImagePoint(arm.WristImage, depthHeuristic, out bool wristHit);
                if (wristHit) depthRaycastHits++;
            }

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
            _hasStableFixedAxisDepth = false;
        }

        private float GetStableFixedAxisDepth(float rawDepth)
        {
            rawDepth = Mathf.Clamp(rawDepth, _minDepthMeters, _maxDepthMeters);
            if (!_stabilizeFixedAxisDepth) return rawDepth;

            if (!_hasStableFixedAxisDepth)
            {
                _stableFixedAxisDepth = rawDepth;
                _hasStableFixedAxisDepth = true;
            }
            else
            {
                float t = 1f - _fixedAxisDepthSmoothing;
                _stableFixedAxisDepth = Mathf.Lerp(_stableFixedAxisDepth, rawDepth, t);
            }

            return _stableFixedAxisDepth;
        }

        /// <summary>
        /// Estimates the arm's 3D axis directly from sensed depth, bypassing the model's
        /// keypoint regression entirely. Samples a grid of points inside <paramref name="imageBounds"/>,
        /// raycasts each against real-world depth via the Depth API, and runs PCA on the
        /// resulting point cloud to find the dominant (long) axis. Returns the two extreme
        /// points along that axis as the arm's endpoints.
        ///
        /// Use this when the bounding box is reliable but the keypoint head has no real
        /// training signal (e.g. only box/class supervision exists for a class) - orientation
        /// comes from sensed geometry instead of an untrained regression output.
        /// </summary>
        private bool TryEstimateArmAxisFromDepth(Rect imageBounds, out Vector3 endA, out Vector3 endB)
        {
            endA = endB = default;

            if (!_useDepthAxisEstimation)
            {
                DepthAxisStatus = "Disabled (_useDepthAxisEstimation = false)";
                return false;
            }
            if (_depthRaycaster == null)
            {
                DepthAxisStatus = "No EnvironmentRaycastManager assigned to _depthRaycaster";
                return false;
            }
            if (!EnvironmentRaycastManager.IsSupported)
            {
                DepthAxisStatus = "EnvironmentRaycastManager.IsSupported = false " +
                                   "(device/OS doesn't support Depth API, or Spatial Data permission not granted)";
                return false;
            }

            int grid = _depthSampleGrid;
            // Shrink sampling inward from the box edges so background pixels (where the box
            // slightly overshoots the real arm silhouette) don't get raycasted.
            float padX = imageBounds.width * 0.1f;
            float padY = imageBounds.height * 0.1f;

            var hits = new List<Vector3>(grid * grid);
            for (int gx = 0; gx < grid; gx++)
            {
                float u = grid > 1 ? gx / (float)(grid - 1) : 0.5f;
                float px = Mathf.Lerp(imageBounds.xMin + padX, imageBounds.xMax - padX, u);

                for (int gy = 0; gy < grid; gy++)
                {
                    float v = grid > 1 ? gy / (float)(grid - 1) : 0.5f;
                    float py = Mathf.Lerp(imageBounds.yMin + padY, imageBounds.yMax - padY, v);

                    var ray = _cameraSource.ImagePointToRay(new Vector2(px, py));
                    if (_depthRaycaster.Raycast(ray, out var hit, _maxDepthMeters))
                        hits.Add(hit.point);
                }
            }

            if (hits.Count < 6)
            {
                DepthAxisStatus = $"Only {hits.Count}/{grid * grid} depth samples hit this frame (<6 needed)";
                return false;
            }

            // Outlier rejection: keep points near the median depth so any background pixels
            // that slipped through don't skew the axis away from the true arm surface.
            Vector3 camPos = _cameraSource.CameraPose.position;
            var depths = new List<float>(hits.Count);
            foreach (var h in hits) depths.Add(Vector3.Distance(camPos, h));
            depths.Sort();
            float medianDepth = depths[depths.Count / 2];

            var filtered = new List<Vector3>(hits.Count);
            foreach (var h in hits)
                if (Mathf.Abs(Vector3.Distance(camPos, h) - medianDepth) < 0.15f)
                    filtered.Add(h);
            if (filtered.Count < 5) filtered = hits;

            Vector3 centroid = Vector3.zero;
            foreach (var p in filtered) centroid += p;
            centroid /= filtered.Count;

            float xx = 0f, xy = 0f, xz = 0f, yy = 0f, yz = 0f, zz = 0f;
            foreach (var p in filtered)
            {
                Vector3 d = p - centroid;
                xx += d.x * d.x; xy += d.x * d.y; xz += d.x * d.z;
                yy += d.y * d.y; yz += d.y * d.z; zz += d.z * d.z;
            }

            Vector3 axis = DominantEigenvector(xx, xy, xz, yy, yz, zz);

            float minProj = float.MaxValue, maxProj = float.MinValue;
            Vector3 minPt = centroid, maxPt = centroid;
            foreach (var p in filtered)
            {
                float proj = Vector3.Dot(p - centroid, axis);
                if (proj < minProj) { minProj = proj; minPt = p; }
                if (proj > maxProj) { maxProj = proj; maxPt = p; }
            }

            endA = _swapDepthAxisEndpoints ? maxPt : minPt;
            endB = _swapDepthAxisEndpoints ? minPt : maxPt;
            DepthAxisStatus = $"OK - {filtered.Count}/{hits.Count} samples, axis length {Vector3.Distance(endA, endB):F2} m";
            return true;
        }

        /// <summary>
        /// Places the two arm endpoints around <paramref name="centerWorld"/> using a fixed,
        /// manually-calibrated real-world direction instead of sensed depth or trained
        /// keypoints. Valid because the mannequin is a stationary prop - calibrate
        /// _fixedArmYawDegrees once in Play mode and it stays correct across sessions.
        /// </summary>
        private void GetFixedAxisEndpoints(Vector3 centerWorld, out Vector3 endA, out Vector3 endB)
        {
            Vector3 dir = Quaternion.Euler(0f, _fixedArmYawDegrees, 0f) * Vector3.forward;
            Vector3 half = dir.normalized * (_fixedArmLengthMeters * 0.5f);
            Vector3 a = centerWorld - half;
            Vector3 b = centerWorld + half;
            endA = _swapFixedAxisEndpoints ? b : a;
            endB = _swapFixedAxisEndpoints ? a : b;
        }

        /// <summary>Dominant eigenvector of a 3x3 symmetric matrix via power iteration (a few
        /// iterations are enough since only the principal axis direction is needed).</summary>
        private static Vector3 DominantEigenvector(float xx, float xy, float xz, float yy, float yz, float zz)
        {
            Vector3 v = new Vector3(1f, 1f, 1f).normalized;
            for (int i = 0; i < 12; i++)
            {
                Vector3 mv = new Vector3(
                    xx * v.x + xy * v.y + xz * v.z,
                    xy * v.x + yy * v.y + yz * v.z,
                    xz * v.x + yz * v.y + zz * v.z);
                if (mv.sqrMagnitude < 1e-9f) break;
                v = mv.normalized;
            }
            return v;
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
