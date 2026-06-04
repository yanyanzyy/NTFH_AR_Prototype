using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Orchestrates the arm-detection pipeline each frame:
    ///   1. Reads MediaPipe hand landmarks from the current passthrough frame.
    ///   2. Converts synthetic arm keypoints to world-space arm endpoints.
    ///   3. Rejects the wearer's own arms via WearerHandFilter.
    ///   4. Among remaining candidates, picks the SINGLE CLOSEST arm to the camera.
    ///   5. Passes it to ArmOverlay to render the red world-space quad.
    /// </summary>
    [DefaultExecutionOrder(100)]
    public class ArmDetectionManager : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private PassthroughCameraSource _cameraSource;
        [SerializeField] private CustomArmDetector _customArmDetector;
        [SerializeField] private MediaPipeHandArmDetector _mediaPipeDetector;
        [SerializeField] private WearerHandFilter        _wearerFilter;
        [SerializeField] private ArmOverlay              _overlay;

        [Header("Detector selection")]
        [Tooltip("Use the trained custom arm model as the primary whole-arm detector.")]
        [SerializeField] private bool _useCustomArmDetector = true;
        [Tooltip("If the custom detector finds no arm, use MediaPipe hand landmarks as a fallback.")]
        [SerializeField] private bool _fallbackToMediaPipe = true;

        [Header("Depth estimation")]
        [Tooltip("Optional: Meta Depth-API raycaster. When present, the manager raycasts the wrist's " +
                 "viewport ray against real-world geometry to get accurate depth, replacing the " +
                 "person-height / arm-length heuristic. Add a Meta.XR.EnvironmentRaycastManager to " +
                 "the scene and wire it here.")]
        [SerializeField] private EnvironmentRaycastManager _depthRaycaster;
        [Tooltip("Assumed adult height in metres, used to derive person depth from bbox height " +
                 "when the depth raycaster is unavailable or misses.")]
        [SerializeField] private float _assumedPersonHeightMeters = 1.7f;
        [Tooltip("Minimum bbox height (pixels) before a detection is considered valid.")]
        [SerializeField] private float _minPersonBboxPixels = 40f;
        [SerializeField] private float _minDepthMeters = 0.3f;
        [SerializeField] private float _maxDepthMeters = 8f;

        [Header("Arm filtering")]
        [Tooltip("Minimum keypoint confidence for shoulder and wrist to accept an arm.")]
        [SerializeField, Range(0f, 1f)] private float _keypointConfidence = 0.05f;
        [Tooltip("Skip wearer-arm rejection while in arm-only mode. Leave off for headset use.")]
        [SerializeField] private bool _skipWearerFilterInArmOnlyMode = false;
        [Tooltip("How far an arm-only detection can jump in image pixels before it loses the previous-arm preference.")]
        [SerializeField] private float _armOnlyStabilityRadiusPixels = 220f;
        [Tooltip("Image-space smoothing for the selected arm-only debug box/keypoints.")]
        [SerializeField, Range(0f, 1f)] private float _armOnlyImageSmoothing = 0.65f;
        [Tooltip("World-space smoothing for selected arm-only overlay endpoints.")]
        [SerializeField, Range(0f, 1f)] private float _armOnlyWorldSmoothing = 0.55f;
        [Tooltip("Keep the last valid non-wearer arm visible through short MediaPipe dropouts.")]
        [SerializeField, Range(0, 90)] private int _armOnlyLockLostFrames = 30;
        [Tooltip("Run custom arm model every N frames. Higher = faster but less responsive.")]
        [SerializeField, Range(1, 12)] private int _customDetectorEveryNFrames = 4;

        [Header("Debug")]
        [Tooltip("Optional debug HUD. Attach ArmDetectionDebugHUD component here.")]
        [SerializeField] private ArmDetectionDebugHUD _debugHUD;
        [Tooltip("Skip the WearerHandFilter entirely. Use this to confirm detection works before enabling filtering.")]
        [SerializeField] private bool _bypassWearerFilter = false;

        // ── Private state ──────────────────────────────────────────────────────────────

        // Readable by the debug HUD.
        public int    InferenceCount    { get; private set; }
        public int    LastPersonCount   { get; private set; }
        public bool   LastFoundArm      { get; private set; }
        /// <summary>One-line status shown in the debug HUD — explains the manager state or any blocking condition.</summary>
        public string ManagerStatus   { get; private set; } = "Not started";
        /// <summary>Explains why no arm was found on the last inference frame.</summary>
        public string LastArmStatus   { get; private set; } = "—";
        /// <summary>The prototype now always runs as an arm-only detector.</summary>
        public bool   IsArmOnlyMode   => true;
        /// <summary>All PersonDetections from the last inference frame (copied from detector scratch list).</summary>
        public List<PersonDetection> LastDetections { get; } = new();
        /// <summary>
        /// Index into <see cref="LastDetections"/> of the detection the manager selected
        /// for overlay rendering this frame, or -1 if no arm was found. Used by
        /// ArmBoundingBoxDebug to draw the bbox/keypoints of the chosen target only.
        /// </summary>
        public int SelectedDetectionIndex { get; private set; } = -1;
        /// <summary>The side (Left/Right) of the selected arm. Undefined when SelectedDetectionIndex is -1.</summary>
        public Side SelectedDetectionSide { get; private set; } = Side.Left;
        /// <summary>Highest detector score from the custom arm model or MediaPipe fallback.</summary>
        public float  LastMaxArmScore => Mathf.Max(
            _customArmDetector != null ? _customArmDetector.LastArmOnlyMaxScore : 0f,
            _mediaPipeDetector != null ? _mediaPipeDetector.LastArmOnlyMaxScore : 0f);

        /// <summary>Toggle detection mode at runtime — called by DetectionModeButton.</summary>
        public void SetArmOnlyMode(bool armOnly)
        {
            Debug.Log("[ArmManager] Arm detection mode is fixed to ARM-ONLY.");
        }

        // Best arm candidate this frame.
        private struct ArmCandidate
        {
            public Vector3 Shoulder;
            public Vector3 Wrist;
            public float   Depth;   // estimated metres from camera
            public float   Score;
            public Vector2 ShoulderImage;
            public Vector2 ElbowImage;
            public Vector2 WristImage;
            public Side    Side;
        }

        private bool    _hasStableArmImage;
        private Side    _stableArmSide;
        private Vector2 _stableArmMidpointImage;
        private bool    _hasSmoothedArmWorld;
        private Vector3 _smoothedShoulderWorld;
        private Vector3 _smoothedWristWorld;
        private PersonDetection _stableArmDetection;
        private int _stableArmLostFrames;
        private bool _currentDetectionsAreArmOnly;
        private bool _currentDetectionsAreMediaPipe;
        private bool _currentDetectionsAreCustom;
        private int _customDetectorFrameCounter;
        private readonly List<PersonDetection> _cachedCustomDetections = new();

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void Reset()
        {
            _cameraSource = GetComponentInChildren<PassthroughCameraSource>();
            _customArmDetector = GetComponentInChildren<CustomArmDetector>();
            _mediaPipeDetector = GetComponentInChildren<MediaPipeHandArmDetector>();
            _wearerFilter = GetComponentInChildren<WearerHandFilter>();
            _overlay      = GetComponentInChildren<ArmOverlay>();
        }

        private void Start()
        {
            Debug.Log($"[ArmManager] Start — cam={_cameraSource != null}  mp={_mediaPipeDetector != null}  " +
                      $"filter={_wearerFilter != null}  overlay={_overlay != null}  hud={_debugHUD != null}");
            if (_customArmDetector == null) _customArmDetector = GetComponentInChildren<CustomArmDetector>();
            if (_mediaPipeDetector == null) _mediaPipeDetector = GetComponentInChildren<MediaPipeHandArmDetector>();
            if (_cameraSource == null) Debug.LogError("[ArmManager] _cameraSource is NULL — drag PassthroughCameraSource here.");
            if (_customArmDetector == null && _mediaPipeDetector == null) Debug.LogError("[ArmManager] No detector assigned — add CustomArmDetector or MediaPipeHandArmDetector.");
            if (_overlay      == null) Debug.LogError("[ArmManager] _overlay is NULL — drag ArmOverlay here.");
        }

        private void Update()
        {
            // Log every ~2 seconds so we can see what is blocking via adb logcat.
            if (Time.frameCount % 120 == 0)
            {
                Debug.Log($"[ArmManager] " +
                          $"cam={_cameraSource != null} " +
                          $"custom={_customArmDetector != null} " +
                          $"mp={_mediaPipeDetector != null} " +
                          $"overlay={_overlay != null} " +
                          $"hasFrame={(_cameraSource != null && _cameraSource.HasFrame)} " +
                          $"modelReady={HasAnyReadyDetector()} " +
                          $"inferenceCount={InferenceCount}  armStatus={LastArmStatus}");
            }

            // Set ManagerStatus at every blocking point so the HUD shows the root cause.
            if (_cameraSource == null) { ManagerStatus = "ERR: CameraSource not assigned"; return; }
            if (_customArmDetector == null && _mediaPipeDetector == null) { ManagerStatus = "ERR: No arm detector assigned"; return; }
            if (_overlay      == null) { ManagerStatus = "ERR: Overlay not assigned";      return; }
            if (!_cameraSource.HasFrame)  { ManagerStatus = "Waiting: no camera frame";    return; }
            if (!HasAnyReadyDetector()) { ManagerStatus = "Waiting: arm detector not ready"; return; }

            ManagerStatus = "Running [Custom arm detector]";

            _currentDetectionsAreCustom = false;
            _currentDetectionsAreMediaPipe = false;
            _currentDetectionsAreArmOnly = true;

            List<PersonDetection> persons = RunPrimaryDetector();
            LastPersonCount = persons.Count;

            // Copy to stable list so debug visualisers can read it next frame.
            LastDetections.Clear();
            LastDetections.AddRange(persons);

            // ── Find the single closest non-wearer arm ─────────────────────────────────
            bool found = false;
            ArmCandidate best = default;
            int  selectedIdx  = -1;
            Side selectedSide = Side.Left;

            int bboxPassCount      = 0;
            int keypointPassCount  = 0;
            int filterPassCount    = 0;
            int depthRaycastHits   = 0;

            for (int i = 0; i < persons.Count; i++)
            {
                var p = persons[i];
                // Use the longer bbox dimension so a horizontal arm still passes.
                float bboxSize = Mathf.Max(p.ImageBounds.width, p.ImageBounds.height);
                if (bboxSize < _minPersonBboxPixels) continue;
                bboxPassCount++;

                float depth = EstimateDepth(p);
                if (TryArm(p, Side.Left, i, depth, ref found, ref best,
                           ref keypointPassCount, ref filterPassCount, ref depthRaycastHits))
                { selectedIdx = i; selectedSide = Side.Left; }
                if (TryArm(p, Side.Right, i, depth, ref found, ref best,
                           ref keypointPassCount, ref filterPassCount, ref depthRaycastHits))
                { selectedIdx = i; selectedSide = Side.Right; }
            }

            if (found && _currentDetectionsAreArmOnly)
            {
                StabilizeArmOnlyCandidate(ref best, selectedSide, selectedIdx);
            }
            else if (!found && IsArmOnlyMode && !_currentDetectionsAreCustom)
            {
                TryUseLockedArmOnlyCandidate(ref found, ref best, ref selectedIdx, ref selectedSide);
            }
            else if (!found)
            {
                ResetArmStability();
            }

            SelectedDetectionIndex = found ? selectedIdx : -1;
            SelectedDetectionSide  = selectedSide;

            // ── Compose arm status string ──────────────────────────────────────────────
            if (found)
            {
                string mode = _currentDetectionsAreMediaPipe
                    ? " [arm-only MediaPipe]"
                    : _currentDetectionsAreArmOnly ? " [arm-only mode]" : "";
                string depthSrc = depthRaycastHits > 0 ? "depth-API" : "heuristic";
                LastArmStatus = $"OK — {best.Depth:F1} m away ({depthSrc}){mode}";
            }
            else if (persons.Count == 0)
            {
                LastArmStatus = _currentDetectionsAreMediaPipe
                    ? $"0 hands ({_mediaPipeDetector.Status})"
                    : $"0 arms ({(_customArmDetector != null ? _customArmDetector.Status : "custom detector missing")})";
            }
            else if (bboxPassCount == 0)
            {
                LastArmStatus = $"{persons.Count} person(s) bbox too small (<{_minPersonBboxPixels}px)";
            }
            else if (keypointPassCount == 0)
            {
                LastArmStatus = $"Keypoints too low (conf<{_keypointConfidence:F2}) — lower threshold";
            }
            else if (filterPassCount == 0)
            {
                LastArmStatus = $"All arms blocked by WearerFilter — try BypassWearerFilter";
            }
            else
            {
                LastArmStatus = "No arm selected (unexpected)";
            }

            // Use the calibrated camera position when available so the overlay's "face
            // the camera" maths matches the pose used to build Shoulder/Wrist world coords.
            var camPose = _cameraSource.CameraPose;
            var camTransform = _cameraSource.CameraTransform;
            // Wrap the cached pose in a transient transform if needed — ArmOverlay reads .position.
            // The existing API takes a Transform; we pass CameraTransform but the world coords
            // were already computed against camPose, so this stays consistent.
            if (found)
                _overlay.Render((best.Shoulder, best.Wrist), camTransform);
            else
                _overlay.Render(null, camTransform);

            LastFoundArm = found;
            _debugHUD?.ReportDetections(persons.Count, found ? 1 : 0);
        }

        // ── Helpers ────────────────────────────────────────────────────────────────────

        /// <summary>Public wrapper used by the bounding-box debug visualiser.</summary>
        public float GetEstimatedDepth(PersonDetection p) => EstimateDepth(p);

        private bool HasAnyReadyDetector()
        {
            bool customReady = _useCustomArmDetector
                            && _customArmDetector != null
                            && _customArmDetector.IsReady;
            bool mediaPipeReady = _mediaPipeDetector != null && _mediaPipeDetector.IsReady;
            return customReady || mediaPipeReady;
        }

        private List<PersonDetection> RunPrimaryDetector()
        {
            bool canRunCustom = _useCustomArmDetector
                            && _customArmDetector != null
                            && _customArmDetector.IsReady;

            if (canRunCustom)
            {
                _customDetectorFrameCounter++;
                bool shouldRunCustom = _cachedCustomDetections.Count == 0
                                    || _customDetectorFrameCounter % Mathf.Max(1, _customDetectorEveryNFrames) == 0;

                if (shouldRunCustom)
                {
                    var customPersons = _customArmDetector.Run(_cameraSource.CurrentTexture);
                    if (_customArmDetector.LastRunScheduledInference) InferenceCount++;

                    if (customPersons.Count > 0)
                    {
                        _cachedCustomDetections.Clear();
                        _cachedCustomDetections.AddRange(customPersons);
                    }
                    else
                    {
                        _cachedCustomDetections.Clear();
                    }
                }

                if (_cachedCustomDetections.Count > 0 || !_fallbackToMediaPipe || _mediaPipeDetector == null || !_mediaPipeDetector.IsReady)
                {
                    _currentDetectionsAreCustom = true;
                    _currentDetectionsAreMediaPipe = false;
                    ManagerStatus = "Running [Custom arm detector]";
                    return _cachedCustomDetections;
                }
            }

            if (_mediaPipeDetector != null && _mediaPipeDetector.IsReady)
            {
                var mediaPipePersons = _mediaPipeDetector.Run(null);
                if (_mediaPipeDetector.LastRunConsumedNewResult) InferenceCount++;
                _currentDetectionsAreCustom = false;
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

            // If this came from the arm-only fallback, the bbox is built from keypoints
            // (not from a full-body anchor), so we use shoulder→wrist pixel distance
            // calibrated against assumed arm length instead.
            if (_currentDetectionsAreArmOnly)
            {
                float armDepth = EstimateDepthFromArmLength(p, focalPx);
                if (armDepth > 0f) return armDepth;
            }

            // Normal full-body path: bbox height ≈ person height.
            float bboxH = Mathf.Max(1f, p.ImageBounds.height);
            float depth = _assumedPersonHeightMeters * focalPx / bboxH;
            return Mathf.Clamp(depth, _minDepthMeters, _maxDepthMeters);
        }

        /// <summary>
        /// Estimates depth using the pixel distance between the detected shoulder and wrist,
        /// calibrated against a typical adult arm length. Used when no torso is visible.
        /// </summary>
        private float EstimateDepthFromArmLength(PersonDetection p, float focalPx)
        {
            const float AssumedArmLengthMeters = 0.65f; // shoulder → wrist, typical adult

            // Pick the arm side with the highest combined shoulder+wrist confidence.
            float bestConf = 0f;
            Vector2 bestShoulder = default, bestWrist = default;

            foreach (var (shIdx, wrIdx) in new[] {
                ((int)CocoKeypoint.LeftShoulder,  (int)CocoKeypoint.LeftWrist),
                ((int)CocoKeypoint.RightShoulder, (int)CocoKeypoint.RightWrist),
            })
            {
                float conf = Mathf.Min(p.Keypoints[shIdx].Confidence,
                                       p.Keypoints[wrIdx].Confidence);
                if (conf > bestConf)
                {
                    bestConf     = conf;
                    bestShoulder = p.Keypoints[shIdx].ImagePos;
                    bestWrist    = p.Keypoints[wrIdx].ImagePos;
                }
            }

            if (bestConf < 0.05f) return 0f;
            float armPx = Vector2.Distance(bestShoulder, bestWrist);
            if (armPx < 5f) return 0f;

            return Mathf.Clamp(AssumedArmLengthMeters * focalPx / armPx,
                               _minDepthMeters, _maxDepthMeters);
        }

        /// <summary>Returns true when this side became the new best candidate.</summary>
        private bool TryArm(PersonDetection p, Side side, int detectionIndex, float depthHeuristic,
                             ref bool found, ref ArmCandidate best,
                             ref int keypointPassCount, ref int filterPassCount,
                             ref int depthRaycastHits)
        {
            CocoKeypoint shoulderId, wristId;
            if (side == Side.Left)
            { shoulderId = CocoKeypoint.LeftShoulder;  wristId = CocoKeypoint.LeftWrist;  }
            else
            { shoulderId = CocoKeypoint.RightShoulder; wristId = CocoKeypoint.RightWrist; }

            var shoulder = p.Keypoints[(int)shoulderId];
            var wrist    = p.Keypoints[(int)wristId];

            if (shoulder.Confidence < _keypointConfidence
             || wrist.Confidence    < _keypointConfidence) return false;

            keypointPassCount++;

            var arm = new ArmDetection
            {
                Side          = side,
                ShoulderImage = shoulder.ImagePos,
                ElbowImage    = p.Keypoints[(int)(side == Side.Left
                                    ? CocoKeypoint.LeftElbow : CocoKeypoint.RightElbow)].ImagePos,
                WristImage    = wrist.ImagePos,
                Confidence    = Mathf.Min(shoulder.Confidence, wrist.Confidence),
            };

            // Reject the wearer's own arms (skip if bypass is enabled for testing).
            bool skipWearerFilter = _bypassWearerFilter || (IsArmOnlyMode && _skipWearerFilterInArmOnlyMode);
            if (!skipWearerFilter && _wearerFilter != null && _wearerFilter.IsWearerArm(arm, _cameraSource)) return false;

            filterPassCount++;

            // Project shoulder + wrist into world space. When the depth raycaster is
            // available we use the actual scene depth from Meta's Depth API — much more
            // accurate than the bbox-height / arm-length heuristic.
            Vector3 shoulderWorld = ProjectImagePoint(arm.ShoulderImage, depthHeuristic, out bool _);
            Vector3 wristWorld    = ProjectImagePoint(arm.WristImage,    depthHeuristic, out bool wristHit);
            if (wristHit) depthRaycastHits++;

            // Effective candidate depth is the distance from the camera to the wrist
            // world position — works whether we used raycast or heuristic.
            var camPose = _cameraSource.CameraPose;
            float effectiveDepth = Vector3.Distance(camPose.position, wristWorld);
            float score = arm.Confidence;

            if (_currentDetectionsAreArmOnly)
            {
                Vector2 midpoint = (arm.ShoulderImage + arm.WristImage) * 0.5f;
                if (_hasStableArmImage && _stableArmSide == side)
                {
                    float jump = Vector2.Distance(midpoint, _stableArmMidpointImage);
                    if (jump > Mathf.Max(1f, _armOnlyStabilityRadiusPixels)) return false;

                    float continuity = Mathf.Clamp01(1f - jump / Mathf.Max(1f, _armOnlyStabilityRadiusPixels));
                    score += continuity * 0.5f;
                }
                else if (_hasStableArmImage)
                {
                    return false;
                }

                score -= effectiveDepth * 0.015f;
            }
            else
            {
                score -= effectiveDepth * 0.05f;
            }

            // Keep the most stable/plausible candidate. In normal mode this still
            // slightly prefers closer arms; in arm-only mode continuity matters more.
            if (!found || score > best.Score)
            {
                best = new ArmCandidate
                {
                    Shoulder      = shoulderWorld,
                    Wrist         = wristWorld,
                    Depth         = effectiveDepth,
                    Score         = score,
                    ShoulderImage = arm.ShoulderImage,
                    ElbowImage    = arm.ElbowImage,
                    WristImage    = arm.WristImage,
                    Side          = side,
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
                best.ShoulderImage = Vector2.Lerp(_stableArmDetection.Keypoints[(int)(selectedSide == Side.Left
                    ? CocoKeypoint.LeftShoulder : CocoKeypoint.RightShoulder)].ImagePos, best.ShoulderImage, imageT);
                best.ElbowImage = Vector2.Lerp(_stableArmDetection.Keypoints[(int)(selectedSide == Side.Left
                    ? CocoKeypoint.LeftElbow : CocoKeypoint.RightElbow)].ImagePos, best.ElbowImage, imageT);
                best.WristImage = Vector2.Lerp(_stableArmDetection.Keypoints[(int)(selectedSide == Side.Left
                    ? CocoKeypoint.LeftWrist : CocoKeypoint.RightWrist)].ImagePos, best.WristImage, imageT);
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
                _stableArmDetection = BuildSmoothedDetection(
                    LastDetections[selectedIdx],
                    best,
                    selectedSide,
                    preserveSourceBounds: false);
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
                ShoulderImage = _stableArmDetection.Keypoints[(int)(selectedSide == Side.Left
                    ? CocoKeypoint.LeftShoulder : CocoKeypoint.RightShoulder)].ImagePos,
                ElbowImage = _stableArmDetection.Keypoints[(int)(selectedSide == Side.Left
                    ? CocoKeypoint.LeftElbow : CocoKeypoint.RightElbow)].ImagePos,
                WristImage = _stableArmDetection.Keypoints[(int)(selectedSide == Side.Left
                    ? CocoKeypoint.LeftWrist : CocoKeypoint.RightWrist)].ImagePos,
                Side = selectedSide,
            };
            found = true;
            return true;
        }

        private static PersonDetection BuildSmoothedDetection(PersonDetection source,
                                                              ArmCandidate arm,
                                                              Side side,
                                                              bool preserveSourceBounds)
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
                ImageBounds = preserveSourceBounds
                    ? source.ImageBounds
                    : BuildArmImageBounds(arm.ShoulderImage, arm.ElbowImage, arm.WristImage),
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

        /// <summary>
        /// Projects an image-space keypoint into world space using (in priority order):
        ///   1. Meta Depth-API raycast — gives the actual world hit point.
        ///   2. Heuristic depth + intrinsics-based projection from PassthroughCameraSource.
        /// Returns true via <paramref name="usedRaycast"/> when the depth API produced a hit.
        /// </summary>
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
