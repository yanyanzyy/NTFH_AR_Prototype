using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Orchestrates the arm-detection pipeline each frame:
    ///   1. Runs YoloPoseDetector on the current passthrough frame.
    ///   2. Converts keypoints to world-space arm endpoints using a depth heuristic.
    ///   3. Rejects the wearer's own arms via WearerHandFilter.
    ///   4. Among remaining candidates, picks the SINGLE CLOSEST arm to the camera.
    ///   5. Passes it to ArmOverlay to render the red world-space quad.
    /// </summary>
    [DefaultExecutionOrder(100)]
    public class ArmDetectionManager : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private PassthroughCameraSource _cameraSource;
        [SerializeField] private YoloPoseDetector        _detector;
        [SerializeField] private WearerHandFilter        _wearerFilter;
        [SerializeField] private ArmOverlay              _overlay;

        [Header("Detection cadence")]
        [Tooltip("Run inference every N frames. 1 = every frame, 2 = every other.")]
        [SerializeField, Range(1, 8)] private int _inferEveryNFrames = 2;

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
        [SerializeField, Range(0f, 1f)] private float _keypointConfidence = 0.15f;

        [Header("Debug")]
        [Tooltip("Optional debug HUD. Attach ArmDetectionDebugHUD component here.")]
        [SerializeField] private ArmDetectionDebugHUD _debugHUD;
        [Tooltip("Skip the WearerHandFilter entirely. Use this to confirm detection works before enabling filtering.")]
        [SerializeField] private bool _bypassWearerFilter = false;

        // ── Private state ──────────────────────────────────────────────────────────────

        private int _frameCounter;

        // Readable by the debug HUD.
        public int    InferenceCount    { get; private set; }
        public int    LastPersonCount   { get; private set; }
        public bool   LastFoundArm      { get; private set; }
        /// <summary>One-line status shown in the debug HUD — explains the manager state or any blocking condition.</summary>
        public string ManagerStatus   { get; private set; } = "Not started";
        /// <summary>Explains why no arm was found on the last inference frame.</summary>
        public string LastArmStatus   { get; private set; } = "—";
        /// <summary>True when the detector is running in arm-only mode (mannequin / isolated arm).</summary>
        public bool   IsArmOnlyMode   { get; private set; } = false;
        /// <summary>All PersonDetections from the last inference frame (copied from detector scratch list).</summary>
        public List<PersonDetection> LastDetections { get; } = new();
        /// <summary>
        /// Index into <see cref="LastDetections"/> of the detection the manager selected
        /// for overlay rendering this frame, or -1 if no arm was found. Used by
        /// YoloBoundingBoxDebug to draw the bbox/keypoints of the chosen target only.
        /// </summary>
        public int SelectedDetectionIndex { get; private set; } = -1;
        /// <summary>The side (Left/Right) of the selected arm. Undefined when SelectedDetectionIndex is -1.</summary>
        public Side SelectedDetectionSide { get; private set; } = Side.Left;
        /// <summary>Passthrough to YoloPoseDetector.LastArmOnlyMaxScore for the HUD and debug visualiser.</summary>
        public float  LastMaxArmScore => _detector != null ? _detector.LastArmOnlyMaxScore : 0f;

        /// <summary>Toggle detection mode at runtime — called by DetectionModeButton.</summary>
        public void SetArmOnlyMode(bool armOnly)
        {
            IsArmOnlyMode = armOnly;
            if (_detector != null) _detector.ArmOnlyMode = armOnly;
            Debug.Log($"[ArmManager] Mode → {(armOnly ? "ARM-ONLY" : "NORMAL")}");
        }

        // Best arm candidate this frame.
        private struct ArmCandidate
        {
            public Vector3 Shoulder;
            public Vector3 Wrist;
            public float   Depth;   // estimated metres from camera
        }

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void Reset()
        {
            _cameraSource = GetComponentInChildren<PassthroughCameraSource>();
            _detector     = GetComponentInChildren<YoloPoseDetector>();
            _wearerFilter = GetComponentInChildren<WearerHandFilter>();
            _overlay      = GetComponentInChildren<ArmOverlay>();
        }

        private void Start()
        {
            Debug.Log($"[ArmManager] Start — cam={_cameraSource != null}  det={_detector != null}  " +
                      $"filter={_wearerFilter != null}  overlay={_overlay != null}  hud={_debugHUD != null}");
            if (_cameraSource == null) Debug.LogError("[ArmManager] _cameraSource is NULL — drag PassthroughCameraSource here.");
            if (_detector     == null) Debug.LogError("[ArmManager] _detector is NULL — drag YoloPoseDetector here.");
            if (_overlay      == null) Debug.LogError("[ArmManager] _overlay is NULL — drag ArmOverlay here.");
        }

        private void Update()
        {
            // Log every ~2 seconds so we can see what is blocking via adb logcat.
            if (Time.frameCount % 120 == 0)
            {
                Debug.Log($"[ArmManager] " +
                          $"cam={_cameraSource != null} " +
                          $"det={_detector != null} " +
                          $"overlay={_overlay != null} " +
                          $"hasFrame={(_cameraSource != null && _cameraSource.HasFrame)} " +
                          $"modelReady={(_detector != null && _detector.IsReady)} " +
                          $"inferenceCount={InferenceCount}  armStatus={LastArmStatus}");
            }

            // Set ManagerStatus at every blocking point so the HUD shows the root cause.
            if (_cameraSource == null) { ManagerStatus = "ERR: CameraSource not assigned"; return; }
            if (_detector     == null) { ManagerStatus = "ERR: Detector not assigned";     return; }
            if (_overlay      == null) { ManagerStatus = "ERR: Overlay not assigned";      return; }
            if (!_cameraSource.HasFrame)  { ManagerStatus = "Waiting: no camera frame";    return; }
            if (!_detector.IsReady)       { ManagerStatus = "Waiting: model loading";      return; }

            _frameCounter++;
            if (_frameCounter % Mathf.Max(1, _inferEveryNFrames) != 0) return;

            ManagerStatus = IsArmOnlyMode ? "Running [ARM-ONLY]" : "Running [Normal]";
            var persons = _detector.Run(_cameraSource.CurrentTexture);
            InferenceCount++;
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

            SelectedDetectionIndex = found ? selectedIdx : -1;
            SelectedDetectionSide  = selectedSide;

            // ── Compose arm status string ──────────────────────────────────────────────
            if (found)
            {
                string mode = (_detector != null && _detector.LastRunWasArmOnlyFallback)
                    ? " [arm-only mode]" : "";
                string depthSrc = depthRaycastHits > 0 ? "depth-API" : "heuristic";
                LastArmStatus = $"OK — {best.Depth:F1} m away ({depthSrc}){mode}";
            }
            else if (persons.Count == 0)
            {
                LastArmStatus = $"0 persons (YOLO conf<{_detector.ConfidenceThreshold:F2}? arm-only threshold too high?)";
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

        private float EstimateDepth(PersonDetection p)
        {
            float imageH = _cameraSource.Height;
            if (imageH <= 0f) return _minDepthMeters;
            float focalPx = (imageH * 0.5f) / Mathf.Tan(_cameraSource.VerticalFovRadians * 0.5f);

            // If this came from the arm-only fallback, the bbox is built from keypoints
            // (not from a full-body anchor), so we use shoulder→wrist pixel distance
            // calibrated against assumed arm length instead.
            if (_detector != null && _detector.LastRunWasArmOnlyFallback)
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
            if (!_bypassWearerFilter && _wearerFilter != null && _wearerFilter.IsWearerArm(arm, _cameraSource)) return false;

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

            // Keep only the closest candidate.
            if (!found || effectiveDepth < best.Depth)
            {
                best = new ArmCandidate
                {
                    Shoulder = shoulderWorld,
                    Wrist    = wristWorld,
                    Depth    = effectiveDepth,
                };
                found = true;
                return true;
            }
            return false;
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
