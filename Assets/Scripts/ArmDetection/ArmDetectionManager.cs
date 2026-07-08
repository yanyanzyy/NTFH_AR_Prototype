using System.Collections.Generic;
using System.Reflection;
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

        [Header("Needle detection (vision)")]
        [Tooltip("Runs the trained needle model (arm-needle-pose-320.onnx) alongside the arm model. " +
                 "Optional: when unassigned, InjectionSiteDetector falls back to hand tracking.")]
        [SerializeField] private NeedleDetector _needleDetector;
        [Tooltip("Only run the needle model once an arm is locked. The needle tip is only consumed " +
                 "during injection (which requires a locked arm), so this keeps the whole GPU budget " +
                 "on arm acquisition until then.")]
        [SerializeField] private bool _runNeedleOnlyWhileLocked = true;
        [Tooltip("How strongly to smooth the needle tip/hub world positions (0 = raw, 1 = frozen).")]
        [SerializeField, Range(0f, 1f)] private float _needleWorldSmoothing = 0.5f;
        [Tooltip("Seconds without a fresh needle detection before TryGetNeedleTip reports lost. " +
                 "Bridges the gap between completed inferences (the model only finishes every few frames).")]
        [SerializeField] private float _needleLostAfterSeconds = 0.75f;
        [Header("MediaPipe pipeline lifecycle")]
        [Tooltip("Manage the heavyweight MediaPipe HandLandmarkerRunner automatically. The runner " +
                 "opens its own camera stream, does a full-frame GPU readback every frame and runs " +
                 "CPU inference, so it is kept OFF while its results would not be consumed (custom " +
                 "detector active and fallback unused) and started/paused on demand.")]
        [SerializeField] private bool _manageMediaPipeLifecycle = true;
        [Tooltip("Seconds the custom detector must be starved (usable but zero detections) before " +
                 "the MediaPipe fallback pipeline is started or resumed.")]
        [SerializeField] private float _mediaPipeActivateAfterSeconds = 2f;
        [Tooltip("Seconds of sustained custom detections before the MediaPipe pipeline is paused again.")]
        [SerializeField] private float _mediaPipePauseAfterSeconds = 2f;

        [Header("Depth estimation")]
        [SerializeField] private EnvironmentRaycastManager _depthRaycaster;
        [Tooltip("Real-world length (m) of the detected arm segment. Used to estimate depth from " +
                 "the detected box's long side (the arm-only model's box IS the arm). The Limbs & " +
                 "Things forearm incl. hand is ~0.5 m.")]
        [SerializeField] private float _assumedArmLengthMeters = 0.5f;
        [SerializeField] private float _minPersonBboxPixels = 10f;
        [SerializeField] private float _minDepthMeters = 0.3f;
        [SerializeField] private float _maxDepthMeters = 8f;

        [Header("Arm filtering")]
        [SerializeField, Range(0f, 1f)] private float _keypointConfidence = 0.01f;
        [SerializeField] private bool _skipWearerFilterInArmOnlyMode = true;
        [SerializeField] private float _armOnlyStabilityRadiusPixels = 420f;
        [SerializeField, Range(0f, 1f)] private float _armOnlyImageSmoothing = 0.65f;
        [SerializeField, Range(0f, 1f)] private float _armOnlyWorldSmoothing = 0.55f;
        [SerializeField, Range(1, 12)] private int _customDetectorEveryNFrames = 1;

        [Header("Target lock")]
        [Tooltip("Consecutive NEW inference results a candidate must stay in roughly the same world " +
                 "position before the overlay locks on. Prevents snapping onto single-frame false " +
                 "positives. Cached detections served between inferences do not count.")]
        [SerializeField, Range(1, 30)] private int _acquireConsistentFrames = 4;
        [Tooltip("World radius (m) a candidate may drift from the running average position and still " +
                 "count toward acquisition.")]
        [SerializeField] private float _acquireRadiusMeters = 0.3f;
        [Tooltip("Minimum detector confidence a candidate needs to START acquiring a lock. Updating " +
                 "an already-locked arm only needs the detector's own threshold; this higher bar " +
                 "keeps marginal detections from initiating a lock.")]
        [SerializeField, Range(0f, 1f)] private float _acquireMinConfidence = 0.4f;
        [Tooltip("During acquisition, how long detection may drop out (seconds) before progress " +
                 "resets. The detector only delivers results every few frames, so a hard per-frame " +
                 "reset would make locking nearly impossible with an intermittent detector.")]
        [SerializeField] private float _acquireGapToleranceSeconds = 0.6f;
        [Tooltip("While locked, only detections whose midpoint lies within this distance (m) of the locked " +
                 "arm can update it. Anything outside is ignored, so the overlay never jumps to another arm.")]
        [SerializeField] private float _lockGateRadiusMeters = 0.35f;
        [Tooltip("The gate radius grows at this rate (m/s) while the locked arm is undetected, so a " +
                 "mannequin arm that was moved during a detection dropout can be re-captured.")]
        [SerializeField] private float _lockGateGrowPerSecondLost = 0.25f;
        [Tooltip("Upper bound (m) for the grown gate radius.")]
        [SerializeField] private float _lockGateMaxRadiusMeters = 0.9f;
        [Tooltip("Automatically release the lock after the arm has been undetected this long (seconds). " +
                 "0 = never auto-unlock; the pose is held in world space until Unlock() is called " +
                 "(UNLOCK ARM button / controller B). Ignored while the lock is frozen.")]
        [SerializeField] private float _autoUnlockAfterLostSeconds = 0f;
        [Tooltip("Freeze the overlay in world space once the lock is acquired (after the refinement " +
                 "window below). The physical arm is stationary, so detections arriving while the " +
                 "user moves and looks around only wobble the overlay or drag it onto other objects. " +
                 "Frozen means: ignore ALL detections and hold the world-anchored pose until " +
                 "UNLOCK ARM / controller B releases it.")]
        [SerializeField] private bool _freezeWhileLocked = true;
        [Tooltip("How long after acquiring the lock detections keep refining the pose (seconds) " +
                 "before it freezes. Lets the smoothed world pose settle onto the arm first. " +
                 "0 = freeze on the very first locked pose.")]
        [SerializeField] private float _lockRefineSeconds = 1.5f;

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

        /// <summary>True while the tracker is locked onto a target arm (endpoints valid).
        /// The lock is held through detection dropouts — the pose stays anchored in world
        /// space — until Unlock() is called or the optional auto-unlock timer expires.</summary>
        public bool IsLocked => _lockState == LockState.Locked && _hasSmoothedArmWorld;

        /// <summary>Human-readable lock state for the HUD and the unlock button.</summary>
        public string LockStatus { get; private set; } = "Searching";

        /// <summary>
        /// Releases the current target lock so a new arm can be acquired. Wired to the
        /// UNLOCK ARM button (and controller B) for when the overlay grabbed the wrong target.
        /// </summary>
        public void Unlock()
        {
            _lockState = LockState.Searching;
            _acquireFrameCount = 0;
            _acquireGapSeconds = 0f;
            _lockLostSeconds = 0f;
            ResetArmStability();
            LockStatus = "Searching (unlocked)";
            Debug.Log("[ArmManager] Target lock released — searching for a new arm.");
        }

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

        /// <summary>Human-readable needle detection state for the HUD.</summary>
        public string NeedleStatus { get; private set; } = "No needle detector";

        /// <summary>
        /// Smoothed world position of the vision-detected needle tip. Returns false when the
        /// needle detector is missing/idle or the needle hasn't been seen recently.
        /// </summary>
        public bool TryGetNeedleTip(out Vector3 tip)
        {
            tip = _smoothedNeedleTipWorld;
            return _hasNeedleWorld;
        }

        /// <summary>Smoothed world positions of the needle tip and hub (the needle axis).</summary>
        public bool TryGetNeedle(out Vector3 tip, out Vector3 hub)
        {
            tip = _smoothedNeedleTipWorld;
            hub = _smoothedNeedleHubWorld;
            return _hasNeedleWorld;
        }

        private struct ArmCandidate
        {
            public Vector3 Shoulder;
            public Vector3 Wrist;
            public float Depth;
            public float Score;
            public float Confidence;   // raw detector confidence, before score shaping
            public Vector2 ShoulderImage;
            public Vector2 ElbowImage;
            public Vector2 WristImage;
            public Side Side;
        }

        private enum LockState { Searching, Acquiring, Locked }

        private LockState _lockState = LockState.Searching;
        private int _acquireFrameCount;
        private Vector3 _acquireMidWorld;
        private float _acquireGapSeconds;
        private bool _freshResultThisFrame;
        private float _lockAcquiredTime;
        private float _lockLostSeconds;
        private bool _hasStableArmImage;
        private Side _stableArmSide;
        private Vector2 _stableArmMidpointImage;
        private bool _hasSmoothedArmWorld;
        private Vector3 _smoothedShoulderWorld;
        private Vector3 _smoothedWristWorld;
        private PersonDetection _stableArmDetection;
        private bool _currentDetectionsAreMediaPipe;
        private int _customDetectorFrameCounter;
        private readonly List<PersonDetection> _cachedCustomDetections = new();

        private bool _hasStableFixedAxisDepth;
        private float _stableFixedAxisDepth;

        private bool _hasNeedleWorld;
        private Vector3 _smoothedNeedleTipWorld;
        private Vector3 _smoothedNeedleHubWorld;
        private float _needleLastSeenTime;
        // Scratch buffers + per-frame result cache for TryEstimateArmAxisFromDepth, so the
        // grid raycasts are never repeated for the same box within a frame and the hot path
        // allocates nothing.
        private readonly List<Vector3> _axisHits = new();
        private readonly List<float> _axisDepths = new();
        private readonly List<Vector3> _axisFilteredPoints = new();
        private int _axisCacheFrame = -1;
        private Rect _axisCacheBounds;
        private bool _axisCacheResult;
        private Vector3 _axisCacheEndA;
        private Vector3 _axisCacheEndB;

        // MediaPipe runner lifecycle (found by type name so there is no compile-time
        // dependency on the MediaPipe sample assembly, mirroring MediaPipeHomulerBridge).
        private Behaviour _mediaPipeRunner;
        private MethodInfo _mediaPipeRunnerPause;
        private MethodInfo _mediaPipeRunnerResume;
        private bool _mediaPipeRunnerStarted;
        private bool _mediaPipeRunnerPaused;
        private float _mediaPipeStarvedSeconds;
        private float _mediaPipeFedSeconds;

        private void Reset()
        {
            _cameraSource = GetComponentInChildren<PassthroughCameraSource>();
            _customArmDetector = GetComponentInChildren<CustomArmDetector>();
            _mediaPipeDetector = GetComponentInChildren<MediaPipeHandArmDetector>();
            _wearerFilter = GetComponentInChildren<WearerHandFilter>();
            _overlay = GetComponentInChildren<ArmOverlay>();
        }

        private void Awake()
        {
            if (!_manageMediaPipeLifecycle) return;

            // Locate the homuler HandLandmarkerRunner and disable it BEFORE its Start() runs
            // (every Awake executes before any Start), so the MediaPipe pipeline — its own
            // camera stream, per-frame readbacks and CPU inference — does not auto-play.
            // Unity defers Start() until a component is first enabled, so enabling the runner
            // later boots the pipeline through exactly the same code path as at scene load.
            foreach (var mb in FindObjectsByType<MonoBehaviour>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            {
                if (mb == null || mb.GetType().Name != "HandLandmarkerRunner") continue;
                _mediaPipeRunner = mb;
                _mediaPipeRunnerPause = mb.GetType().GetMethod("Pause");
                _mediaPipeRunnerResume = mb.GetType().GetMethod("Resume");
                mb.enabled = false;
                Debug.Log("[ArmManager] MediaPipe HandLandmarkerRunner found - deferred until its results are needed.");
                break;
            }
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

            UpdateMediaPipeLifecycle();

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
            int gateRejectCount = 0;
            int depthRaycastHits = 0;

            // The custom detector fills the Left and Right keypoint slots with IDENTICAL data
            // (see CustomArmDetector.ToPersonDetection), so evaluating the Right side can only
            // change the outcome when the continuity bonus keys on a previously selected Right
            // side. Skipping the duplicate halves the per-candidate work (and, when depth-axis
            // estimation is enabled, its raycasts) without changing the selected result.
            bool sidesIdentical = !_currentDetectionsAreMediaPipe;
            bool evaluateRight = !sidesIdentical || (_hasStableArmImage && _stableArmSide == Side.Right);

            for (int i = 0; i < persons.Count; i++)
            {
                var p = persons[i];
                float bboxSize = Mathf.Max(p.ImageBounds.width, p.ImageBounds.height);
                if (bboxSize < _minPersonBboxPixels) continue;
                bboxPassCount++;

                float depth = EstimateDepth(p);
                if (TryArm(p, Side.Left, depth, ref found, ref best, ref keypointPassCount, ref filterPassCount, ref gateRejectCount, ref depthRaycastHits))
                {
                    selectedIdx = i;
                    selectedSide = Side.Left;
                }
                if (evaluateRight &&
                    TryArm(p, Side.Right, depth, ref found, ref best, ref keypointPassCount, ref filterPassCount, ref gateRejectCount, ref depthRaycastHits))
                {
                    selectedIdx = i;
                    selectedSide = Side.Right;
                }
            }

            bool renderOverlay = false;
            if (_lockState == LockState.Locked)
            {
                float refineElapsed = Time.time - _lockAcquiredTime;
                bool frozen = _freezeWhileLocked && refineElapsed >= _lockRefineSeconds;

                if (frozen)
                {
                    // Hard lock: the pose stays world-anchored where refinement left it and ALL
                    // detections are ignored. The physical arm hasn't moved - any pose change
                    // from here on would come from detection shift as the user moves and looks
                    // around, which is exactly the wobble/drift the freeze exists to prevent.
                    UseHeldLockPose(ref found, ref best, ref selectedIdx, ref selectedSide);
                    LockStatus = "Locked (frozen in place)";
                    renderOverlay = found;
                }
                else if (found)
                {
                    _lockLostSeconds = 0f;
                    StabilizeArmOnlyCandidate(ref best, selectedSide, selectedIdx);
                    LockStatus = _freezeWhileLocked
                        ? $"Locked (refining {refineElapsed:F1}/{_lockRefineSeconds:F1}s)"
                        : "Locked (tracking)";
                    renderOverlay = true;
                }
                else
                {
                    _lockLostSeconds += Time.deltaTime;
                    if (_autoUnlockAfterLostSeconds > 0f && _lockLostSeconds >= _autoUnlockAfterLostSeconds)
                    {
                        Unlock();
                    }
                    else
                    {
                        // Hold the last pose: it is anchored in world space, so it stays on
                        // the (stationary) arm through dropouts and while the user walks around.
                        UseHeldLockPose(ref found, ref best, ref selectedIdx, ref selectedSide);
                        LockStatus = $"Locked (holding {_lockLostSeconds:F1}s, gate {CurrentLockGateRadius():F2} m)";
                        renderOverlay = found;
                    }
                }
            }
            else if (found)
            {
                UpdateAcquisition(best);
                if (_lockState == LockState.Locked)
                {
                    StabilizeArmOnlyCandidate(ref best, selectedSide, selectedIdx);
                    LockStatus = "Locked (tracking)";
                    renderOverlay = true;
                }
                // else: UpdateAcquisition set LockStatus.
            }
            else
            {
                TickAcquisitionGap();
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
            else if (gateRejectCount > 0)
            {
                LastArmStatus = $"{gateRejectCount} detection(s) outside lock gate (ignored)";
            }
            else
            {
                LastArmStatus = "No arm selected";
            }

            var camTransform = _cameraSource.CameraTransform;
            _overlay.Render(renderOverlay ? (best.Shoulder, best.Wrist) : ((Vector3, Vector3)?)null, camTransform);

            LastFoundArm = found;
            UpdateNeedleDetection();
            _debugHUD?.ReportDetections(persons.Count, found ? 1 : 0);
        }

        /// <summary>
        /// Runs the needle model on the same camera frame and maintains a smoothed world-space
        /// needle tip/hub for InjectionSiteDetector. The needle is a separate, second worker —
        /// the arm model stays untouched — so it only runs while an arm is locked by default.
        /// </summary>
        private void UpdateNeedleDetection()
        {
            if (_needleDetector == null || !_needleDetector.IsReady)
            {
                NeedleStatus = _needleDetector == null ? "No needle detector" : _needleDetector.Status;
                _hasNeedleWorld = false;
                return;
            }

            if (_runNeedleOnlyWhileLocked && _lockState != LockState.Locked)
            {
                NeedleStatus = "Idle until arm locked";
                _hasNeedleWorld = false;
                return;
            }

            _needleDetector.Run(_cameraSource.CurrentTexture);

            if (_needleDetector.LastRunConsumedNewResult && _needleDetector.TryGetLatest(out var needle))
            {
                // The needle only matters near the locked arm, so the arm's depth is the best
                // fallback when the depth raycast misses (thin needle/hand rarely in the mesh).
                float fallbackDepth = _hasSmoothedArmWorld
                    ? Vector3.Distance(_cameraSource.CameraPose.position,
                                       (_smoothedShoulderWorld + _smoothedWristWorld) * 0.5f)
                    : Mathf.Clamp(0.6f, _minDepthMeters, _maxDepthMeters);

                Vector3 tipWorld = ProjectImagePoint(needle.TipImage, fallbackDepth, out _);
                Vector3 hubWorld = ProjectImagePoint(needle.HubImage, fallbackDepth, out _);

                if (!_hasNeedleWorld)
                {
                    _smoothedNeedleTipWorld = tipWorld;
                    _smoothedNeedleHubWorld = hubWorld;
                }
                else
                {
                    float t = 1f - _needleWorldSmoothing;
                    _smoothedNeedleTipWorld = Vector3.Lerp(_smoothedNeedleTipWorld, tipWorld, t);
                    _smoothedNeedleHubWorld = Vector3.Lerp(_smoothedNeedleHubWorld, hubWorld, t);
                }

                _hasNeedleWorld = true;
                _needleLastSeenTime = Time.time;
                NeedleStatus = $"Needle {needle.Confidence:F2} @ {Vector3.Distance(_cameraSource.CameraPose.position, _smoothedNeedleTipWorld):F2} m";
            }
            else if (_hasNeedleWorld && Time.time - _needleLastSeenTime > _needleLostAfterSeconds)
            {
                _hasNeedleWorld = false;
                NeedleStatus = $"Needle lost ({_needleDetector.Status})";
            }
            else if (!_hasNeedleWorld)
            {
                NeedleStatus = $"Searching ({_needleDetector.Status})";
            }
        }

        public float GetEstimatedDepth(PersonDetection p) => EstimateDepth(p);

        /// <summary>
        /// Starts / pauses the MediaPipe pipeline so it only costs anything while its output
        /// can actually be consumed by RunPrimaryDetector:
        ///   - custom detector unusable  -> MediaPipe is the only detector, run it;
        ///   - fallback disabled          -> results would be thrown away, keep it off;
        ///   - fallback enabled           -> run only after the custom detector has been starved
        ///                                  for a while, pause again once it recovers.
        /// While the target lock is frozen every detection is ignored anyway, so the pipeline
        /// is not woken up during a frozen lock.
        /// </summary>
        private void UpdateMediaPipeLifecycle()
        {
            if (!_manageMediaPipeLifecycle || _mediaPipeRunner == null ||
                _mediaPipeDetector == null || !_mediaPipeDetector.isActiveAndEnabled) return;

            bool customUsable = _useCustomArmDetector && _customArmDetector != null && _customArmDetector.IsReady;
            bool resultsUsable = !customUsable || _fallbackToMediaPipe;
            bool lockFrozen = _lockState == LockState.Locked && _freezeWhileLocked &&
                              Time.time - _lockAcquiredTime >= _lockRefineSeconds;
            bool starving = (!customUsable || _cachedCustomDetections.Count == 0) && !lockFrozen;

            if (resultsUsable && starving)
            {
                _mediaPipeFedSeconds = 0f;
                _mediaPipeStarvedSeconds += Time.deltaTime;
                if (_mediaPipeStarvedSeconds >= _mediaPipeActivateAfterSeconds)
                    SetMediaPipeRunning(true);
            }
            else
            {
                _mediaPipeStarvedSeconds = 0f;
                _mediaPipeFedSeconds += Time.deltaTime;
                if (!resultsUsable || _mediaPipeFedSeconds >= _mediaPipePauseAfterSeconds)
                    SetMediaPipeRunning(false);
            }
        }

        private void SetMediaPipeRunning(bool run)
        {
            if (run)
            {
                if (!_mediaPipeRunnerStarted)
                {
                    _mediaPipeRunner.enabled = true;   // fires the runner's deferred Start() -> bootstrap -> Play()
                    _mediaPipeRunnerStarted = true;
                    _mediaPipeRunnerPaused = false;
                    Debug.Log("[ArmManager] MediaPipe fallback pipeline started.");
                }
                else if (_mediaPipeRunnerPaused)
                {
                    _mediaPipeRunnerResume?.Invoke(_mediaPipeRunner, null);
                    _mediaPipeRunnerPaused = false;
                    Debug.Log("[ArmManager] MediaPipe fallback pipeline resumed.");
                }
            }
            else if (_mediaPipeRunnerStarted && !_mediaPipeRunnerPaused)
            {
                _mediaPipeRunnerPause?.Invoke(_mediaPipeRunner, null);
                _mediaPipeRunnerPaused = true;
                Debug.Log("[ArmManager] MediaPipe fallback pipeline paused (custom detector active).");
            }
        }

        private bool HasReadyDetector()
        {
            bool customReady = _useCustomArmDetector && _customArmDetector != null && _customArmDetector.IsReady;
            bool mediaPipeReady = _mediaPipeDetector != null && _mediaPipeDetector.IsReady;
            return customReady || mediaPipeReady;
        }

        private List<PersonDetection> RunPrimaryDetector()
        {
            _freshResultThisFrame = false;
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
                    _freshResultThisFrame = _customArmDetector.LastRunConsumedNewResult;

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
                _freshResultThisFrame = _mediaPipeDetector.LastRunConsumedNewResult;
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

            // The arm-only model's box IS the arm, so its long side maps directly to the
            // known real arm length. (The old fallback assumed a full 1.7 m person filled
            // the box, which placed the forearm-sized box ~3x too far away.)
            float longSidePx = Mathf.Max(1f, Mathf.Max(p.ImageBounds.width, p.ImageBounds.height));
            float depth = _assumedArmLengthMeters * focalPx / longSidePx;
            return Mathf.Clamp(depth, _minDepthMeters, _maxDepthMeters);
        }

        private static readonly (int Shoulder, int Wrist)[] ArmKeypointPairs =
        {
            ((int)CocoKeypoint.LeftShoulder, (int)CocoKeypoint.LeftWrist),
            ((int)CocoKeypoint.RightShoulder, (int)CocoKeypoint.RightWrist),
        };

        private float EstimateDepthFromArmLength(PersonDetection p, float focalPx)
        {
            float bestConf = 0f;
            Vector2 bestShoulder = default;
            Vector2 bestWrist = default;

            foreach (var pair in ArmKeypointPairs)
            {
                float conf = Mathf.Min(p.Keypoints[pair.Shoulder].Confidence, p.Keypoints[pair.Wrist].Confidence);
                if (conf > bestConf)
                {
                    bestConf = conf;
                    bestShoulder = p.Keypoints[pair.Shoulder].ImagePos;
                    bestWrist = p.Keypoints[pair.Wrist].ImagePos;
                }
            }

            if (bestConf < 0.05f) return 0f;
            float armPx = Vector2.Distance(bestShoulder, bestWrist);

            // Sanity-gate against the box: the current model's keypoint head is untrained
            // (pose mAP ~0), so its keypoints can collapse to a point or land anywhere.
            // A real proximal→distal span is comparable to the box's long side; anything
            // far off means the keypoints are garbage and the box heuristic should win.
            float longSidePx = Mathf.Max(p.ImageBounds.width, p.ImageBounds.height);
            if (armPx < 5f || armPx < longSidePx * 0.3f || armPx > longSidePx * 1.5f) return 0f;

            return Mathf.Clamp(_assumedArmLengthMeters * focalPx / armPx, _minDepthMeters, _maxDepthMeters);
        }

        private bool TryArm(PersonDetection p, Side side, float depthHeuristic,
                            ref bool found, ref ArmCandidate best,
                            ref int keypointPassCount, ref int filterPassCount,
                            ref int gateRejectCount, ref int depthRaycastHits)
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

            // Keep endpoint order consistent with the tracked arm: the depth-axis PCA has a
            // sign ambiguity, so its min/max endpoints can swap between frames and flip the
            // model 180°. Align each candidate with the previous smoothed direction instead.
            if (_hasSmoothedArmWorld &&
                Vector3.Dot(wristWorld - shoulderWorld, _smoothedWristWorld - _smoothedShoulderWorld) < 0f)
            {
                (shoulderWorld, wristWorld) = (wristWorld, shoulderWorld);
            }

            // Hard gate while locked: a detection far from the locked arm is a DIFFERENT arm.
            // Reject it outright so nothing can steal an existing lock, no matter its score.
            if (_lockState == LockState.Locked && _hasSmoothedArmWorld)
            {
                Vector3 lockedMid = (_smoothedShoulderWorld + _smoothedWristWorld) * 0.5f;
                Vector3 candidateMid = (shoulderWorld + wristWorld) * 0.5f;
                if (Vector3.Distance(lockedMid, candidateMid) > CurrentLockGateRadius())
                {
                    gateRejectCount++;
                    return false;
                }
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
                    Confidence = arm.Confidence,
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

        /// <summary>Reuses the smoothed world pose of the locked arm on frames where no
        /// gated detection came in, keeping the overlay world-anchored on the target.</summary>
        private void UseHeldLockPose(ref bool found, ref ArmCandidate best, ref int selectedIdx, ref Side selectedSide)
        {
            if (!_hasStableArmImage || !_hasSmoothedArmWorld) return;

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
                Confidence = _stableArmDetection.Confidence,
                ShoulderImage = _stableArmDetection.Keypoints[(int)(selectedSide == Side.Left ? CocoKeypoint.LeftShoulder : CocoKeypoint.RightShoulder)].ImagePos,
                ElbowImage = _stableArmDetection.Keypoints[(int)(selectedSide == Side.Left ? CocoKeypoint.LeftElbow : CocoKeypoint.RightElbow)].ImagePos,
                WristImage = _stableArmDetection.Keypoints[(int)(selectedSide == Side.Left ? CocoKeypoint.LeftWrist : CocoKeypoint.RightWrist)].ImagePos,
                Side = selectedSide,
            };
            found = true;
        }

        /// <summary>Counts consecutive NEW inference results with a spatially consistent
        /// candidate and promotes Searching → Acquiring → Locked once enough have been seen.
        /// Cached detections served between inferences neither advance nor reset progress —
        /// otherwise a single inference result, re-served over several rendered frames, would
        /// satisfy the consistency requirement on its own.</summary>
        private void UpdateAcquisition(in ArmCandidate candidate)
        {
            if (candidate.Confidence < _acquireMinConfidence)
            {
                // Too weak to start a lock; treat like a dropout so brief dips don't reset progress.
                TickAcquisitionGap();
                if (_lockState == LockState.Searching)
                    LockStatus = $"Searching (weak detection {candidate.Confidence:F2} < {_acquireMinConfidence:F2})";
                return;
            }

            _acquireGapSeconds = 0f;

            if (!_freshResultThisFrame)
            {
                if (_lockState == LockState.Acquiring)
                    LockStatus = $"Acquiring {_acquireFrameCount}/{_acquireConsistentFrames}";
                return;
            }

            Vector3 mid = (candidate.Shoulder + candidate.Wrist) * 0.5f;
            bool continues = _acquireFrameCount > 0 &&
                             Vector3.Distance(mid, _acquireMidWorld) <= _acquireRadiusMeters;
            if (continues)
            {
                _acquireFrameCount++;
                // Anchor on the running average so a drifting false-positive chain (each step
                // inside the radius, total drift unbounded) cannot ratchet its way to a lock.
                _acquireMidWorld = Vector3.Lerp(_acquireMidWorld, mid, 1f / _acquireFrameCount);
            }
            else
            {
                _acquireFrameCount = 1;
                _acquireMidWorld = mid;
            }

            if (_acquireFrameCount >= _acquireConsistentFrames)
            {
                _lockState = LockState.Locked;
                _lockLostSeconds = 0f;
                _lockAcquiredTime = Time.time;
                Debug.Log($"[ArmManager] Target lock acquired at {mid} after {_acquireFrameCount} consistent results.");
            }
            else
            {
                _lockState = LockState.Acquiring;
                LockStatus = $"Acquiring {_acquireFrameCount}/{_acquireConsistentFrames}";
            }
        }

        /// <summary>Called on frames with no usable candidate. During acquisition a short
        /// dropout is tolerated (the detector only reports every few frames); progress only
        /// resets after _acquireGapToleranceSeconds without a strong candidate.</summary>
        private void TickAcquisitionGap()
        {
            if (_lockState != LockState.Acquiring)
            {
                _lockState = LockState.Searching;
                _acquireFrameCount = 0;
                _acquireGapSeconds = 0f;
                LockStatus = "Searching";
                return;
            }

            _acquireGapSeconds += Time.deltaTime;
            if (_acquireGapSeconds > _acquireGapToleranceSeconds)
            {
                _lockState = LockState.Searching;
                _acquireFrameCount = 0;
                _acquireGapSeconds = 0f;
                LockStatus = "Searching";
            }
            else
            {
                LockStatus = $"Acquiring {_acquireFrameCount}/{_acquireConsistentFrames} " +
                             $"(gap {_acquireGapSeconds:F1}s)";
            }
        }

        /// <summary>Gate radius for lock updates; grows while the arm is undetected so a
        /// target that moved during a dropout can still be re-captured.</summary>
        private float CurrentLockGateRadius() =>
            Mathf.Min(_lockGateMaxRadiusMeters,
                      _lockGateRadiusMeters + _lockLostSeconds * _lockGateGrowPerSecondLost);

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

            // Same box already estimated this frame (e.g. the Right-side pass of the same
            // detection): reuse the result instead of re-raycasting the whole grid.
            if (_axisCacheFrame == Time.frameCount && _axisCacheBounds == imageBounds)
            {
                endA = _axisCacheEndA;
                endB = _axisCacheEndB;
                return _axisCacheResult;
            }
            _axisCacheFrame = Time.frameCount;
            _axisCacheBounds = imageBounds;
            _axisCacheResult = false;

            int grid = _depthSampleGrid;
            // Shrink sampling inward from the box edges so background pixels (where the box
            // slightly overshoots the real arm silhouette) don't get raycasted.
            float padX = imageBounds.width * 0.1f;
            float padY = imageBounds.height * 0.1f;

            var hits = _axisHits;
            hits.Clear();
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
            var depths = _axisDepths;
            depths.Clear();
            foreach (var h in hits) depths.Add(Vector3.Distance(camPos, h));
            depths.Sort();
            float medianDepth = depths[depths.Count / 2];

            var filtered = _axisFilteredPoints;
            filtered.Clear();
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
            _axisCacheResult = true;
            _axisCacheEndA = endA;
            _axisCacheEndB = endB;
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
