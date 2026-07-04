using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Tracks the mannequin arm's full 6-DoF world pose from the ArUco marker band,
    /// using the passthrough camera. The camera-vision path that makes a MOVING
    /// prop trackable without a controller:
    ///
    ///   corners (IMarkerCornerProvider, e.g. OpenCV ArUco)
    ///     -> normalized rays via PassthroughCameraSource.ImagePointToRay
    ///        (real intrinsics + capture-time camera pose, so head motion during
    ///        camera latency does not smear the result)
    ///     -> MarkerPnP rigid solve against MarkerBandLayout
    ///     -> One-Euro filter (low jitter at rest, low lag in motion)
    ///     -> freeze-when-stationary (overlay is rock solid between movements).
    ///
    /// Consumers read <see cref="TryGetArmPose"/>. <see cref="TrackedArmModel"/> uses
    /// it to drive the 3D arm model, falling back to the YOLO detector when no
    /// markers are visible.
    /// </summary>
    public class MarkerArmTracker : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private PassthroughCameraSource _cameraSource;

        [Header("Marker layout (must match the printed band)")]
        [SerializeField] private MarkerBandLayout _layout = new();

        [Header("Detection")]
        [Tooltip("Run marker detection every N rendered frames (1 = every frame). " +
                 "Detection is a few ms of CPU; raise this if CPU-bound.")]
        [SerializeField, Range(1, 6)] private int _detectEveryNFrames = 1;
        [Tooltip("Reject solves whose RMS reprojection error (normalized coords) exceeds this. " +
                 "With subpixel corners good solves sit near 0.001; noisy/ambiguous ones above " +
                 "0.005 carry large rotation errors and must be dropped.")]
        [SerializeField] private float _maxRmsError = 0.005f;
        [Tooltip("Drop the tracked pose after this many seconds without an accepted solve.")]
        [SerializeField] private float _lostAfterSeconds = 0.6f;

        [Header("Filtering (One-Euro)")]
        [SerializeField] private float _filterMinCutoff = 1.2f;
        [SerializeField] private float _filterBeta = 0.08f;

        [Header("Freeze when stationary")]
        [Tooltip("Lock the pose after the arm has been still this long (0 = never freeze).")]
        [SerializeField] private float _freezeAfterSeconds = 1.0f;
        [SerializeField] private float _stillSpeedThreshold = 0.015f;      // m/s
        [SerializeField] private float _stillAngularThreshold = 3f;        // deg/s
        [Tooltip("Unfreeze when a fresh solve deviates from the frozen pose by more than this.")]
        [SerializeField] private float _unfreezePositionDelta = 0.02f;     // m
        [SerializeField] private float _unfreezeAngleDelta = 4f;           // deg

        private IMarkerCornerProvider _cornerProvider;
        private OneEuroPoseFilter _filter;
        private readonly List<MarkerObservation> _observations = new();
        private readonly List<MarkerPnP.Correspondence> _correspondences = new();
        private readonly List<MarkerPnP.Correspondence> _seedQuad = new();
        private readonly Vector3[] _cornerScratch = new Vector3[4];

        private bool _hasPose;
        private Pose _armPoseWorld;           // filtered (or frozen) output
        private Pose _lastRawPoseWorld;       // last accepted unfiltered solve
        private Pose _lastCamFromArm;         // camera-local solve, used as next init
        private bool _hasCamFromArm;
        private float _lastAcceptTime = -999f;
        private float _stillSince = -1f;
        private bool _isFrozen;
        private int _frameCounter;

        public bool IsTracking => _hasPose && Time.time - _lastAcceptTime < _lostAfterSeconds;
        public bool IsFrozen => _isFrozen && IsTracking;
        public int LastMarkerCount { get; private set; }
        public float LastRmsError { get; private set; }
        public string Status { get; private set; } = "Not started";

        /// <summary>World pose of the arm-local frame (band centre, +Z toward wrist).</summary>
        public bool TryGetArmPose(out Pose pose)
        {
            pose = _armPoseWorld;
            return IsTracking;
        }

        private void OnEnable()
        {
            _filter = new OneEuroPoseFilter(_filterMinCutoff, _filterBeta);
            if (_cameraSource == null) _cameraSource = FindFirstObjectByType<PassthroughCameraSource>();
            foreach (var mb in GetComponents<MonoBehaviour>())
                if (mb is IMarkerCornerProvider p) { _cornerProvider = p; break; }
        }

        private void Update()
        {
            if (_cameraSource == null) { Status = "No PassthroughCameraSource"; return; }
            if (_cornerProvider == null)
            {
                Status = "No marker corner provider on this GameObject (add ArUcoCornerProvider; " +
                         "requires OpenCV for Unity + OPENCV_FOR_UNITY define)";
                return;
            }
            if (!_cornerProvider.IsReady) { Status = $"Provider not ready: {_cornerProvider.Status}"; return; }
            if (!_cameraSource.HasFrame) { Status = "Waiting: no camera frame"; return; }

            _frameCounter++;
            if (_frameCounter % Mathf.Max(1, _detectEveryNFrames) != 0) return;

            _observations.Clear();
            if (!_cornerProvider.TryDetect(_cameraSource.CurrentTexture, _observations))
            {
                Status = $"Detection failed: {_cornerProvider.Status}";
                return;
            }

            Pose camPose = _cameraSource.CameraPose;
            BuildCorrespondences(camPose);
            LastMarkerCount = _correspondences.Count / 4;

            if (_correspondences.Count < 4)
            {
                Status = $"tracking={IsTracking} markers=0 ({_observations.Count} unknown ids)";
                if (!IsTracking) { _hasCamFromArm = false; _filter.Reset(); }
                return;
            }

            Pose? init = _hasCamFromArm && IsTracking ? _lastCamFromArm : (Pose?)null;
            if (!MarkerPnP.TrySolve(_correspondences, _seedQuad, init, out Pose camFromArm, out float rms)
                || rms > _maxRmsError)
            {
                // A stale init can trap the solver — retry once from scratch.
                if (init.HasValue &&
                    MarkerPnP.TrySolve(_correspondences, _seedQuad, null, out camFromArm, out rms)
                    && rms <= _maxRmsError)
                {
                    AcceptSolve(camPose, camFromArm, rms);
                    return;
                }
                LastRmsError = rms;
                Status = $"tracking={IsTracking} solve rejected (rms={rms:F4}, markers={LastMarkerCount})";
                return;
            }

            AcceptSolve(camPose, camFromArm, rms);
        }

        private void AcceptSolve(Pose camPose, Pose camFromArm, float rms)
        {
            LastRmsError = rms;
            _lastCamFromArm = camFromArm;
            _hasCamFromArm = true;

            var raw = new Pose(
                camPose.position + camPose.rotation * camFromArm.position,
                camPose.rotation * camFromArm.rotation);

            float now = Time.time;
            float dt = Mathf.Clamp(now - _lastAcceptTime, 1e-3f, 0.5f);
            bool wasTracking = IsTracking;
            _lastAcceptTime = now;

            if (!wasTracking)
            {
                _filter.Reset();
                _isFrozen = false;
                _stillSince = -1f;
            }

            // Freeze logic: measure raw motion so filter lag can't mask movement.
            float speed = wasTracking ? Vector3.Distance(raw.position, _lastRawPoseWorld.position) / dt : 0f;
            float angSpeed = wasTracking ? Quaternion.Angle(raw.rotation, _lastRawPoseWorld.rotation) / dt : 0f;
            _lastRawPoseWorld = raw;

            if (_isFrozen)
            {
                bool moved = Vector3.Distance(raw.position, _armPoseWorld.position) > _unfreezePositionDelta
                          || Quaternion.Angle(raw.rotation, _armPoseWorld.rotation) > _unfreezeAngleDelta;
                if (moved)
                {
                    _isFrozen = false;
                    _stillSince = -1f;
                    _filter.Reset();
                }
                else
                {
                    _hasPose = true;
                    Status = $"FROZEN markers={LastMarkerCount} rms={rms:F4}";
                    return; // keep the locked pose — zero jitter while the arm rests
                }
            }

            bool still = speed < _stillSpeedThreshold && angSpeed < _stillAngularThreshold;
            if (still && wasTracking)
            {
                if (_stillSince < 0f) _stillSince = now;
                if (_freezeAfterSeconds > 0f && now - _stillSince >= _freezeAfterSeconds)
                {
                    _isFrozen = true;
                    Status = $"FROZEN markers={LastMarkerCount} rms={rms:F4}";
                    _hasPose = true;
                    return; // _armPoseWorld keeps its last filtered value
                }
            }
            else
            {
                _stillSince = -1f;
            }

            _armPoseWorld = _filter.Filter(raw, dt);
            _hasPose = true;
            Status = $"TRACKING markers={LastMarkerCount} rms={rms:F4} v={speed:F2}m/s";
        }

        /// <summary>
        /// Converts pixel observations into (arm-local 3D corner, normalized camera ray)
        /// pairs. Rays come from ImagePointToRay so device intrinsics/crop are respected;
        /// they are brought into the capture-time camera frame here.
        /// </summary>
        private void BuildCorrespondences(Pose camPose)
        {
            _correspondences.Clear();
            _seedQuad.Clear();
            Quaternion invCamRot = Quaternion.Inverse(camPose.rotation);

            float bestSeedArea = 0f;
            int bestSeedStart = -1;

            for (int i = 0; i < _observations.Count; i++)
            {
                var obs = _observations[i];
                if (!_layout.TryGetMarkerCorners(obs.Id, _cornerScratch)) continue;

                int start = _correspondences.Count;
                bool valid = true;
                for (int c = 0; c < 4 && valid; c++)
                {
                    Ray ray = _cameraSource.ImagePointToRay(obs.Corner(c));
                    Vector3 dir = invCamRot * ray.direction;
                    if (dir.z < 0.05f) { valid = false; break; }
                    _correspondences.Add(new MarkerPnP.Correspondence
                    {
                        ObjectPoint = _cornerScratch[c],
                        Observed = new Vector2(dir.x / dir.z, dir.y / dir.z),
                    });
                }
                if (!valid)
                {
                    _correspondences.RemoveRange(start, _correspondences.Count - start);
                    continue;
                }

                // Seed the homography init with the biggest (best-conditioned) marker.
                float area = Mathf.Abs((obs.C1.x - obs.C0.x) * (obs.C3.y - obs.C0.y)
                                     - (obs.C3.x - obs.C0.x) * (obs.C1.y - obs.C0.y));
                if (area > bestSeedArea) { bestSeedArea = area; bestSeedStart = start; }
            }

            if (bestSeedStart >= 0)
                for (int c = 0; c < 4; c++)
                    _seedQuad.Add(_correspondences[bestSeedStart + c]);
        }
    }
}
