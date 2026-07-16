using System;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Staged assessment of a venipuncture attempt, evaluated in clinical order once the
    /// vision needle tip touches the locked arm:
    ///
    ///   1. CONTACT — tip within _contactDistanceMeters of the arm-cylinder surface
    ///   2. SPOT    — the surface contact point lies on a VeinMap vein zone
    ///   3. ANGLE   — insertion angle within SyringeAngleEstimator's acceptable band
    ///   4. DEPTH   — tip penetrated below the surface within the accepted range
    ///   => SUCCESS (latched until the needle leaves the arm)
    ///
    /// Every check is re-validated each frame and the reported stage is the FIRST failing
    /// one, so sliding off the vein while adjusting the angle drops the sequence back to
    /// SPOT instead of grading the angle at the wrong site. Success latches so a small
    /// depth wobble after a good insertion doesn't revoke the pass; ending contact (or
    /// losing the needle beyond a short grace period) resets the attempt.
    ///
    /// DEPTH CAVEAT: penetration is measured as how far the DETECTED tip sits below the
    /// modelled arm surface. Once the physical needle is inside the mannequin the camera
    /// can no longer see the real tip, so this relies on the pose model extrapolating the
    /// tip from the visible barrel (plus registration accuracy of the locked arm). Treat
    /// the depth band as a coarse plausibility window, not a millimetre measurement.
    /// </summary>
    public class InjectionSequenceEvaluator : MonoBehaviour
    {
        public enum Stage { Idle, Contact, Spot, Angle, Depth, Success }

        [Header("References")]
        [SerializeField] private ArmDetectionManager _armManager;
        [SerializeField] private VeinMap _veinMap;
        [Tooltip("Group 2's SyringeAngleEstimator (under SyringePosePrototype / SyringeLabelContainer). " +
                 "Supplies the insertion-angle stage.")]
        [SerializeField] private SyringeAngleEstimator _angleEstimator;
        [Tooltip("Optional, PREFERRED — the in-namespace NeedleAngleEstimator, which measures the " +
                 "angle from the manager's needle axis (works for BOTH the vision syringe and the " +
                 "simulated finger/pen needle). When it has an angle it wins over the estimator above.")]
        [SerializeField] private NeedleAngleEstimator _needleAngleEstimator;

        [Header("Finger test (direct vein proximity)")]
        [Tooltip("Grade the attempt from the needle TIP's distance to the visible vein paths " +
                 "directly, instead of the arm collision cylinder. Recommended for the finger test: " +
                 "the cylinder (from the vision lock) is usually offset from the veins, which leaves " +
                 "the cylinder-based flow stuck on 'Approach the arm'. Depth is not measured with a " +
                 "finger, so success here = on the vein (angle shown as info, gated only if required).")]
        [SerializeField] private bool  _useDirectVeinProximity = true;
        [Tooltip("Direct mode: tip within this distance (m) of a vein path counts as being AT the arm " +
                 "(the contact stage). A vein's own hitRadius then decides right vs wrong spot.")]
        [SerializeField] private float _directContactMeters = 0.06f;
        [Tooltip("Direct mode: also require an acceptable insertion angle for success. Off by default " +
                 "— a fingertip points nearly flat, so demanding a 15–30° needle angle would keep the " +
                 "finger test from ever succeeding. The angle is still shown as info.")]
        [SerializeField] private bool  _requireAngleInDirectMode = false;
        [Tooltip("Direct mode: grade the tip against the veins AS SEEN from the headset — the miss " +
                 "distance is measured perpendicular to the view ray. The locked overlay can sit " +
                 "several cm off in DEPTH while looking perfectly aligned, so raw 3D distance fails " +
                 "trainees for a registration error they cannot see; this grades what they actually " +
                 "aim at. The ignored depth offset is still bounded by the tolerance below.")]
        [SerializeField] private bool  _gradeFromView = true;
        [Tooltip("Maximum offset (m) between tip and vein ALONG the view ray for a poke to count. " +
                 "Covers lock registration depth error + hand-tracking depth error, while still " +
                 "rejecting a hand that is nowhere near the arm.")]
        [SerializeField] private float _viewDepthToleranceMeters = 0.12f;

        [Header("Arm model")]
        [Tooltip("Physical radius of the mannequin arm at injection sites (forearm ~4.25 cm). " +
                 "Keep identical to VeinMap / InjectionSiteDetector.")]
        [SerializeField] private float _armRadiusMeters = 0.0425f;

        [Header("Stage 1 — contact")]
        [Tooltip("Tip must be within this distance (m) of the arm surface to count as contact. " +
                 "Tighter than InjectionSiteDetector's 7 cm approach radius on purpose: this is " +
                 "'touching the skin', not 'hovering near the arm'. Allow ~1 cm for vision jitter.")]
        [SerializeField] private float _contactDistanceMeters = 0.015f;
        [Tooltip("Seconds the tip must stay in contact before the sequence starts (debounce).")]
        [SerializeField] private float _contactDwellSeconds = 0.15f;
        [Tooltip("Seconds contact may drop out (needle briefly undetected / jitter past the " +
                 "threshold) before the attempt resets. Bridges the gaps between inferences.")]
        [SerializeField] private float _contactLossGraceSeconds = 0.4f;

        [Header("Stage 4 — depth")]
        [Tooltip("Minimum penetration (m) below the arm surface to count as inserted. ~2 mm.")]
        [SerializeField] private float _minDepthMeters = 0.002f;
        [Tooltip("Maximum penetration (m) before it counts as 'too deep' (through the vein). ~15 mm.")]
        [SerializeField] private float _maxDepthMeters = 0.015f;

        // ── Public state (read by HUD / facilitator scripts) ────────────────────────────

        public Stage CurrentStage { get; private set; } = Stage.Idle;
        /// <summary>One-line human-readable state incl. guidance, for the HUD.</summary>
        public string StatusText { get; private set; } = "Waiting for needle";

        public bool ContactOk { get; private set; }
        public bool SpotOk { get; private set; }
        public bool AngleOk { get; private set; }
        public bool DepthOk { get; private set; }

        /// <summary>Nearest vein while in contact (empty when idle).</summary>
        public string NearestVeinName { get; private set; } = "";
        public float VeinDistanceMeters { get; private set; }
        /// <summary>How far the tip sits below the arm surface (negative = still outside).</summary>
        public float PenetrationMeters { get; private set; }

        // ── Events ──────────────────────────────────────────────────────────────────────

        /// <summary>Fired whenever the sequence advances or regresses to a different stage.</summary>
        public event Action<Stage> OnStageChanged;
        /// <summary>Fired once per attempt when all four checks pass.</summary>
        public event Action OnSuccess;
        /// <summary>Fired when an attempt ends (needle left the arm), with whether it succeeded.</summary>
        public event Action<bool> OnAttemptEnded;

        // ── Private state ───────────────────────────────────────────────────────────────

        private float _contactDwellTimer;
        private float _contactLossTimer;
        private bool _attemptActive;
        private bool _succeeded;

        private void Update()
        {
            if (_useDirectVeinProximity)
            {
                UpdateDirect();
                return;
            }

            ContactOk = SpotOk = AngleOk = DepthOk = false;

            // Prerequisites: locked arm + tracked needle.
            if (_armManager == null ||
                !_armManager.TryGetArmEndpoints(out var shoulder, out var wrist))
            {
                EndAttemptIfActive("Waiting for arm lock");
                return;
            }
            if (!_armManager.TryGetNeedleTip(out var tip))
            {
                // The needle detector drops out between inferences and when the barrel is
                // occluded by the hand — give the attempt the same grace as contact loss.
                if (_attemptActive && TickLossGrace()) return;
                EndAttemptIfActive("Waiting for needle");
                return;
            }

            float signedDistance = InjectionSiteDetector.DistanceToArmSurface(
                tip, shoulder, wrist, _armRadiusMeters, out Vector3 surfacePoint);
            PenetrationMeters = -signedDistance;

            // ── Stage 1: contact (with dwell debounce and loss grace) ───────────────────
            bool touching = signedDistance <= _contactDistanceMeters;
            if (touching)
            {
                _contactLossTimer = 0f;
                _contactDwellTimer += Time.deltaTime;
            }
            else if (_attemptActive)
            {
                if (TickLossGrace()) return;   // still inside grace: hold current stage
                EndAttemptIfActive("Needle left the arm");
                return;
            }
            else
            {
                _contactDwellTimer = 0f;
                SetStage(Stage.Contact,
                    $"Approach the arm ({Mathf.Max(0f, signedDistance) * 100f:F1} cm from skin)");
                return;
            }

            if (!_attemptActive && _contactDwellTimer < _contactDwellSeconds)
            {
                SetStage(Stage.Contact, "Touching — hold steady…");
                return;
            }
            _attemptActive = true;
            ContactOk = true;

            // Success is latched for the rest of the attempt.
            if (_succeeded)
            {
                SetStage(Stage.Success, StatusText);
                return;
            }

            // ── Stage 2: right spot (vein) ──────────────────────────────────────────────
            if (_veinMap == null || !_veinMap.QueryNearestVein(surfacePoint, out var vein))
            {
                SetStage(Stage.Spot, "No vein map available");
                return;
            }
            NearestVeinName = vein.Vein.name;
            VeinDistanceMeters = vein.DistanceMeters;
            SpotOk = vein.IsOnVein;
            if (!SpotOk)
            {
                SetStage(Stage.Spot,
                    $"Wrong spot — {vein.Vein.name} is {vein.DistanceMeters * 100f:F1} cm away");
                return;
            }

            // ── Stage 3: angle ──────────────────────────────────────────────────────────
            if (!TryGetInsertionAngle(out float angle, out bool angleOk, out float minA, out float maxA))
            {
                SetStage(Stage.Angle, $"On {vein.Vein.name} — angle unavailable");
                return;
            }
            AngleOk = angleOk;
            if (!AngleOk)
            {
                SetStage(Stage.Angle,
                    $"On {vein.Vein.name} — fix angle: {angle:F0}° (need {minA:F0}–{maxA:F0}°)");
                return;
            }

            // ── Stage 4: depth ──────────────────────────────────────────────────────────
            if (PenetrationMeters < _minDepthMeters)
            {
                SetStage(Stage.Depth,
                    $"Good spot & angle — insert the needle ({PenetrationMeters * 1000f:F0} mm deep)");
                return;
            }
            if (PenetrationMeters > _maxDepthMeters)
            {
                SetStage(Stage.Depth,
                    $"TOO DEEP — pull back ({PenetrationMeters * 1000f:F0} mm, max {_maxDepthMeters * 1000f:F0} mm)");
                return;
            }
            DepthOk = true;

            // ── All checks passed ───────────────────────────────────────────────────────
            _succeeded = true;
            SetStage(Stage.Success,
                $"SUCCESS — {vein.Vein.name} at {angle:F0}°, " +
                $"{PenetrationMeters * 1000f:F0} mm deep");
            OnSuccess?.Invoke();
        }

        /// <summary>Reads the insertion angle from whichever estimator currently has one —
        /// the needle-axis estimator first (vision + simulated needles), then Group 2's
        /// keypoint-sphere estimator. Returns false when neither can measure.</summary>
        private bool TryGetInsertionAngle(out float angle, out bool acceptable,
                                          out float min, out float max)
        {
            if (_needleAngleEstimator != null && _needleAngleEstimator.HasAngle)
            {
                angle      = _needleAngleEstimator.CurrentInsertionAngle;
                acceptable = _needleAngleEstimator.IsAngleAcceptable;
                min        = _needleAngleEstimator.MinAcceptableAngle;
                max        = _needleAngleEstimator.MaxAcceptableAngle;
                return true;
            }
            if (_angleEstimator != null && _angleEstimator.HasAngle)
            {
                angle      = _angleEstimator.CurrentInsertionAngle;
                acceptable = _angleEstimator.IsAngleAcceptable;
                min        = _angleEstimator.MinAcceptableAngle;
                max        = _angleEstimator.MaxAcceptableAngle;
                return true;
            }
            angle = 0f; acceptable = false; min = 0f; max = 0f;
            return false;
        }

        /// <summary>
        /// Finger-test grading: measures the needle TIP straight against the visible vein paths,
        /// bypassing the arm collision cylinder (which the vision lock usually places offset from
        /// the veins). Stages: Contact (near a vein) → Spot (on the vein) → Success. Depth is not
        /// measurable with a fingertip, so it is not gated; the insertion angle is shown as info and
        /// only gated when <see cref="_requireAngleInDirectMode"/> is set.
        /// </summary>
        private void UpdateDirect()
        {
            ContactOk = SpotOk = AngleOk = DepthOk = false;
            PenetrationMeters = 0f;

            if (_armManager == null || !_armManager.IsLocked ||
                _veinMap == null || !_armManager.TryGetNeedleTip(out var tip) ||
                !QueryVein(tip, out var vein))
            {
                if (_attemptActive && TickLossGrace()) return;
                EndAttemptIfActive("Waiting for needle / vein map");
                return;
            }

            float dist = vein.DistanceMeters;
            NearestVeinName    = vein.Vein.name;
            VeinDistanceMeters = dist;

            // ── Stage 1: at the arm (near a vein path) ──────────────────────────────────
            // In view-grading mode `dist` is the LATERAL miss (what the trainee sees) and the
            // depth offset along the view ray is gated separately — see _gradeFromView.
            bool depthOk = vein.ViewDepthMeters <= _viewDepthToleranceMeters;
            bool near = depthOk && dist <= _directContactMeters;
            if (near)
            {
                _contactLossTimer = 0f;
                _contactDwellTimer += Time.deltaTime;
            }
            else if (_attemptActive)
            {
                if (TickLossGrace()) return;
                EndAttemptIfActive(ApproachText(dist, vein.ViewDepthMeters, depthOk));
                return;
            }
            else
            {
                _contactDwellTimer = 0f;
                SetStage(Stage.Contact, ApproachText(dist, vein.ViewDepthMeters, depthOk));
                return;
            }

            if (!_attemptActive && _contactDwellTimer < _contactDwellSeconds)
            {
                SetStage(Stage.Contact, "At the arm — hold steady…");
                return;
            }
            _attemptActive = true;
            ContactOk = true;

            if (_succeeded) { SetStage(Stage.Success, StatusText); return; }

            // ── Stage 2: right spot (on the vein) ───────────────────────────────────────
            SpotOk = vein.IsOnVein;
            if (!SpotOk)
            {
                SetStage(Stage.Spot,
                    $"Wrong spot — {vein.Vein.name} is {dist * 100f:F1} cm away");
                return;
            }

            // ── Stage 3: angle (info by default; gated only when required) ──────────────
            bool haveAngle = TryGetInsertionAngle(out float angle, out bool angleOk, out float minA, out float maxA);
            AngleOk = haveAngle && angleOk;
            if (_requireAngleInDirectMode && haveAngle && !angleOk)
            {
                SetStage(Stage.Angle,
                    $"On {vein.Vein.name} — fix angle: {angle:F0}° (need {minA:F0}–{maxA:F0}°)");
                return;
            }

            // ── Success: a fingertip can't penetrate, so on-vein is the goal ────────────
            DepthOk = true;
            _succeeded = true;
            string angleNote = haveAngle
                ? $" at {angle:F0}°{(angleOk ? "" : " (angle off)")}"
                : "";
            SetStage(Stage.Success, $"SUCCESS — on {vein.Vein.name}{angleNote}");
            OnSuccess?.Invoke();
        }

        /// <summary>Runs the vein query in the configured metric: view-perpendicular (grading
        /// what the trainee sees) when enabled and a camera exists, else raw 3D.</summary>
        private bool QueryVein(Vector3 tip, out VeinMap.QueryResult result)
        {
            var cam = Camera.main;
            if (_gradeFromView && cam != null)
                return _veinMap.QueryNearestVeinFromView(tip, cam.transform.position,
                                                         _viewDepthToleranceMeters, out result);
            return _veinMap.QueryNearestVein(tip, out result);
        }

        /// <summary>Approach guidance incl. a depth hint when the depth gate is the blocker —
        /// otherwise a tiny lateral cm reads as "basically there" while the stage refuses to
        /// advance, which looks like a bug.</summary>
        private static string ApproachText(float lateral, float viewDepth, bool depthOk) =>
            depthOk
                ? $"Approach a vein ({lateral * 100f:F1} cm away)"
                : $"Approach a vein ({lateral * 100f:F1} cm across, {viewDepth * 100f:F0} cm off in depth)";

        /// <summary>Counts contact/needle dropout time. Returns true while still within the
        /// grace window (attempt held), false once the grace is exhausted.</summary>
        private bool TickLossGrace()
        {
            _contactLossTimer += Time.deltaTime;
            return _contactLossTimer <= _contactLossGraceSeconds;
        }

        private void EndAttemptIfActive(string idleStatus)
        {
            if (_attemptActive)
            {
                bool succeeded = _succeeded;
                _attemptActive = false;
                _succeeded = false;
                OnAttemptEnded?.Invoke(succeeded);
            }
            _contactDwellTimer = 0f;
            _contactLossTimer = 0f;
            NearestVeinName = "";
            VeinDistanceMeters = 0f;
            SetStage(Stage.Idle, idleStatus);
        }

        private void SetStage(Stage stage, string status)
        {
            StatusText = status;
            if (stage == CurrentStage) return;
            CurrentStage = stage;
            OnStageChanged?.Invoke(stage);
        }
    }
}
