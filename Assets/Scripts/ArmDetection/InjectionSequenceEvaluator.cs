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
            if (_angleEstimator == null || !_angleEstimator.HasAngle)
            {
                SetStage(Stage.Angle, $"On {vein.Vein.name} — angle unavailable");
                return;
            }
            AngleOk = _angleEstimator.IsAngleAcceptable;
            if (!AngleOk)
            {
                SetStage(Stage.Angle,
                    $"On {vein.Vein.name} — fix angle: {_angleEstimator.CurrentInsertionAngle:F0}° " +
                    $"(need {_angleEstimator.MinAcceptableAngle:F0}–{_angleEstimator.MaxAcceptableAngle:F0}°)");
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
                $"SUCCESS — {vein.Vein.name} at {_angleEstimator.CurrentInsertionAngle:F0}°, " +
                $"{PenetrationMeters * 1000f:F0} mm deep");
            OnSuccess?.Invoke();
        }

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
