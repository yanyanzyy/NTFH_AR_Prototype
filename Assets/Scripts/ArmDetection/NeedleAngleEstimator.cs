using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Estimates the needle's insertion angle relative to the horizontal plane.
    /// Port of the Jin-Rui branch's SyringeAngleEstimator: instead of reading four
    /// manually-wired keypoint sphere transforms, it reads the smoothed world-space
    /// needle axis from <see cref="ArmDetectionManager.TryGetNeedle"/> (tip = kpt0
    /// NeedleTip, hub = kpt3 Plunger).
    ///
    /// Sign convention differs from the branch version on purpose: there,
    /// 90 - Angle(dir, up) went NEGATIVE when the needle pointed down into the arm —
    /// the actual insertion case — so the 15-30° acceptance band could never pass.
    /// Here downward tilt below the horizon is POSITIVE, so a needle angled 20°
    /// down into a flat-lying mannequin arm reads +20°.
    /// </summary>
    public class NeedleAngleEstimator : MonoBehaviour
    {
        [SerializeField] private ArmDetectionManager _armManager;

        [Header("Facilitator Target Metrics")]
        [Tooltip("Venipuncture insertion angle bounds (degrees below horizontal).")]
        [SerializeField] private float _minAcceptableAngle = 15f;
        [SerializeField] private float _maxAcceptableAngle = 30f;

        /// <summary>Degrees the needle points below the horizontal plane (negative = tip
        /// above the hub). Only meaningful while HasAngle is true.</summary>
        public float CurrentInsertionAngle { get; private set; }
        public bool IsAngleAcceptable { get; private set; }
        /// <summary>True while a vision needle is tracked and its axis is long enough to measure.</summary>
        public bool HasAngle { get; private set; }

        public float MinAcceptableAngle => _minAcceptableAngle;
        public float MaxAcceptableAngle => _maxAcceptableAngle;

        private void Update()
        {
            HasAngle = false;
            IsAngleAcceptable = false;
            CurrentInsertionAngle = 0f;

            if (_armManager == null || !_armManager.TryGetNeedle(out var tip, out var hub)) return;

            Vector3 axis = tip - hub;   // plunger -> needle tip
            // Below ~2 cm the two smoothed points are effectively coincident (e.g. depth
            // raycasts snapped both to the same surface) and the direction is noise.
            if (axis.sqrMagnitude < 0.02f * 0.02f) return;

            Vector3 dir = axis.normalized;
            // Angle(dir, up): 90° = horizontal; > 90° = pointing downward. Shift so that
            // downward tilt (insertion) is positive.
            CurrentInsertionAngle = Vector3.Angle(dir, Vector3.up) - 90f;
            IsAngleAcceptable = CurrentInsertionAngle >= _minAcceptableAngle &&
                                CurrentInsertionAngle <= _maxAcceptableAngle;
            HasAngle = true;
        }
    }
}
