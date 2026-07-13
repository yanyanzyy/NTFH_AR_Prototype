using System;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Detects when the needle tip is close to the mannequin arm surface and fires
    /// events with the projected injection site in world space.
    ///
    /// The needle tip comes from the trained needle model (via
    /// <see cref="ArmDetectionManager.TryGetNeedleTip"/>) when a NeedleDetector is wired
    /// into the manager. When the vision needle isn't detected, the OVR right-hand index
    /// fingertip is used as a proxy for the tip (within ~3 cm of the insertion point in
    /// a standard IV grip).
    ///
    /// SETUP:
    ///   Drag the ArmDetectionManager into _armManager and the OVRSkeleton of the
    ///   nurse's right hand (from the Hand Tracking building block) into
    ///   _rightHandSkeleton.
    /// </summary>
    public class InjectionSiteDetector : MonoBehaviour
    {
        [SerializeField] private ArmDetectionManager _armManager;

        [Header("Needle-tip source")]
        [Tooltip("Prefer the trained needle model's tip (ArmDetectionManager.TryGetNeedleTip).")]
        [SerializeField] private bool _useVisionNeedle = true;
        [Tooltip("Fall back to the OVR right-hand index fingertip when the vision needle isn't detected.")]
        [SerializeField] private bool _useHandFallback = true;
        [Tooltip("OVRSkeleton component on the nurse's right hand (from the Hand Tracking building block).")]
        [SerializeField] private OVRSkeleton _rightHandSkeleton;

        [Tooltip("Physical radius of the mannequin arm in metres (forearm ≈ 4.25 cm). Used to project the fingertip onto the arm surface.")]
        [SerializeField] private float _armRadiusMeters = 0.0425f;

        [Header("Detection thresholds")]
        [Tooltip("Fingertip must be within this distance (m) of the arm surface to start the dwell timer")]
        [SerializeField] private float _approachRadiusMeters = 0.07f;

        [Tooltip("Seconds the fingertip must remain within approach radius before events fire")]
        [SerializeField] private float _dwellSeconds = 0.25f;

        // ── Events ────────────────────────────────────────────────────────────────────

        /// <summary>Fired once when dwell threshold is first reached. Payload = surface projection of fingertip.</summary>
        public event Action<Vector3> OnInjectionStarted;

        /// <summary>Fired every frame while injecting with the latest surface projection.</summary>
        public event Action<Vector3> OnInjectionUpdated;

        /// <summary>Fired when the fingertip leaves the approach radius or the arm is lost.</summary>
        public event Action OnInjectionEnded;

        // ── Public state ──────────────────────────────────────────────────────────────

        public bool    IsInjecting    { get; private set; }
        public Vector3 InjectionPoint { get; private set; }

        // ── Private state ─────────────────────────────────────────────────────────────

        private Transform _indexTipBone;
        private float     _dwellTimer;

        // ── Unity lifecycle ───────────────────────────────────────────────────────────

        private void Update()
        {
            if (_armManager == null || !_armManager.IsLocked) { EndInjection(); return; }
            if (!_armManager.TryGetArmEndpoints(out var shoulder, out var wrist)) { EndInjection(); return; }

            if (!TryGetNeedleTip(out Vector3 needleTip)) { EndInjection(); return; }

            float distToSurface = DistanceToArmSurface(needleTip, shoulder, wrist,
                                                        _armRadiusMeters, out Vector3 surfacePoint);

            if (distToSurface <= _approachRadiusMeters)
            {
                _dwellTimer += Time.deltaTime;
                if (_dwellTimer >= _dwellSeconds)
                {
                    InjectionPoint = surfacePoint;
                    if (!IsInjecting)
                    {
                        IsInjecting = true;
                        OnInjectionStarted?.Invoke(surfacePoint);
                    }
                    else
                    {
                        OnInjectionUpdated?.Invoke(surfacePoint);
                    }
                }
            }
            else
            {
                _dwellTimer = Mathf.Max(0f, _dwellTimer - Time.deltaTime * 3f);
                if (_dwellTimer == 0f) EndInjection();
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────────────

        /// <summary>
        /// Resolves the needle tip: vision needle model first, then the OVR right-hand
        /// index fingertip as fallback.
        /// </summary>
        private bool TryGetNeedleTip(out Vector3 tip)
        {
            if (_useVisionNeedle && _armManager != null && _armManager.TryGetNeedleTip(out tip))
                return true;

            if (_useHandFallback)
            {
                if (_indexTipBone == null) TryResolveBone();
                if (_indexTipBone != null)
                {
                    tip = _indexTipBone.position;
                    return true;
                }
            }

            tip = default;
            return false;
        }

        private void TryResolveBone()
        {
            if (_rightHandSkeleton == null) return;
            foreach (var bone in _rightHandSkeleton.Bones)
            {
                if (bone.Id == OVRSkeleton.BoneId.Hand_IndexTip)
                {
                    _indexTipBone = bone.Transform;
                    Debug.Log("[InjectionSiteDetector] Resolved Hand_IndexTip bone.");
                    return;
                }
            }
        }

        /// <summary>
        /// Returns the signed distance from <paramref name="point"/> to the surface of the arm
        /// cylinder (negative = inside). Also outputs the nearest point on the arm surface.
        /// Public so InjectionSequenceEvaluator shares the same arm-cylinder model.
        /// </summary>
        public static float DistanceToArmSurface(Vector3 point,
                                                   Vector3 shoulder, Vector3 wrist,
                                                   float   armRadius,
                                                   out Vector3 surfacePoint)
        {
            Vector3 axis     = wrist - shoulder;
            float   len      = axis.magnitude;
            Vector3 axisNorm = len > 0.001f ? axis / len : Vector3.up;

            float   t         = Mathf.Clamp01(Vector3.Dot(point - shoulder, axisNorm) / len);
            Vector3 axisPoint = shoulder + axisNorm * (t * len);
            Vector3 radial    = point - axisPoint;
            float   radialDist = radial.magnitude;

            Vector3 radialDir = radialDist > 0.001f ? radial / radialDist : Vector3.up;
            surfacePoint = axisPoint + radialDir * armRadius;

            return radialDist - armRadius;
        }

        private void EndInjection()
        {
            if (!IsInjecting) return;
            IsInjecting = false;
            OnInjectionEnded?.Invoke();
        }
    }
}
