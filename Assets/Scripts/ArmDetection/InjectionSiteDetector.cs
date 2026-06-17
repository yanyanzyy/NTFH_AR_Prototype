using System;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Detects when the nurse's right-hand index fingertip is close to the mannequin
    /// arm surface and fires events with the projected injection site in world space.
    ///
    /// Uses OVR hand tracking (index fingertip bone) as a proxy for the needle tip.
    /// When holding a syringe in standard IV insertion grip the fingertip is within
    /// ~3 cm of the actual insertion point — close enough for directional feedback.
    ///
    /// SETUP:
    ///   Drag the OVRSkeleton component of the nurse's right hand into _rightHandSkeleton.
    ///   The bone is resolved automatically once hand tracking initialises.
    /// </summary>
    public class InjectionSiteDetector : MonoBehaviour
    {
        [SerializeField] private ArmDetectionManager _armManager;

        [Tooltip("OVRSkeleton component on the nurse's right hand (from the Hand Tracking building block)")]
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
            if (_indexTipBone == null) { TryResolveBone(); return; }
            if (_armManager == null || !_armManager.IsLocked) { EndInjection(); return; }
            if (!_armManager.TryGetArmEndpoints(out var shoulder, out var wrist)) { EndInjection(); return; }

            Vector3 fingertip     = _indexTipBone.position;
            float   distToSurface = DistanceToArmSurface(fingertip, shoulder, wrist,
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
        /// </summary>
        private static float DistanceToArmSurface(Vector3 point,
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
