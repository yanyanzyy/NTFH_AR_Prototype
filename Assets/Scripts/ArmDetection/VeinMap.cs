using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Defines the fixed vein layout on the mannequin arm and provides nearest-vein
    /// queries against a world-space injection point.
    ///
    /// Vein positions are expressed as normalised coordinates:
    ///   tAlongArm     0 = shoulder end, 1 = wrist end
    ///   angleDegrees  0 = dorsal/top of arm, 90 = lateral, 180 = ventral, 270 = medial
    ///
    /// These are converted to world space each frame using the arm endpoints from
    /// ArmDetectionManager. Because the mannequin never moves during a session, the
    /// world positions remain stable once the arm is locked.
    ///
    /// HOW TO CALIBRATE:
    ///   1. Run the app, lock the arm overlay.
    ///   2. Enable Debug Gizmos — coloured spheres appear at each vein zone.
    ///   3. Adjust tAlongArm and angleDegrees in the Inspector until the spheres
    ///      sit on top of the correct palpation sites on the physical mannequin.
    ///   4. Set hitRadiusMeters to the acceptable tolerance (default 2 cm).
    /// </summary>
    public class VeinMap : MonoBehaviour
    {
        [System.Serializable]
        public class VeinZone
        {
            [Tooltip("Label shown in the feedback UI (e.g. \"Median Cubital Vein\")")]
            public string name = "Unnamed Vein";

            [Range(0f, 1f)]
            [Tooltip("0 = shoulder end of the arm, 1 = wrist end")]
            public float tAlongArm = 0.35f;

            [Range(0f, 360f)]
            [Tooltip("Rotation around the arm long axis. 0 = top/dorsal (palm-up), 90 = lateral (thumb side), 180 = bottom/ventral, 270 = medial (pinky side)")]
            public float angleDegrees = 0f;

            [Tooltip("Acceptable hit radius in metres — how close the injection point must be to count as 'on vein'")]
            public float hitRadiusMeters = 0.02f;

            [Tooltip("Colour used for gizmo sphere in Scene view")]
            public Color debugColor = Color.green;
        }

        [SerializeField] private List<VeinZone> _veins = new List<VeinZone>
        {
            new VeinZone { name = "Median Cubital Vein",  tAlongArm = 0.30f, angleDegrees = 0f,   hitRadiusMeters = 0.025f, debugColor = Color.green  },
            new VeinZone { name = "Cephalic Vein",        tAlongArm = 0.50f, angleDegrees = 90f,  hitRadiusMeters = 0.020f, debugColor = Color.cyan   },
            new VeinZone { name = "Basilic Vein",         tAlongArm = 0.50f, angleDegrees = 270f, hitRadiusMeters = 0.020f, debugColor = Color.yellow },
        };

        [SerializeField] private ArmDetectionManager _armManager;

        [Tooltip("Physical radius of the mannequin arm at injection sites (forearm ~4.25 cm)")]
        [SerializeField] private float _armRadiusMeters = 0.0425f;

        [Tooltip("Draw coloured spheres at each vein position in the Scene / Game view")]
        [SerializeField] private bool _debugGizmos = true;

        // ── State ─────────────────────────────────────────────────────────────────────

        public bool    HasArm   { get; private set; }
        public Vector3 Shoulder { get; private set; }
        public Vector3 Wrist    { get; private set; }

        public IReadOnlyList<VeinZone> Veins => _veins;

        // ── Unity lifecycle ───────────────────────────────────────────────────────────

        private void Update()
        {
            if (_armManager != null && _armManager.TryGetArmEndpoints(out var s, out var w))
            {
                HasArm   = true;
                Shoulder = s;
                Wrist    = w;
            }
            else
            {
                HasArm = false;
            }
        }

        // ── Public API ────────────────────────────────────────────────────────────────

        /// <summary>
        /// Finds the nearest vein to <paramref name="injectionPoint"/> in world space.
        /// Returns false when the arm isn't locked or no veins are defined.
        /// </summary>
        public bool QueryNearestVein(Vector3 injectionPoint, out QueryResult result)
        {
            result = default;
            if (!HasArm || _veins == null || _veins.Count == 0) return false;

            float     bestDist = float.MaxValue;
            VeinZone  bestVein = null;
            Vector3   bestPos  = default;

            foreach (var vein in _veins)
            {
                Vector3 pos  = GetVeinWorldPosition(vein);
                float   dist = Vector3.Distance(injectionPoint, pos);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestVein = vein;
                    bestPos  = pos;
                }
            }

            if (bestVein == null) return false;

            result = new QueryResult
            {
                Vein           = bestVein,
                VeinWorldPos   = bestPos,
                InjectionPoint = injectionPoint,
                Delta          = bestPos - injectionPoint,
                DistanceMeters = bestDist,
                IsOnVein       = bestDist <= bestVein.hitRadiusMeters,
            };
            return true;
        }

        /// <summary>Converts a vein's normalised coordinates to world space.</summary>
        public Vector3 GetVeinWorldPosition(VeinZone vein)
            => NormalisedToWorld(vein.tAlongArm, vein.angleDegrees, _armRadiusMeters, Shoulder, Wrist);

        /// <summary>
        /// Converts normalised arm coordinates to a world-space point on the arm surface.
        /// </summary>
        public static Vector3 NormalisedToWorld(float t, float angleDeg, float radius,
                                                Vector3 shoulder, Vector3 wrist)
        {
            Vector3 armAxis    = (wrist - shoulder).normalized;
            float   armLength  = Vector3.Distance(shoulder, wrist);
            Vector3 basePos    = shoulder + armAxis * (t * armLength);

            Vector3 worldRef   = Mathf.Abs(Vector3.Dot(armAxis, Vector3.up)) < 0.99f
                               ? Vector3.up : Vector3.forward;
            Vector3 radialX    = Vector3.Cross(armAxis, worldRef).normalized;
            Vector3 radialY    = Vector3.Cross(radialX, armAxis).normalized;

            float   rad        = angleDeg * Mathf.Deg2Rad;
            Vector3 radial     = (Mathf.Cos(rad) * radialY + Mathf.Sin(rad) * radialX) * radius;

            return basePos + radial;
        }

        // ── Gizmos ────────────────────────────────────────────────────────────────────

        private void OnDrawGizmos()
        {
            if (!_debugGizmos || !HasArm || _veins == null) return;

            foreach (var vein in _veins)
            {
                Vector3 pos = GetVeinWorldPosition(vein);
                Gizmos.color = vein.debugColor;
                Gizmos.DrawSphere(pos, vein.hitRadiusMeters);
                Gizmos.DrawLine(pos, pos + Vector3.up * 0.04f);

#if UNITY_EDITOR
                UnityEditor.Handles.Label(pos + Vector3.up * 0.05f, vein.name);
#endif
            }
        }

        // ── Result struct ─────────────────────────────────────────────────────────────

        public struct QueryResult
        {
            public VeinZone Vein;
            public Vector3  VeinWorldPos;
            public Vector3  InjectionPoint;
            /// <summary>Vector from injection point to the nearest vein centre (world space).</summary>
            public Vector3  Delta;
            public float    DistanceMeters;
            public bool     IsOnVein;
        }
    }
}
