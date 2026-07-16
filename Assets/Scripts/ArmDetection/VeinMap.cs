using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

namespace ARArmDetection
{
    /// <summary>
    /// Defines where the veins RUN on the mannequin arm (as line segments / polylines,
    /// not points — veins are long) and answers nearest-vein queries against a
    /// world-space injection point. "On vein" means within hitRadiusMeters of the vein
    /// LINE, i.e. each vein is a capsule around its path.
    ///
    /// Each vein can be defined two ways (checked in this order):
    ///
    ///  A) PREFAB PATH — recommended, no manual on-device marking.
    ///     Author the vein once in the Unity editor against the 3D arm model: inside the
    ///     arm overlay prefab, add an empty GameObject named after pathObjectName with a
    ///     chain of child empties tracing the vein along the mesh surface. At runtime the
    ///     waypoints are read from the instantiated overlay model (ArmOverlay.ModelRoot),
    ///     so the veins inherit the overlay's full alignment (position, rotation, scale)
    ///     automatically — wherever the overlay sits, the veins sit with it, and they
    ///     stay consistent with what the trainee sees.
    ///
    ///  B) CYLINDER SEGMENT — fallback needing no prefab work.
    ///     A start and end point in normalised arm coordinates (t: 0 = shoulder end,
    ///     1 = wrist end; angle: rotation around the arm axis, 0 = dorsal/top) mapped
    ///     onto the arm cylinder between the locked endpoints. If the end coordinates
    ///     are left at (0, 0) the vein degrades to a single point at the start (the old
    ///     sphere behaviour).
    /// </summary>
    public class VeinMap : MonoBehaviour
    {
        [System.Serializable]
        public class VeinZone
        {
            [Tooltip("Label shown in the feedback UI (e.g. \"Median Cubital Vein\")")]
            public string name = "Unnamed Vein";

            [Tooltip("OPTION A: name of a GameObject inside the arm overlay prefab whose CHILDREN " +
                     "(in order) trace this vein along the 3D arm mesh. When set and found, it " +
                     "overrides the cylinder-segment coordinates below.")]
            public string pathObjectName = "";

            [Header("Option B: cylinder segment (used when no prefab path)")]
            [FormerlySerializedAs("tAlongArm")]
            [Range(0f, 1f)]
            [Tooltip("Segment START along the arm: 0 = shoulder end, 1 = wrist end")]
            public float tStart = 0.35f;

            [FormerlySerializedAs("angleDegrees")]
            [Range(0f, 360f)]
            [Tooltip("Segment START rotation around the arm axis. 0 = top/dorsal, 90 = lateral, 270 = medial")]
            public float angleStart = 0f;

            [Range(0f, 1f)]
            [Tooltip("Segment END along the arm. Leave BOTH end values at 0 for a single-point vein.")]
            public float tEnd = 0f;

            [Range(0f, 360f)]
            [Tooltip("Segment END rotation around the arm axis.")]
            public float angleEnd = 0f;

            [Tooltip("Acceptable distance (m) from the vein LINE to count as 'on vein' — the vein " +
                     "is a capsule of this radius around its path.")]
            public float hitRadiusMeters = 0.02f;

            [Tooltip("Colour used for the gizmo line in Scene view")]
            public Color debugColor = Color.green;
        }

        [SerializeField] private List<VeinZone> _veins = new List<VeinZone>
        {
            new VeinZone { name = "Median Cubital Vein", pathObjectName = "Vein_MedianCubital",
                           tStart = 0.28f, angleStart = 350f, tEnd = 0.34f, angleEnd = 20f,
                           hitRadiusMeters = 0.025f, debugColor = Color.green },
            new VeinZone { name = "Cephalic Vein",       pathObjectName = "Vein_Cephalic",
                           tStart = 0.30f, angleStart = 90f, tEnd = 0.75f, angleEnd = 90f,
                           hitRadiusMeters = 0.020f, debugColor = Color.cyan },
            new VeinZone { name = "Basilic Vein",        pathObjectName = "Vein_Basilic",
                           tStart = 0.30f, angleStart = 270f, tEnd = 0.75f, angleEnd = 270f,
                           hitRadiusMeters = 0.020f, debugColor = Color.yellow },
        };

        [SerializeField] private ArmDetectionManager _armManager;

        [Tooltip("Arm overlay whose instantiated 3D model hosts the prefab-authored vein paths " +
                 "(Option A). Optional — without it only cylinder segments are used.")]
        [SerializeField] private ArmOverlay _overlay;

        [Tooltip("Physical radius of the mannequin arm at injection sites (forearm ~4.25 cm)")]
        [SerializeField] private float _armRadiusMeters = 0.0425f;

        [Tooltip("Draw the vein paths in the Scene view (editor only)")]
        [SerializeField] private bool _debugGizmos = true;

        // ── State ─────────────────────────────────────────────────────────────────────

        public bool    HasArm   { get; private set; }
        public Vector3 Shoulder { get; private set; }
        public Vector3 Wrist    { get; private set; }

        public IReadOnlyList<VeinZone> Veins => _veins;

        private readonly Dictionary<VeinZone, Transform> _pathRootCache = new();
        private readonly List<Vector3> _workPoints = new();

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
        /// Finds the vein whose PATH passes closest to <paramref name="injectionPoint"/> as SEEN
        /// from <paramref name="viewOrigin"/> (the headset): the miss distance is measured
        /// PERPENDICULAR to the view ray, ignoring the component along it. Use this for grading
        /// pokes — the locked overlay can sit several cm off in DEPTH (registration error along
        /// the view ray) while looking perfectly aligned, so a trainee who touches the vein they
        /// SEE would otherwise be failed for an error they cannot perceive or correct. The
        /// discarded depth offset is still reported (ViewDepthMeters) and bounded by
        /// <paramref name="depthToleranceMeters"/> so a hand nowhere near the arm cannot score.
        /// Delta is the IN-PLANE vector to the vein — the direction the trainee actually sees.
        /// </summary>
        public bool QueryNearestVeinFromView(Vector3 injectionPoint, Vector3 viewOrigin,
                                             float depthToleranceMeters, out QueryResult result)
        {
            result = default;
            if (!HasArm || _veins == null || _veins.Count == 0) return false;

            Vector3 viewDir = injectionPoint - viewOrigin;
            if (viewDir.sqrMagnitude < 1e-6f)
                return QueryNearestVein(injectionPoint, out result);
            viewDir.Normalize();

            float    bestLateral = float.MaxValue;
            VeinZone bestVein    = null;
            Vector3  bestWorld   = default;
            Vector3  bestInPlane = default;
            float    bestDepth   = 0f;

            foreach (var vein in _veins)
            {
                if (GetVeinPolyline(vein, _workPoints) == 0) continue;

                ClosestPointOnPolylineFromView(injectionPoint, viewDir, _workPoints,
                                               out Vector3 world, out Vector3 inPlane,
                                               out float lateral, out float depth);
                if (lateral < bestLateral)
                {
                    bestLateral = lateral;
                    bestVein    = vein;
                    bestWorld   = world;
                    bestInPlane = inPlane;
                    bestDepth   = depth;
                }
            }

            if (bestVein == null) return false;

            result = new QueryResult
            {
                Vein            = bestVein,
                VeinWorldPos    = bestWorld,
                InjectionPoint  = injectionPoint,
                Delta           = bestInPlane - injectionPoint,   // in-plane: what the trainee sees
                DistanceMeters  = bestLateral,
                ViewDepthMeters = bestDepth,
                IsOnVein        = bestLateral <= bestVein.hitRadiusMeters &&
                                  bestDepth   <= depthToleranceMeters,
            };
            return true;
        }

        /// <summary>
        /// Finds the vein whose PATH passes closest to <paramref name="injectionPoint"/>.
        /// Returns false when the arm isn't locked or no veins are defined.
        /// </summary>
        public bool QueryNearestVein(Vector3 injectionPoint, out QueryResult result)
        {
            result = default;
            if (!HasArm || _veins == null || _veins.Count == 0) return false;

            float    bestDist = float.MaxValue;
            VeinZone bestVein = null;
            Vector3  bestPos  = default;

            foreach (var vein in _veins)
            {
                if (GetVeinPolyline(vein, _workPoints) == 0) continue;

                Vector3 closest = ClosestPointOnPolyline(injectionPoint, _workPoints);
                float   dist    = Vector3.Distance(injectionPoint, closest);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestVein = vein;
                    bestPos  = closest;
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

        /// <summary>
        /// Writes the vein's current world-space waypoints into <paramref name="points"/>
        /// (prefab path when available, else the cylinder segment). Returns the count.
        /// </summary>
        public int GetVeinPolyline(VeinZone vein, List<Vector3> points)
        {
            points.Clear();

            // Option A: waypoints authored inside the arm overlay prefab.
            if (!string.IsNullOrEmpty(vein.pathObjectName) && TryGetPathRoot(vein, out var root))
            {
                if (root.childCount > 0)
                {
                    for (int i = 0; i < root.childCount; i++)
                        points.Add(root.GetChild(i).position);
                }
                else
                {
                    points.Add(root.position);
                }
                return points.Count;
            }

            // Option B: segment on the arm cylinder between the locked endpoints.
            if (!HasArm) return 0;
            points.Add(NormalisedToWorld(vein.tStart, vein.angleStart, _armRadiusMeters, Shoulder, Wrist));
            bool degeneratePoint = vein.tEnd == 0f && vein.angleEnd == 0f;
            if (!degeneratePoint)
                points.Add(NormalisedToWorld(vein.tEnd, vein.angleEnd, _armRadiusMeters, Shoulder, Wrist));
            return points.Count;
        }

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

        // ── Helpers ───────────────────────────────────────────────────────────────────

        private bool TryGetPathRoot(VeinZone vein, out Transform root)
        {
            if (_pathRootCache.TryGetValue(vein, out root) && root != null) return true;

            root = null;
            Transform modelRoot = _overlay != null ? _overlay.ModelRoot : null;
            if (modelRoot == null) return false;

            foreach (var t in modelRoot.GetComponentsInChildren<Transform>(true))
            {
                if (t.name == vein.pathObjectName)
                {
                    root = t;
                    _pathRootCache[vein] = t;
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Closest approach of a polyline to <paramref name="point"/> measured in the plane
        /// perpendicular to <paramref name="viewDir"/> (i.e. ignoring depth along the view ray).
        /// Outputs the point ON the original polyline (for arrows/markers), its flattened
        /// in-plane counterpart, the lateral distance, and the |depth| offset along the ray.
        /// </summary>
        private static void ClosestPointOnPolylineFromView(Vector3 point, Vector3 viewDir,
                                                           List<Vector3> polyline,
                                                           out Vector3 worldPoint,
                                                           out Vector3 inPlanePoint,
                                                           out float lateralDist,
                                                           out float viewDepth)
        {
            // Projects x onto the plane through `point` perpendicular to viewDir.
            Vector3 Flatten(Vector3 x) => x - viewDir * Vector3.Dot(x - point, viewDir);

            Vector3 bestWorld   = polyline[0];
            Vector3 bestInPlane = Flatten(polyline[0]);
            float   bestSqr     = (bestInPlane - point).sqrMagnitude;

            for (int i = 0; i < polyline.Count - 1; i++)
            {
                Vector3 aF = Flatten(polyline[i]);
                Vector3 bF = Flatten(polyline[i + 1]);
                Vector3 ab = bF - aF;
                float lenSqr = ab.sqrMagnitude;
                float t = lenSqr < 1e-8f ? 0f
                        : Mathf.Clamp01(Vector3.Dot(point - aF, ab) / lenSqr);

                Vector3 candInPlane = aF + ab * t;
                float sqr = (candInPlane - point).sqrMagnitude;
                if (sqr < bestSqr)
                {
                    bestSqr     = sqr;
                    bestInPlane = candInPlane;
                    bestWorld   = Vector3.Lerp(polyline[i], polyline[i + 1], t);
                }
            }

            worldPoint   = bestWorld;
            inPlanePoint = bestInPlane;
            lateralDist  = Mathf.Sqrt(bestSqr);
            viewDepth    = Mathf.Abs(Vector3.Dot(bestWorld - point, viewDir));
        }

        private static Vector3 ClosestPointOnPolyline(Vector3 point, List<Vector3> polyline)
        {
            if (polyline.Count == 1) return polyline[0];

            Vector3 best = polyline[0];
            float bestSqr = float.MaxValue;
            for (int i = 0; i < polyline.Count - 1; i++)
            {
                Vector3 candidate = ClosestPointOnSegment(point, polyline[i], polyline[i + 1]);
                float sqr = (candidate - point).sqrMagnitude;
                if (sqr < bestSqr)
                {
                    bestSqr = sqr;
                    best = candidate;
                }
            }
            return best;
        }

        private static Vector3 ClosestPointOnSegment(Vector3 point, Vector3 a, Vector3 b)
        {
            Vector3 ab = b - a;
            float lengthSqr = ab.sqrMagnitude;
            if (lengthSqr < 1e-8f) return a;
            float t = Mathf.Clamp01(Vector3.Dot(point - a, ab) / lengthSqr);
            return a + ab * t;
        }

        // ── Gizmos ────────────────────────────────────────────────────────────────────

        private void OnDrawGizmos()
        {
            if (!_debugGizmos || !HasArm || _veins == null) return;

            var points = new List<Vector3>();
            foreach (var vein in _veins)
            {
                if (GetVeinPolyline(vein, points) == 0) continue;

                Gizmos.color = vein.debugColor;
                for (int i = 0; i < points.Count - 1; i++)
                    Gizmos.DrawLine(points[i], points[i + 1]);
                foreach (var p in points)
                    Gizmos.DrawWireSphere(p, vein.hitRadiusMeters);

#if UNITY_EDITOR
                UnityEditor.Handles.Label(points[0] + Vector3.up * 0.05f, vein.name);
#endif
            }
        }

        // ── Result struct ─────────────────────────────────────────────────────────────

        public struct QueryResult
        {
            public VeinZone Vein;
            /// <summary>Closest point ON the vein path to the injection point (world space).</summary>
            public Vector3  VeinWorldPos;
            public Vector3  InjectionPoint;
            /// <summary>Vector from injection point to the closest point on the vein (world space).
            /// From <see cref="QueryNearestVeinFromView"/> this is the IN-PLANE (view-perpendicular)
            /// vector — the correction direction as the trainee sees it.</summary>
            public Vector3  Delta;
            /// <summary>Miss distance (m). From <see cref="QueryNearestVeinFromView"/> this is the
            /// LATERAL distance perpendicular to the view ray; from the plain query it is raw 3D.</summary>
            public float    DistanceMeters;
            /// <summary>Offset (m) between tip and vein ALONG the view ray — the registration/depth
            /// error the trainee cannot see. Always 0 from the plain 3D query.</summary>
            public float    ViewDepthMeters;
            public bool     IsOnVein;
        }
    }
}
