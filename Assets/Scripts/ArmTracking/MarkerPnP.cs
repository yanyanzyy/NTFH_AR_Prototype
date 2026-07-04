using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Dependency-free rigid pose solver (PnP) for the marker band.
    ///
    /// Works entirely in NORMALIZED camera coordinates: each 2D observation is
    /// (x/z, y/z) of the ray direction in the camera's local frame (+X right,
    /// +Y up, +Z forward — Unity camera convention). Because the observations come
    /// from PassthroughCameraSource.ImagePointToRay, real device intrinsics, sensor
    /// crop and image-origin handling are already baked in and never appear here.
    ///
    /// Solve strategy:
    ///  1. init from the previous pose when tracking, otherwise from a planar
    ///     homography over one marker's 4 corners (Zhang-style decomposition);
    ///  2. Gauss-Newton refinement of (rotation, translation) over ALL visible
    ///     corners, minimizing reprojection error in normalized coords.
    /// With two or more band facets visible the problem is well conditioned and the
    /// single-marker planar ambiguity disappears.
    /// </summary>
    public static class MarkerPnP
    {
        public struct Correspondence
        {
            public Vector3 ObjectPoint;   // arm-local 3D
            public Vector2 Observed;      // normalized camera coords (x/z, y/z)
        }

        private const int GaussNewtonIterations = 10;
        private const float Damping = 1e-4f;

        /// <summary>
        /// Solves the arm-local → camera-local rigid transform.
        /// <paramref name="seedQuad"/> supplies exactly 4 coplanar correspondences
        /// (one physical marker) used for initialization when <paramref name="initialPose"/>
        /// is null. Returns false when the solve diverges or ends up behind the camera.
        /// </summary>
        public static bool TrySolve(List<Correspondence> all, List<Correspondence> seedQuad,
                                    Pose? initialPose, out Pose camFromArm, out float rmsError)
        {
            camFromArm = Pose.identity;
            rmsError = float.MaxValue;
            if (all == null || all.Count < 4) return false;

            Quaternion q;
            Vector3 t;
            if (initialPose.HasValue)
            {
                q = initialPose.Value.rotation;
                t = initialPose.Value.position;
            }
            else if (seedQuad == null || seedQuad.Count != 4 || !TryHomographyInit(seedQuad, out q, out t))
            {
                return false;
            }

            if (!Refine(all, ref q, ref t, out rmsError)) return false;

            // Sanity: every model point must be in front of the camera.
            for (int i = 0; i < all.Count; i++)
            {
                Vector3 p = q * all[i].ObjectPoint + t;
                if (p.z < 0.05f) return false;
            }

            camFromArm = new Pose(t, q);
            return true;
        }

        // ── Initialization: planar homography from one marker ─────────────────────────

        private static bool TryHomographyInit(List<Correspondence> quad, out Quaternion q, out Vector3 t)
        {
            q = Quaternion.identity;
            t = Vector3.zero;

            // Build an in-plane 2D basis for the marker from its 3D corners
            // (order: TL, TR, BR, BL).
            Vector3 origin = (quad[0].ObjectPoint + quad[1].ObjectPoint +
                              quad[2].ObjectPoint + quad[3].ObjectPoint) * 0.25f;
            Vector3 ex = (quad[1].ObjectPoint - quad[0].ObjectPoint).normalized;   // right
            Vector3 eyRaw = quad[0].ObjectPoint - quad[3].ObjectPoint;             // up
            Vector3 ez = Vector3.Cross(ex, eyRaw).normalized;                      // plane normal
            Vector3 ey = Vector3.Cross(ez, ex).normalized;                         // re-orthogonalized up
            if (ez.sqrMagnitude < 1e-8f) return false;

            var planePts = new Vector2[4];
            for (int i = 0; i < 4; i++)
            {
                Vector3 d = quad[i].ObjectPoint - origin;
                planePts[i] = new Vector2(Vector3.Dot(d, ex), Vector3.Dot(d, ey));
            }

            // Homography h (h33 = 1) mapping plane coords -> normalized image coords.
            // Two equations per point: 8x8 linear system.
            var A = new float[8, 9];
            for (int i = 0; i < 4; i++)
            {
                float X = planePts[i].x, Y = planePts[i].y;
                float u = quad[i].Observed.x, v = quad[i].Observed.y;
                int r0 = i * 2, r1 = i * 2 + 1;
                A[r0, 0] = X; A[r0, 1] = Y; A[r0, 2] = 1; A[r0, 6] = -u * X; A[r0, 7] = -u * Y; A[r0, 8] = u;
                A[r1, 3] = X; A[r1, 4] = Y; A[r1, 5] = 1; A[r1, 6] = -v * X; A[r1, 7] = -v * Y; A[r1, 8] = v;
            }
            var h = new float[8];
            if (!SolveLinear(A, 8, h)) return false;

            Vector3 h1 = new Vector3(h[0], h[3], h[6]);
            Vector3 h2 = new Vector3(h[1], h[4], h[7]);
            Vector3 h3 = new Vector3(h[2], h[5], 1f);

            // The marker must be in front of the camera (positive z translation).
            if (h3.z < 0f) { h1 = -h1; h2 = -h2; h3 = -h3; }

            float lambda = 2f / Mathf.Max(1e-6f, h1.magnitude + h2.magnitude);
            Vector3 r1v = h1 * lambda;
            Vector3 r2v = h2 * lambda;
            Vector3 tPlane = h3 * lambda;

            // Orthonormalize into a proper rotation (plane frame -> camera frame).
            Vector3 r3v = Vector3.Cross(r1v, r2v).normalized;
            Vector3 r2o = Vector3.Cross(r3v, r1v.normalized).normalized;
            Quaternion planeToCam = Quaternion.LookRotation(r3v, r2o);

            // Compose full arm -> camera transform from the plane frame.
            Quaternion planeInArm = Quaternion.LookRotation(ez, ey); // plane frame -> arm frame
            q = planeToCam * Quaternion.Inverse(planeInArm);
            t = tPlane - q * origin;
            return true;
        }

        // ── Gauss-Newton refinement over all corners ───────────────────────────────────

        private static bool Refine(List<Correspondence> pts, ref Quaternion q, ref Vector3 t, out float rms)
        {
            rms = float.MaxValue;
            int n = pts.Count;
            var residual = new float[n * 2];
            var jac = new float[n * 2, 6];

            for (int iter = 0; iter < GaussNewtonIterations; iter++)
            {
                float cost = ComputeResiduals(pts, q, t, residual);
                if (float.IsNaN(cost) || float.IsInfinity(cost)) return false;

                // Numeric Jacobian (central differences) over 3 rotation + 3 translation params.
                const float epsR = 0.1f;    // degrees
                const float epsT = 0.0005f; // metres
                var rPlus = new float[n * 2];
                var rMinus = new float[n * 2];

                for (int p = 0; p < 6; p++)
                {
                    Quaternion qp = q, qm = q;
                    Vector3 tp = t, tm = t;
                    float step;
                    if (p < 3)
                    {
                        Vector3 axis = p == 0 ? Vector3.right : p == 1 ? Vector3.up : Vector3.forward;
                        qp = Quaternion.AngleAxis(epsR, axis) * q;
                        qm = Quaternion.AngleAxis(-epsR, axis) * q;
                        step = 2f * epsR;
                    }
                    else
                    {
                        Vector3 axis = p == 3 ? Vector3.right : p == 4 ? Vector3.up : Vector3.forward;
                        tp = t + axis * epsT;
                        tm = t - axis * epsT;
                        step = 2f * epsT;
                    }
                    ComputeResiduals(pts, qp, tp, rPlus);
                    ComputeResiduals(pts, qm, tm, rMinus);
                    for (int r = 0; r < n * 2; r++)
                        jac[r, p] = (rPlus[r] - rMinus[r]) / step;
                }

                // Normal equations: (JtJ + damping I) delta = -Jt r
                var JtJ = new float[6, 7];
                for (int a = 0; a < 6; a++)
                {
                    for (int b = 0; b < 6; b++)
                    {
                        float s = 0f;
                        for (int r = 0; r < n * 2; r++) s += jac[r, a] * jac[r, b];
                        JtJ[a, b] = s + (a == b ? Damping : 0f);
                    }
                    float g = 0f;
                    for (int r = 0; r < n * 2; r++) g += jac[r, a] * residual[r];
                    JtJ[a, 6] = -g;
                }

                var delta = new float[6];
                if (!SolveLinear(JtJ, 6, delta)) return false;

                q = Quaternion.AngleAxis(delta[0], Vector3.right)
                  * Quaternion.AngleAxis(delta[1], Vector3.up)
                  * Quaternion.AngleAxis(delta[2], Vector3.forward)
                  * q;
                t += new Vector3(delta[3], delta[4], delta[5]);

                float stepMag = Mathf.Abs(delta[0]) + Mathf.Abs(delta[1]) + Mathf.Abs(delta[2])
                              + (Mathf.Abs(delta[3]) + Mathf.Abs(delta[4]) + Mathf.Abs(delta[5])) * 1000f;
                if (stepMag < 1e-4f) break;
            }

            float finalCost = ComputeResiduals(pts, q, t, residual);
            rms = Mathf.Sqrt(finalCost / (n * 2));
            return !float.IsNaN(rms);
        }

        /// <summary>Fills residuals (predicted - observed, normalized coords). Returns summed squared error.</summary>
        private static float ComputeResiduals(List<Correspondence> pts, Quaternion q, Vector3 t, float[] outResiduals)
        {
            float cost = 0f;
            for (int i = 0; i < pts.Count; i++)
            {
                Vector3 p = q * pts[i].ObjectPoint + t;
                float z = Mathf.Max(0.01f, p.z);
                float ru = p.x / z - pts[i].Observed.x;
                float rv = p.y / z - pts[i].Observed.y;
                outResiduals[i * 2] = ru;
                outResiduals[i * 2 + 1] = rv;
                cost += ru * ru + rv * rv;
            }
            return cost;
        }

        /// <summary>Gaussian elimination with partial pivoting. A is n x (n+1) augmented. Solution into x.</summary>
        private static bool SolveLinear(float[,] A, int n, float[] x)
        {
            for (int col = 0; col < n; col++)
            {
                int pivot = col;
                float best = Mathf.Abs(A[col, col]);
                for (int r = col + 1; r < n; r++)
                {
                    float v = Mathf.Abs(A[r, col]);
                    if (v > best) { best = v; pivot = r; }
                }
                if (best < 1e-10f) return false;
                if (pivot != col)
                    for (int c = col; c <= n; c++)
                        (A[col, c], A[pivot, c]) = (A[pivot, c], A[col, c]);

                for (int r = col + 1; r < n; r++)
                {
                    float f = A[r, col] / A[col, col];
                    for (int c = col; c <= n; c++) A[r, c] -= f * A[col, c];
                }
            }
            for (int r = n - 1; r >= 0; r--)
            {
                float s = A[r, n];
                for (int c = r + 1; c < n; c++) s -= A[r, c] * x[c];
                x[r] = s / A[r, r];
            }
            return true;
        }
    }
}
