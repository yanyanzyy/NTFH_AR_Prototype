using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Renders world-space debug geometry for the SELECTED arm only — never for
    /// candidates that the manager rejected and never while searching. The selection
    /// comes from ArmDetectionManager.SelectedDetectionIndex / SelectedDetectionSide;
    /// when no arm is selected this frame, nothing is drawn.
    ///
    /// Geometry drawn for the selected arm:
    ///   • White/green rectangle — person bounding box (green = arm-only fallback)
    ///   • Yellow diamond        — shoulder keypoint
    ///   • Cyan diamond          — elbow keypoint
    ///   • Red diamond           — wrist keypoint
    ///
    /// Keypoints are only drawn when their confidence exceeds _keypointVisThreshold.
    /// Disable or remove this component before shipping.
    /// </summary>
    public class ArmBoundingBoxDebug : MonoBehaviour
    {
        [SerializeField] private ArmDetectionManager      _manager;
        [SerializeField] private PassthroughCameraSource  _cameraSource;
        [Tooltip("World-space width of the debug lines (metres).")]
        [SerializeField] private float _lineWidth = 0.007f;
        [Tooltip("Minimum keypoint confidence to draw a keypoint marker.")]
        [SerializeField, Range(0f, 1f)] private float _keypointVisThreshold = 0.05f;

        // ── Line-renderer pool ─────────────────────────────────────────────────────────

        private struct PoolEntry
        {
            public LineRenderer Lr;
            public Material     Mat;
        }

        private readonly List<PoolEntry> _pool      = new();
        private int                      _usedCount = 0;

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void LateUpdate()
        {
            _usedCount = 0;

            if (_manager == null || _cameraSource == null || !_cameraSource.HasFrame)
            {
                RetireAll(); return;
            }

            // Only draw when the manager has SELECTED a detection this frame.
            // While searching (no arm chosen) we render nothing — this keeps the
            // view clean and avoids the "flying boxes everywhere" effect.
            int  idx  = _manager.SelectedDetectionIndex;
            var  dets = _manager.LastDetections;
            if (idx < 0 || dets == null || idx >= dets.Count)
            {
                RetireAll(); return;
            }

            var det = dets[idx];
            bool armOnlyRun = _manager.LastArmStatus != null &&
                              _manager.LastArmStatus.Contains("arm-only");
            float depth = _manager.GetEstimatedDepth(det);

            Color boxCol = armOnlyRun ? Color.green : Color.white;
            DrawRect(det.ImageBounds, depth, boxCol);

            // Draw keypoints only for the side that was actually picked, so the
            // visualisation matches the red ArmOverlay quad rather than spraying
            // markers for both arms of a person.
            var side = _manager.SelectedDetectionSide;
            CocoKeypoint shoulderId = side == Side.Left ? CocoKeypoint.LeftShoulder : CocoKeypoint.RightShoulder;
            CocoKeypoint elbowId    = side == Side.Left ? CocoKeypoint.LeftElbow    : CocoKeypoint.RightElbow;
            CocoKeypoint wristId    = side == Side.Left ? CocoKeypoint.LeftWrist    : CocoKeypoint.RightWrist;

            DrawKp(det.Keypoints[(int)shoulderId], depth, Color.yellow);
            DrawKp(det.Keypoints[(int)elbowId],    depth, Color.cyan);
            DrawKp(det.Keypoints[(int)wristId],    depth, Color.red);

            RetireFrom(_usedCount);
        }

        private void OnDestroy()
        {
            foreach (var e in _pool)
                if (e.Mat != null) Destroy(e.Mat);
        }

        // ── Draw helpers ───────────────────────────────────────────────────────────────

        /// <summary>Draws a closed 4-sided rectangle at the given image-space Rect projected to <paramref name="depth"/> metres.</summary>
        private void DrawRect(Rect rect, float depth, Color col)
        {
            var lr = Acquire(col);
            lr.positionCount = 5;
            lr.SetPosition(0, W(rect.xMin, rect.yMin, depth));
            lr.SetPosition(1, W(rect.xMax, rect.yMin, depth));
            lr.SetPosition(2, W(rect.xMax, rect.yMax, depth));
            lr.SetPosition(3, W(rect.xMin, rect.yMax, depth));
            lr.SetPosition(4, W(rect.xMin, rect.yMin, depth)); // close loop
        }

        /// <summary>Draws a small diamond marker at the keypoint's image position.</summary>
        private void DrawKp(Keypoint kp, float depth, Color col)
        {
            if (kp.Confidence < _keypointVisThreshold) return;
            const float R = 12f; // half-size in camera pixels
            var p = kp.ImagePos;
            var lr = Acquire(col);
            lr.positionCount = 5;
            lr.SetPosition(0, W(p.x - R, p.y,     depth));
            lr.SetPosition(1, W(p.x,     p.y - R, depth));
            lr.SetPosition(2, W(p.x + R, p.y,     depth));
            lr.SetPosition(3, W(p.x,     p.y + R, depth));
            lr.SetPosition(4, W(p.x - R, p.y,     depth)); // close
        }

        private Vector3 W(float px, float py, float depth) =>
            _cameraSource.ImagePointToWorld(new Vector2(px, py), depth);

        // ── Pool management ────────────────────────────────────────────────────────────

        private LineRenderer Acquire(Color col)
        {
            // Grow pool on demand.
            if (_usedCount >= _pool.Count)
            {
                var go = new GameObject($"BBoxLine_{_pool.Count}");
                go.transform.SetParent(transform, false);

                var lr  = go.AddComponent<LineRenderer>();
                var mat = new Material(
                    Shader.Find("Universal Render Pipeline/Unlit")
                 ?? Shader.Find("Unlit/Color"));

                lr.material          = mat;
                lr.useWorldSpace     = true;
                lr.loop              = false;
                lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                lr.receiveShadows    = false;
                lr.startWidth        = _lineWidth;
                lr.endWidth          = _lineWidth;

                _pool.Add(new PoolEntry { Lr = lr, Mat = mat });
            }

            var entry = _pool[_usedCount++];

            // Apply colour.
            if (entry.Mat.HasProperty("_BaseColor")) entry.Mat.SetColor("_BaseColor", col);
            if (entry.Mat.HasProperty("_Color"))     entry.Mat.SetColor("_Color",     col);
            entry.Lr.startColor = col;
            entry.Lr.endColor   = col;
            entry.Lr.startWidth = _lineWidth;
            entry.Lr.endWidth   = _lineWidth;
            entry.Lr.gameObject.SetActive(true);
            return entry.Lr;
        }

        private void RetireAll()  => RetireFrom(0);

        private void RetireFrom(int startIdx)
        {
            for (int i = startIdx; i < _pool.Count; i++)
                _pool[i].Lr.gameObject.SetActive(false);
        }
    }
}
