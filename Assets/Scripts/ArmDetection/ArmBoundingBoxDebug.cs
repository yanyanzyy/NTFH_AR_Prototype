using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Renders world-space debug geometry for EVERY raw detection the arm model produced
    /// this frame, so the debug view is a faithful picture of what the model is detecting —
    /// not the single arm the manager selected, and not the temporally-smoothed box the
    /// manager tracks once locked. Boxes and keypoints are read from
    /// ArmDetectionManager.RawDetections, which is the detector's own output captured before
    /// any selection / lock-gating / smoothing touches it. When the model returns nothing (or
    /// is suspended during a frozen lock), nothing is drawn.
    ///
    /// The model regresses exactly TWO keypoints, so two diamonds are drawn per detection —
    /// no synthetic midpoint. Geometry drawn per detection:
    ///   • White rectangle       — model bounding box
    ///   • Green rectangle       — the box the manager currently selected / locked onto
    ///   • Yellow diamond        — proximal keypoint (near elbow)
    ///   • Red diamond           — distal keypoint (wrist)
    ///
    /// Boxes are projected to a per-detection heuristic depth for placement only; the box and
    /// keypoint positions themselves are the model's raw image-space output. Keypoints are only
    /// drawn when their confidence exceeds _keypointVisThreshold.
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

            // Draw the detector's RAW output — every box the model returned this frame — so
            // the debug view reflects what the model is actually detecting. RawDetections is
            // untouched by the manager's selection / smoothing (unlike LastDetections, whose
            // selected entry is overwritten with a smoothed box once locked), and is empty
            // while the model is suspended during a frozen lock.
            var dets = _manager.RawDetections;
            if (dets == null || dets.Count == 0)
            {
                RetireAll(); return;
            }

            int selectedIdx = _manager.SelectedDetectionIndex;

            for (int i = 0; i < dets.Count; i++)
            {
                var det = dets[i];
                float depth = _manager.GetEstimatedDepth(det);

                // Green = the detection the manager currently selected / locked onto;
                // white = every other raw detection the model returned this frame.
                Color boxCol = i == selectedIdx ? Color.green : Color.white;
                DrawRect(det.ImageBounds, depth, boxCol);

                // The model regresses exactly TWO keypoints — proximal (near elbow) and distal
                // (wrist). CustomArmDetector maps proximal → the shoulder slot and distal → the
                // wrist slot (the elbow slot it fills is a synthetic midpoint, NOT a model
                // output), so only these two slots are drawn — one diamond per real keypoint.
                var kps = det.Keypoints;
                if (kps != null && kps.Length > (int)CocoKeypoint.LeftWrist)
                {
                    DrawKp(kps[(int)CocoKeypoint.LeftShoulder], depth, Color.yellow); // proximal (near elbow)
                    DrawKp(kps[(int)CocoKeypoint.LeftWrist],    depth, Color.red);    // distal (wrist)
                }
            }

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
