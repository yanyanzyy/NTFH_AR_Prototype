using UnityEngine;
using ZXing;
using ZXing.Common;

namespace ARArmDetection
{
    /// <summary>
    /// Detects a QR code in the passthrough camera texture using ZXing.Net.
    /// Uses the calibrated PCA intrinsics to project the marker centre into world space.
    ///
    /// SETUP
    /// -----
    /// 1. Drop zxing.dll into Assets/Plugins/.
    /// 2. Print a QR code at 10 × 10 cm, tape it to the shoulder end of the arm.
    /// 3. Assign _cameraSource in the Inspector.
    ///
    /// The QR code text content does not matter — any QR code works.
    /// </summary>
    public class MarkerDetector : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private PassthroughCameraSource _cameraSource;

        [Header("Marker settings")]
        [Tooltip("Physical width of the printed QR code in metres (e.g. 0.10 = 10 cm).")]
        [SerializeField] private float _markerSizeMetres = 0.10f;

        [Tooltip("How many frames to skip between detection runs. " +
                 "3 is a good balance — detects fast enough while avoiding frame drops.")]
        [SerializeField, Range(1, 15)] private int _detectEveryNFrames = 3;

        [Tooltip("How many recent detection attempts to consider for the sliding window.\n" +
                 "e.g. window=5, required=3 means 3 successes in the last 5 attempts = confirmed.")]
        [SerializeField, Range(2, 10)] private int _windowSize = 5;

        [Tooltip("How many successes within the window are needed to confirm detection.\n" +
                 "Lower = faster but more false positives. 3 out of 5 is a good default.")]
        [SerializeField, Range(1, 10)] private int _requiredHits = 3;

        // ── Public result ──────────────────────────────────────────────────────────────

        /// <summary>True when the QR code has been reliably confirmed.</summary>
        public bool       IsDetected     { get; private set; }
        /// <summary>World-space position of the QR code centre when IsDetected is true.</summary>
        public Vector3    MarkerWorldPos { get; private set; }
        /// <summary>World-space rotation of the marker plane when IsDetected is true.</summary>
        public Quaternion MarkerWorldRot { get; private set; }

        // ── Private state ──────────────────────────────────────────────────────────────

        private BarcodeReader _reader;
        private Texture2D     _readbackTex;
        private int           _frameCounter;

        // Sliding window: stores the last _windowSize detection results (true/false).
        private bool[] _window;
        private int    _windowIndex;

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void Awake()
        {
            _reader = new BarcodeReader
            {
                // AutoRotate tries all 4 rotations — essential for a marker on a curved
                // arm surface that may be tilted at any angle.
                AutoRotate = true,
                Options = new DecodingOptions
                {
                    // TryHarder makes ZXing attempt multiple binarization thresholds,
                    // which helps in uneven lighting and on curved/reflective surfaces.
                    TryHarder       = true,
                    // TryInverted also attempts to decode a light-on-dark version of
                    // the image — useful if lighting washes out the QR code contrast.
                    TryInverted     = true,
                    PossibleFormats = new[] { BarcodeFormat.QR_CODE },
                }
            };

            _window = new bool[Mathf.Max(2, _windowSize)];
        }

        private void OnDestroy()
        {
            if (_readbackTex != null) Destroy(_readbackTex);
        }

        private void Update()
        {
            if (_cameraSource == null || !_cameraSource.HasFrame) return;

            _frameCounter++;
            if (_frameCounter % _detectEveryNFrames != 0) return;

            var tex = _cameraSource.CurrentTexture;
            if (tex == null) return;

            // Copy the GPU texture to a CPU-readable Texture2D.
            // Half resolution keeps detection fast while still giving enough pixels
            // for small markers (6–7 cm) to be readable at arm viewing distance.
            int w = Mathf.Max(1, tex.width  / 2);
            int h = Mathf.Max(1, tex.height / 2);
            EnsureReadbackTex(w, h);

            var prev = RenderTexture.active;
            var rt   = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(tex, rt);
            RenderTexture.active = rt;
            _readbackTex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
            _readbackTex.Apply(false);
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);

            // Run ZXing detection.
            Color32[] pixels = _readbackTex.GetPixels32();
            var result = _reader.Decode(pixels, w, h);

            bool detected = result?.ResultPoints != null && result.ResultPoints.Length >= 3;

            // Sliding window: record this attempt, count successes in the window.
            // One missed frame no longer resets everything — much more robust on
            // a curved surface where the QR briefly goes out of the optimal angle.
            _window[_windowIndex % _window.Length] = detected;
            _windowIndex++;

            int hits = 0;
            foreach (bool b in _window) if (b) hits++;

            if (hits >= _requiredHits)
            {
                IsDetected = true;
                if (detected)
                    ComputeWorldPose(result.ResultPoints, w, h);
            }
            else
            {
                IsDetected = false;
            }
        }

        // ── Pose estimation ────────────────────────────────────────────────────────────

        /// <summary>
        /// Estimates the marker's world-space pose from its detected image corners
        /// using the calibrated PCA camera intrinsics.
        /// </summary>
        private void ComputeWorldPose(ZXing.ResultPoint[] pts, int sampleW, int sampleH)
        {
            // ZXing returns 3 finder-pattern centres:
            //   pts[0] = top-left corner of QR code
            //   pts[1] = top-right corner
            //   pts[2] = bottom-left corner
            float fullW = _cameraSource.Width;
            float fullH = _cameraSource.Height;
            float sx    = fullW / sampleW;
            float sy    = fullH / sampleH;

            // Centre of the QR code in full camera-image pixels.
            float cx = ((pts[0].X + pts[1].X + pts[2].X) / 3f) * sx;
            float cy = ((pts[0].Y + pts[1].Y + pts[2].Y) / 3f) * sy;

            // Depth from apparent marker size.
            float p0p1     = Dist(pts[0], pts[1]);
            float p0p2     = Dist(pts[0], pts[2]);
            float markerPx = Mathf.Max(p0p1, p0p2) * sx;
            float focalPx  = (fullH * 0.5f) / Mathf.Tan(_cameraSource.VerticalFovRadians * 0.5f);
            float depth    = Mathf.Clamp(_markerSizeMetres * focalPx / Mathf.Max(markerPx, 1f),
                                         0.1f, 5f);

            // Project the three finder patterns and the centre into world space.
            // Using the same depth for all gives a consistent planar estimate.
            Vector3 worldCentre = _cameraSource.ImagePointToWorld(new Vector2(cx, cy), depth);
            Vector3 worldTL     = _cameraSource.ImagePointToWorld(
                                      new Vector2(pts[0].X * sx, pts[0].Y * sy), depth);
            Vector3 worldTR     = _cameraSource.ImagePointToWorld(
                                      new Vector2(pts[1].X * sx, pts[1].Y * sy), depth);
            Vector3 worldBL     = _cameraSource.ImagePointToWorld(
                                      new Vector2(pts[2].X * sx, pts[2].Y * sy), depth);

            MarkerWorldPos = worldCentre;

            // Compute the QR code's real-world orientation from its corner positions.
            // right  = direction from top-left to top-right along the QR code.
            // down   = direction from top-left to bottom-left along the QR code.
            // normal = perpendicular to the marker surface, pointing away from the arm.
            Vector3 right  = (worldTR - worldTL);
            Vector3 down   = (worldBL - worldTL);

            if (right.sqrMagnitude < 1e-8f || down.sqrMagnitude < 1e-8f)
            {
                // Degenerate detection — fall back to camera-facing orientation.
                var camPose = _cameraSource.CameraPose;
                Vector3 toCam = (camPose.position - worldCentre).normalized;
                MarkerWorldRot = Quaternion.LookRotation(toCam, Vector3.up);
                return;
            }

            right = right.normalized;
            down  = down.normalized;
            Vector3 normal = Vector3.Cross(right, down).normalized;

            // LookRotation(forward, up):
            //   forward = normal to QR surface (pointing toward camera = out of arm)
            //   up      = NEGATIVE down direction = from bottom-left to top-left
            //             This means "up" on the marker points AWAY from the wrist.
            MarkerWorldRot = Quaternion.LookRotation(normal, -down);
        }

        // ── Helpers ────────────────────────────────────────────────────────────────────

        private void EnsureReadbackTex(int w, int h)
        {
            if (_readbackTex != null && _readbackTex.width == w && _readbackTex.height == h) return;
            if (_readbackTex != null) Destroy(_readbackTex);
            _readbackTex = new Texture2D(w, h, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode   = TextureWrapMode.Clamp,
            };
        }

        private static float Dist(ZXing.ResultPoint a, ZXing.ResultPoint b)
            => Mathf.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));
    }
}
