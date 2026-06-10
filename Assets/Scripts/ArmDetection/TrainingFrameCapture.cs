using System.Collections;
using System.IO;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Saves passthrough camera frames as PNGs for training-data capture.
    ///
    /// USAGE
    /// -----
    /// 1. Add this component to the ArmDetectionPrototype GameObject (or any GameObject)
    ///    and drag the PassthroughCameraSource into _cameraSource.
    /// 2. Build to the headset and slowly walk around the mannequin arm while frames
    ///    are captured every _intervalSeconds (vary angle, distance, lighting).
    /// 3. Pull the images to the PC (see Training/README.md):
    ///      adb pull /sdcard/Android/data/&lt;your.package.name&gt;/files/ArmCaptures Training/dataset/raw
    /// 4. Disable or remove this component when not capturing.
    ///
    /// Frames are saved at the camera's native resolution so the training images match
    /// the exact optics/colour profile the model sees at inference time.
    /// </summary>
    public class TrainingFrameCapture : MonoBehaviour
    {
        [SerializeField] private PassthroughCameraSource _cameraSource;
        [Tooltip("Seconds between captures. ~1 s while walking around the arm gives good variety.")]
        [SerializeField] private float _intervalSeconds = 1.0f;
        [Tooltip("Stop after this many captures (disk safety).")]
        [SerializeField] private int _maxCaptures = 600;
        [Tooltip("Begin capturing as soon as the scene starts.")]
        [SerializeField] private bool _captureOnStart = true;
        [Tooltip("Flip saved PNGs vertically. Enable this if pulled captures appear upside-down " +
                 "(can happen depending on the graphics API's render-texture origin).")]
        [SerializeField] private bool _flipVertical = false;

        public int  CaptureCount { get; private set; }
        public bool IsCapturing  { get; private set; }

        private string    _outputDir;
        private Texture2D _readback;

        private void Awake()
        {
            if (_cameraSource == null) _cameraSource = FindFirstObjectByType<PassthroughCameraSource>();
            _outputDir = Path.Combine(Application.persistentDataPath, "ArmCaptures");
            Directory.CreateDirectory(_outputDir);
            Debug.Log($"[FrameCapture] Saving to: {_outputDir}");
        }

        private void Start()
        {
            if (_captureOnStart) StartCapture();
        }

        public void StartCapture()
        {
            if (IsCapturing) return;
            IsCapturing = true;
            StartCoroutine(CaptureLoop());
        }

        public void StopCapture() => IsCapturing = false;

        private IEnumerator CaptureLoop()
        {
            while (IsCapturing && CaptureCount < _maxCaptures)
            {
                yield return new WaitForSeconds(_intervalSeconds);
                if (_cameraSource == null || !_cameraSource.HasFrame) continue;

                // End of frame so the camera texture is fully updated before we read it.
                yield return new WaitForEndOfFrame();
                SaveFrame(_cameraSource.CurrentTexture);
            }
            IsCapturing = false;
            Debug.Log($"[FrameCapture] Capture finished — {CaptureCount} frames in {_outputDir}");
        }

        private void SaveFrame(Texture source)
        {
            if (source == null) return;
            int w = source.width, h = source.height;
            if (w < 16 || h < 16) return;

            var tmp = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(source, tmp);

            if (_readback == null || _readback.width != w || _readback.height != h)
            {
                if (_readback != null) Destroy(_readback);
                _readback = new Texture2D(w, h, TextureFormat.RGB24, false);
            }

            var prev = RenderTexture.active;
            RenderTexture.active = tmp;
            _readback.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            _readback.Apply(false);
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(tmp);

            if (_flipVertical) FlipVertically(_readback);

            string file = Path.Combine(_outputDir,
                $"arm_{System.DateTime.Now:yyyyMMdd_HHmmss_fff}.png");
            File.WriteAllBytes(file, _readback.EncodeToPNG());
            CaptureCount++;

            if (CaptureCount % 10 == 0)
                Debug.Log($"[FrameCapture] {CaptureCount}/{_maxCaptures} frames captured");
        }

        private static void FlipVertically(Texture2D tex)
        {
            var pixels = tex.GetPixels32();
            int w = tex.width, h = tex.height;
            var flipped = new Color32[pixels.Length];
            for (int y = 0; y < h; y++)
                System.Array.Copy(pixels, y * w, flipped, (h - 1 - y) * w, w);
            tex.SetPixels32(flipped);
            tex.Apply(false);
        }

        private void OnDestroy()
        {
            if (_readback != null) Destroy(_readback);
        }
    }
}
