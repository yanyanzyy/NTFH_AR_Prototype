using UnityEngine;
using UnityEngine.UI;

namespace ARArmDetection
{
    /// <summary>
    /// Floating debug panel that appears 1.5 m in front of the camera.
    /// Add this component to any GameObject in the scene while diagnosing
    /// detection issues. Remove or disable it before final builds.
    /// </summary>
    public class ArmDetectionDebugHUD : MonoBehaviour
    {
        [SerializeField] private PassthroughCameraSource _cameraSource;
        [SerializeField] private MediaPipeHandArmDetector _mediaPipeDetector;
        [SerializeField] private ArmDetectionManager     _manager;
        [SerializeField] private float                   _distanceMeters = 1.5f;

        private Canvas    _canvas;
        private Text      _label;
        private int       _detectionCount;
        private int       _frameCount;
        private float     _lastRunTime;

        // Called by ArmDetectionManager each frame it runs inference.
        public void ReportDetections(int personCount, int armCount)
        {
            _detectionCount = armCount;
            _frameCount++;
            _lastRunTime = Time.time;
        }

        private void Awake()
        {
            BuildHUD();
        }

        private void Update()
        {
            if (_label == null) return;

            // Float in front of camera.
            var cam = Camera.main;
            if (cam != null)
            {
                transform.position = cam.transform.position + cam.transform.forward * _distanceMeters;
                transform.rotation = cam.transform.rotation;
            }

            bool hasFrame   = _cameraSource != null && _cameraSource.HasFrame;
            bool mediaPipeReady = _mediaPipeDetector != null && _mediaPipeDetector.IsReady;
            bool modelReady = mediaPipeReady;

            // Read directly from manager so this works even if ReportDetections isn't wired.
            int  inferCount  = _manager != null ? _manager.InferenceCount  : _frameCount;
            int  personCount = _manager != null ? _manager.LastPersonCount : _detectionCount;
            bool foundArm    = _manager != null ? _manager.LastFoundArm    : _detectionCount > 0;

            string camStatus  = _cameraSource != null ? _cameraSource.CameraManagerStatus : "—";
            string mgrStatus  = _manager     != null ? _manager.ManagerStatus            : "<color=red>(manager not assigned!)</color>";
            string armStatus  = _manager     != null ? _manager.LastArmStatus            : "—";
            string modeStr    = "<color=orange>ARM DETECTION</color>";
            bool   calibrated = _cameraSource != null && _cameraSource.HasCalibratedProjection;
            string projStr    = calibrated
                ? "<color=lime>CALIBRATED</color> (PCA intrinsics)"
                : "<color=red>FALLBACK</color> (FOV heuristic — overlay may be misplaced)";

            _label.text =
                $"=== ARM DETECTION DEBUG ===\n" +
                $"Mode      : {modeStr}\n" +
                $"Manager   : {mgrStatus}\n" +
                $"Camera    : {camStatus}\n" +
                $"HasFrame  : {(hasFrame   ? "<color=lime>YES</color>" : "<color=red>NO</color>")}  " +
                $"{(_cameraSource != null ? $"{_cameraSource.Width}x{_cameraSource.Height}" : "—")}\n" +
                $"Projection: {projStr}\n" +
                $"Model     : {(modelReady ? "<color=lime>READY</color>" : "<color=red>NOT READY</color>")}  " +
                $"MediaPipe={(mediaPipeReady ? "on" : "off")}\n" +
                $"Inference : {inferCount} runs\n" +
                $"Persons   : {personCount} detected\n" +
                $"Arm status: {armStatus}\n" +
                $"Arm found : {(foundArm ? "<color=lime>YES</color>" : "no")}\n" +
                $"Lock      : {(_manager != null ? (_manager.IsLocked ? $"<color=lime>{_manager.LockStatus}</color>" : _manager.LockStatus) : "—")}\n" +
                $"Needle    : {(_manager != null ? _manager.NeedleStatus : "—")}\n" +
                $"DepthAxis : {(_manager != null ? _manager.DepthAxisStatus : "—")}\n" +
                $"MaxArmKP  : {(_manager != null ? _manager.LastMaxArmScore.ToString("F3") : "—")}  " +
                $"<color=grey>(lower threshold if < threshold)</color>\n" +
                $"Time      : {Time.time:F1}s";
        }

        private void BuildHUD()
        {
            // World-space canvas so it appears inside the headset view.
            var canvasGO = new GameObject("DebugCanvas");
            canvasGO.transform.SetParent(transform, false);

            _canvas = canvasGO.AddComponent<Canvas>();
            _canvas.renderMode    = RenderMode.WorldSpace;
            _canvas.worldCamera   = Camera.main;
            canvasGO.AddComponent<CanvasScaler>();
            canvasGO.AddComponent<GraphicRaycaster>();

            var rt = canvasGO.GetComponent<RectTransform>();
            rt.sizeDelta      = new Vector2(640, 400);
            rt.localPosition  = Vector3.zero;
            rt.localRotation  = Quaternion.identity;
            rt.localScale     = Vector3.one * 0.003f; // 3 mm per canvas unit → ~1.2 m wide

            // Background panel.
            var bgGO = new GameObject("Background");
            bgGO.transform.SetParent(canvasGO.transform, false);
            var bgImg = bgGO.AddComponent<Image>();
            bgImg.color = new Color(0, 0, 0, 0.75f);
            var bgRT = bgGO.GetComponent<RectTransform>();
            bgRT.anchorMin = Vector2.zero;
            bgRT.anchorMax = Vector2.one;
            bgRT.offsetMin = Vector2.zero;
            bgRT.offsetMax = Vector2.zero;

            // Text label.
            var textGO = new GameObject("Label");
            textGO.transform.SetParent(canvasGO.transform, false);
            _label = textGO.AddComponent<Text>();
            _label.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")
                             ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
            _label.fontSize  = 18;
            _label.color     = Color.white;
            _label.supportRichText = true;
            var txtRT = textGO.GetComponent<RectTransform>();
            txtRT.anchorMin = Vector2.zero;
            txtRT.anchorMax = Vector2.one;
            txtRT.offsetMin = new Vector2(10, 10);
            txtRT.offsetMax = new Vector2(-10, -10);
        }
    }
}
