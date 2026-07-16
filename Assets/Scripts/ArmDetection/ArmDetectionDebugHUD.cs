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
        [SerializeField] private NeedleAngleEstimator    _angleEstimator;
        [SerializeField] private InjectionSequenceEvaluator _injectionEvaluator;
        [Tooltip("Optional — the finger/pen test rig. Shows whether the simulated needle is " +
                 "actually feeding (hand tracked + bones resolved) so 'nothing happens' can be diagnosed.")]
        [SerializeField] private SimulatedNeedleProvider  _simulatedNeedle;
        [SerializeField] private float                   _distanceMeters = 1.5f;

        private Canvas    _canvas;
        private Text      _label;
        private int       _detectionCount;
        private int       _frameCount;
        private float     _lastRunTime;
        private float     _nextTextRefresh;

        // Live process memory, read from /proc/self/status (works in release builds,
        // unlike the Profiler API). RSS+swap is the same number the Android
        // low-memory killer acts on, so if THIS climbs, the app is heading for a kill.
        private float _nextMemSample;
        private long  _rssKb;
        private long  _swapKb;
        private long  _prevTotalKb;
        private float _prevMemTime;
        private float _memRateMBs;

        // Rebuilding the rich-text block every frame allocates a large string AND forces a
        // Canvas mesh rebuild each frame; 4 Hz is plenty for a diagnostic readout.
        private const float TextRefreshInterval = 0.25f;

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

        private void SampleMemory()
        {
            if (Time.unscaledTime < _nextMemSample) return;
            _nextMemSample = Time.unscaledTime + 1f;

            try
            {
                foreach (var line in System.IO.File.ReadLines("/proc/self/status"))
                {
                    if (line.StartsWith("VmRSS:")) _rssKb = ParseKbLine(line);
                    else if (line.StartsWith("VmSwap:")) { _swapKb = ParseKbLine(line); break; }
                }
            }
            catch { /* not available on this platform (e.g. Editor on Windows) */ }

            long totalKb = _rssKb + _swapKb;
            if (_prevMemTime > 0f && Time.unscaledTime > _prevMemTime)
                _memRateMBs = (totalKb - _prevTotalKb) / 1024f / (Time.unscaledTime - _prevMemTime);
            _prevTotalKb = totalKb;
            _prevMemTime = Time.unscaledTime;
        }

        private static long ParseKbLine(string line)
        {
            long value = 0;
            foreach (char c in line)
            {
                if (c >= '0' && c <= '9') value = value * 10 + (c - '0');
                else if (value > 0) break;
            }
            return value;
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

            SampleMemory();

            if (Time.unscaledTime < _nextTextRefresh) return;
            _nextTextRefresh = Time.unscaledTime + TextRefreshInterval;

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
                $"Anchor    : {(_manager != null ? _manager.LockAnchorStatus : "—")}\n" +
                $"Needle    : {(_manager != null ? _manager.NeedleStatus : "—")}\n" +
                $"Finger rig: {SimulatedNeedleText()}\n" +
                $"Angle     : {NeedleAngleText()}\n" +
                $"Inject    : {InjectionText()}\n" +
                $"DepthAxis : {(_manager != null ? _manager.DepthAxisStatus : "—")}\n" +
                $"MaxArmKP  : {(_manager != null ? _manager.LastMaxArmScore.ToString("F3") : "—")}  " +
                $"<color=grey>(lower threshold if < threshold)</color>\n" +
                $"Memory    : RSS+swap {(_rssKb + _swapKb) / 1048576f:F2} GB  " +
                $"<color={(_memRateMBs > 0.5f ? "red" : "lime")}>Δ{_memRateMBs:+0.00;-0.00} MB/s</color>  " +
                $"GC {System.GC.GetTotalMemory(false) / 1048576f:F0} MB\n" +
                $"Time      : {Time.time:F1}s";
        }

        private string InjectionText()
        {
            if (_injectionEvaluator == null) return "— (no evaluator)";
            string color = _injectionEvaluator.CurrentStage switch
            {
                InjectionSequenceEvaluator.Stage.Success => "lime",
                InjectionSequenceEvaluator.Stage.Idle    => "grey",
                _                                        => "orange",
            };
            return $"<color={color}>[{_injectionEvaluator.CurrentStage}]</color> {_injectionEvaluator.StatusText}";
        }

        private string SimulatedNeedleText()
        {
            if (_simulatedNeedle == null) return "<color=grey>— (not wired)</color>";
            if (!_simulatedNeedle.isActiveAndEnabled) return "<color=grey>disabled</color>";
            string s = _simulatedNeedle.Status;
            bool feeding = s != null && s.StartsWith("Feeding");
            return $"<color={(feeding ? "lime" : "orange")}>{s}</color>";
        }

        private string NeedleAngleText()
        {
            if (_angleEstimator == null) return "— (no estimator)";
            if (!_angleEstimator.HasAngle) return "no needle axis";
            string verdict = _angleEstimator.IsAngleAcceptable
                ? "<color=lime>ACCEPTABLE</color>"
                : "<color=red>OUT OF RANGE</color>";
            return $"{_angleEstimator.CurrentInsertionAngle:F1}° {verdict}";
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
            rt.sizeDelta      = new Vector2(640, 500);
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
            // Default is Truncate, which SILENTLY drops rows that don't fit the rect —
            // diagnostic lines must never vanish, so overflow past the panel instead.
            _label.verticalOverflow = VerticalWrapMode.Overflow;
            var txtRT = textGO.GetComponent<RectTransform>();
            txtRT.anchorMin = Vector2.zero;
            txtRT.anchorMax = Vector2.one;
            txtRT.offsetMin = new Vector2(10, 10);
            txtRT.offsetMax = new Vector2(-10, -10);
        }
    }
}
