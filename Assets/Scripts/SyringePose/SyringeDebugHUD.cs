using UnityEngine;
using UnityEngine.UI;

public class SyringeDebugHUD : MonoBehaviour
{
    [Header("Dependencies to Monitor")]
    [SerializeField] private CustomSyringeDetector detector;
    [SerializeField] private SyringeAngleEstimator angleEstimator;

    private GameObject _panelRoot;
    private Text _hudText;

    // Rebuilding the rich-text block (StringBuilder + string allocs + UI Text re-layout)
    // every frame feeds the GC and costs frame time on Quest for no visible benefit -
    // refresh the text at ~5 Hz instead. The panel transform still follows the head
    // every frame so it doesn't visibly lag behind.
    private const float TextRefreshInterval = 0.2f;
    private float _nextTextRefreshTime;
    private readonly System.Text.StringBuilder _sb = new System.Text.StringBuilder(1024);

    // FindAnyObjectByType is a scene scan - retry missing references on a slow timer,
    // not every frame.
    private float _nextSceneSearchTime;

    private void Awake()
    {
        // 1. Build a clean, lightweight UI Canvas that doesn't rely on outside prefabs
        _panelRoot = new GameObject("SyringeDebugCanvas");
        _panelRoot.transform.SetParent(transform, false);

        var canvas = _panelRoot.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        var canvasRt = (RectTransform)_panelRoot.transform;
        canvasRt.sizeDelta = new Vector2(500f, 300f);
        canvasRt.localScale = Vector3.one * 0.0015f; // Scale down so it fits comfortably in VR view

        // 2. Create a solid dark background card so the text is readable
        var bgGo = new GameObject("Background", typeof(RectTransform), typeof(Image));
        bgGo.transform.SetParent(_panelRoot.transform, false);
        var bgRt = (RectTransform)bgGo.transform;
        bgRt.anchorMin = Vector2.zero; bgRt.anchorMax = Vector2.one;
        bgRt.offsetMin = Vector2.zero; bgRt.offsetMax = Vector2.zero;
        bgGo.GetComponent<Image>().color = new Color(0.02f, 0.02f, 0.04f, 0.90f);

        // 3. Create the text block component
        var textGo = new GameObject("DebugText", typeof(RectTransform), typeof(Text));
        textGo.transform.SetParent(_panelRoot.transform, false);
        var textRt = (RectTransform)textGo.transform;
        textRt.anchorMin = new Vector2(0.05f, 0.05f); textRt.anchorMax = new Vector2(0.95f, 0.95f);
        textRt.offsetMin = Vector2.zero; textRt.offsetMax = Vector2.zero;

        _hudText = textGo.GetComponent<Text>();
        _hudText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        _hudText.fontSize = 16;
        _hudText.color = Color.white;
        _hudText.alignment = TextAnchor.UpperLeft;
    }

    void Start()
    {
        if (detector == null) detector = FindAnyObjectByType<CustomSyringeDetector>();
        if (angleEstimator == null) angleEstimator = FindAnyObjectByType<SyringeAngleEstimator>();
    }

    private void LateUpdate()
    {
        if ((detector == null || angleEstimator == null) && Time.unscaledTime >= _nextSceneSearchTime)
        {
            _nextSceneSearchTime = Time.unscaledTime + 1f;
            if (detector == null) detector = FindAnyObjectByType<CustomSyringeDetector>();
            if (angleEstimator == null) angleEstimator = FindAnyObjectByType<SyringeAngleEstimator>();
        }

        // Anchor the debug panel directly in front of your head/camera view space
        var cam = Camera.main;
        if (cam == null) return;

        Transform ct = cam.transform;
        transform.position = ct.position + ct.forward * 1.2f + ct.up * -0.2f; // 1.2 meters away, slightly lowered
        transform.rotation = Quaternion.LookRotation((transform.position - ct.position).normalized, ct.up);

        // 5. Update data display strings (throttled - see TextRefreshInterval)
        if (_hudText == null) return;
        if (Time.unscaledTime < _nextTextRefreshTime) return;
        _nextTextRefreshTime = Time.unscaledTime + TextRefreshInterval;

        System.Text.StringBuilder sb = _sb;
        sb.Clear();
        sb.AppendLine("<b>== SYRINGE PIPELINE DEBUGGER ==</b>\n");

        if (detector == null)
        {
            sb.AppendLine("<color=red>Status: Detector Reference Missing in Inspector</color>");
        }
        else
        {
            sb.AppendLine($"Live Inference Output Loop: {(detector.isActiveAndEnabled ? "<color=green>ACTIVE</color>" : "<color=yellow>PAUSED</color>")}");
            sb.AppendLine($"Max Box Confidence: {detector.HighestConfidence:F4}");
            sb.AppendLine($"Detection State: {(detector.IsSyringeDetected ? "<color=green>FOUND</color>" : "<color=orange>SEARCHING...</color>")}");

            // Render insertion angle metrics from your standalone estimator script
            if (angleEstimator != null)
            {
                string angleColor = angleEstimator.IsAngleAcceptable ? "green" : "red";
                string statusText = angleEstimator.IsAngleAcceptable ? "ACCEPTABLE" : "INVALID BOUNDS";

                sb.AppendLine($"Insertion Angle: <b>{angleEstimator.CurrentInsertionAngle:F1}°</b>");
                sb.AppendLine($"Facilitator Rule: <color={angleColor}><b>{statusText}</b></color>");
            }
            else
            {
                sb.AppendLine("<color=yellow>Angle Estimator: Script reference unassigned</color>");
            }

            sb.AppendLine("\nNormalized Anchor Keypoints:");

            for (int i = 0; i < 4; i++)
            {
                Vector2 pt = detector.NormalizedKeypoints[i];
                float kptConf = detector.KeypointConfidences[i];
                // Low per-keypoint confidence doesn't mean "wrong code" - VPIC found this model's
                // 4th keypoint specifically runs ~0.03-0.05 even on clean detections while 0-2 run
                // ~0.98-0.99. Colour-coding here so a low number reads as "known model limitation",
                // not "something broke".
                string confColor = kptConf >= 0.5f ? "green" : (kptConf >= 0.15f ? "yellow" : "red");
                sb.AppendLine($"  Point {i + 1}: ({pt.x:F3}, {pt.y:F3})  conf=<color={confColor}>{kptConf:F3}</color>");
            }
        }

        _hudText.text = sb.ToString();
    }
}
