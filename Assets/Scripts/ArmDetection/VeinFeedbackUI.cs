using UnityEngine;
using UnityEngine.UI;

namespace ARArmDetection
{
    /// <summary>
    /// World-space AR panel that floats above the injection site and shows:
    ///   • Green "CORRECT" indicator when the nurse is on the vein
    ///   • Red indicator + directional text when off-target
    ///   • A LineRenderer arrow pointing from injection site to the correct vein
    ///
    /// This script builds its own Canvas hierarchy at runtime — no prefab required.
    ///
    /// IMPORTANT: Keep this component (and its GameObject) ACTIVE at all times.
    /// The visual panel is a child object that gets enabled/disabled; this script
    /// itself must stay active so it can receive Show/Hide calls.
    /// </summary>
    public class VeinFeedbackUI : MonoBehaviour
    {
        // ── Input data struct ─────────────────────────────────────────────────────────

        public struct FeedbackData
        {
            public bool    IsOnVein;
            public string  VeinName;
            public Vector3 VeinWorldPos;
            public Vector3 InjectionPoint;
            /// <summary>Positive = must move toward wrist; negative = toward shoulder/elbow.</summary>
            public float   AlongArmMeters;
            public float   CrossArmMeters;
            /// <summary>True = must move toward the lateral/right side of the arm.</summary>
            public bool    CrossArmRight;
            public float   TotalDistance;
        }

        // ── Inspector wiring (optional — auto-built if null) ──────────────────────────

        [Header("Optional manual wiring — leave empty for auto-build")]
        [SerializeField] private Text _statusText;
        [SerializeField] private Text _instructionText;

        [Header("Layout")]
        [Tooltip("Height in metres above the injection point where the panel floats")]
        [SerializeField] private float _floatHeight = 0.15f;
        [Tooltip("How quickly the panel follows the injection point")]
        [SerializeField] private float _followSmooth = 0.08f;

        [Header("Thresholds")]
        [Tooltip("Offsets smaller than this (m) are not mentioned in the text to avoid noise")]
        [SerializeField] private float _minGuidanceMeters = 0.005f;

        [Header("Arrow")]
        [SerializeField] private float _arrowWidth = 0.004f;
        [SerializeField] private Color _arrowColor = new Color(1f, 0.8f, 0f);

        // ── Private state ─────────────────────────────────────────────────────────────

        private bool         _visible;
        private Vector3      _targetPos;
        private Vector3      _posVel;
        private LineRenderer _arrow;
        /// <summary>Child object that holds the Canvas — toggled on/off without disabling this component.</summary>
        private GameObject   _panelRoot;

        // ── Unity lifecycle ───────────────────────────────────────────────────────────

        private void Awake()
        {
            // Create a dedicated child for the visual panel so we can hide it
            // without disabling VeinFeedbackUI itself (which would break Show/Hide calls).
            _panelRoot = new GameObject("VeinFeedbackPanel");
            _panelRoot.transform.SetParent(transform, false);

            BuildArrow();
            if (_statusText == null) BuildCanvas();

            _panelRoot.SetActive(false);   // start hidden; Show() reveals it
        }

        private void LateUpdate()
        {
            if (!_visible) return;

            _panelRoot.transform.position = Vector3.SmoothDamp(
                _panelRoot.transform.position, _targetPos, ref _posVel, _followSmooth);

            var cam = Camera.main;
            if (cam != null)
                _panelRoot.transform.rotation = Quaternion.LookRotation(
                    _panelRoot.transform.position - cam.transform.position);
        }

        // ── Public API ────────────────────────────────────────────────────────────────

        public void Show(FeedbackData data)
        {
            _panelRoot.SetActive(true);
            _visible   = true;
            _targetPos = data.InjectionPoint + Vector3.up * _floatHeight;

            if (data.IsOnVein)
            {
                SetText(_statusText,      "<color=#00FF00>CORRECT</color>");
                SetText(_instructionText, $"On {data.VeinName}\nGood insertion site.");
            }
            else
            {
                SetText(_statusText,      "<color=#FF4444>ADJUST POSITION</color>");
                SetText(_instructionText, BuildInstructionText(data));
            }

            if (_arrow != null)
            {
                _arrow.enabled = !data.IsOnVein;
                if (!data.IsOnVein)
                {
                    _arrow.SetPosition(0, data.InjectionPoint);
                    _arrow.SetPosition(1, data.VeinWorldPos);
                }
            }
        }

        public void Hide()
        {
            _visible = false;
            if (_arrow != null) _arrow.enabled = false;
            _panelRoot.SetActive(false);
        }

        // ── Helpers ───────────────────────────────────────────────────────────────────

        private string BuildInstructionText(FeedbackData d)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Target: {d.VeinName}");
            sb.AppendLine($"Off by {d.TotalDistance * 1000f:F0} mm");
            sb.AppendLine();

            float alongMm = Mathf.Abs(d.AlongArmMeters) * 1000f;
            if (Mathf.Abs(d.AlongArmMeters) > _minGuidanceMeters)
            {
                string dir = d.AlongArmMeters > 0f ? "toward wrist" : "toward elbow";
                sb.AppendLine($"• Move {alongMm:F0} mm {dir}");
            }

            float crossMm = d.CrossArmMeters * 1000f;
            if (d.CrossArmMeters > _minGuidanceMeters)
            {
                string dir = d.CrossArmRight ? "laterally (thumb side)" : "medially (pinky side)";
                sb.AppendLine($"• Move {crossMm:F0} mm {dir}");
            }

            return sb.ToString();
        }

        private static void SetText(Text t, string value)
        {
            if (t != null) t.text = value;
        }

        // ── Runtime canvas builder ────────────────────────────────────────────────────

        private void BuildCanvas()
        {
            // Canvas lives on _panelRoot so the panel can be toggled independently
            var canvas = _panelRoot.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;

            var scaler = _panelRoot.AddComponent<CanvasScaler>();
            scaler.dynamicPixelsPerUnit = 300f;

            var rt = (RectTransform)_panelRoot.transform;
            rt.sizeDelta = new Vector2(0.28f, 0.20f);

            var bg    = new GameObject("Background");
            bg.transform.SetParent(_panelRoot.transform, false);
            var bgRT  = bg.AddComponent<RectTransform>();
            bgRT.anchorMin = Vector2.zero;
            bgRT.anchorMax = Vector2.one;
            bgRT.offsetMin = Vector2.zero;
            bgRT.offsetMax = Vector2.zero;
            var img   = bg.AddComponent<Image>();
            img.color = new Color(0f, 0f, 0f, 0.75f);

            _statusText      = CreateTextChild("StatusText",
                anchorMin: new Vector2(0f, 0.65f), anchorMax: new Vector2(1f, 1f),
                fontSize: 18, alignment: TextAnchor.MiddleCenter, bold: true);

            _instructionText = CreateTextChild("InstructionText",
                anchorMin: new Vector2(0.04f, 0f), anchorMax: new Vector2(0.96f, 0.65f),
                fontSize: 13, alignment: TextAnchor.UpperLeft, bold: false);
        }

        private Text CreateTextChild(string goName,
                                     Vector2 anchorMin, Vector2 anchorMax,
                                     int fontSize, TextAnchor alignment, bool bold)
        {
            var go  = new GameObject(goName);
            go.transform.SetParent(_panelRoot.transform, false);
            var rt  = go.AddComponent<RectTransform>();
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.offsetMin = new Vector2(4f, 4f);
            rt.offsetMax = new Vector2(-4f, -4f);

            var t             = go.AddComponent<Text>();
            t.font            = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")
                             ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
            t.fontSize        = fontSize;
            t.color           = Color.white;
            t.alignment       = alignment;
            t.fontStyle       = bold ? FontStyle.Bold : FontStyle.Normal;
            t.supportRichText = true;
            t.horizontalOverflow = HorizontalWrapMode.Wrap;
            t.verticalOverflow   = VerticalWrapMode.Overflow;
            return t;
        }

        private void BuildArrow()
        {
            var go   = new GameObject("FeedbackArrow");
            go.transform.SetParent(_panelRoot.transform, false);
            _arrow   = go.AddComponent<LineRenderer>();
            _arrow.positionCount = 2;
            _arrow.useWorldSpace = true;
            _arrow.startWidth    = _arrowWidth;
            _arrow.endWidth      = _arrowWidth * 2.5f;
            _arrow.startColor    = _arrowColor;
            _arrow.endColor      = new Color(_arrowColor.r, _arrowColor.g, _arrowColor.b, 0.5f);
            _arrow.material      = new Material(Shader.Find("Sprites/Default")
                                ?? Shader.Find("Unlit/Color"));
            _arrow.enabled       = false;
        }
    }
}
