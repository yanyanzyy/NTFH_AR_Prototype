using UnityEngine;
using UnityEngine.UI;

namespace ARArmDetection
{
    /// <summary>
    /// Clean, Facilitator-style status panel for the vein injection trainer. Unlike the raw
    /// <see cref="ArmDetectionDebugHUD"/> (a dense always-on diagnostics dump), this panel:
    ///   • only appears once the arm is LOCKED (nothing to inject into before that), and
    ///   • presents the finger/needle → vein result the trainee actually cares about:
    ///     which vein is nearest, how far off, whether the current poke is the CORRECT spot,
    ///     the insertion angle, and a big colour-coded verdict (APPROACH / WRONG / ON VEIN /
    ///     SUCCESS) that mirrors the InjectionSequenceEvaluator stages.
    ///
    /// It builds its own world-space canvas at runtime (no prefab) in the same visual language
    /// as the Facilitator panel — dark rounded card, teal accent, stage pips — and floats in a
    /// comfortable spot in front of the viewer. All references self-heal, so simply having the
    /// component in the scene (or letting <see cref="VeinTrainerHUDBootstrap"/> add it) is enough.
    /// </summary>
    public class VeinTrainerHUD : MonoBehaviour
    {
        [Header("Sources (auto-found when empty)")]
        [SerializeField] private ArmDetectionManager      _manager;
        [SerializeField] private InjectionSequenceEvaluator _evaluator;
        [SerializeField] private SimulatedNeedleProvider   _simulatedNeedle;
        [SerializeField] private NeedleAngleEstimator      _angleEstimator;

        [Header("Placement")]
        [Tooltip("Distance (m) the panel floats in front of the viewer.")]
        [SerializeField] private float _distanceMeters = 1.05f;
        [Tooltip("Vertical offset (m) from eye level. Negative sits it below the eyeline so it " +
                 "doesn't cover the arm.")]
        [SerializeField] private float _heightOffsetMeters = -0.28f;
        [Tooltip("Horizontal offset (m). Positive nudges it right of centre, clear of the debug HUD.")]
        [SerializeField] private float _rightOffsetMeters = 0.34f;
        [Tooltip("Follow smoothing (0 = snap, 1 = frozen). Keeps the panel from jittering with head motion.")]
        [SerializeField, Range(0f, 0.99f)] private float _followSmoothing = 0.5f;

        // ── Colours (match the Facilitator panel language) ──────────────────────────────
        private static readonly Color Bg        = new Color(0.055f, 0.065f, 0.075f, 0.96f);
        private static readonly Color Ink       = new Color(0.95f, 0.97f, 1f);
        private static readonly Color Teal      = new Color(0.14f, 0.78f, 0.66f);
        private static readonly Color Muted     = new Color(0.62f, 0.67f, 0.72f);
        private static readonly Color Good       = new Color(0.20f, 0.85f, 0.42f);
        private static readonly Color Warn      = new Color(1f, 0.62f, 0.12f);
        private static readonly Color Bad       = new Color(0.96f, 0.32f, 0.26f);
        private static readonly Color TrackDim  = new Color(0.20f, 0.22f, 0.25f, 1f);

        private const float TextRefreshInterval = 0.15f;

        private GameObject _panelRoot;
        private Image      _verdictBar;
        private Text       _verdictText;
        private Text       _badgeText;
        private Text       _needleText;
        private Text       _veinText;
        private Text       _angleText;
        private Text       _guidanceText;
        private Text       _lockText;
        private Image[]    _stagePips;
        private Material   _alwaysOnTop;
        private float      _nextRefresh;
        private bool       _placed;

        // Stage order shown as pips (Depth is skipped for the finger test).
        private static readonly InjectionSequenceEvaluator.Stage[] PipStages =
        {
            InjectionSequenceEvaluator.Stage.Contact,
            InjectionSequenceEvaluator.Stage.Spot,
            InjectionSequenceEvaluator.Stage.Angle,
            InjectionSequenceEvaluator.Stage.Success,
        };
        private static readonly string[] PipLabels = { "NEAR", "SPOT", "ANGLE", "DONE" };

        // ── Unity lifecycle ─────────────────────────────────────────────────────────────

        private void Awake()
        {
            BuildUI();
            SetPanelVisible(false);
        }

        private void OnDestroy()
        {
            if (_alwaysOnTop != null) Destroy(_alwaysOnTop);
        }

        private void Update()
        {
            SelfHeal();

            bool locked = _manager != null && _manager.IsLocked;
            if (_panelRoot.activeSelf != locked) SetPanelVisible(locked);
            if (!locked) return;

            PlacePanel();

            if (Time.unscaledTime < _nextRefresh) return;
            _nextRefresh = Time.unscaledTime + TextRefreshInterval;
            RefreshContent();
        }

        // ── Content ─────────────────────────────────────────────────────────────────────

        private void RefreshContent()
        {
            var stage = _evaluator != null ? _evaluator.CurrentStage
                                           : InjectionSequenceEvaluator.Stage.Idle;

            // Verdict banner — the headline the trainee reads at a glance.
            string verdict; Color color;
            if (_evaluator == null)
            {
                verdict = "NO EVALUATOR"; color = Muted;
            }
            else if (stage == InjectionSequenceEvaluator.Stage.Success)
            {
                verdict = "SUCCESS"; color = Good;
            }
            else if (_evaluator.SpotOk)
            {
                verdict = "ON VEIN"; color = Good;
            }
            else if (_evaluator.ContactOk)
            {
                verdict = "WRONG SPOT"; color = Bad;
            }
            else
            {
                verdict = "APPROACH A VEIN"; color = Muted;
            }
            SetText(_verdictText, verdict, color);
            if (_verdictBar != null) _verdictBar.color = new Color(color.r, color.g, color.b, 0.22f);

            // SIM / LIVE badge.
            bool sim = _manager != null && _manager.NeedleIsSimulated;
            SetText(_badgeText, sim ? "SIMULATION" : "LIVE NEEDLE", sim ? Teal : Warn);

            // Needle / finger feeding status.
            string needle;
            Color needleColor;
            if (_simulatedNeedle != null && _simulatedNeedle.isActiveAndEnabled &&
                _simulatedNeedle.Status != null && _simulatedNeedle.Status.StartsWith("Feeding"))
            {
                needle = $"Finger: {_simulatedNeedle.Status}";
                needleColor = Good;
            }
            else
            {
                needle = $"Needle: {(_manager != null ? _manager.NeedleStatus : "-")}";
                needleColor = _manager != null && _manager.TryGetNeedleTip(out _) ? Ink : Warn;
            }
            SetText(_needleText, needle, needleColor);

            // Nearest vein + distance.
            if (_evaluator != null && !string.IsNullOrEmpty(_evaluator.NearestVeinName))
            {
                float cm = _evaluator.VeinDistanceMeters * 100f;
                bool onVein = _evaluator.SpotOk;
                SetText(_veinText,
                    $"Nearest vein: {_evaluator.NearestVeinName}   " +
                    $"<color=#{ColorHex(onVein ? Good : Warn)}>{cm:F1} cm</color>",
                    Ink);
            }
            else
            {
                SetText(_veinText, "Nearest vein: —", Muted);
            }

            // Insertion angle.
            if (_angleEstimator != null && _angleEstimator.HasAngle)
            {
                bool ok = _angleEstimator.IsAngleAcceptable;
                SetText(_angleText,
                    $"Angle: {_angleEstimator.CurrentInsertionAngle:F0}° " +
                    $"({_angleEstimator.MinAcceptableAngle:F0}–{_angleEstimator.MaxAcceptableAngle:F0}°)  " +
                    $"<color=#{ColorHex(ok ? Good : Warn)}>{(ok ? "OK" : "off")}</color>",
                    Ink);
            }
            else
            {
                SetText(_angleText, "Angle: —", Muted);
            }

            // Guidance line straight from the evaluator.
            SetText(_guidanceText, _evaluator != null ? _evaluator.StatusText : "", Muted);

            // Lock footer + anchor state (world-locked = survives headset re-donning).
            if (_manager != null)
            {
                string anchor = _manager.LockAnchorStatus;
                bool anchored = anchor != null && anchor.Contains("world-locked");
                SetText(_lockText,
                    $"{_manager.LockStatus}   <color=#{ColorHex(anchored ? Good : Muted)}>{anchor}</color>",
                    Muted);
            }
            else
            {
                SetText(_lockText, "", Muted);
            }

            UpdateStagePips(stage);
        }

        private void UpdateStagePips(InjectionSequenceEvaluator.Stage stage)
        {
            if (_stagePips == null) return;

            bool success = stage == InjectionSequenceEvaluator.Stage.Success;
            bool spotOk  = _evaluator != null && _evaluator.SpotOk;
            bool angleOk = _evaluator != null && _evaluator.AngleOk;
            bool contact = _evaluator != null && _evaluator.ContactOk;

            // Which pips are "done".
            bool[] done =
            {
                contact || spotOk || success,
                spotOk || success,
                angleOk || success,
                success,
            };

            for (int i = 0; i < _stagePips.Length && i < done.Length; i++)
            {
                if (_stagePips[i] == null) continue;
                _stagePips[i].color = done[i] ? Good
                    : (stage == PipStages[i] ? Warn : TrackDim);
            }
        }

        // ── Placement ───────────────────────────────────────────────────────────────────

        private void PlacePanel()
        {
            var cam = Camera.main;
            if (cam == null) return;

            Transform c = cam.transform;
            Vector3 target = c.position
                           + c.forward * _distanceMeters
                           + c.up      * _heightOffsetMeters
                           + c.right   * _rightOffsetMeters;

            if (!_placed)
            {
                transform.position = target;
                _placed = true;
            }
            else
            {
                transform.position = Vector3.Lerp(target, transform.position, _followSmoothing);
            }
            transform.rotation = Quaternion.LookRotation(transform.position - c.position, c.up);
        }

        // ── Build ───────────────────────────────────────────────────────────────────────

        private void BuildUI()
        {
            _panelRoot = new GameObject("VeinTrainerCanvas");
            _panelRoot.transform.SetParent(transform, false);

            var canvas = _panelRoot.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            _panelRoot.AddComponent<CanvasScaler>();

            var rt = (RectTransform)_panelRoot.transform;
            rt.sizeDelta = new Vector2(560f, 360f);
            rt.localScale = Vector3.one * 0.0013f;   // ~0.73 m wide

            CreateImage("Background", rt, Vector2.zero, Vector2.one, Bg);

            // Header: title + SIM badge.
            CreateText("Title", rt, new Vector2(0.04f, 0.87f), new Vector2(0.70f, 0.98f),
                24, FontStyle.Bold, TextAnchor.MiddleLeft, Ink).text = "VEIN INJECTION TRAINER";
            _badgeText = CreateText("Badge", rt, new Vector2(0.66f, 0.87f), new Vector2(0.96f, 0.98f),
                16, FontStyle.Bold, TextAnchor.MiddleRight, Teal);

            // Verdict banner.
            _verdictBar = CreateImage("VerdictBar", rt, new Vector2(0.04f, 0.66f), new Vector2(0.96f, 0.85f),
                new Color(0.5f, 0.5f, 0.5f, 0.2f));
            _verdictText = CreateText("Verdict", rt, new Vector2(0.06f, 0.66f), new Vector2(0.94f, 0.85f),
                34, FontStyle.Bold, TextAnchor.MiddleCenter, Ink);

            // Stage pips row.
            BuildStagePips(rt, 0.575f, 0.64f);

            // Detail lines.
            _needleText   = CreateText("Needle",   rt, new Vector2(0.05f, 0.45f), new Vector2(0.95f, 0.55f),
                18, FontStyle.Normal, TextAnchor.MiddleLeft, Ink);
            _veinText     = CreateText("Vein",     rt, new Vector2(0.05f, 0.35f), new Vector2(0.95f, 0.45f),
                18, FontStyle.Bold,   TextAnchor.MiddleLeft, Ink);
            _angleText    = CreateText("Angle",    rt, new Vector2(0.05f, 0.25f), new Vector2(0.95f, 0.35f),
                18, FontStyle.Normal, TextAnchor.MiddleLeft, Ink);

            _guidanceText = CreateText("Guidance", rt, new Vector2(0.05f, 0.10f), new Vector2(0.95f, 0.24f),
                17, FontStyle.Normal, TextAnchor.UpperLeft, Muted);
            _lockText     = CreateText("Lock",     rt, new Vector2(0.05f, 0.02f), new Vector2(0.95f, 0.09f),
                13, FontStyle.Normal, TextAnchor.MiddleLeft, Muted);

            ApplyAlwaysOnTop();
        }

        private void BuildStagePips(RectTransform parent, float yMin, float yMax)
        {
            int n = PipStages.Length;
            _stagePips = new Image[n];
            float pad = 0.05f;
            float span = (0.96f - 0.04f) - pad * (n - 1);
            float w = span / n;

            for (int i = 0; i < n; i++)
            {
                float x0 = 0.04f + i * (w + pad);
                float x1 = x0 + w;
                _stagePips[i] = CreateImage($"Pip{i}", parent,
                    new Vector2(x0, yMin), new Vector2(x1, yMax), TrackDim);
                CreateText($"PipLabel{i}", parent, new Vector2(x0, yMin), new Vector2(x1, yMax),
                    13, FontStyle.Bold, TextAnchor.MiddleCenter, new Color(0.05f, 0.06f, 0.07f))
                    .text = PipLabels[i];
            }
        }

        private void ApplyAlwaysOnTop()
        {
            Shader shader = Resources.Load<Shader>("Facilitator/FacilitatorUIAlwaysOnTop")
                         ?? Shader.Find("Facilitator/UI Depth Aware");
            if (shader == null) return;   // fall back to default UI material (may be occluded by hands)

            _alwaysOnTop = new Material(shader) { name = "VeinTrainer UI Depth Aware (Runtime)" };
            foreach (var g in _panelRoot.GetComponentsInChildren<Graphic>(true))
                g.material = _alwaysOnTop;
        }

        // ── Helpers ─────────────────────────────────────────────────────────────────────

        private void SelfHeal()
        {
            if (_manager == null)        _manager        = FindFirstObjectByType<ArmDetectionManager>();
            if (_evaluator == null)      _evaluator      = FindFirstObjectByType<InjectionSequenceEvaluator>();
            if (_simulatedNeedle == null) _simulatedNeedle = FindFirstObjectByType<SimulatedNeedleProvider>();
            if (_angleEstimator == null) _angleEstimator = FindFirstObjectByType<NeedleAngleEstimator>();
        }

        private void SetPanelVisible(bool visible)
        {
            if (_panelRoot != null) _panelRoot.SetActive(visible);
            if (visible) _placed = false;   // re-anchor cleanly next time it shows
        }

        private static void SetText(Text t, string value, Color color)
        {
            if (t == null) return;
            t.text = value;
            t.color = color;
        }

        private static string ColorHex(Color c) => ColorUtility.ToHtmlStringRGB(c);

        private static Image CreateImage(string name, Transform parent, Vector2 min, Vector2 max, Color color)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = min; rt.anchorMax = max;
            rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
            var img = go.GetComponent<Image>();
            img.color = color;
            img.raycastTarget = false;
            return img;
        }

        private static Text CreateText(string name, Transform parent, Vector2 min, Vector2 max,
            int fontSize, FontStyle style, TextAnchor align, Color color)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Text));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = min; rt.anchorMax = max;
            rt.offsetMin = new Vector2(6f, 2f); rt.offsetMax = new Vector2(-6f, -2f);
            var t = go.GetComponent<Text>();
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")
                  ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
            t.fontSize = fontSize;
            t.fontStyle = style;
            t.alignment = align;
            t.color = color;
            t.supportRichText = true;
            t.horizontalOverflow = HorizontalWrapMode.Wrap;
            t.verticalOverflow = VerticalWrapMode.Truncate;
            t.raycastTarget = false;
            return t;
        }
    }

    /// <summary>
    /// Auto-adds the VeinTrainerHUD to any scene that runs arm detection AND has the vein
    /// injection evaluator (i.e. the vein test rig is set up), so it appears without manual
    /// wiring — same pattern as ArmLockButtonBootstrap.
    /// </summary>
    public static class VeinTrainerHUDBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Create()
        {
            if (Object.FindFirstObjectByType<ArmDetectionManager>() == null) return;
            if (Object.FindFirstObjectByType<InjectionSequenceEvaluator>() == null) return;
            if (Object.FindFirstObjectByType<VeinTrainerHUD>() != null) return;

            var go = new GameObject("VeinTrainerHUD");
            go.AddComponent<VeinTrainerHUD>();
        }
    }
}
