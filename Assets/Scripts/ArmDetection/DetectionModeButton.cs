using UnityEngine;
using UnityEngine.UI;

namespace ARArmDetection
{
    /// <summary>
    /// Floating world-space status panel for the fixed arm-detection mode.
    /// The project no longer switches between normal and arm-only modes.
    /// </summary>
    public class DetectionModeButton : MonoBehaviour
    {
        [SerializeField] private ArmDetectionManager _manager;
        [Tooltip("How far in front of the camera the panel floats.")]
        [SerializeField] private float _distanceMeters = 1.4f;
        [Tooltip("How high in the user's view the panel sits.")]
        [SerializeField] private float _heightInViewMeters = 0.55f;
        [Tooltip("How far to the right of the user's view the grouped panels sit.")]
        [SerializeField] private float _rightInViewMeters = 0.48f;

        private Image _background;
        private Text _label;
        private string _lastStatus;
        private float _nextTextRefresh;

        private static readonly Color PanelColor = new Color(0.90f, 0.45f, 0.05f, 0.92f);
        private const float TextRefreshInterval = 0.25f;

        private void Awake() => BuildUI();

        private void Update()
        {
            FollowCamera();

            // Text updates are throttled and change-gated: a fresh string every frame both
            // allocates and forces a Canvas rebuild. The background colour is constant and
            // set once in BuildUI.
            if (Time.unscaledTime < _nextTextRefresh) return;
            _nextTextRefresh = Time.unscaledTime + TextRefreshInterval;
            RefreshVisuals();
        }

        private void RefreshVisuals()
        {
            if (_label == null) return;

            string status = _manager != null ? _manager.ManagerStatus : "Manager not assigned";
            if (status == _lastStatus) return;
            _lastStatus = status;
            _label.text = $"<b>ARM DETECTION</b>\n<size=14>{status}</size>";
        }

        private void FollowCamera()
        {
            var cam = Camera.main;
            if (cam == null) return;

            Vector3 target = cam.transform.position
                + cam.transform.forward * _distanceMeters
                + cam.transform.up * _heightInViewMeters
                + cam.transform.right * _rightInViewMeters;
            transform.position = target;

            Vector3 toCamera = cam.transform.position - target;
            if (toCamera.sqrMagnitude > 0.001f)
                transform.rotation = Quaternion.LookRotation(-toCamera.normalized, cam.transform.up);
        }

        private void BuildUI()
        {
            var canvasGO = new GameObject("ArmDetectionStatusCanvas");
            canvasGO.transform.SetParent(transform, false);

            var canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            canvasGO.AddComponent<CanvasScaler>();
            canvasGO.AddComponent<GraphicRaycaster>();

            var rt = canvasGO.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(260, 96);
            rt.localPosition = Vector3.zero;
            rt.localRotation = Quaternion.identity;
            rt.localScale = Vector3.one * 0.003f;

            var bgGO = new GameObject("Background");
            bgGO.transform.SetParent(canvasGO.transform, false);
            _background = bgGO.AddComponent<Image>();
            _background.color = PanelColor;
            var bgRT = bgGO.GetComponent<RectTransform>();
            bgRT.anchorMin = Vector2.zero;
            bgRT.anchorMax = Vector2.one;
            bgRT.offsetMin = Vector2.zero;
            bgRT.offsetMax = Vector2.zero;

            var lblGO = new GameObject("Label");
            lblGO.transform.SetParent(canvasGO.transform, false);
            _label = lblGO.AddComponent<Text>();
            _label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")
                       ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
            _label.fontSize = 18;
            _label.color = Color.white;
            _label.alignment = TextAnchor.MiddleCenter;
            _label.supportRichText = true;
            var lblRT = lblGO.GetComponent<RectTransform>();
            lblRT.anchorMin = Vector2.zero;
            lblRT.anchorMax = Vector2.one;
            lblRT.offsetMin = new Vector2(8, 8);
            lblRT.offsetMax = new Vector2(-8, -8);

            RefreshVisuals();
        }
    }
}
