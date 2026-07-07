using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;

namespace ARArmDetection
{
    /// <summary>
    /// World-space UNLOCK button for the arm target lock.
    ///
    /// Once ArmDetectionManager locks onto an arm it deliberately ignores every
    /// other detection, so if it grabbed the wrong target the only way out is an
    /// explicit release — this button (or the controller B button) calls
    /// ArmDetectionManager.Unlock() so a new target can be acquired.
    ///
    /// Groups itself below the ARM DETECTION status panel when one exists,
    /// otherwise floats in front of the camera.
    /// </summary>
    public class ArmLockButton : MonoBehaviour
    {
        [SerializeField] private ArmDetectionManager _manager;
        [Tooltip("Vertical offset (m) below the ARM DETECTION status panel.")]
        [SerializeField] private float _belowStatusPanelMeters = 0.34f;
        [Tooltip("Also release the lock with the controller B button.")]
        [SerializeField] private bool _unlockWithControllerB = true;

        [Header("Fallback placement (no status panel in scene)")]
        [SerializeField] private float _distanceMeters = 1.4f;
        [SerializeField] private float _heightInViewMeters = 0.21f;
        [SerializeField] private float _rightInViewMeters = 0.48f;

        private Image _background;
        private Text _label;
        private DetectionModeButton _statusPanel;
        private bool _grouped;

        private static readonly Color LockedColor    = new Color(0.72f, 0.16f, 0.16f, 0.92f);
        private static readonly Color SearchingColor = new Color(0.24f, 0.27f, 0.31f, 0.92f);

        private void Awake()
        {
            if (_manager == null) _manager = FindFirstObjectByType<ArmDetectionManager>();
            BuildUI();
            EnsureEventSystem();
        }

        private void Update()
        {
            if (_unlockWithControllerB && OVRInput.GetDown(OVRInput.RawButton.B))
                ReleaseLock();
            RefreshVisuals();
        }

        private void LateUpdate() => PlacePanel();

        /// <summary>Wired to the button; releases the manager's target lock.</summary>
        public void ReleaseLock()
        {
            if (_manager == null) return;
            _manager.Unlock();
        }

        private void RefreshVisuals()
        {
            bool locked = _manager != null && _manager.IsLocked;
            if (_background != null)
                _background.color = locked ? LockedColor : SearchingColor;

            if (_label != null)
            {
                string status = _manager != null ? _manager.LockStatus : "Manager not assigned";
                _label.text = locked
                    ? $"<b>UNLOCK ARM</b>\n<size=13>{status} — tap or press B</size>"
                    : $"<b>ARM LOCK</b>\n<size=13>{status}</size>";
            }
        }

        private void PlacePanel()
        {
            if (_statusPanel == null)
            {
                _statusPanel = FindFirstObjectByType<DetectionModeButton>();
                _grouped = false;
            }

            if (_statusPanel != null)
            {
                Transform statusTransform = _statusPanel.transform;
                if (!_grouped || transform.parent != statusTransform)
                {
                    transform.SetParent(statusTransform, false);
                    _grouped = true;
                }

                transform.localPosition = Vector3.down * _belowStatusPanelMeters;
                transform.localRotation = Quaternion.identity;
                transform.localScale = Vector3.one;
                return;
            }

            var cam = Camera.main;
            if (cam == null) return;
            if (transform.parent != null) transform.SetParent(null, true);
            _grouped = false;

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
            var canvasGO = new GameObject("ArmLockCanvas");
            canvasGO.transform.SetParent(transform, false);

            var canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            canvasGO.AddComponent<CanvasScaler>();
            canvasGO.AddComponent<GraphicRaycaster>();
            canvasGO.AddComponent<TrackedDeviceGraphicRaycaster>();

            var rt = canvasGO.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(260, 96);
            rt.localPosition = Vector3.zero;
            rt.localRotation = Quaternion.identity;
            rt.localScale = Vector3.one * 0.003f;

            var buttonGO = new GameObject("UnlockButton", typeof(RectTransform), typeof(Image), typeof(Button));
            buttonGO.transform.SetParent(canvasGO.transform, false);
            var btnRT = (RectTransform)buttonGO.transform;
            btnRT.anchorMin = Vector2.zero;
            btnRT.anchorMax = Vector2.one;
            btnRT.offsetMin = Vector2.zero;
            btnRT.offsetMax = Vector2.zero;

            _background = buttonGO.GetComponent<Image>();
            _background.color = SearchingColor;
            var button = buttonGO.GetComponent<Button>();
            button.targetGraphic = _background;
            button.onClick.AddListener(ReleaseLock);
            var colors = button.colors;
            colors.highlightedColor = new Color(0.95f, 0.35f, 0.30f, 1f);
            colors.pressedColor = new Color(1f, 0.55f, 0.45f, 1f);
            colors.selectedColor = colors.highlightedColor;
            button.colors = colors;

            var lblGO = new GameObject("Label", typeof(RectTransform), typeof(Text));
            lblGO.transform.SetParent(buttonGO.transform, false);
            _label = lblGO.GetComponent<Text>();
            _label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _label.fontSize = 18;
            _label.color = Color.white;
            _label.alignment = TextAnchor.MiddleCenter;
            _label.supportRichText = true;
            _label.raycastTarget = false;
            var lblRT = (RectTransform)lblGO.transform;
            lblRT.anchorMin = Vector2.zero;
            lblRT.anchorMax = Vector2.one;
            lblRT.offsetMin = new Vector2(8, 8);
            lblRT.offsetMax = new Vector2(-8, -8);

            RefreshVisuals();
        }

        private static void EnsureEventSystem()
        {
            if (EventSystem.current != null) return;
            var go = new GameObject("XR EventSystem", typeof(EventSystem), typeof(XRUIInputModule));
            DontDestroyOnLoad(go);
        }
    }

    /// <summary>
    /// Auto-creates the unlock button in any scene that runs arm detection, so scenes
    /// built before the target-lock feature get it without re-running the editor setup
    /// (same pattern as FacilitatorModeBootstrap).
    /// </summary>
    public static class ArmLockButtonBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void CreateForArmDetectionScene()
        {
            if (Object.FindFirstObjectByType<ArmDetectionManager>() == null) return;
            if (Object.FindFirstObjectByType<ArmLockButton>() != null) return;

            var go = new GameObject("ArmLockButton");
            go.AddComponent<ArmLockButton>();
        }
    }
}
