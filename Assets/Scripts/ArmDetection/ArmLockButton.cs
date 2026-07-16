using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;

namespace ARArmDetection
{
    /// <summary>
    /// Unlock control for the arm target lock.
    ///
    /// Once ArmDetectionManager locks onto an arm it deliberately ignores every
    /// other detection, so if it grabbed the wrong target the only way out is an
    /// explicit release — the hand pinch-hold gesture, the controller B button,
    /// or (when _showPanel is on) a world-space UNLOCK ARM button all call
    /// ArmDetectionManager.Unlock() so a new target can be acquired.
    ///
    /// The visible panel is OFF by default to keep the trainee's view clean; the
    /// gesture and controller unlock work without it. Tick _showPanel to bring
    /// the tappable button (and its lock-status readout) back.
    ///
    /// HAND-TRACKING UNLOCK: pinch the thumb against _pinchFinger (middle by
    /// default) on either tracked hand and hold for _pinchHoldSeconds. Middle is
    /// deliberate enough that ordinary index-pinch UI interaction and holding the
    /// needle don't trigger it by accident.
    ///
    /// When shown, the panel groups itself below the ARM DETECTION status panel
    /// when one exists, otherwise floats in front of the camera.
    /// </summary>
    public class ArmLockButton : MonoBehaviour
    {
        [SerializeField] private ArmDetectionManager _manager;
        [Tooltip("Build and show the world-space UNLOCK ARM panel. Off keeps the view clean — " +
                 "the pinch-hold gesture and controller B unlock still work without it.")]
        [SerializeField] private bool _showPanel = false;
        [Tooltip("Vertical offset (m) below the ARM DETECTION status panel.")]
        [SerializeField] private float _belowStatusPanelMeters = 0.34f;
        [Tooltip("Also release the lock with the controller B button (harmless when using hands).")]
        [SerializeField] private bool _unlockWithControllerB = true;

        [Header("Hand-gesture unlock (hand tracking)")]
        [Tooltip("Release the lock by pinching thumb + _pinchFinger on either hand and holding.")]
        [SerializeField] private bool _unlockWithHandPinch = true;
        [Tooltip("Which finger must pinch against the thumb. Middle (default) avoids clashing " +
                 "with index pinches used for UI and with the needle grip.")]
        [SerializeField] private OVRHand.HandFinger _pinchFinger = OVRHand.HandFinger.Middle;
        [Tooltip("How long the pinch must be held (seconds) before the lock releases. " +
                 "Prevents accidental unlocks from momentary pinches.")]
        [SerializeField] private float _pinchHoldSeconds = 1.0f;
        [Tooltip("Minimum pinch strength (0..1) that counts as pinching.")]
        [SerializeField, Range(0f, 1f)] private float _pinchStrengthThreshold = 0.8f;

        [Header("Fallback placement (no status panel in scene)")]
        [SerializeField] private float _distanceMeters = 1.4f;
        [SerializeField] private float _heightInViewMeters = 0.21f;
        [SerializeField] private float _rightInViewMeters = 0.48f;

        private Image _background;
        private Text _label;
        private DetectionModeButton _statusPanel;
        private bool _grouped;
        private bool _lastLocked;
        private string _lastStatus;
        private float _nextTextRefresh;
        private OVRHand[] _hands;
        private float _nextHandSearchTime;
        private float _pinchHeldSeconds;
        private bool _pinchUnlockFired;

        private static readonly Color LockedColor    = new Color(0.72f, 0.16f, 0.16f, 0.92f);
        private static readonly Color SearchingColor = new Color(0.24f, 0.27f, 0.31f, 0.92f);
        private const float TextRefreshInterval = 0.25f;

        private void Awake()
        {
            if (_manager == null) _manager = FindFirstObjectByType<ArmDetectionManager>();
            if (_showPanel) BuildUI();
            // The event system is still needed by other world-space UI (e.g. VeinTrainerHUD),
            // so ensure it exists even when this panel is hidden.
            EnsureEventSystem();
        }

        private void Update()
        {
            if (_unlockWithControllerB && OVRInput.GetDown(OVRInput.RawButton.B))
                ReleaseLock();

            UpdateHandPinchUnlock();

            if (!_showPanel) return;   // headless: unlock inputs only, no label to refresh

            // Throttled: the lock status string changes near-continuously while holding or
            // refining, and rebuilding the label every frame allocates + rebuilds the Canvas.
            if (Time.unscaledTime < _nextTextRefresh) return;
            _nextTextRefresh = Time.unscaledTime + TextRefreshInterval;
            RefreshVisuals();
        }

        /// <summary>
        /// Pinch-and-hold unlock for hand tracking: thumb + _pinchFinger held for
        /// _pinchHoldSeconds on either tracked hand releases the lock. Requires the
        /// pinch to be released before it can fire again.
        /// </summary>
        private void UpdateHandPinchUnlock()
        {
            if (!_unlockWithHandPinch) return;

            bool locked = _manager != null && _manager.IsLocked;
            if (!locked)
            {
                _pinchHeldSeconds = 0f;
                _pinchUnlockFired = false;
                return;
            }

            // OVRHand instances can spawn after this component (hand-tracking building
            // blocks initialise asynchronously), so re-scan occasionally until found.
            if ((_hands == null || _hands.Length == 0) && Time.unscaledTime >= _nextHandSearchTime)
            {
                _nextHandSearchTime = Time.unscaledTime + 2f;
                _hands = FindObjectsByType<OVRHand>(FindObjectsSortMode.None);
            }
            if (_hands == null || _hands.Length == 0) return;

            bool pinching = false;
            foreach (var hand in _hands)
            {
                if (hand == null || !hand.IsTracked) continue;
                if (hand.GetFingerIsPinching(_pinchFinger) &&
                    hand.GetFingerPinchStrength(_pinchFinger) >= _pinchStrengthThreshold)
                {
                    pinching = true;
                    break;
                }
            }

            if (!pinching)
            {
                _pinchHeldSeconds = 0f;
                _pinchUnlockFired = false;
                return;
            }

            if (_pinchUnlockFired) return;

            _pinchHeldSeconds += Time.deltaTime;
            if (_pinchHeldSeconds >= _pinchHoldSeconds)
            {
                _pinchUnlockFired = true;
                _pinchHeldSeconds = 0f;
                Debug.Log("[ArmLockButton] Pinch-hold gesture released the arm lock.");
                ReleaseLock();
            }
        }

        /// <summary>0..1 progress of the current pinch-hold, for the button label.</summary>
        private float PinchProgress =>
            _pinchHoldSeconds <= 0f ? 0f : Mathf.Clamp01(_pinchHeldSeconds / _pinchHoldSeconds);

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
            string status = _manager != null ? _manager.LockStatus : "Manager not assigned";

            // Fold the pinch progress into the change gate (rounded so the label only
            // rebuilds a few times during the hold, not every refresh tick).
            string hint;
            if (!locked)
            {
                hint = null;
            }
            else if (_pinchHeldSeconds > 0f)
            {
                hint = $"unlocking… {Mathf.RoundToInt(PinchProgress * 100f)}%  (keep pinching)";
            }
            else
            {
                hint = _unlockWithHandPinch
                    ? $"pinch thumb+{_pinchFinger.ToString().ToLowerInvariant()} & hold, or tap"
                    : "tap to unlock";
            }

            string composed = locked ? $"{status}|{hint}" : status;
            if (locked == _lastLocked && composed == _lastStatus) return;
            _lastLocked = locked;
            _lastStatus = composed;

            if (_background != null)
                _background.color = locked ? LockedColor : SearchingColor;

            if (_label != null)
            {
                _label.text = locked
                    ? $"<b>UNLOCK ARM</b>\n<size=13>{status}\n{hint}</size>"
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
