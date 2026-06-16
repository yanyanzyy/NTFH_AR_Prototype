using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;

namespace ARArmDetection
{
    /// <summary>
    /// Floating world-space button that switches between Normal and Arm-Only detection modes.
    ///
    /// INTERACTION
    /// -----------
    /// Pinch your index finger + thumb on either hand while looking at the button.
    /// The button changes colour to confirm the new mode:
    ///   Blue  → Normal mode   (full-body YOLO, arm-only fallback if no person found)
    ///   Orange → Arm-Only mode (skips full-body, goes straight to arm-keypoint scan)
    ///
    /// PLACEMENT
    /// ---------
    /// The button floats _distanceMeters in front of the user (projected horizontally)
    /// at _heightAboveCamera metres above the camera. It billboards to always face the
    /// user, so it remains readable regardless of head pitch.
    /// </summary>
    public class DetectionModeButton : MonoBehaviour
    {
        [SerializeField] private ArmDetectionManager _manager;
        [Tooltip("How far in front of the camera the button floats.")]
        [SerializeField] private float _distanceMeters  = 1.4f;
        [Tooltip("How many metres above the camera origin the button sits (world-space Y).")]
        [SerializeField] private float _heightAboveCamera = 1.0f;

        // ── Internal references ────────────────────────────────────────────────────────
        private Image _background;
        private Text  _label;
        private float _cooldown;

        // OVRHand components — looked up lazily because they're added at runtime
        // by the Hand Tracking building block.
        private OVRHand[] _hands;
        private bool[]    _wasPinching;
        private bool      _wasXrPressed;

        // ── Colours ────────────────────────────────────────────────────────────────────
        private static readonly Color ColNormal  = new Color(0.15f, 0.45f, 0.90f, 0.92f); // blue
        private static readonly Color ColArmOnly = new Color(0.90f, 0.45f, 0.05f, 0.92f); // orange

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void Awake() => BuildUI();

        private void Update()
        {
            FollowCamera();

            if (_cooldown > 0f) { _cooldown -= Time.deltaTime; return; }

            if (DetectPinch()) Toggle();
        }

        // ── Interaction ────────────────────────────────────────────────────────────────

        /// <summary>
        /// True on the rising edge of a pinch (index + thumb) on either hand, or when the
        /// user presses the A button / right index trigger on a controller. We query
        /// OVRHand.GetFingerIsPinching directly because the Meta MRUK 201 Hand-Tracking
        /// building block doesn't always publish pinch state through OVRInput.Button.One.
        /// </summary>
        private bool DetectPinch()
        {
            // Lazy-find OVRHand components — they're instantiated at runtime by the
            // [BuildingBlock] Hand Tracking children, so they may not exist in Awake().
            if (_hands == null || _hands.Length == 0)
            {
                _hands = FindObjectsByType<OVRHand>(FindObjectsSortMode.None);
                _wasPinching = new bool[_hands.Length];
            }

            // Edge-triggered pinch from either hand.
            for (int i = 0; i < _hands.Length; i++)
            {
                var hand = _hands[i];
                if (hand == null || !hand.IsTracked) { _wasPinching[i] = false; continue; }

                bool pinchNow = hand.GetFingerIsPinching(OVRHand.HandFinger.Index);
                bool rising   = pinchNow && !_wasPinching[i];
                _wasPinching[i] = pinchNow;
                if (rising) return true;
            }

            // Controller fallbacks so the button is testable without hand tracking.
            return OVRInput.GetDown(OVRInput.Button.One)                    // A button (any controller)
                || OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger)    // right trigger
                || OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger)  // left trigger
                || DetectXrControllerButton();
        }

        private bool DetectXrControllerButton()
        {
            bool pressed = false;
            var right = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
            var left = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);

            pressed |= IsPressed(right, CommonUsages.primaryButton);
            pressed |= IsPressed(right, CommonUsages.triggerButton);
            pressed |= IsPressed(left, CommonUsages.primaryButton);
            pressed |= IsPressed(left, CommonUsages.triggerButton);

            bool rising = pressed && !_wasXrPressed;
            _wasXrPressed = pressed;
            return rising;
        }

        private static bool IsPressed(InputDevice device, InputFeatureUsage<bool> usage)
        {
            return device.isValid && device.TryGetFeatureValue(usage, out bool pressed) && pressed;
        }

        private void Toggle()
        {
            bool newMode = !(_manager != null && _manager.IsArmOnlyMode);
            _manager?.SetArmOnlyMode(newMode);
            Debug.Log($"[DetectionModeButton] Toggled mode -> {(newMode ? "ARM-ONLY" : "NORMAL")}");
            RefreshVisuals();
            _cooldown = 1.2f; // prevent double-toggle
        }

        // ── Visuals ────────────────────────────────────────────────────────────────────

        private void RefreshVisuals()
        {
            bool armOnly = _manager != null && _manager.IsArmOnlyMode;

            if (_background != null)
                _background.color = armOnly ? ColArmOnly : ColNormal;

            if (_label != null)
            {
                _label.text = armOnly
                    ? "<b>ARM-ONLY MODE</b>\n<size=14>Mannequin / isolated arm</size>\n\n<size=11>Pinch to switch</size>"
                    : "<b>NORMAL MODE</b>\n<size=14>Full-body detection</size>\n\n<size=11>Pinch to switch</size>";
            }
        }

        private void FollowCamera()
        {
            var cam = Camera.main;
            if (cam == null) return;

            // Project the camera's forward onto the horizontal plane so the button
            // doesn't drift forward/back when the user looks up or down.
            Vector3 flatForward = cam.transform.forward;
            flatForward.y = 0f;
            if (flatForward.sqrMagnitude < 0.001f) flatForward = Vector3.forward;
            flatForward.Normalize();

            // Place the button in front of the user at a fixed world-space height.
            Vector3 target = cam.transform.position
                + flatForward          * _distanceMeters
                + Vector3.up           * _heightAboveCamera;
            transform.position = target;

            // Billboard: always face the camera so the text is readable from any angle.
            Vector3 toCamera = cam.transform.position - target;
            if (toCamera.sqrMagnitude > 0.001f)
                transform.rotation = Quaternion.LookRotation(-toCamera.normalized, Vector3.up);
        }

        // ── UI construction ────────────────────────────────────────────────────────────

        private void BuildUI()
        {
            var canvasGO = new GameObject("ModeButtonCanvas");
            canvasGO.transform.SetParent(transform, false);

            var canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode  = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            canvasGO.AddComponent<CanvasScaler>();
            canvasGO.AddComponent<GraphicRaycaster>();

            var rt = canvasGO.GetComponent<RectTransform>();
            rt.sizeDelta     = new Vector2(240, 120);
            rt.localPosition = Vector3.zero;
            rt.localRotation = Quaternion.identity;
            rt.localScale    = Vector3.one * 0.003f; // 3 mm per canvas unit → ~72 cm wide

            // ── Background ─────────────────────────────────────────────────────────────
            var bgGO = new GameObject("Background");
            bgGO.transform.SetParent(canvasGO.transform, false);
            _background       = bgGO.AddComponent<Image>();
            _background.color = ColNormal;
            var bgRT = bgGO.GetComponent<RectTransform>();
            bgRT.anchorMin = Vector2.zero;
            bgRT.anchorMax = Vector2.one;
            bgRT.offsetMin = Vector2.zero;
            bgRT.offsetMax = Vector2.zero;

            // ── Rounded corner hint (thin white border) ─────────────────────────────────
            var borderGO = new GameObject("Border");
            borderGO.transform.SetParent(canvasGO.transform, false);
            var border = borderGO.AddComponent<Image>();
            border.color = new Color(1, 1, 1, 0.25f);
            var borderRT = borderGO.GetComponent<RectTransform>();
            borderRT.anchorMin = Vector2.zero;
            borderRT.anchorMax = Vector2.one;
            borderRT.offsetMin = new Vector2(-2, -2);
            borderRT.offsetMax = new Vector2( 2,  2);

            // ── Label ──────────────────────────────────────────────────────────────────
            var lblGO = new GameObject("Label");
            lblGO.transform.SetParent(canvasGO.transform, false);
            _label = lblGO.AddComponent<Text>();
            _label.font     = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")
                           ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
            _label.fontSize        = 19;
            _label.color           = Color.white;
            _label.alignment       = TextAnchor.MiddleCenter;
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
